from collections import defaultdict
import numpy as np
import os
import sys

from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.VLM.qwen import Qwen
from src.VLM.policy import MacroActionPolicy
import src.MCTS.utils as utils
DISCOUNT_FACTOR = 0.95

"""
To be determined
"""


class StateNode:
    def __init__(self, reward=0, done=False):
        self.ob = None
        self.state = {} # current state of the vehicle, speed, lane, acceleration, and steering angle
        self.env_state = None
        self.prev_action = None
        self.id = None
        self.valid_actions = ["overtaking", "keeping_lane", "turning_left", 
                             "turning_right", "left_change", "right_change", "brake"]
        self.history = []
        self.navi_info = None
        self.camera_images = None
        self.observation = None

        self.parent = None
        self.parent_action_id = None
        self.best_action_node = None
        

        self.N = 0
        self.children = []
        self.children_probs = []
        self.reward = reward/(1-DISCOUNT_FACTOR)
        self.score = 0
        self.done = done
        self.use_llm = False


class ActionNode:
    def __init__(self, action):
        self.action = action
        self.N = 0
        self.Q = 0
        self.Q_hat = 0
        self.Rs = []
        self.children = None
        self.children_id = None


class MCTSAgent:
    def __init__(self, model, env=None, args=None, name='MCTS', uct_type='PUCT', use_llm=True):
        """
        Initialize the MCTS agent for high-level planning
        
        Args:
            model: Vision Language Model for policy generation
            env: Environment simulator (MCTSEnv instance)
            args: Configuration arguments
            name: Agent name
            uct_type: UCT algorithm type ('UCT' or 'PUCT')
            use_llm: Whether to use language model for policy generation
        """
        self.env = env
        self.name = name
        self.best_action_node = None
        self.uct_type = uct_type
        self.root = None
        
        # Set default args if none provided
        if args is None:
            from argparse import Namespace
            args = Namespace(
                seed=42,
                round=0,
                exploration_constant=2.0,
                max_depth=5,
                discount_factor=0.95,
                simulation_num=10
            )
            
        self.seed = args.seed
        self.round = args.round
        self.exploration_constant = args.exploration_constant
        self.max_depth = args.max_depth
        self.discount_factor = args.discount_factor
        self.simulation_num = args.simulation_num
        
        # Temperature for action selection
        self.action_selection_temp = 0.1 / (self.round + 1)
        
        # VLM policy
        self.use_llm = use_llm
        if use_llm and model is not None:
            self.vlm_policy = MacroActionPolicy(model)
            
        # State dictionary to avoid duplicating states
        self.state_dict = {}
        
        # Store observation for Highway Environment
        self.observation = None
    
    def search(self, ob, history, cur_depth, valid_actions, done, camera_image=None, navi_info=None):
        """
        Search for the best action using MCTS
        
        Args:
            ob: Current observation
            history: Action history
            cur_depth: Current search depth
            valid_actions: List of valid actions
            done: Whether the episode is done
            camera_image: Camera images for VLM policy
            navi_info: Navigation information
            
        Returns:
            best_action: The selected best action
        """
        # Copy history to avoid modifying the original
        init_history = history.copy()
        
        # Save observation for Highway Environment
        self.observation = ob
        
        # Build initial state
        self.root = self.build_state(
            ob=ob, 
            history=history, 
            valid_actions=valid_actions, 
            done=done, 
            reward=0,
            navi_info=navi_info, 
            camera_image=camera_image, 
            observation=ob,  # Pass text observation
            use_llm=self.use_llm
        )
        
        # Run simulations
        for _ in tqdm(range(self.simulation_num)):
            # Reset environment for new simulation
            if self.env:
                self.env.reset()
                self.env.history = init_history.copy()
                _, root = self.simulate(self.root, 0)
                self.root = root
            else:
                # If no environment is provided, skip simulation
                print("Warning: No environment provided, skipping simulation")
                break
                
        # Select best action using greedy selection (no exploration)
        best_action_node_idx = self.greedy_action_node(self.root, 0, if_print=True)
        best_action_node = self.root.children[best_action_node_idx]
        self.root.best_action_node = best_action_node
        
        return self.root.best_action_node.action
        
    def build_state(self, ob, history, valid_actions, done, reward=0, navi_info=None, 
                    camera_image=None, observation=None, prev_action='<s>', use_llm=False, state=None):
        """
        Build a state node with the given information
        
        Args:
            ob: Observation data structure
            history: Action history
            valid_actions: List of valid actions
            done: Whether the episode is done
            reward: Reward value
            navi_info: Navigation information
            camera_image: Camera images for CARLA
            observation: Text observation for Highway Environment
            prev_action: Previous action
            use_llm: Whether to use LLM for action probabilities
            state: State information (optional)
            
        Returns:
            state_node: Constructed state node
        """
        state_node = StateNode()
        state_node.ob = ob
        state_node.state = state or {}
        state_node.done = done
        state_node.reward = reward
        state_node.prev_action = prev_action
        state_node.history = history
        state_node.camera_images = camera_image
        state_node.observation = observation  # Store the text observation
        state_node.id = self.state_id(history)
        state_node.valid_actions = valid_actions
        state_node.navi_info = navi_info
        state_node.use_llm = use_llm
        
        # Set action probabilities
        if use_llm and hasattr(self, 'vlm_policy'):
            # For Highway Environment, use the text observation
            state_description = observation if observation else str(ob)
            
            # Extract history description
            if history and isinstance(history[0], list):
                history_description = "\n".join([str(item) for sublist in history for item in sublist]) if history else "No previous actions"
            else:
                history_description = "\n".join(map(str, history)) if history else "No previous actions"
            
            # Get navigation info as string
            navigation_info = str(navi_info) if navi_info else "No navigation information"
            
            # Get probabilities from VLM policy - emphasize observation for Highway Environment
            state_node.children_probs = self.vlm_policy.calculate_probabilities(
                state_description, 
                history_description,
                valid_actions, 
                navigation_info, 
                observation=observation  # Pass text observation
            )
        else:
            # Uniform distribution if not using LLM
            state_node.children_probs = np.ones(len(valid_actions)) / len(valid_actions)
        
        # Store state in dictionary to avoid duplicates
        self.state_dict[state_node.id] = state_node
        
        # Create child action nodes
        for valid_action in valid_actions:
            if isinstance(valid_actions, dict):
                state_node.children.append(ActionNode(valid_actions[valid_action]))
            else:
                state_node.children.append(ActionNode(valid_action))
                
        return state_node

    @staticmethod
    def state_id(historys: list):
        print("History: ", historys) # List of
        def low2high_action(action):
            """
            Convert low-level action to high-level action
            high level action = ["overtaking", "keeping_lane", "left_change", "right_change"]
            """
            # action[0]: acceleration
            # action[1]: steering
            if action[0] > 0 and action[1] > 0:
                return "overtaking"
            elif action[0] > 0 and action[1] == 0:
                return "keeping_lane"
            elif action[0] < 0 and action[1] > 0:
                return "left_change"
            elif action[0] < 0 and action[1] < 0:
                return "right_change"
            else:
                return "keeping_lane"
        
        if len(historys) == 0:
            return 'None'
        
        action_history = []
        if isinstance(historys[0], list):
            for i in range(len(historys)):
                action_history.append(low2high_action(historys[i]))
            return ' '.join(action_history)
        else:
            return ' '.join(historys)
        
                
    def low2high_action(self, action):
        """
        Convert low-level action to high-level action
        high level action = ["overtaking", "keeping_lane", "left_change", "right_change"]
        """
        # action[0]: acceleration
        # action[1]: steering
        if action[0] > 0 and action[1] > 0:
            return "overtaking"
        elif action[0] > 0 and action[1] == 0:
            return "keeping_lane"
        elif action[0] < 0 and action[1] > 0:
            return "left_change"
        elif action[0] < 0 and action[1] < 0:
            return "right_change"
        else:
            return "keeping_lane"
            

    def simulate(self, state_node, depth):

        if state_node.done or depth == self.max_depth:
            return 0, state_node

        best_action_node_idx = self.greedy_action_node(state_node, self.exploration_constant)
        best_action_node = state_node.children[best_action_node_idx]
        rollout_next = False
        ob, reward, done, history, valid_actions = self.env.mcts_step(best_action_node.action)
        next_state_id = self.state_id(history)
        if next_state_id == best_action_node.children_id:
            next_state_node = best_action_node.children
            if next_state_node.use_llm == False:
                next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=best_action_node.action, use_llm=self.use_llm)
                next_state_node.parent = state_node
                rollout_next = True
        else: 
            next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=best_action_node.action, use_llm=self.use_llm)
            next_state_node.parent = state_node
            best_action_node.children = next_state_node
            best_action_node.children_id = next_state_node.id
            rollout_next = True


        if rollout_next:
            if self.use_llm:
                rollout_r = []
                for _ in range(1):
                    random_r = reward + self.discount_factor * self.rollout(next_state_node, depth+1)
                    rollout_r.append(random_r)  
                R = sum(rollout_r)/len(rollout_r)
            else:
                rollout_r = []
                for _ in range(1):
                    random_r = reward + self.discount_factor * self.rollout(next_state_node, depth+1)
                    rollout_r.append(random_r)  
                R = sum(rollout_r)/len(rollout_r)
        else:
            r, next_state_node = self.simulate(next_state_node, depth+1)
            R = reward + self.discount_factor * r

        state_node.N += 1
        best_action_node.N += 1
        best_action_node.children = next_state_node
        best_action_node.Rs.append(R)
        best_action_node.Q = np.sum(np.array(best_action_node.Rs) * utils.softmax(best_action_node.Rs, T=10))
        state_node.best_action_node = best_action_node       
        return R, state_node

    def max_visit_action_node(self, state_node):
        children_count = []

        for i in range(len(state_node.children)):
            child = state_node.children[i]
            children_count.append(child.N)

        children_count = children_count / np.max(children_count)
        count_based_probs = children_count ** (1/self.action_selection_temp) / (np.sum(children_count ** (1/self.action_selection_temp)))
        return np.random.choice(state_node.children, p=count_based_probs)

    def greedy_action_node(self, state_node, exploration_constant, if_print=False):
        best_value = -np.inf
        best_children = []
        best_children_prob = []
        for i in range(len(state_node.children)):
            child = state_node.children[i]
            # print(f"child node: {state_node.children}")
            assert len(state_node.children_probs) == len(state_node.children), print(state_node.children_probs)
            child_prob = state_node.children_probs[i]
            
            if exploration_constant == 0:
                ucb_value = child.Q

            elif self.uct_type == 'PUCT':
                ucb_value = child.Q + exploration_constant * child_prob * np.sqrt(state_node.N) / (child.N + 1)

            else:
                raise NotImplementedError

            if ucb_value == best_value:
                best_children.append(i)
                best_children_prob.append(child_prob)
            elif ucb_value > best_value:
                best_value = ucb_value
                best_children = [i]
                best_children_prob = [child_prob]
        if if_print:
            for c in state_node.children:
                if c.N > 0:
                    print(c.action, c.Q, c.N)
        best_children_prob = np.array(best_children_prob) / np.sum(best_children_prob)
        output_action_index = np.argmax(best_children_prob)
        return best_children[output_action_index]

    def rollout(self, state_node, depth):
        if state_node.done or depth == self.max_depth:
            return 0
        action_node = np.random.choice(state_node.children, 1)[0]
        action = action_node.action

        ob, reward, done, history, valid_actions = self.env.mcts_step(action)
        if done:
            print("Done!")
        next_state_id = self.state_id(history)


        if next_state_id == action_node.children_id:
            next_state_node = action_node.children
        else:
            # Pass both camera_image and observation text to the build_state method
            next_state_node = self.build_state(
                ob, 
                history, 
                valid_actions, 
                done, 
                reward=reward,
                prev_action=action,
                camera_image=state_node.camera_images,
                observation=ob
            )
            next_state_node.parent = state_node
            action_node.children = next_state_node
            action_node.children_id = next_state_node.id
        r = reward + self.discount_factor * self.rollout(next_state_node, depth+1)
        return r
    


if __name__ == "__main__":
    model = Qwen()
    print("Model loaded")
    agent = MCTSAgent(model)
    root_state = StateNode()
    best_action = agent.search(root_state, 10)
    print(best_action)