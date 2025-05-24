import math
import string
import sys
import os
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
from collections import deque
from copy import deepcopy
import random
import time
import gymnasium as gym
import highway_env
from highway_env.vehicle.kinematics import Vehicle
from gymnasium.wrappers import RecordVideo

# Fix: Replace relative imports with absolute imports
from src.VLM.belief import OtherVehicleAgent, Belief
from src.MCTS.utils import get_simulation_params, get_vehicle_parameters, highway_lanes_to_path, highway_to_numpy_transform
from src.MCTS.high_level_mcts import MCTSAgent as HighLevelMCTS
from src.MCTS.dqn import DQN, ReplayBuffer # Added DQN imports
import torch # Added for DQN
# from src.MCTS.tree import Tree as LowLevelMCTS
from src.HighwayEnv.goal_checker import GoalChecker
from src.VLM.policy import get_mid_level_action
from src.HighwayEnv.envScenario import EnvScenario
from src.HighwayEnv.stategraph import StateGraph, EnvironmentState


class MCTSEnv:
    def __init__(self, env_id='highway-v0', model=None, config=None, road_network=None):
        # Highway Environment connection
        self.env_id = env_id
        self.config = config or {}
        self._setup_default_config() # Initializes self.config if it's empty
        
        # Merge provided config with defaults, then apply to env
        env_gym_config = {**self.config, **config} if config else self.config
        self.config = env_gym_config # Update self.config to be the final merged one

        self.env = gym.make(self.env_id, render_mode="rgb_array")
        self.env.configure(self.config) # Configure gym env with final config
        self.video_path = "/home/ubuntu/sjh04/MCTS-based-VLM-Numerical-Prediction-Planner/video"
        # Ensure video_path directory exists
        os.makedirs(self.video_path, exist_ok=True)
        self.env = RecordVideo(self.env, video_folder=self.video_path, episode_trigger=lambda e: True)  # record all episodes
        self.env_scenario = None
        
        self.frames = self.config.get("simulation_frequency", 10) # Use .get for safety
        
        # Environment state management
        self.state_graph = StateGraph()
        self.agent_history = deque(maxlen=100)
        self.belief_states = {}
        self.agent_ids = None

        self.acc_target_range = self.config.get("action", {}).get("acceleration_range", [-3, 3])
        self.steering_target_range = self.config.get("action", {}).get("steering_range", [-np.pi/4, np.pi/4])
        
        # DQN related params from config or defaults
        self.acc_values = self.config.get("num_throttle_bins", 13)
        self.steering_values = self.config.get("num_steering_bins", 13)

        self.acc_coef = (self.acc_target_range[1] - self.acc_target_range[0]) / (self.acc_values - 1) if self.acc_values > 1 else 0
        self.steer_coef = (self.steering_target_range[1] - self.steering_target_range[0]) / (self.steering_values - 1) if self.steering_values > 1 else 0
        
        # Goal tracking
        self.current_goal = None
        self.goal_checker = GoalChecker()
        
        # Environment parameters
        self.low_level_action_continuous = [0.0, 0.0] 
        self.mid_level_action = None
        self.ego_vehicle = None
        self.ego_vehicle_id = 0
        self.sensors = {}
        self.camera_images = {} 
        self.timestep = 0
        self.road_network = road_network # or self.env.unwrapped.road (careful, unwrapped might not be ready)
        
        self.model = model
        self.belief = None
        self.belief_agent = None
        
        self.current_observation = None 
        self.current_numerical_observation = None 
        self.last_discrete_dqn_action = None 
        
        self.planning_horizon = 5.0 
        self.action_space = self._get_action_space()
        self.reward_weights = {
            'safety': 10.0, 'progress': 1.0, 'comfort': 0.5, 'efficiency': 1.0
        }
        
        self.high_level_mcts = None

        # DQN agent configuration
        # Default DQN config, can be overridden by self.config["dqn_config"]
        default_dqn_params = {
            "state_dim": int(np.prod(self.env.observation_space.shape)) if self.env.observation_space is not None else 60,
            "hidden_dim": 256,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "epsilon": 0.9, 
            "epsilon_decay": 0.995,
            "min_epsilon": 0.05,
            "target_update": 20, 
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "buffer_capacity": 50000,
            "batch_size": 128,
            "num_throttle_bins": self.acc_values,
            "num_steering_bins": self.steering_values,
            "p_follow_llm": 0.3 
        }
        
        # Merge default DQN params with those from self.config (e.g., passed from main.py)
        # self.config might contain "dqn_config" from main.py or individual dqn params
        passed_dqn_config = self.config.get("dqn_config", {}) # Get the dqn_config dict if present
        # Also allow overriding individual dqn params if they are directly in self.config
        for key in default_dqn_params:
            if key in self.config:
                passed_dqn_config[key] = self.config[key]

        self.dqn_config = {**default_dqn_params, **passed_dqn_config}

        # Ensure state_dim is correctly set after env is available
        if self.env.observation_space is not None:
            self.dqn_config["state_dim"] = int(np.prod(self.env.observation_space.shape))
        else: # Fallback if observation_space is somehow None after gym.make
             self.dqn_config["state_dim"] = self.config.get("dqn_state_dim", 60)


        self.dqn_agent = DQN(state_dim=self.dqn_config["state_dim"],
                             hidden_dim=self.dqn_config["hidden_dim"],
                             num_throttle_bins=self.dqn_config["num_throttle_bins"],
                             num_steering_bins=self.dqn_config["num_steering_bins"],
                             learning_rate=self.dqn_config["learning_rate"],
                             gamma=self.dqn_config["gamma"],
                             epsilon=self.dqn_config["epsilon"],
                             target_update=self.dqn_config["target_update"],
                             device=self.dqn_config["device"])
        self.dqn_agent_replay_buffer = ReplayBuffer(self.dqn_config["buffer_capacity"])
        
        self.dqn_sub_steps = self.config.get("DQN_SUB_STEPS", 10)
        self.use_llm_for_dqn_guidance = self.config.get("USE_LLM_FOR_DQN_GUIDANCE", False)

        self.high_to_low_action_mapping = {
            'overtaking': {'acceleration': [0.5, 1.0], 'steering': [-0.1, 0.1]},
            'keeping_lane': {'acceleration': [0.0, 0.5], 'steering': [-0.1, 0.1]},
            'turning_left': {'acceleration': [0.0, 0.3], 'steering': [-0.3, -0.1]},
            'turning_right': {'acceleration': [0.0, 0.3], 'steering': [0.1, 0.3]},
            'left_change': {'acceleration': [0.0, 0.5], 'steering': [-0.3, -0.1]},
            'right_change': {'acceleration': [0.0, 0.5], 'steering': [0.1, 0.3]},
            'brake': {'acceleration': [-0.5, -0.1], 'steering': [-0.1, 0.1]},
        }
        
        self.current_high_level_action = None
        self.high_level_action_completed = True 
        self.high_level_action_steps = 0
        self.max_steps_per_high_level_action = 160 

    def _setup_default_config(self):
        """Set up default configuration for Highway Environment if not already configured"""
        if not self.config: # Only set defaults if self.config is empty
            self.config = {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": ["presence", "x", "y", "vx", "vy", "heading"],
                    "absolute": True,
                    "normalize": False,
                    "order": "sorted"
                },
                "action": {
                    "type": "ContinuousAction",
                    "acceleration_range": [-3.0, 3.0], # Default, can be overridden
                    "steering_range": [-np.pi/4, np.pi/4], # Default
                },
                "lanes_count": 3,
                "vehicles_count": 15,
                "duration": 400, # Increased default duration for longer episodes
                "initial_spacing": 2,
                "collision_reward": -20, # More significant collision penalty
                "reward_speed_range": [20, 30],
                "simulation_frequency": 10,
                "policy_frequency": 10, # Corresponds to high-level policy
                "screen_width": 600,
                "screen_height": 150,
                "centering_position": [0.3, 0.5],
                "scaling": 5.5,
                "show_trajectories": True,
                "render_agent": True,
                "offscreen_rendering": False, # Set to True for headless
                "manual_control": False,
                "real_time_rendering": False,
            }

    def _get_action_space(self):
        return {
            'overtaking': {'acceleration': 0.8, 'steering': 0.0},
            'keeping_lane': {'acceleration': 0.3, 'steering': 0.0},
            'turning_left': {'acceleration': 0.2, 'steering': -0.3},
            'turning_right': {'acceleration': 0.2, 'steering': 0.3},
            'left_change': {'acceleration': 0.3, 'steering': -0.2},
            'right_change': {'acceleration': 0.3, 'steering': 0.2},
            'brake': {'acceleration': -0.3, 'steering': 0.0}
        }

    def connect_to_MCTS(self, high_level_agent, low_level_agent=None):
        self.high_level_mcts = high_level_agent
        high_level_params = get_simulation_params(
            action_space=self.action_space,
            planning_horizon=self.planning_horizon,
            reward_weights=self.reward_weights
        )
        if hasattr(self.high_level_mcts, 'configure'):
            self.high_level_mcts.configure(high_level_params)
        return self

    def reset(self, episode=None, step=None, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.env.observation_space is not None and hasattr(self.env.observation_space, 'shape'):
             self.current_numerical_observation = obs.flatten()
        else: # Fallback for non-standard observation spaces
            self.current_numerical_observation = np.array(obs).flatten()

        if self.road_network is None and hasattr(self.env.unwrapped, 'road'):
            self.road_network = self.env.unwrapped.road

        self.env_scenario = EnvScenario(self.env, self.env_id, seed=kwargs.get('seed', 42))
        if hasattr(self.env.unwrapped, 'set_record_video_wrapper'): # Some envs might not have this
            self.env.unwrapped.set_record_video_wrapper(self.env)

        self.ego_vehicle = self.env.unwrapped.vehicle
        self.ego_vehicle_id = id(self.ego_vehicle) % 1000
        
        self.timestep = 0
        self.agent_history.clear()
        initial_state = self._capture_current_state()
        self.state_graph.add_node(initial_state)
        self._update_belief_state(initial_state, episode=episode, step=step)
        
        self.low_level_action_continuous = [0.0, 0.0]
        self.last_discrete_dqn_action = None
        self.mid_level_action = None
        self.current_high_level_action = None
        self.high_level_action_completed = True
        self.high_level_action_steps = 0

        # Reset DQN epsilon at the start of an episode or based on external control
        # If args.dqn_epsilon_start is used, it should be set by the training script.
        # This is for inter-episode decay if not overridden.
        self.dqn_agent.epsilon = max(
            self.dqn_config["min_epsilon"], 
            self.dqn_agent.epsilon * self.dqn_config["epsilon_decay"]
        )
        
        observation_text = self._get_observation() 
        valid_actions = self.get_valid_actions()
        
        return observation_text, valid_actions

    def step(self, action_continuous, episode=None, step_num_low_level=None):
        print(f"Action taken: {action_continuous}")
        obs_numerical, reward, terminated, truncated, info = self.env.step(action_continuous)
        
        if self.env.observation_space is not None and hasattr(self.env.observation_space, 'shape'):
            next_numerical_observation = obs_numerical.flatten()
        else:
            next_numerical_observation = np.array(obs_numerical).flatten()


        if hasattr(self.env.unwrapped, 'vehicle') and self.env.unwrapped.vehicle:
             ego_position = self.env.unwrapped.vehicle.position
        else:
            ego_position = np.array([0.0, 0.0]) # Fallback

        # self.env.render() # Rendering can be slow, consider conditional rendering
        if hasattr(self.env.unwrapped, 'automatic_rendering_callback') and self.env.video_recorder:
            self.env.unwrapped.automatic_rendering_callback = self.env.video_recorder.capture_frame()

        self.timestep += 1
        done_flag = terminated or truncated

        if self.dqn_agent_replay_buffer is not None and \
           self.last_discrete_dqn_action is not None and \
           self.current_numerical_observation is not None:
            self.dqn_agent_replay_buffer.add(self.current_numerical_observation, 
                                             self.last_discrete_dqn_action, 
                                             reward, 
                                             next_numerical_observation, 
                                             done_flag)
        
        self.current_numerical_observation = next_numerical_observation

        if self.dqn_agent is not None and \
           self.dqn_agent_replay_buffer.size() >= self.dqn_config["batch_size"] and \
           self.timestep % self.dqn_config.get("train_frequency", 1) == 0: # Optional: train every N steps
            transitions_sample = self.dqn_agent_replay_buffer.sample(self.dqn_config["batch_size"])
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*transitions_sample)
            transition_dict = {
                'states': np.array(states_batch), 'actions': np.array(actions_batch), 
                'rewards': np.array(rewards_batch), 'next_states': np.array(next_states_batch),
                'dones': np.array(dones_batch)
            }
            self.dqn_agent.update(transition_dict)

        current_env_state = self._capture_current_state()
        self.state_graph.add_node(current_env_state)
        self._update_belief_state(current_env_state, episode=episode, step=step_num_low_level)
        
        observation_text = self._get_observation()
        valid_actions = self.get_valid_actions()

        # Store the high-level action name that was active during this low-level step
        hl_action_name_for_history = self.current_high_level_action if self.current_high_level_action else "initial_driving"
        self.agent_history.append((hl_action_name_for_history, observation_text, reward)) # Changed from action_continuous
        
        if self.current_high_level_action is not None:
            self.high_level_action_steps += 1
            self.high_level_action_completed = self._check_high_level_action_completed(
                self.current_high_level_action, obs_numerical, current_env_state
            )
        
        info_dict = {
            'timestep': self.timestep,
            'belief_states': deepcopy(self.belief_states),
            'camera_images': self.camera_images,
            'risk_assessment': self._assess_risk() if self.belief else None,
            'original_info': info, # Original info from gym env step
            'high_level_action': self.current_high_level_action,
            'high_level_completed': self.high_level_action_completed,
            'ego_speed': self.ego_vehicle.speed if self.ego_vehicle else 0,
            'on_road': self.ego_vehicle.on_road if self.ego_vehicle else True,
        }
        
        return observation_text, reward, done_flag, info_dict, valid_actions, ego_position

    def plan_next_action(self, planning_time=0.1, step=None, previous_action=None):
        if not self.high_level_mcts:
            raise ValueError("MCTS agents not connected. Call connect_to_MCTS first.")
        if not self.dqn_agent:
            raise ValueError("DQN agent not initialized.")
            
        metrics = {}
        start_time = time.time()

        if self.high_level_action_completed or self.current_high_level_action is None:
            observation_text = self._get_observation() 
            # History will now be a list of high-level action strings
            history_for_vlm = [item[0] for item in self.agent_history] if self.agent_history else []
            valid_hl_actions = self.get_valid_actions()
            
            high_level_start = time.time()
            # Ensure camera_images and navi_info are available or have defaults
            cam_images = self.camera_images if hasattr(self, 'camera_images') else {}
            nav_info = self._get_navigation_info() if hasattr(self, '_get_navigation_info') else {}

            high_level_action_name = self.high_level_mcts.search(
                observation_text, history_for_vlm, self.timestep, valid_hl_actions, 
                False, cam_images, nav_info
            )
            high_level_time = time.time() - high_level_start
            
            self.current_high_level_action = high_level_action_name
            self.high_level_action_completed = False
            self.high_level_action_steps = 0
            
            metrics['high_level_time'] = high_level_time
            metrics['high_level_action'] = self.current_high_level_action
            # print(f"New high-level action: {self.current_high_level_action}")

            if self.use_llm_for_dqn_guidance:
                # Ensure observation_text and history_for_vlm are current for mid_level_action call
                current_obs_text_for_mid = observation_text # observation_text is set if new HL action planned
                current_history_for_mid = history_for_vlm # history_for_vlm is set if new HL action planned
                
                self.mid_level_action = get_mid_level_action(
                    self.model, current_obs_text_for_mid, current_history_for_mid, 
                    self.current_high_level_action, observation=current_obs_text_for_mid
                )
        else:
            metrics['high_level_action'] = self.current_high_level_action
            metrics['high_level_time'] = 0.0
            # print(f"Continuing high-level action: {self.current_high_level_action} (step {self.high_level_action_steps})")
            if self.use_llm_for_dqn_guidance:
                # HL action not re-planned, so get current observation and history
                current_obs_text_for_mid = self._get_observation()
                current_history_for_mid = [item[0] for item in self.agent_history] if self.agent_history else []
                self.mid_level_action = get_mid_level_action(
                    self.model, current_obs_text_for_mid, current_history_for_mid,
                    self.current_high_level_action, observation=current_obs_text_for_mid
                )
        


        dqn_planning_start_time = time.time()
        llm_suggested_continuous_action = None
        if self.use_llm_for_dqn_guidance and self.mid_level_action and \
           'acceleration' in self.mid_level_action and 'steering' in self.mid_level_action:
            
            raw_llm_acc = self.mid_level_action['acceleration']
            raw_llm_steer = self.mid_level_action['steering']

            # Normalize LLM action to [-1, 1] for DQN guidance
            # Assumes symmetric ranges, e.g., acc_target_range is [-max_acc, max_acc]
            # self.acc_target_range[1] would be max_acc (e.g., 3.0)
            # self.steering_target_range[1] would be max_steer (e.g., np.pi/4)
            
            norm_llm_acc = 0.0
            if self.acc_target_range[1] != 0: # Avoid division by zero
                # If LLM output is already scaled, e.g. -3 to 3, divide by the max of that scale
                norm_llm_acc = raw_llm_acc / self.acc_target_range[1]
            
            norm_llm_steer = 0.0
            if self.steering_target_range[1] != 0: # Avoid division by zero
                # If LLM output is e.g. -0.79 to 0.79, divide by the max of that scale
                norm_llm_steer = raw_llm_steer / self.steering_target_range[1]
            
            # Clip to ensure they are strictly within [-1, 1] as expected by DQN's discrete action mapping
            norm_llm_acc = np.clip(norm_llm_acc, -1.0, 1.0)
            norm_llm_steer = np.clip(norm_llm_steer, -1.0, 1.0)
            
            llm_suggested_continuous_action = (norm_llm_acc, norm_llm_steer)

        if self.current_numerical_observation is None:
            print("Warning: current_numerical_observation is None in plan_next_action. Using default action.")
            # Choose a neutral discrete action (e.g., middle action if action_dim is odd, or specific index)
            neutral_throttle_idx = self.dqn_config["num_throttle_bins"] // 2
            neutral_steering_idx = self.dqn_config["num_steering_bins"] // 2
            discrete_action = neutral_throttle_idx * self.dqn_config["num_steering_bins"] + neutral_steering_idx
        else:
            discrete_action = self.dqn_agent.take_action(
                self.current_numerical_observation,
                llm_suggested_continuous_action=llm_suggested_continuous_action,
                p_follow_llm=self.dqn_config["p_follow_llm"] if self.use_llm_for_dqn_guidance else 0.0
            )
        
        self.last_discrete_dqn_action = discrete_action 
        throttle, steering = self.dqn_agent.get_continuous_action_pair(discrete_action)
        continuous_action_for_env = [throttle, steering]
        self.low_level_action_continuous = continuous_action_for_env

        dqn_planning_time = time.time() - dqn_planning_start_time
        metrics['low_level_time'] = dqn_planning_time
        metrics['low_level_planner'] = "DQN"
        
        total_time = time.time() - start_time
        metrics['total_time'] = total_time
        
        return [throttle, steering], metrics

    def execute_policy_step(self, episode=None, hl_step_num=None):
        """
        Executes one high-level policy step, which involves planning and 
        executing low-level DQN actions for a number of sub-steps.
        """
        # 1. Plan next low-level action using the hierarchy
        continuous_action_from_dqn, planning_metrics = self.plan_next_action(
            planning_time=self.config.get("planning_time_budget", 0.1),
            step=self.timestep 
        )

        accumulated_reward_hl_step = 0.0
        done_hl_step = False
        info_from_last_low_level_step = {}
        
        # 2. Execute the chosen low-level action for dqn_sub_steps
        actual_sub_steps_taken = 0
        for i in range(self.dqn_sub_steps):
            actual_sub_steps_taken +=1
            # self.step executes the continuous_action_from_dqn, updates DQN, etc.
            text_obs_ll, reward_ll, done_ll, info_ll, _, _ = self.step(
                continuous_action_from_dqn, 
                episode=episode, 
                step_num_low_level=self.timestep # self.step increments self.timestep
            )
            
            accumulated_reward_hl_step += reward_ll
            info_from_last_low_level_step = info_ll 
            if done_ll:
                done_hl_step = True
                break 
        
        # 3. Get the new high-level text observation after all sub-steps
        next_high_level_text_observation = self._get_observation()
        
        # 4. Final info dictionary for this high-level step
        final_info_hl_step = {
            **planning_metrics, 
            **info_from_last_low_level_step, 
            'dqn_steps_executed': actual_sub_steps_taken,
            'accumulated_hl_reward': accumulated_reward_hl_step
        }
        
        return next_high_level_text_observation, accumulated_reward_hl_step, done_hl_step, final_info_hl_step

    def _check_high_level_action_completed(self, high_level_action, observation_numerical, current_env_state):
        if self.ego_vehicle is None: return True
        vehicle = self.ego_vehicle
        
        # Simplified completion: after a fixed number of low-level steps
        # More sophisticated checks can be added based on high_level_action type
        # For example, for 'left_change', check if lane has actually changed.
        # The self.high_level_action_steps counts low-level steps since HL action started.
        
        # Max steps condition
        if self.high_level_action_steps >= self.config.get("max_ll_steps_per_hl_action", self.dqn_sub_steps * 3): # e.g., 3 HL policy frequencies
            # print(f"HL action '{high_level_action}' timed out after {self.high_level_action_steps} LL steps.")
            return True

        # Goal-based completion (example for lane change)
        if high_level_action in ['left_change', 'right_change']:
            # This requires tracking previous lane or target lane.
            # For simplicity, we can rely on a timeout or a more direct check.
            # Let's assume GoalChecker or a similar mechanism could be used if a specific goal was set.
            # For now, rely on timeout or a simple heuristic.
            # Example: if vehicle is stable in a new lane.
            # This is complex to check robustly without more state.
            pass # Placeholder for more specific checks

        # Default: complete after a certain number of low-level steps (e.g., policy_frequency)
        # This is implicitly handled by the high-level loop in main.py calling execute_policy_step
        # and plan_next_action deciding if a new HL action is needed.
        # The self.high_level_action_steps is incremented in self.step().
        # If plan_next_action is called, and high_level_action_completed is True, it plans a new one.
        # So, this function should determine if the *current* HL action's objective is met.

        # For now, let's use a simple step count for demonstration.
        # A more robust way is to check if the *purpose* of the HL action is achieved.
        if self.high_level_action_steps > self.config.get("policy_frequency", 10) * 0.8: # e.g. 80% of policy frequency
             # This is a simple heuristic. Real completion should be goal-oriented.
             # print(f"HL action '{high_level_action}' considered complete after {self.high_level_action_steps} LL steps (heuristic).")
             return True
        
        return False # By default, not completed until timeout or specific condition.

    def _get_observation(self):
        if self.env_scenario is not None:
            # Ensure ego vehicle is available for scenario description
            if hasattr(self.env.unwrapped, 'vehicle') and self.env.unwrapped.vehicle:
                 return self.env_scenario.describe(0) # Assuming ego is agent 0
            else: # Fallback if ego vehicle not ready
                return "Ego vehicle not available for scenario description."


        if not hasattr(self.env.unwrapped, 'vehicle') or not self.env.unwrapped.vehicle:
            return "No vehicle information available."
        
        vehicle = self.env.unwrapped.vehicle
        speed = getattr(vehicle, 'speed', 0.0) 
        current_lane_tuple = getattr(vehicle, 'lane_index', None) # e.g., ('o', 'e', 0)
        current_lane_idx = current_lane_tuple[2] if current_lane_tuple else -1

        total_lanes = 0
        if hasattr(self.env.unwrapped, 'road') and self.env.unwrapped.road:
            total_lanes = getattr(self.env.unwrapped.road, 'lanes_count', 0)
            if total_lanes == 0 and hasattr(self.env.unwrapped.road.network, 'lanes_on_road'):
                # Try to get from network if road.lanes_count is 0 or not set
                lanes_on_road = self.env.unwrapped.road.network.lanes_on_road(vehicle.position)
                if lanes_on_road: total_lanes = len(set(idx[2] for idx in lanes_on_road))


        observation = f"Ego vehicle in lane {current_lane_idx} of {total_lanes} lanes. "
        observation += f"Moving at {speed:.1f} m/s with heading {vehicle.heading:.2f} rad. "

        other_vehicles = []
        if hasattr(self.env.unwrapped, 'road') and self.env.unwrapped.road:
            # Get vehicles sorted by distance, excluding ego
            other_vehicles_raw = self.env.unwrapped.road.close_vehicles_to(
                vehicle, 
                self.env.unwrapped.config.get("observation", {}).get("vehicles_count", 5) * 20.0, # perception distance
                count=self.env.unwrapped.config.get("observation", {}).get("vehicles_count", 5) -1, # count for others
                see_behind=True,
                sort='sorted'
            )
            other_vehicles = [v for v in other_vehicles_raw if v is not vehicle]
            
            observation += f"There are {len(other_vehicles)} other vehicles nearby. "
            
            for i, other_v in enumerate(other_vehicles):
                if hasattr(other_v, 'position') and hasattr(other_v, 'speed'):
                    other_v_id = id(other_v) % 1000
                    other_lane_tuple = getattr(other_v, 'lane_index', None)
                    other_lane_idx = other_lane_tuple[2] if other_lane_tuple else -1
                    other_speed = getattr(other_v, 'speed', 0.0)
                    dist_to_ego = np.linalg.norm(np.array(other_v.position) - np.array(vehicle.position))

                    observation += f"Vehicle {other_v_id} in lane {other_lane_idx}, dist: {dist_to_ego:.1f}m. "
                    if other_speed > speed + 1.0: observation += "Faster. "
                    elif other_speed < speed - 1.0: observation += "Slower. "
                    else: observation += "Similar speed. "
                    
                    if self.belief_states and other_v_id in self.belief_states:
                        belief_state = self.belief_states[other_v_id]
                        if belief_state.get('predicted_actions'):
                            pred_actions = belief_state['predicted_actions']
                            if isinstance(pred_actions, dict) and 'most_likely' in pred_actions:
                                observation += f"Predicted: {pred_actions['most_likely']}. "
                            elif isinstance(pred_actions, (list, np.ndarray)): # if it's a distribution
                                 # Find most likely action from distribution if not already processed
                                 action_names = self._get_action_space() # Assuming this returns list of names
                                 if len(pred_actions) == len(action_names):
                                     most_likely_idx = np.argmax(pred_actions)
                                     observation += f"Predicted: {action_names[most_likely_idx]}. "
                                 else:
                                     observation += f"Predicted actions available (raw: {pred_actions}). "
                            else:
                                observation += f"Predicted actions available (raw: {pred_actions}). "
        return observation

    def render(self, mode='human'):
        if self.env:
            # Ensure render_mode is compatible if 'human' is requested but env is 'rgb_array'
            current_render_mode = self.env.render_mode
            if mode == 'human' and current_render_mode == 'rgb_array':
                # print("Cannot render in 'human' mode if env is 'rgb_array'. Returning array.")
                img = self.env.render() # Will use its native mode
            else: # if mode is 'rgb_array' or matches env's mode
                img = self.env.render() # Call with env's configured mode or requested 'rgb_array'
            
            if isinstance(img, np.ndarray): # If render returns an array
                self.camera_images['front'] = img
            return img
        return None

    def _capture_current_state(self):
        state = EnvironmentState()
        if self.ego_vehicle:
            state.vehicle_state = {
                'position': self.ego_vehicle.position, 'velocity': self.ego_vehicle.velocity,
                'heading': self.ego_vehicle.heading, 'speed': self.ego_vehicle.speed,
                'lane_index': self.ego_vehicle.lane_index if hasattr(self.ego_vehicle, 'lane_index') else None,
                'lane': self.ego_vehicle.lane_index[2] if hasattr(self.ego_vehicle, 'lane_index') else -1
            }
        
        if hasattr(self.env.unwrapped, 'road') and hasattr(self.env.unwrapped.road, 'vehicles'):
            other_vehicles_list = [v for v in self.env.unwrapped.road.vehicles if v is not self.ego_vehicle]
            for vehicle_obj in other_vehicles_list:
                v_id = id(vehicle_obj) % 1000
                if hasattr(vehicle_obj, 'position') and hasattr(vehicle_obj, 'velocity'):
                    state.env_state[v_id] = {
                        'position': vehicle_obj.position, 'velocity': vehicle_obj.velocity,
                        'heading': getattr(vehicle_obj, 'heading', 0.0),
                        'speed': getattr(vehicle_obj, 'speed', np.linalg.norm(vehicle_obj.velocity)),
                        'lane_index': getattr(vehicle_obj, 'lane_index', None),
                        'lane': getattr(vehicle_obj, 'lane_index', (None,None,-1))[2],
                        'type': 'vehicle'
                    }
        
        if hasattr(self.env.unwrapped, 'road') and self.env.unwrapped.road:
            road_config = self.env.unwrapped.config
            state.env_state['road'] = {
                'lanes_count': road_config.get("lanes_count", 0),
                'lanes': [] # Placeholder, fill if detailed lane geometry is needed by belief
            }
            if hasattr(self.env.unwrapped.road.network, 'lanes'):
                 state.env_state['road']['lanes'] = [
                     (lane.index, lane.length, lane.width) for lane in self.env.unwrapped.road.network.lanes
                 ]
        return state
    
    def _update_belief_state(self, current_state, episode=None, step=None):
        if self.belief is None:
            self.belief = Belief(
                road_graph=self.road_network if self.road_network else (self.env.unwrapped.road if hasattr(self.env.unwrapped, 'road') else None),
                ego_vehicle_id=self.ego_vehicle_id,
                forget_rate=0.05 
            )
        
        vehicle_states_for_belief_agent = {}
        agent_ids_list = []
        
        for v_id, v_data in current_state.env_state.items():
            if isinstance(v_id, int) and v_id != self.ego_vehicle_id and v_data.get('type') == 'vehicle':
                agent_ids_list.append(v_id)
                vehicle_states_for_belief_agent[v_id] = {
                    'lane': v_data.get('lane', -1), 'speed': v_data.get('speed', 0.0),
                    'position': v_data.get('position'), 'heading': v_data.get('heading', 0.0)
                }
        self.agent_ids = agent_ids_list

        if self.model: # Only init/update belief_agent if VLM model is present
            if self.belief_agent is None:
                self.belief_agent = OtherVehicleAgent(self.model, self.action_space, 
                                                    vehicle_states_for_belief_agent, 
                                                    self.agent_ids, self.belief)
            else:
                self.belief_agent.current_states = vehicle_states_for_belief_agent
                self.belief_agent.agent_ids = self.agent_ids
                self.belief_agent.belief = self.belief # Ensure belief is current
        
        observation_for_belief_update = {'timestamp': self.timestep, 'vehicles': {}}
        for v_id, data in current_state.env_state.items():
            if isinstance(v_id, int) and v_id != self.ego_vehicle_id and data.get('type') == 'vehicle':
                observation_for_belief_update['vehicles'][v_id] = {
                    'position': data.get('position'), 'velocity': data.get('velocity'),
                    'heading': data.get('heading', 0.0), 'lane_id': data.get('lane', -1)
                }
        
        if 'road' in current_state.env_state and 'lanes' in current_state.env_state['road']:
            road_cfg = self.env.unwrapped.config
            observation_for_belief_update['lanes'] = {
                i: {'drivable': 1.0, 
                    'speed_limit': road_cfg.get("reward_speed_range", [20,30])[1],
                    'width': lane_geom[2] if len(lane_geom) > 2 else 4.0 }
                for i, lane_geom in enumerate(current_state.env_state['road']['lanes'])
            }
        self.belief.update_from_observation(observation_for_belief_update)
        
        if self.belief_agent: # Only call get_action if belief_agent exists
            self.belief_agent.get_action(episode, step)
        
        self.belief_states = {
            v_id: {
                'position': self.belief.vehicle_beliefs.get(v_id, {}).get('position'),
                'velocity': self.belief.vehicle_beliefs.get(v_id, {}).get('velocity'),
                'heading': self.belief.vehicle_beliefs.get(v_id, {}).get('heading'),
                'predicted_actions': self.belief.vehicle_beliefs.get(v_id, {}).get('predicted_actions', {})
            } for v_id in self.agent_ids
        }

    def _assess_risk(self):
        if not self.belief: return {'overall_risk': 0.0, 'vehicles': {}}
        if self.ego_vehicle and hasattr(self.ego_vehicle, 'position') and hasattr(self.ego_vehicle, 'velocity'):
            dt = 0.1; steps = 10
            ego_traj = np.array([self.ego_vehicle.position + self.ego_vehicle.velocity * dt * i for i in range(steps)])
        else: return {'overall_risk': 0.0, 'vehicles': {}}
        
        collision_probs = self.belief.get_collision_probabilities(ego_traj)
        risky_vehicles = self.belief.get_risky_vehicles(threshold=0.3) # Uses its own ego_traj generation
        overall_risk = max([0.0] + [probs.get('probability', 0.0) for probs in collision_probs.values()])
        return {'overall_risk': overall_risk, 'risky_vehicles': risky_vehicles, 'vehicles': collision_probs}
    
    def _calculate_reward(self, action_hl_name): # MCTS simulation reward
        # This reward is for MCTS rollouts, not the primary reward for DQN training.
        # DQN uses the reward directly from env.step().
        if not hasattr(self.env.unwrapped, 'vehicle') or not self.env.unwrapped.vehicle: return 0.0
        
        # Use info from the actual (simulated) ego vehicle state if available
        ego_v = self.env.unwrapped.vehicle
        speed = ego_v.speed
        target_speed_range = self.config.get("reward_speed_range", [20, 30])
        collision = ego_v.crashed if hasattr(ego_v, 'crashed') else False # Check 'crashed' status
        on_road = ego_v.on_road if hasattr(ego_v, 'on_road') else True
        
        # Speed reward
        speed_reward = 0
        if target_speed_range[0] <= speed <= target_speed_range[1]:
            speed_reward = 1.0
        elif speed < target_speed_range[0]:
            speed_reward = speed / target_speed_range[0] # Penalize being too slow
        else: # speed > target_speed_range[1]
            speed_reward = max(0, 1.0 - (speed - target_speed_range[1]) / (target_speed_range[1] * 0.5)) # Penalize overspeeding

        # Penalties
        collision_penalty = -10.0 if collision else 0.0
        off_road_penalty = -5.0 if not on_road else 0.0
        
        # Action related reward (e.g., penalize harsh braking if not needed)
        action_reward = 0.0
        if action_hl_name == 'brake' and speed > 5: # Gentle brake is fine
            action_reward = -0.1 * (speed / target_speed_range[1]) # Penalize braking at high speed
        
        # Progress (can be complex, e.g., distance to goal or lane progress)
        # For MCTS simulation, a simple speed-based progress can work
        progress_reward = speed / target_speed_range[1] if speed <= target_speed_range[1] else 1.0
        
        total_reward = (
            self.reward_weights['efficiency'] * speed_reward +
            self.reward_weights['safety'] * (collision_penalty + off_road_penalty) +
            self.reward_weights['progress'] * progress_reward +
            self.reward_weights['comfort'] * action_reward # Comfort can be related to action smoothness
        )
        return float(total_reward)
        
    def get_valid_actions(self): # High-level valid actions
        default_actions = list(self._get_action_space().keys())
        valid_actions = default_actions.copy()
        # Add logic to prune actions based on env state if needed (e.g., no left_change from leftmost lane)
        if self.ego_vehicle and hasattr(self.ego_vehicle, 'lane_index'):
            lane_idx_tuple = self.ego_vehicle.lane_index
            if lane_idx_tuple:
                current_lane_idx_val = lane_idx_tuple[2]
                num_lanes = self.config.get("lanes_count", 3)

                if current_lane_idx_val == 0: # Leftmost lane
                    if "left_change" in valid_actions: valid_actions.remove("left_change")
                    if "turning_left" in valid_actions: valid_actions.remove("turning_left") # If turns imply lane change
                if current_lane_idx_val == num_lanes - 1: # Rightmost lane
                    if "right_change" in valid_actions: valid_actions.remove("right_change")
                    if "turning_right" in valid_actions: valid_actions.remove("turning_right")
        return valid_actions
    
    def _is_at_intersection(self): # Placeholder
        return False # Most highway-v0 scenarios don't have functional intersections for MCTS decisions
    
    def _get_navigation_info(self): # Placeholder for MCTS
        return {
            'road_option': 'STRAIGHT', 
            'target_speed': self.config.get("reward_speed_range", [20,30])[1],
            'distance_to_goal': float('inf')
        }

    # mcts_step, _copy_state_for_simulation, _simulate_step, _update_belief_with_simulation,
    # _get_observation_from_state, _check_terminal_state, _action_to_control
    # These are primarily for MCTS internal simulation and seem okay.
    # Ensure _calculate_reward used by mcts_step is appropriate for simulated steps.

    def mcts_step(self, action_hl_name):
        sim_state = self._copy_state_for_simulation()
        control_params = self._action_to_control(action_hl_name) # Get [acc, steer] for HL action
        
        # Simulate multiple low-level steps for one HL action in MCTS rollout
        # This makes MCTS rollouts more realistic if HL actions span time.
        # For simplicity, let's assume one HL action corresponds to one effective change.
        # A more advanced MCTS would simulate for a short duration.
        # Here, we apply a simplified kinematic update.
        sim_next_state = self._simulate_step(sim_state, control_params) # Applies one dt step
        
        sim_timestep = self.timestep + 1 # Hypothetical next step
        # self._update_belief_with_simulation(sim_next_state) # Belief update in simulation can be complex
        
        reward = self._calculate_reward(action_hl_name) # Use the MCTS specific reward
        done = self._check_terminal_state(sim_next_state) # Check terminal conditions in sim
        observation_text = self._get_observation_from_state(sim_next_state)
        
        # History for MCTS needs to be managed carefully if it affects state_id
        # For now, assume history is a list of HL action names for MCTS state_id
        # Let's assume high_level_mcts.py handles history for its state IDs.
        # This history is for the MCTS agent's perspective.
        sim_history = self.agent_history.copy() # This is LL history
        mcts_history_hl_actions = [item[0] if isinstance(item[0], str) else "ll_action" for item in sim_history]
        mcts_history_hl_actions.append(action_hl_name)

        valid_actions_sim = self.get_valid_actions() # Valid actions from the new sim_state (can be simplified)
        
        return observation_text, reward, done, mcts_history_hl_actions, valid_actions_sim

    def _copy_state_for_simulation(self):
        current_state_obj = self._capture_current_state()
        return deepcopy(current_state_obj)

    def _simulate_step(self, state_obj, control): # control is [acc, steer]
        # Simplified kinematic update on a state_obj (EnvironmentState)
        # This is a very basic forward model for MCTS.
        dt = 1.0 / self.config.get("policy_frequency", 10) # Time duration of one HL action

        pos = np.array(state_obj.vehicle_state['position'])
        vel = np.array(state_obj.vehicle_state['velocity'])
        heading = state_obj.vehicle_state['heading']
        speed = state_obj.vehicle_state['speed']

        acc, steer = control[0], control[1]

        # Update speed and heading
        new_speed = speed + acc * dt
        new_speed = max(0, new_speed) # No negative speed
        
        # Simplified steering effect: change heading
        # More accurate model would use bicycle model (speed * tan(steer) / wheelbase)
        angular_velocity = 0
        if speed > 0.1: # Avoid division by zero or large changes at low speed
            wheelbase = self.config.get("wheelbase", 2.5) # Approx vehicle wheelbase
            angular_velocity = (new_speed / wheelbase) * np.tan(steer) 
        
        new_heading = heading + angular_velocity * dt
        new_heading = (new_heading + np.pi) % (2 * np.pi) - np.pi # Normalize heading to [-pi, pi]

        # Update velocity vector
        new_vel_vec = np.array([new_speed * np.cos(new_heading), new_speed * np.sin(new_heading)])
        
        # Update position
        # Average velocity over dt for position update: (vel + new_vel_vec)/2 * dt
        # Or simpler: new_pos = pos + new_vel_vec * dt (assuming new_vel_vec is constant over dt)
        new_pos = pos + vel * dt + 0.5 * np.array([acc * np.cos(heading), acc * np.sin(heading)]) * dt**2 # Kinematic
        # Or even simpler for MCTS:
        # new_pos = pos + new_vel_vec * dt


        sim_next_state_obj = deepcopy(state_obj)
        sim_next_state_obj.vehicle_state['position'] = new_pos
        sim_next_state_obj.vehicle_state['velocity'] = new_vel_vec
        sim_next_state_obj.vehicle_state['heading'] = new_heading
        sim_next_state_obj.vehicle_state['speed'] = new_speed
        
        # TODO: Update other vehicle states in sim_next_state_obj.env_state if MCTS needs to reason about them.
        # For now, MCTS rollouts primarily focus on ego's simulated evolution.
        return sim_next_state_obj

    def _get_observation_from_state(self, state_obj: EnvironmentState):
        # Simplified observation for MCTS simulation from an EnvironmentState object
        vs = state_obj.vehicle_state
        obs_str = f"SimEgo at {vs['position'][0]:.1f},{vs['position'][1]:.1f}, speed {vs['speed']:.1f}m/s, head {vs['heading']:.2f}rad. "
        # Could add simplified info about a few other vehicles if state_obj.env_state is populated for them.
        return obs_str

    def _check_terminal_state(self, state_obj: EnvironmentState):
        # Simplified terminal check for MCTS simulation
        # e.g., if ego goes off a predefined road boundary, or speed is too low for too long.
        # For highway-v0, off-road is a key terminal state.
        # This requires defining road boundaries or using lane information.
        # For now, assume not terminal unless a very obvious condition.
        if state_obj.vehicle_state['speed'] < 0.1 and self.timestep > 100 : # Stuck
            return True
        # Add collision check if other vehicles are simulated in state_obj.env_state
        return False

    def _action_to_control(self, action_hl_name): # Returns [acc, steer] for the HL action
        if action_hl_name in self.action_space: # self.action_space is from _get_action_space()
            params = self.action_space[action_hl_name]
            # These are target values, not ranges for LLM.
            # For simulation, we can use these directly.
            return np.array([params['acceleration'], params['steering']])
        else: # Fallback for unknown action
            print(f"Warning: Unknown HL action '{action_hl_name}' in _action_to_control. Defaulting.")
            return np.array([0.0, 0.0]) # Neutral action

    def close(self):
        if self.env:
            self.env.close()
