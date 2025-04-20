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
from src.MCTS.tree import Tree as LowLevelMCTS
from src.HighwayEnv.goal_checker import GoalChecker
from src.VLM.policy import get_mid_level_action
from src.HighwayEnv.envScenario import EnvScenario


class MCTSEnv:
    def __init__(self, env_id='highway-v0', model=None, config=None, road_network=None):
        # Highway Environment connection
        self.env_id = env_id
        self.config = config or {}
        self._setup_default_config()
        self.env = gym.make(self.env_id, render_mode="rgb_array")
        self.env.configure(self.config)
        self.video_path = "/home/ubuntu/sjh04/MCTS-based-VLM-Numerical-Prediction-Planner/video"
        self.env = RecordVideo(self.env, video_folder=self.video_path, episode_trigger=lambda e: True)  # record all episodes
        self.env_scenario = None
        # self.env.unwrapped.set_record_video_wrapper(self.env)

        self.frames = config["simulation_frequency"]
        
        # Environment state management
        self.state_graph = StateGraph()
        self.agent_history = deque(maxlen=100)
        self.belief_states = {}
        self.agent_ids = None

        self.acc_target_range = [-3, 3]  # m/s² (Highway Environment typical range)
        self.steering_target_range = [-np.pi/4, np.pi/4]  # rad (Highway Environment typical range)
        self.acc_values = 13  # Number of discrete acceleration values
        self.steering_values = 13  # Number of discrete steering values
        self.acc_coef = (self.acc_target_range[1] - self.acc_target_range[0]) / (self.acc_values - 1)
        self.steer_coef = (self.steering_target_range[1] - self.steering_target_range[0]) / (self.steering_values - 1)
        
        # Goal tracking
        self.current_goal = None
        self.goal_checker = GoalChecker()
        
        # Environment parameters
        self.low_level_action = [0, 0]
        self.mid_level_action = None
        self.ego_vehicle = None
        self.ego_vehicle_id = 0
        self.sensors = {}
        self.camera_images = {}  # Empty placeholder for compatibility
        self.timestep = 0
        self.road_network = road_network or self.env.unwrapped.road
        
        # VLM model
        self.model = model

        # Belief system
        self.belief = None
        self.belief_agent = None
        
        # Observation and state
        self.current_observation = None  # Store latest text observation
        
        # MCTS parameters
        self.planning_horizon = 5.0  # seconds
        self.action_space = self._get_action_space()
        self.reward_weights = {
            'safety': 10.0,
            'progress': 1.0, 
            'comfort': 0.5,
            'efficiency': 1.0
        }
        
        # MCTS agents
        self.high_level_mcts = None
        self.low_level_mcts = None
        
        # Translation dictionaries for high-level to low-level actions
        self.high_to_low_action_mapping = {
            'overtaking': {'acceleration': [0.5, 1.0], 'steering': [-0.1, 0.1]},
            'keeping_lane': {'acceleration': [0.0, 0.5], 'steering': [-0.1, 0.1]},
            'turning_left': {'acceleration': [0.0, 0.3], 'steering': [-0.3, -0.1]},
            'turning_right': {'acceleration': [0.0, 0.3], 'steering': [0.1, 0.3]},
            'left_change': {'acceleration': [0.0, 0.5], 'steering': [-0.3, -0.1]},
            'right_change': {'acceleration': [0.0, 0.5], 'steering': [0.1, 0.3]},
            'brake': {'acceleration': [-0.5, -0.1], 'steering': [-0.1, 0.1]},
        }
        
        # High-level instruction status tracking
        self.current_high_level_action = None
        self.high_level_action_completed = True  # Start with True to generate first high-level action
        self.high_level_action_steps = 0
        self.max_steps_per_high_level_action = 160  # Maximum steps before forcing new high-level action

    def _setup_default_config(self):
        """Set up default configuration for Highway Environment"""
        if not self.config:
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
                    "type": "ContinuousAction"
                },
                "lanes_count": 3,
                "vehicles_count": 15,
                "duration": 40,
                "initial_spacing": 2,
                "collision_reward": -1,
                "reward_speed_range": [20, 30],
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "screen_width": 600,
                "screen_height": 150,
                "centering_position": [0.3, 0.5],
                "scaling": 5.5,
                "show_trajectories": True,
                "render_agent": True,
                "offscreen_rendering": False
            }

    def _get_action_space(self):
        """
        Define the high-level action space for planning in Highway Environment
        
        Returns:
            Dictionary of high-level actions
        """
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
        """
        Connect this environment to both high-level and low-level MCTS planning agents
        
        Args:
            high_level_agent: High-level MCTS agent (from high-level-mcts.py)
            low_level_agent: Low-level MCTS agent (from tree.py)
        """
        # Connect high-level MCTS
        self.high_level_mcts = high_level_agent
        
        # Pass environment configuration to high-level MCTS
        high_level_params = get_simulation_params(
            action_space=self.action_space,
            planning_horizon=self.planning_horizon,
            reward_weights=self.reward_weights
        )
        if hasattr(self.high_level_mcts, 'configure'):
            self.high_level_mcts.configure(high_level_params)
        
        # Connect low-level MCTS if provided
        if low_level_agent:
            self.low_level_mcts = low_level_agent
        
        return self

    def initialize_low_level_mcts(self, policy=None, mid_level_action=None, high_level_action=None, action=None):
        """
        Initialize the low-level MCTS Tree with current environment state
        
        Args:
            policy: Optional policy network for low-level MCTS
            mid_level_action: Mid-level action parameters
            action: Initial action
        """
        if self.env is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        # Prepare initial state for Tree
        init_state = self._create_tree_init_state()
        
        # Initialize Tree with current world state
        self.low_level_mcts = LowLevelMCTS(
            env=self.env,
            mcts_env=self,
            ego_vehicle=self.env.unwrapped.vehicle,
            init_state=init_state,
            policy=policy,
            high_level_action=high_level_action,
            mid_action=mid_level_action,
            action=action,
        )
        
        return self.low_level_mcts


    def reset(self, episode=None, step=None, **kwargs):
        """Reset environment to initial state"""
        # Reset the Highway Environment
        obs, info = self.env.reset(**kwargs)
        self.env_scenario = EnvScenario(self.env, self.env_id, 42)
        self.env.unwrapped.set_record_video_wrapper(self.env)

        # Get vehicle IDs
        # Get ego vehicle reference
        self.ego_vehicle = self.env.unwrapped.vehicle
        print(f"ego_vehicle_type: {type(self.ego_vehicle)}")
        print(f"ego_vehicle: {self.ego_vehicle}")
        self.ego_vehicle_id = id(self.ego_vehicle) % 1000
        print(f"ego_vehicle_id: {self.ego_vehicle_id}")
        
        # Initialize state and belief
        self.timestep = 0
        self.agent_history.clear()
        initial_state = self._capture_current_state()
        self.state_graph.add_node(initial_state)
        self._update_belief_state(initial_state, episode=episode, step=step)
        
        # Reset low-level action
        self.low_level_action = [0, 0]

        # Reset mid-level action
        self.mid_level_action = None

        # Reset high-level action tracking
        self.current_high_level_action = None
        self.high_level_action_completed = True
        self.high_level_action_steps = 0
        
        # Get observation for agent
        observation = self._get_observation()
        valid_actions = self.get_valid_actions()
        
        # Return initial observation and action space
        return observation, valid_actions

    def step(self, action, episode=None, step=None):
        """Execute action and return environment feedback"""
        # Validate action
        # if not self._is_action_valid(action):
        #     raise ValueError(f"Invalid action: {action}")
        
        # Execute action in simulator
        print(f"Executing action: {action}")
        # control = self._action_to_control(action)
        print(f"Control: {action}")
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get position of the ego vehicle
        ego_position = self.env.unwrapped.vehicle.position

        self.env.render()

        self.env.unwrapped.automatic_rendering_callback = self.env.video_recorder.capture_frame()


        self.timestep += 1
        # print("========================")
        # print(f"obs: {obs}")
        # print("=================")
        # Update environment state
        current_state = self._capture_current_state()
        self.state_graph.add_node(current_state)
        
        # Update belief system
        self._update_belief_state(current_state, episode=episode, step=step)
        
        # Calculate reward and check if done
        observation = self._get_observation()
        # reward = self._calculate_reward(action)
        done = terminated or truncated
        valid_actions = self.get_valid_actions()
        
        # Update history
        self.agent_history.append((action, observation, reward))
        
        # Check if current high-level action is completed
        if self.current_high_level_action is not None:
            self.high_level_action_steps += 1
            self.high_level_action_completed = self._check_high_level_action_completed(
                self.current_high_level_action, 
                obs, 
                current_state
            )
            
            # Force high-level action completion after max steps
            # if self.high_level_action_steps >= self.max_steps_per_high_level_action:
            #     self.high_level_action_completed = True
        
        # Create info dict with additional data for MCTS
        info_dict = {
            'timestep': self.timestep,
            'belief_states': deepcopy(self.belief_states),
            'camera_images': self.camera_images,
            'risk_assessment': self._assess_risk() if self.belief else None,
            'original_info': info,
            'high_level_action': self.current_high_level_action,
            'high_level_completed': self.high_level_action_completed
        }
        
        return observation, reward, done, info_dict, valid_actions, ego_position

    def mcts_step(self, action):
        """
        Execute a high-level action in a simulated environment (not the real environment)
        and update beliefs accordingly. Used by MCTS for planning.
        
        Args:
            action: High-level action to execute
            
        Returns:
            tuple: (observation, reward, done, history, valid_actions)
        """
        # Create a copy of the current state to simulate without affecting real environment
        sim_state = self._copy_state_for_simulation()
        
        # Convert high-level action to control commands
        control = self._action_to_control(action)
        
        # Simulate the action's effect on the copied state
        sim_next_state = self._simulate_step(sim_state, control)
        
        # Update simulated timestep
        sim_timestep = self.timestep + 1
        
        # Update belief system with simulated state
        self._update_belief_with_simulation(sim_next_state)
        
        # Calculate reward for the simulated action
        reward = self._calculate_reward(action)
        
        # Check if the simulated state represents a terminal state
        done = self._check_terminal_state(sim_next_state)
        
        # Get observation from simulated state
        observation = self._get_observation_from_state(sim_next_state)
        
        # Update simulated history
        sim_history = self.agent_history.copy()
        sim_history.append((action, observation, reward))
        history = [item[0] for item in sim_history]  # Extract just the actions
        
        # Get valid actions in the new state
        valid_actions = self.get_valid_actions()
        
        return observation, reward, done, history, valid_actions
    
    def _copy_state_for_simulation(self):
        """
        Create a deep copy of the current environment state for simulation purposes.
        
        Returns:
            dict: Copied state for simulation
        """
        # Capture the current state to use as a base
        current_state = self._capture_current_state()
        
        # Return a copy to avoid modifying the original
        return deepcopy(current_state)
    
    def _simulate_step(self, state, control):
        """
        Simulate a step in the environment without affecting the real environment.
        
        Args:
            state: Current state to simulate from
            control: Control action to apply
            
        Returns:
            dict: Next simulated state
        """
        # Extract vehicle state from current state
        ego_position = state.vehicle_state['position']
        ego_velocity = state.vehicle_state['velocity']
        ego_heading = state.vehicle_state['heading']
        
        # Extract acceleration and steering from control
        acceleration = control[0]
        steering = control[1]
        
        # Apply simple kinematic model to update position and velocity
        dt = 0.1  # Default time step
        
        # Update velocity based on acceleration
        new_speed = np.linalg.norm(ego_velocity) + acceleration * dt
        new_speed = max(0, new_speed)  # Ensure non-negative speed
        
        # Update heading based on steering and velocity
        wheel_base = 3.5  # Default value for Highway Environment
        angular_velocity = new_speed * np.tan(steering) / wheel_base
        new_heading = ego_heading + angular_velocity * dt
        
        # Calculate new velocity vector
        new_velocity = np.array([
            new_speed * np.cos(new_heading),
            new_speed * np.sin(new_heading)
        ])
        
        # Update position
        new_position = ego_position + new_velocity * dt
        
        # Create new state with updated values
        new_state = deepcopy(state)
        new_state.vehicle_state['position'] = new_position
        new_state.vehicle_state['velocity'] = new_velocity
        new_state.vehicle_state['heading'] = new_heading
        
        return new_state
    
    def _update_belief_with_simulation(self, simulated_state):
        """
        Update belief system with simulated state information.
        This is used for maintaining consistent beliefs during MCTS planning.
        
        Args:
            simulated_state: The simulated next state
        """
        # Skip if belief system is not initialized
        if not hasattr(self, 'belief') or self.belief is None:
            return
            
        # Create observation data structure for belief update
        observation_data = {
            'vehicles': {},
            'timestamp': self.timestep + 1  # Simulated next timestep
        }
        
        # Extract vehicle information from simulated state
        if 'vehicles' in simulated_state.env_state.keys():
            for i, vehicle in enumerate(simulated_state.env_state['vehicles']):
                vehicle_id = vehicle.get('id', i)
                
                # Skip ego vehicle
                if vehicle_id == self.ego_vehicle_id:
                    continue
                    
                # Add vehicle data to observation
                observation_data['vehicles'][vehicle_id] = {
                    'position': vehicle.get('position', [0, 0]),
                    'velocity': vehicle.get('velocity', [0, 0]),
                    'heading': vehicle.get('heading', 0),
                    'lane': vehicle.get('lane', 0),
                    'speed': np.linalg.norm(vehicle.get('velocity', [0, 0]))
                }
        
        # Update belief with this observation data (without changing the main belief state)
        # We're creating a temporary belief copy for simulation
        sim_belief = deepcopy(self.belief)
        sim_belief.update_from_observation(observation_data)
        
        # Store the updated belief state for this simulation
        self.sim_belief = sim_belief
    
    def _get_observation_from_state(self, state):
        """
        Convert a state dictionary to an observation string for the agent.
        
        Args:
            state: State dictionary
            
        Returns:
            str: Observation text
        """
        # Create a text observation similar to _get_observation but using the provided state
        if not state:
            return "Environment not initialized"
            
        # Extract key information
        ego_position = state.vehicle_state['position']
        ego_velocity = state.vehicle_state['velocity']
        ego_heading = state.vehicle_state['heading']
        
        # Calculate speed
        speed = np.linalg.norm(ego_velocity) * 3.6  # Convert to km/h
        
        # Build observation string
        observation = f"Vehicle at location ({ego_position[0]:.1f}, {ego_position[1]:.1f}) "
        observation += f"moving at {speed:.1f} km/s with heading {ego_heading:.1f} radians. "
        
        return observation
    
    def _check_terminal_state(self, state):
        """
        Check if a state represents a terminal condition (collision, off-road, goal reached).
        
        Args:
            state: State to check
            
        Returns:
            bool: True if state is terminal, False otherwise
        """
        # Check for collision
        # if 'collision' in state and state.vehicle_state.keys():
        #     return True
            
        # # Check for off-road
        # if 'off_road' in state and state.vehicle_state.keys():
        #     return True
            
        # # Check for goal reached
        # if 'goal_reached' in state and state.vehicle_state.keys():
        #     return True
            
        # Default: not terminal
        return False

    def plan_next_action(self, planning_time=0.1, step=None, previous_action=None):
        """
        Plan the next action using both high-level and low-level MCTS
        
        Args:
            planning_time: Planning time budget in seconds
            
        Returns:
            action: The low-level action to execute
            metrics: Dictionary with planning metrics
        """
        if not self.high_level_mcts:
            raise ValueError("MCTS agents not connected. Call connect_to_MCTS first.")
        
        metrics = {}
        start_time = time.time()

        # Check if we need to run high-level MCTS to get a new high-level action
        if self.high_level_action_completed:
            # Get current observation for high-level planning
            observation = self._get_observation()
            history = [item[0] for item in self.agent_history] if self.agent_history else []
            valid_actions = self.get_valid_actions()
            
            # Run high-level MCTS
            high_level_start = time.time()
            high_level_action = self.high_level_mcts.search(
                observation,      # Text observation for Highway Environment
                history, 
                self.timestep, 
                valid_actions, 
                False,            # not done
                self.camera_images,  # Empty placeholder for compatibility 
                self._get_navigation_info()
            )
            high_level_time = time.time() - high_level_start
            
            # Store new high-level action and reset step counter
            self.current_high_level_action = high_level_action
            self.high_level_action_completed = False
            self.high_level_action_steps = 0
            
            metrics['high_level_time'] = high_level_time
            metrics['high_level_action'] = high_level_action
            print(f"New high-level action: {high_level_action}")

            # Get mid-level action from VLM model
            self.state_description = observation
            self.history_description = history
            self.mid_level_action = get_mid_level_action(self.model, self.state_description, self.history_description, high_level_action, observation=observation)

            # self.mid_level_action = self._high2mid_level_action(high_level_action)

            # initialize low-level MCTS with mid-level action
            initialize_low_level_mcts = self.initialize_low_level_mcts(
                policy=self.model,
                mid_level_action=self.mid_level_action,
                action=self.low_level_action,
                high_level_action=self.current_high_level_action
            )

        else:
            metrics['high_level_action'] = self.current_high_level_action
            metrics['high_level_time'] = 0.0
            print(f"Continuing high-level action: {self.current_high_level_action} (step {self.high_level_action_steps})")
        
        # Convert high-level action to constraints for low-level MCTS
        constraints = self._high_to_low_level_constraints(self.current_high_level_action)
        
        # Run low-level MCTS with remaining planning time
        low_level_action = {}
        if self.low_level_mcts:
            self.low_level_action = previous_action
            
            if step != 0:
                observation = self._get_observation()
                history = [item[0] for item in self.agent_history] if self.agent_history else []
                self.mid_level_action = get_mid_level_action(self.model, observation, 
                                                             history, self.current_high_level_action, observation=self.current_observation)
                initialize_low_level_mcts = self.initialize_low_level_mcts(
                    policy=self.model,
                    mid_level_action=self.mid_level_action,
                    action=self.low_level_action,
                    high_level_action=self.current_high_level_action
                )
            
            # Get history for context
            history = [item[0] for item in self.agent_history] if self.agent_history else []
            # self.low_level_mcts.update_sensors(self.camera_images, history)
            
            # Apply constraints to low-level MCTS
            # This would involve setting appropriate action ranges based on the high-level action
            if hasattr(self.low_level_mcts, 'acc_target_range') and 'acc_range' in constraints:
                self.low_level_mcts.acc_target_range = constraints['acc_range']
            if hasattr(self.low_level_mcts, 'steering_target_range') and 'steering_range' in constraints:
                self.low_level_mcts.steering_target_range = constraints['steering_range']
            
            # Run low-level MCTS simulation
            remaining_time = planning_time - (time.time() - start_time)
            low_level_start = time.time()
            k = max(32, int(remaining_time * 100))  # Adaptive simulation count
            # k = 2
            acc_seq, steering_seq = self.low_level_mcts.simulate(k=k)
            print("=======================")
            print(f"acc_seq shape: {acc_seq.shape}")
            print(f"steering_seq shape: {steering_seq.shape}")
            acc_seq[:, 6] += (acc_seq.sum(-1) == 0) * 0.1
            steering_seq[:, 6] += (steering_seq.sum(-1) == 0) * 0.1
            # print(f"acc_seq: {acc_seq}")
            # print(f"steering_seq: {steering_seq}")
            # Extract the immediate control action from the sequences
            # acc = acc_seq[0]  # First acceleration value
            # steering = steering_seq[0]  # First steering value
            
            # Convert to Highway Environment control format
            low_level_action['acceleration'] = acc_seq
            low_level_action['steering'] = steering_seq
            low_level_action['num_steps'] = 20
            low_level_time = time.time() - low_level_start
            metrics['low_level_time'] = low_level_time
        else:
            # If no low-level MCTS, convert high-level action directly to control
            low_level_action = self._high_level_to_control(self.current_high_level_action)
        
        # self.low_level_action = low_level_action

        total_time = time.time() - start_time
        metrics['total_time'] = total_time
        
        self.low_level_action = [low_level_action['acceleration'], low_level_action['steering']]

        return low_level_action, metrics

    def _check_high_level_action_completed(self, high_level_action, observation, current_state):
        """
        Check if the current high-level action has been completed
        
        Args:
            high_level_action: Current high-level action
            observation: Current observation
            current_state: Current environment state
            
        Returns:
            completed: Whether the high-level action is completed
        """
        # Get ego vehicle state
        if self.ego_vehicle is None:
            return True
            
        vehicle = self.ego_vehicle
        
        # Get information based on high-level action type
        if high_level_action == 'overtaking':
            # Check if we've passed a vehicle
            if hasattr(vehicle, 'lane') and hasattr(vehicle, 'position'):
                # Get surrounding vehicles
                surrounding_vehicles = []
                for v in self.env.unwrapped.road.vehicles:
                    if v is not vehicle:
                        v_lane = self.env.unwrapped.road.network.get_lane(v.lane_index)
                        if v_lane and vehicle.lane == v_lane:
                            # Check if vehicle is ahead and close
                            if v.position[0] > vehicle.position[0] and v.position[0] - vehicle.position[0] < 20:
                                surrounding_vehicles.append(v)
                                
                # Overtaking completed if no vehicles ahead in same lane
                if not surrounding_vehicles:
                    return True
                    
                # If we've moved to a different lane to overtake, that's progress
                if hasattr(self.agent_history, 'lane_idx'):
                    if vehicle.lane_index[2] != self.agent_history.lane_idx:
                        # Changed lanes during overtaking
                        return self.high_level_action_steps > 5  # Require some steps in new lane
            
        elif high_level_action == 'lane_change' or high_level_action == 'left_change' or high_level_action == 'right_change':
            # Check if lane has changed
            if len(self.agent_history) > 0:
                # Get previous lane
                prev_lane = None
                for action, obs, reward in reversed(list(self.agent_history)[:-1]):
                    # Check if the observation is a string (text observation)
                    if isinstance(obs, str):
                        # Try to extract lane information from the text observation
                        if "Currently in lane" in obs:
                            try:
                                # Extract the lane number from text like "Currently in lane 2. "
                                lane_text = obs.split("Currently in lane")[1].split(".")[0].strip()
                                prev_lane = int(lane_text)
                                break
                            except (IndexError, ValueError):
                                continue
                    # If observation is a dictionary (as originally expected)
                    elif isinstance(obs, dict) and 'lane' in obs and obs['lane'] is not None:
                        prev_lane = obs['lane']
                        break
                
                # Get current lane
                current_lane = None
                if hasattr(vehicle, 'lane_index'):
                    current_lane = vehicle.lane_index[2]
                
                # Lane change completed if lane changed and some steps elapsed
                if prev_lane is not None and current_lane is not None and prev_lane != current_lane:
                    return self.high_level_action_steps > 3  # Require some steps in new lane
        
        elif high_level_action == 'keeping_lane':
            # Keeping lane is continuous task, consider it "completed" after certain time
            return self.high_level_action_steps > 10
            
        elif high_level_action == 'brake':
            # Brake completed when speed is low
            if hasattr(vehicle, 'speed'):
                return vehicle.speed < 5  # m/s
        
        # Default completion check based on step count
        return self.high_level_action_steps > 15

    def update_sensors(self, camera_images=None, history=None):
        """
        Update sensor data for planning algorithms
        
        This is a compatibility method that works with both CARLA and Highway
        environments. In Highway, we provide the observation text instead of
        camera images.
        """
        # For Highway Environment, we don't need actual camera images
        # Just create a placeholder dictionary to maintain compatibility with CARLA code
        if camera_images:
            self.camera_images = camera_images
        else:
            # Create placeholder camera image - this won't be used for actual perception
            # The system will prioritize the text observation for Highway Environment
            self.camera_images = {
                'front': np.zeros((100, 100, 3), dtype=np.uint8)
            }
        
        if history:
            self.agent_history = history
        
        # Generate and store the current text observation
        self.current_observation = self._get_observation()

    def _action_to_control(self, action):
        """Convert action to Highway Environment control"""
        if isinstance(action, str) and action in self.action_space:
            return np.array([
                self.action_space[action]['acceleration'],
                self.action_space[action]['steering']
            ])
        elif isinstance(action, list) and len(action) == 2:
            return np.array(action)
        elif isinstance(action, tuple) and len(action) == 2:
            return np.array(action)
        else:
            raise ValueError(f"Invalid action format: {action}")

    def _high_to_low_level_constraints(self, high_level_action):
        """
        Convert high-level action to constraints for low-level MCTS
        
        Args:
            high_level_action: High-level action string
            
        Returns:
            constraints: Dictionary of constraints for low-level MCTS
        """
        # Get action mapping
        action_mapping = self.high_to_low_action_mapping.get(
            high_level_action, 
            {'acceleration': [-0.5, 0.5], 'steering': [-0.2, 0.2]}  # Default if action not found
        )
        
        # Create constraints for Tree
        constraints = {
            'acc_range': action_mapping['acceleration'],
            'steering_range': action_mapping['steering']
        }
        
        return constraints

    def _high2mid_level_action(self, high_level_action):
        """
        Convert high-level action to mid-level action for VLM model
        
        Args:
            high_level_action: High-level action string
            
        Returns:
            mid_level_action: Mid-level action dictionary
        """
        # Map high-level actions to mid-level actions
        mapping = {
            'overtaking': {'acceleration': 1.5, 'steering': 0.2},
            'keeping_lane': {'acceleration': 0.0, 'steering': 0.0},
            'turning_left': {'acceleration': 0.5, 'steering': -0.3},
            'turning_right': {'acceleration': 0.2, 'steering': 0.3},
            'left_change': {'acceleration': 0.5, 'steering': -0.2},
            'right_change': {'acceleration': 0.5, 'steering': 0.2},
            'brake': {'acceleration': -0.5, 'steering': 0.0}
        }
        
        return mapping.get(high_level_action, {'acceleration': 0.3, 'steering': 0.0})

    def _high_level_to_control(self, high_level_action):
        """
        Directly convert high-level action to low-level control
        Used as fallback when low-level MCTS is not available
        
        Args:
            high_level_action: High-level action string
            
        Returns:
            control: List of [acceleration, steering]
        """
        # Map high-level actions to control values
        mapping = {
            'overtaking': [0.8, 0.0],  # [acceleration, steering]
            'keeping_lane': [0.3, 0.0],
            'turning_left': [0.2, -0.2],
            'turning_right': [0.2, 0.2],
            'left_change': [0.3, -0.2],
            'right_change': [0.3, 0.2],
            'brake': [-0.3, 0.0]
        }
        
        return mapping.get(high_level_action, [0.3, 0.0])  # Default to maintain

    def _create_tree_init_state(self):
        """
        Create initial state dictionary for the Tree (low-level MCTS)

        States:
            Map:
            - map: [batch, lane, points, 2]
            - map_mask: [batch, lane, points]
            - additional_map: [batch, lane, points, 2]
            - additional_map_mask: [batch, lane, points]
            Agents:
            - agents: [batch, agent, time, 5]
            - agents_mask: [batch, agent, time, 1]
            - agents_dim: [batch, agent, 2]
            - prediction: [1, n_valid_agents, 1, 80, feature_dimension]
            Ego:
            - ego: [batch, time, 5]
            - ego_pos: [batch, time, 2]
            - ego_speed: [batch, time, 1]
            - ego_heading: [batch, time, 1]
            - ego_pred: [batch, frames, 2]
            Other Road Information:
            - static_objects: [batch, static_objects, 2]
            - static_objects_mask: [batch, static_objects, 1]
            - pedestrian: [batch, pedestrian, 2]
            - pedestrian_mask: [batch, pedestrian, 1]
            Parameters:
            - max_speed: Maximum speed of the vehicle
            - start_time: Start time of the simulation

        
        Returns:
            init_state: Dictionary with state information for Tree
        """
        # Current ego vehicle state
        ego_pos = self.env.unwrapped.vehicle.position
        ego_vel = self.env.unwrapped.vehicle.velocity
        ego_heading = self.env.unwrapped.vehicle.heading
        ego_speed = np.linalg.norm(ego_vel)
        ego_array = np.array([ego_pos[0], ego_pos[1], ego_heading])
        ego = ego_array.reshape(1, 1, 3)  # Reshape to [batch, time, features]

        # Get road and lane information
        if hasattr(self.env.unwrapped, 'road'):
            road = self.env.unwrapped.road
            lanes = []
            
            # Collect lanes from road network
            if hasattr(road, 'network'):
                for _from in road.network.graph:
                    for _to in road.network.graph[_from]:
                        for lane in road.network.graph[_from][_to]:
                            lanes.append(lane)
            
            print(f"Number of lanes: {len(lanes)}")

            # Get vehicle dimensions
            veh_dims = get_vehicle_parameters(self.env.unwrapped.vehicle)
            veh_length = veh_dims['dimensions'][0] if veh_dims else 5.0
            veh_width = veh_dims['dimensions'][1] if veh_dims else 2.0
            
            # Extract waypoints from lanes
            waypoints = []
            for lane in lanes:
                if hasattr(lane, 'position'):
                    # Sample points along the lane
                    # print(f"lane length: {lane.length}")
                    for i in range(0, int(lane.length) - 8, 10):
                        waypoint_pos = lane.position(i, 0)
                        waypoints.append(waypoint_pos)

            print(f"Number of waypoints: {len(waypoints)}")
            print(f"waypoints/length: {len(waypoints) / len(lanes)}")
            # Create map representation from waypoints
            if waypoints:
                # print(f"Extracted {len(waypoints)} waypoints from lanes.")
                map_data = np.array([waypoints]).reshape(1, len(lanes), int(len(waypoints)/len(lanes)), 2)
                map_mask = np.ones((1, len(lanes), int(len(waypoints)/len(lanes))))
            else:
                # Default empty map
                print("No waypoints found in lanes.")
                map_data = np.zeros((1, 0, 1, 2))
                map_mask = np.zeros((1, 0, 1))
                
            # print(f"Map data shape: {map_data.shape}, Map mask shape: {map_mask.shape}")
            # Extract other vehicles information
            vehicles = [v for v in road.vehicles if v is not self.env.unwrapped.vehicle]
            print(f"Vehicles lens: {len(vehicles)}")
            agents_data = []
            
            for i, vehicle in enumerate(vehicles):
                if hasattr(vehicle, 'position') and hasattr(vehicle, 'velocity') and hasattr(vehicle, 'heading'):
                    v_pos = vehicle.position
                    v_vel = vehicle.velocity
                    v_hdg = vehicle.heading
                    v_length = getattr(vehicle, 'LENGTH', 5.0)
                    v_width = getattr(vehicle, 'WIDTH', 2.0)
                    
                    agents_data.append({
                        'id': i,
                        'position': [v_pos[0], v_pos[1]],
                        'velocity': [v_vel[0], v_vel[1]],
                        'heading': v_hdg,
                        'dimensions': [v_length, v_width]
                    })
                    
            # Format agent data for Tree
            # print(f"Number of agents: {len(agents_data)}")
            agents = np.zeros((1, len(agents_data), self.frames, 3))  # Format: [batch, agent, time, features]
            agents_mask = np.ones((1, len(agents_data), self.frames, 1))
            agents_dim = np.zeros((1, len(agents_data), 2))
            
            for i, agent in enumerate(agents_data):
                agents[0, i, :, 0:2] = agent['position'][0:2]  # x, y position
                agents[0, i, :, 2] = agent['heading']  # heading
                agents_dim[0, i, 0:2] = agent['dimensions'][0:2]  # length, width
            
            # Ego vehicle trajectory prediction (simplified)
            ego_pred = np.zeros((1, self.frames, 2))  # frames timesteps of predicted trajectory
            dt = 0.1  # prediction time step
            for i in range(self.frames):
                ego_pred[0, i, 0] = ego_pos[0] + ego_vel[0] * dt * i
                ego_pred[0, i, 1] = ego_pos[1] + ego_vel[1] * dt * i
                
            
            # Create state dictionary for Tree
            init_state = {
                "map": map_data,
                "map_mask": map_mask,
                "additional_map": map_data,  # Same as primary map for simplicity
                "additional_map_mask": map_mask,
                "agents": agents,
                "agents_mask": agents_mask,
                "agents_dim": agents_dim,
                "ego": ego,
                "ego_pos": np.array([[[ego_pos[0], ego_pos[1]]]]).reshape(1, 1, 2),
                "ego_speed": np.array([[[ego_speed]]]).reshape(1, 1, 1),
                "ego_yaw": np.array([[[ego_heading]]]).reshape(1, 1, 1),
                "ego_pred": ego_pred,
                "prediction": np.zeros((1, len(agents_data), 3, 20, 2)),  # [batch, agent, mode, time, xy]
                "static_objects": np.zeros((1, 0, 2)),  # No static objects for Highway Env
                "static_objects_dims": np.zeros((1, 0, 3)),
                "pedestrians": np.zeros((1, 0, 2)),  # No pedestrians in Highway Env
                "pedestrians_dims": np.zeros((1, 0, 3)),
                "max_speed": 30.0,  # Maximum allowed speed (m/s)
                "prior": [None],
                "start_time": time.time()
            }

            # Add belief-based predictions for other vehicles if available
            if hasattr(self, 'belief') and self.belief is not None:
                # Get trajectory predictions from belief system
                prediction_horizon = 2.0  # seconds
                dt = 0.1  # time step
                trajectory_predictions = self.belief.sample_vehicle_trajectories(prediction_horizon, dt)
                
                # Convert to format expected by Tree
                num_agents = len(trajectory_predictions)
                if num_agents > 0:
                    # Get first trajectory to determine shape
                    first_traj = list(trajectory_predictions.values())[0]
                    num_timesteps = first_traj.shape[0]
                    
                    # Initialize prediction array with shape [batch=1, agent, mode=1, time, xy=2]
                    predictions = np.zeros((1, num_agents, 1, num_timesteps, 3))
                    
                    # Fill in predictions
                    agent_ids = list(trajectory_predictions.keys())
                    for i, vehicle_id in enumerate(agent_ids):
                        trajectory = trajectory_predictions[vehicle_id]
                        predictions[0, i, 0, :, :] = trajectory
                    
                    # Add to state dictionary
                    init_state["prediction"] = predictions

            # print(f"====== Initial State ======")
            # # print(f"Agents: {agents}")
            # # print("==============================")
            # # print dimensions of the state
            # # print(f"====== State dimensions ======")
            # print(f"Map shape: {init_state['map'].shape}")
            # print(f"Map mask shape: {init_state['map_mask'].shape}")
            # print(f"Agents shape: {init_state['agents'].shape}")
            # print(f"Agents mask shape: {init_state['agents_mask'].shape}")
            # print(f"Agents dim shape: {init_state['agents_dim'].shape}")
            # print(f"Ego shape: {init_state['ego'].shape}")
            # print(f"Ego position shape: {init_state['ego_pos'].shape}")
            # print(f"Ego speed shape: {init_state['ego_speed'].shape}")
            # print(f"Ego yaw shape: {init_state['ego_yaw'].shape}")
            # print(f"Ego prediction shape: {init_state['ego_pred'].shape}")
            # print(f"Prediction shape: {init_state['prediction'].shape}")
            # print(f"Static objects shape: {init_state['static_objects'].shape}")
            # print(f"Static objects dims shape: {init_state['static_objects_dims'].shape}")
            # print(f"Pedestrians shape: {init_state['pedestrians'].shape}")
            # print(f"Pedestrians dims shape: {init_state['pedestrians_dims'].shape}")
            # print(f"Max speed: {init_state['max_speed']}")
            # print(f"Prior shape: {len(init_state['prior'])}")
            # print(f"Start time: {init_state['start_time']}")
            # print(f"==============================")

            return init_state
        else:
            # Default state if road not available
            return {
                "map": np.zeros((1, 0, 2)),
                "map_mask": np.zeros((1, 0, 1)),
                "additional_map": np.zeros((1, 0, 2)),
                "additional_map_mask": np.zeros((1, 0, 1)),
                "agents": np.zeros((1, 0, 1, 5)),
                "agents_mask": np.zeros((1, 0, 1, 1)),
                "agents_dim": np.zeros((1, 0, 3)),
                "ego_pos": np.array([[[0, 0]]]),
                "ego_speed": np.array([[[0]]]),
                "ego_yaw": np.array([[[0]]]),
                "ego_pred": np.zeros((1, 10, 2)),
                "prediction": np.zeros((1, 0, 3, 20, 2)),
                "static_objects": np.zeros((1, 0, 2)),
                "static_objects_dims": np.zeros((1, 0, 3)),
                "pedestrians": np.zeros((1, 0, 2)),
                "pedestrians_dims": np.zeros((1, 0, 3)),
                "max_speed": 30.0,
                "prior": [None],
                "start_time": time.time()
            }

    def _get_observation(self):
        """Convert current state to text observation for MCTS"""

        if self.env_scenario is not None:
            # Use the scenario's observation method
            return self.env_scenario.describe(0)

        if not hasattr(self.env.unwrapped, 'vehicle'):
            return "No vehicle information available."
        
        # Get vehicle and road information
        vehicle = self.env.unwrapped.vehicle
        speed = getattr(vehicle, 'speed', 0.0)  # Convert m/s to km/h
        current_lane = getattr(vehicle, 'lane_index', None)
        total_lanes = getattr(self.env.unwrapped.road, 'lanes_count', 0)
        
        # Create initial observation text
        observation = f"Ego vehicle in lane {current_lane[2]} of {total_lanes} lanes.  "
        observation += f"moving at {speed:.1f} m/h with heading {vehicle.heading:.1f} radians. "
        observation += f"Currently in lane {current_lane[2]}. "

        def calculate_distance(x, y):
            temp_x = x - vehicle.position[0]
            temp_y = y - vehicle.position[1]
            return np.sqrt(temp_x**2 + temp_y**2)

        # Add other vehicles information
        other_vehicles = []
        if hasattr(self.env.unwrapped, 'road'):
            other_vehicles = [v for v in self.env.unwrapped.road.vehicles if v is not vehicle]
            
            observation += f"There are {len(other_vehicles)} other vehicles nearby. "
            
            for i, other_vehicle in enumerate(other_vehicles):
                if hasattr(other_vehicle, 'position') and hasattr(other_vehicle, 'speed'):
                    vehicle_id = id(other_vehicle) % 1000
                    lane_idx = other_vehicle.lane_index[2]
                    other_speed = getattr(other_vehicle, 'speed', 0.0)
                    observation += f"Vehicle {vehicle_id} in lane {lane_idx}. "
                    observation += f"Distance to ego vehicle: {calculate_distance(other_vehicle.position[0], other_vehicle.position[1]):.1f} m. "
                    if other_speed == speed:
                        observation += f"Moving at the same speed. "
                    elif other_speed > speed:
                        observation += f"Moving faster than ego. "
                    else:
                        observation += f"Moving slower than ego. "
                    
                if self.belief_states and vehicle_id in self.belief_states:
                    belief_state = self.belief_states[vehicle_id]
                    print(f"vehicle_id: {vehicle_id}, predicted action: {belief_state['predicted_actions']}")
                    observation += f"Posible action about vehicle {vehicle_id}: {belief_state['predicted_actions']}. "   
                
        
        # Add navigation information
        nav_info = self._get_navigation_info()
        # observation += f"Next road option: {nav_info['road_option']}. "
        # observation += f"Target speed: {nav_info['target_speed'] * 3.6} km/h. "
        
        return observation

    def render(self, mode='human'):
        """Render the environment"""
        if self.env:
            img = self.env.render(mode=mode)
            # Update front camera image with rendered frame
            if mode == 'rgb_array':
                self.camera_images['front'] = img
            return img
        return None

    def _capture_current_state(self):
        """
        Capture the current state of the environment
        
        Returns:
            state: Dictionary of current state
        """
        # Create state object to store environment state
        state = EnvironmentState()
        
        # Extract ego vehicle state
        if self.ego_vehicle:
            state.vehicle_state = {
                'position': self.ego_vehicle.position,
                'velocity': self.ego_vehicle.velocity,
                'heading': self.ego_vehicle.heading,
                'speed': self.ego_vehicle.speed,
                'lane_index': self.ego_vehicle.lane_index if hasattr(self.ego_vehicle, 'lane_index') else None,
                'lane': self.ego_vehicle.lane_index[2] if hasattr(self.ego_vehicle, 'lane_index') else -1
            }
        
        # Extract information about other vehicles
        if hasattr(self.env.unwrapped, 'road') and hasattr(self.env.unwrapped.road, 'vehicles'):
            other_vehicles = [v for v in self.env.unwrapped.road.vehicles if v is not self.ego_vehicle]
            
            for i, vehicle in enumerate(other_vehicles):
                vehicle_id = id(vehicle) % 1000
                if hasattr(vehicle, 'position') and hasattr(vehicle, 'velocity'):
                    state.env_state[vehicle_id] = {
                        'position': vehicle.position,
                        'velocity': vehicle.velocity,
                        'heading': vehicle.heading if hasattr(vehicle, 'heading') else 0.0,
                        'speed': vehicle.speed if hasattr(vehicle, 'speed') else np.linalg.norm(vehicle.velocity),
                        'lane_index': vehicle.lane_index if hasattr(vehicle, 'lane_index') else None,
                        'lane': vehicle.lane_index[2] if hasattr(vehicle, 'lane_index') else -1,
                        'type': 'vehicle'
                    }
        
        # Extract road and traffic information for the belief system
        if hasattr(self.env.unwrapped, 'road'):
            state.env_state['road'] = {
                'lanes_count': self.env.unwrapped.config.get("lanes_count", 0),
                'lanes': [(lane.index, lane.length, lane.width) 
                         for lane in self.env.unwrapped.road.network.lanes]
                         if hasattr(self.env.unwrapped.road.network, 'lanes') else []
            }
        
        return state
    
    def _update_belief_state(self, current_state, episode=None, step=None):
        """
        Update the belief state based on current observations
        
        Args:
            current_state: Current environment state
        """
        # Initialize belief system if not already done
        self.belief = Belief(
            road_graph=self.env.unwrapped.road if hasattr(self.env.unwrapped, 'road') else None,
            ego_vehicle_id=self.ego_vehicle_id,
            forget_rate=0.05  # Adjust as needed
        )
        
        # Extract vehicle states in the format expected by OtherVehicleAgent
        vehicle_states = {}
        agent_ids = []
        
        for vehicle_id, vehicle_data in current_state.env_state.items():
            if isinstance(vehicle_id, int) and vehicle_id != self.ego_vehicle_id:
                # Only process vehicles, not road or other environment elements
                if vehicle_data.get('type') == 'vehicle':
                    agent_ids.append(vehicle_id)
                    
                    # Extract relevant state information for the belief system
                    vehicle_states[vehicle_id] = {
                        'lane': vehicle_data.get('lane', -1),
                        'speed': vehicle_data.get('speed', 0.0),  # Convert to km/h
                        'position': vehicle_data.get('position'),
                        'heading': vehicle_data.get('heading', 0.0)
                    }
        
        self.agent_ids = agent_ids

        # Create or update OtherVehicleAgent to maintain beliefs about other vehicles
        if not hasattr(self, 'belief_agent') or self.belief_agent is None:
            self.belief_agent = OtherVehicleAgent(self.model, self.action_space, vehicle_states, agent_ids, self.belief)
        else:
            # Update existing belief agent with new observations
            self.belief_agent.current_states = vehicle_states
            self.belief_agent.agent_ids = agent_ids
            self.belief_agent.belief = self.belief
        
        # Update belief system with new observations
        observation = {
            'timestamp': self.timestep,
            'vehicles': {
                vehicle_id: {
                    'position': data.get('position'),
                    'velocity': data.get('velocity'),
                    'heading': data.get('heading', 0.0),
                    'lane_id': data.get('lane', -1)
                }
                for vehicle_id, data in current_state.env_state.items()
                if isinstance(vehicle_id, int) and vehicle_id != self.ego_vehicle_id
                and data.get('type') == 'vehicle'
            }
        }
        
        # Add lane information to observation
        if 'road' in current_state.env_state:
            observation['lanes'] = {
                i: {
                    'drivable': 1.0,
                    'speed_limit': self.env.unwrapped.config.get("reward_speed_range", [20, 30])[1],
                    'width': lane[2] if len(lane) > 2 else 4.0
                }
                for i, lane in enumerate(current_state.env_state['road'].get('lanes', []))
            }
        
        # Update belief with observation
        self.belief.update_from_observation(observation)
        
        # Generate action probabilities for other vehicles
        self.belief_agent.get_action(episode, step)
        
        # Store belief states for planning
        self.belief_states = {
            vehicle_id: {
                'position': self.belief.vehicle_beliefs.get(vehicle_id, {}).get('position'),
                'velocity': self.belief.vehicle_beliefs.get(vehicle_id, {}).get('velocity'),
                'heading': self.belief.vehicle_beliefs.get(vehicle_id, {}).get('heading'),
                'predicted_actions': self.belief.vehicle_beliefs.get(vehicle_id, {}).get('predicted_actions', {})
            }
            for vehicle_id in agent_ids
        }
    
    def _assess_risk(self):
        """
        Assess the risk level of the current state using the belief system
        
        Returns:
            risk_assessment: Dictionary with risk levels for nearby vehicles
        """
        if not hasattr(self, 'belief') or self.belief is None:
            return {'overall_risk': 0.0, 'vehicles': {}}
        
        # Generate a simple ego trajectory prediction based on current velocity
        if self.ego_vehicle and hasattr(self.ego_vehicle, 'position') and hasattr(self.ego_vehicle, 'velocity'):
            dt = 0.1  # Prediction time step
            steps = 10  # Prediction horizon
            ego_trajectory = np.array([
                self.ego_vehicle.position + self.ego_vehicle.velocity * dt * i
                for i in range(steps)
            ])
        else:
            # If no ego vehicle, return minimal risk assessment
            return {'overall_risk': 0.0, 'vehicles': {}}
        
        # Get collision probabilities from belief system
        collision_probs = self.belief.get_collision_probabilities(ego_trajectory)
        
        # Identify risky vehicles
        risky_vehicles = self.belief.get_risky_vehicles(threshold=0.3)
        
        # Calculate overall risk level
        overall_risk = max([0.0] + [probs.get('probability', 0.0) for probs in collision_probs.values()])
        
        # Prepare detailed risk assessment
        risk_assessment = {
            'overall_risk': overall_risk,
            'risky_vehicles': risky_vehicles,
            'vehicles': collision_probs
        }
        
        return risk_assessment
    
    def _execute_action(self, action):
        """
        Execute an action in the environment
        
        Args:
            action: Action to execute (continuous or discrete)
        """
        # Convert action to control format expected by Highway-Env
        control = self._action_to_control(action)
        
        # Apply control to environment
        self.env.unwrapped.vehicle.act(control)
    
    def _calculate_reward(self, action):
        """
        Calculate reward for the current state and action
        
        Args:
            action: Action taken
            
        Returns:
            reward: Calculated reward value
        """
        # Extract components for reward calculation
        if not hasattr(self.env.unwrapped, 'vehicle'):
            return 0.0
            
        speed = self.env.unwrapped.vehicle.speed
        target_speed_range = self.env.unwrapped.config.get("reward_speed_range", [20, 30])
        min_speed, max_speed = target_speed_range
        
        # Check if collision occurred
        collision = self.env.unwrapped._is_terminal_for_other_vehicle_collision() if hasattr(self.env.unwrapped, '_is_terminal_for_other_vehicle_collision') else False
        
        # Check if out of road
        on_road = self.env.unwrapped.vehicle.on_road if hasattr(self.env.unwrapped.vehicle, 'on_road') else True
        
        # Get risk assessment from belief system
        risk = self._assess_risk()
        
        # Calculate reward components
        speed_reward = 0.0
        if min_speed <= speed <= max_speed:
            # Normalized speed reward when within target range
            speed_reward = (speed - min_speed) / (max_speed - min_speed)
        elif speed < min_speed:
            # Penalty for too slow
            speed_reward = -0.5 * (min_speed - speed) / min_speed
        else:
            # Penalty for too fast
            speed_reward = -0.5 * (speed - max_speed) / max_speed
        
        # Safety penalty from belief system
        risk_penalty = -2.0 * risk.get('overall_risk', 0.0)
        
        # Severe penalties for collision or going off-road
        collision_penalty = -10.0 if collision else 0.0
        off_road_penalty = -5.0 if not on_road else 0.0
        
        # Progress reward if applicable
        progress_reward = 0.0
        if hasattr(self, 'goal_checker') and self.goal_checker and hasattr(self.env.unwrapped, 'vehicle'):
            # Calculate progress toward goal
            distance_to_goal = self.goal_checker.get_distance_to_goal(self.env.unwrapped.vehicle)
            if hasattr(self, '_prev_distance_to_goal'):
                progress_made = self._prev_distance_to_goal - distance_to_goal
                progress_reward = max(0, progress_made)
            self._prev_distance_to_goal = distance_to_goal
        
        # Combine reward components with weights
        total_reward = (
            self.reward_weights.get('efficiency', 1.0) * speed_reward +
            self.reward_weights.get('safety', 10.0) * (risk_penalty + collision_penalty + off_road_penalty) +
            self.reward_weights.get('progress', 1.0) * progress_reward
        )
        
        return float(total_reward)
    
    def _check_task_completion(self):
        """
        Check if the current task is completed
        
        Returns:
            done: Whether the task is completed
        """
        # Check if environment reports episode termination
        if hasattr(self.env.unwrapped, 'done'):
            return self.env.unwrapped.done
        
        # Check for collisions
        collision = self.env.unwrapped._is_terminal_for_other_vehicle_collision() if hasattr(self.env.unwrapped, '_is_terminal_for_other_vehicle_collision') else False
        
        # Check if vehicle is off-road
        off_road = not self.env.unwrapped.vehicle.on_road if hasattr(self.env.unwrapped.vehicle, 'on_road') else False
        
        # Check if goal has been reached
        goal_reached = False
        if hasattr(self, 'goal_checker') and self.goal_checker:
            goal_reached = self.goal_checker.check_goal(self.env.unwrapped.vehicle)
        
        # Consider task completed if collision, off-road, or goal reached
        return collision or off_road or goal_reached
    
    def get_valid_actions(self):
        """
        Get valid actions for the current state
        
        Returns:
            valid_actions: List of valid actions
        """
        # Default high-level actions for Highway-Env
        default_actions = ["keeping_lane", "left_change", "right_change"]
        
        # Filter actions based on current state
        valid_actions = default_actions.copy()
        
        # Remove turning actions if not at an intersection
        if not self._is_at_intersection():
            valid_actions = [a for a in valid_actions if "turning" not in a]
        
        # Remove lane change actions if not appropriate
        if self.ego_vehicle and hasattr(self.ego_vehicle, 'lane_index'):
            lane_idx = self.ego_vehicle.lane_index[2]
            lane_count = self.env.unwrapped.config.get("lanes_count", 3)
            
            # Can't change left from leftmost lane
            if lane_idx == 0:
                valid_actions = [a for a in valid_actions if "left_change" not in a]
                
            # Can't change right from rightmost lane
            if lane_idx == lane_count - 1:
                valid_actions = [a for a in valid_actions if "right_change" not in a]
        
        return valid_actions
    
    def _is_at_intersection(self):
        """Check if the vehicle is at an intersection"""
        # Highway-Env typically doesn't have intersections in highway scenarios
        # For scenarios like intersection-v0, we would implement this differently
        return True
    
    def _get_navigation_info(self):
        """Get navigation information for planning"""
        # Get current waypoint
        if not hasattr(self.env.unwrapped, 'vehicle') or not hasattr(self.env.unwrapped, 'road'):
            return {
                'road_option': 'STRAIGHT',
                'target_speed': 30.0,
                'distance_to_goal': float('inf')
            }
        
        vehicle = self.env.unwrapped.vehicle
        
        # Determine road option (lane following direction)
        road_option = 'STRAIGHT'  # Default
        
        # For lane changes, determine direction based on neighboring vehicles
        if hasattr(vehicle, 'lane_index') and hasattr(self.env.unwrapped.road, 'network'):
            lane_idx = vehicle.lane_index[2]
            lane_count = self.env.unwrapped.config.get("lanes_count", 3)
            
            # Check for vehicles in current lane
            vehicles_in_lane = [v for v in self.env.unwrapped.road.vehicles 
                               if v is not vehicle and hasattr(v, 'lane_index') 
                               and v.lane_index[2] == lane_idx
                               and v.position[0] > vehicle.position[0]  # vehicle ahead
                               and v.position[0] - vehicle.position[0] < 50]  # within reasonable distance
            
            if vehicles_in_lane:
                # Consider changing lanes if vehicles ahead
                if lane_idx > 0:  # Can change left
                    left_lane_vehicles = [v for v in self.env.unwrapped.road.vehicles 
                                         if v is not vehicle and hasattr(v, 'lane_index') 
                                         and v.lane_index[2] == lane_idx - 1]
                    if not left_lane_vehicles:
                        road_option = 'LEFT_CHANGE'
                
                if lane_idx < lane_count - 1 and road_option == 'STRAIGHT':  # Can change right and haven't chosen left
                    right_lane_vehicles = [v for v in self.env.unwrapped.road.vehicles 
                                          if v is not vehicle and hasattr(v, 'lane_index') 
                                          and v.lane_index[2] == lane_idx + 1]
                    if not right_lane_vehicles:
                        road_option = 'RIGHT_CHANGE'
        
        # Get distance to goal if available
        distance_to_goal = float('inf')
        if hasattr(self, 'goal_checker') and self.goal_checker:
            distance_to_goal = self.goal_checker.get_distance_to_goal(vehicle)
        
        # Return navigation info
        return {
            'road_option': road_option,
            'target_speed': self.env.unwrapped.config.get("reward_speed_range", [20, 30])[1],
            'distance_to_goal': distance_to_goal
        }

    # def get_vehicle_density(self):
    #     """Get vehicle density in the environment"""
    #     if hasattr(self.env.unwrapped, 'road') and hasattr(self.env.unwrapped.road, 'vehicles'):
    #         vehicles = self.env.unwrapped.road.vehicles
    #         road_length = self.env.unwrapped.road.length
    #         vehicle_density = len(vehicles) / road_length if road_length > 0 else 0
    #         return vehicle_density
    #     return 0

class StateGraph:
    """Graph to store environment states and transitions"""
    def __init__(self):
        self.nodes = []
        self.current_node = None
        self.edges = {}  # Map from node index to list of connected node indices
        self.node_metadata = {}  # Additional data associated with each node

    def add_node(self, node):
        """Add a node to the graph and set it as current"""
        self.nodes.append(node)
        node_idx = len(self.nodes) - 1
        self.current_node = node
        
        # Initialize edges for new node
        if node_idx not in self.edges:
            self.edges[node_idx] = []
        
        # Connect to previous node if exists
        if node_idx > 0:
            self.add_edge(node_idx - 1, node_idx)
            
        return node_idx
        
    def add_edge(self, from_idx, to_idx):
        """Add a directed edge between nodes"""
        if from_idx in self.edges:
            self.edges[from_idx].append(to_idx)
            
    def get_node(self, idx):
        """Get node by index"""
        if 0 <= idx < len(self.nodes):
            return self.nodes[idx]
        return None
        
    def find_path(self, start_idx, end_idx):
        """Find a path between two nodes using BFS"""
        visited = set()
        queue = [[start_idx]]
        
        if start_idx == end_idx:
            return [start_idx]
            
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if node not in visited:
                for neighbor in self.edges.get(node, []):
                    new_path = list(path)
                    new_path.append(neighbor)
                    
                    if neighbor == end_idx:
                        return new_path
                        
                    queue.append(new_path)
                    
                visited.add(node)
                
        return None  # No path found
        
    def add_metadata(self, node_idx, key, value):
        """Associate metadata with a node"""
        if node_idx not in self.node_metadata:
            self.node_metadata[node_idx] = {}
        self.node_metadata[node_idx][key] = value
        
    def get_metadata(self, node_idx, key=None):
        """Get metadata for a node"""
        if node_idx not in self.node_metadata:
            return None
            
        if key is None:
            return self.node_metadata[node_idx]
        return self.node_metadata[node_idx].get(key)
        
    def current_node_idx(self):
        """Get index of current node"""
        if self.current_node is None:
            return None
        return self.nodes.index(self.current_node)


class EnvironmentState:
    """Container for environment state data"""
    def __init__(self):
        self.vehicle_state = {}
        self.env_state = {}
        self.timestamp = time.time()