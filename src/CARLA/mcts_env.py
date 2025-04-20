import math
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append('/home/ubuntu/dockerCarla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')

import carla
import numpy as np
from collections import deque
from copy import deepcopy
import random
import time

# Fix: Replace relative imports with absolute imports
from src.VLM.belief import OtherVehicleAgent, Belief
from src.MCTS.utils import get_simulation_params
from src.MCTS.high_level_mcts import MCTSAgent as HighLevelMCTS
from src.MCTS.tree import Tree as LowLevelMCTS
from src.CARLA.state_graph import StateGraph
from src.CARLA.goal_checker import GoalChecker
from src.VLM.policy import refinement_policy_id


class MCTSEnv:
    def __init__(self, carla_host='localhost', carla_port=2000, road_network=None):
        # Carla connection
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        
        # Environment state management
        self.state_graph = StateGraph()
        self.agent_history = deque(maxlen=100)
        self.belief_states = {}
        
        # Goal tracking
        self.current_goal = None
        self.goal_checker = GoalChecker()
        
        # Environment parameters
        self.ego_vehicle = None
        self.ego_vehicle_id = 0
        self.sensors = {}
        self.camera_images = {}
        self.timestep = 0
        self.road_network = road_network or self.map
        
        # Belief system
        self.belief = None
        self.belief_agent = None
        
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
            'turning_left': {'acceleration': [0.0, 0.3], 'steering': [-0.5, -0.2]},
            'turning_right': {'acceleration': [0.0, 0.3], 'steering': [0.2, 0.5]},
            'left_change': {'acceleration': [0.0, 0.5], 'steering': [-0.3, -0.1]},
            'right_change': {'acceleration': [0.0, 0.5], 'steering': [0.1, 0.3]},
            'brake': {'acceleration': [-3.0, -0.5], 'steering': [-0.1, 0.1]},
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

    def initialize_low_level_mcts(self, policy=None):
        """
        Initialize the low-level MCTS Tree with current environment state
        
        Args:
            policy: Optional policy network for low-level MCTS
        """
        if not self.ego_vehicle:
            raise ValueError("Ego vehicle not initialized. Call reset() first.")
        
        # Prepare initial state for Tree
        init_state = self._create_tree_init_state()
        
        # Initialize Tree with current world state
        self.low_level_mcts = LowLevelMCTS(
            world=self.world,
            ego_vehicle=self.ego_vehicle,
            init_state=init_state,
            policy=policy
        )
        
        return self.low_level_mcts

    def reset(self, start_transform=None):
        """Reset environment to initial state"""
        # Clean up existing actors
        if self.ego_vehicle:
            for sensor in self.sensors.values():
                sensor.destroy()
            self.ego_vehicle.destroy()
        
        # Create new vehicle
        vehicle_bp = self.blueprint_lib.filter('vehicle.tesla.model3')[0]
        if start_transform:
            transform = start_transform
        else:
            transform = random.choice(self.map.get_spawn_points())
            
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, transform)
        self.ego_vehicle.set_autopilot(False)
        
        # Setup sensors (camera, lidar, etc.)
        self._setup_sensors()
        
        # Initialize state and belief
        self.timestep = 0
        self.agent_history.clear()
        initial_state = self._capture_current_state()
        self.state_graph.add_node(initial_state)
        self._update_belief_state(initial_state)
        
        # Get observation for agent
        observation = self._get_observation()
        valid_actions = self.get_valid_actions()
        
        # Return initial observation and action space
        return observation, valid_actions

    def step(self, action):
        """Execute action and return environment feedback"""
        # Validate action
        if not self._is_action_valid(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Execute action in simulator
        self._execute_action(action)
        self.world.tick()  # Progress simulation
        self.timestep += 1
        
        # Update environment state
        current_state = self._capture_current_state()
        self.state_graph.add_node(current_state)
        
        # Update belief system
        self._update_belief_state(current_state)
        
        # Calculate reward and check if done
        observation = self._get_observation()
        reward = self._calculate_reward(action)
        done = self._check_task_completion()
        valid_actions = self.get_valid_actions()
        
        # Update history
        self.agent_history.append((action, observation, reward))
        
        # Create info dict with additional data for MCTS
        info = {
            'timestep': self.timestep,
            'belief_states': deepcopy(self.belief_states),
            'camera_images': self.camera_images,
            'risk_assessment': self._assess_risk() if self.belief else None
        }
        
        return observation, reward, done, info, valid_actions

    def plan_next_action(self, planning_time=0.1):
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
        
        # 1. First, run high-level MCTS to get a high-level action
        observation = self._get_observation()
        history = [item[0] for item in self.agent_history]  # Extract actions from history
        valid_actions = self.get_valid_actions()
        
        high_level_start = time.time()
        high_level_action = self.high_level_mcts.search(
            observation, 
            history, 
            self.timestep, 
            valid_actions, 
            False,  # not done
            self.camera_images, 
            self._get_navigation_info()
        )
        high_level_time = time.time() - high_level_start
        metrics['high_level_time'] = high_level_time
        
        # 2. If we have a low-level MCTS, use it to refine the high-level action
        low_level_action = None
        if self.low_level_mcts:
            # Convert high-level action to constraints for low-level MCTS
            constraints = self._high_to_low_level_constraints(high_level_action)
            
            # Update low-level MCTS state and sensor data
            tree_state = self._create_tree_init_state()
            self.low_level_mcts.update_state(tree_state)
            self.low_level_mcts.update_sensors(self.camera_images, history)
            
            # Run low-level MCTS with remaining planning time
            remaining_time = planning_time - (time.time() - start_time)
            low_level_start = time.time()
            
            # Calculate number of simulations based on remaining time
            k = max(32, int(remaining_time * 100))  # Adaptive simulation count
            
            # Apply constraints to low-level MCTS (implementation specific)
            # For example, could modify action ranges in the tree
            
            # Run low-level MCTS simulation
            acc_seq, steering_seq = self.low_level_mcts.simulate(k=k)
            
            # Extract the immediate control action from the sequences
            acc = acc_seq[0, 0]  # First acceleration value
            steering = steering_seq[0, 0]  # First steering value
            
            # Convert to CARLA control (throttle/brake split from acceleration)
            if acc >= 0:
                throttle, brake = acc, 0.0
            else:
                throttle, brake = 0.0, -acc
                
            low_level_action = (throttle, brake, steering)
            low_level_time = time.time() - low_level_start
            metrics['low_level_time'] = low_level_time
        else:
            # If no low-level MCTS, convert high-level action directly to control
            low_level_action = self._high_level_to_control(high_level_action)
        
        total_time = time.time() - start_time
        metrics['total_time'] = total_time
        metrics['high_level_action'] = high_level_action
        
        return low_level_action, metrics

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

    def _high_level_to_control(self, high_level_action):
        """
        Directly convert high-level action to low-level control
        Used as fallback when low-level MCTS is not available
        
        Args:
            high_level_action: High-level action string
            
        Returns:
            control: Tuple of (throttle, brake, steering)
        """
        # Map high-level actions to control values
        mapping = {
            'overtaking': (0.8, 0.0, 0.0),  # (throttle, brake, steer)
            'keeping_lane': (0.5, 0.0, 0.0),
            'turning_left': (0.4, 0.0, -0.3),
            'turning_right': (0.4, 0.0, 0.3),
            'left_change': (0.5, 0.0, -0.2),
            'right_change': (0.5, 0.0, 0.2),
            'brake': (0.0, 0.8, 0.0)
        }
        
        return mapping.get(high_level_action, (0.5, 0.0, 0.0))  # Default to maintain

    def _create_tree_init_state(self):
        """
        Create initial state dictionary for the Tree (low-level MCTS)
        
        Returns:
            init_state: Dictionary with state information for Tree
        """
        # Current ego vehicle state
        ego_pos = self.ego_vehicle.get_location()
        ego_vel = self.ego_vehicle.get_velocity()
        ego_rot = self.ego_vehicle.get_transform().rotation
        ego_speed = np.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
        
        # Get waypoints for map information
        current_wp = self.map.get_waypoint(ego_pos)
        waypoints = []
        wp = current_wp
        for _ in range(20):  # Get 20 waypoints ahead
            next_wps = wp.next(5.0)  # 5 meters between waypoints
            if not next_wps:
                break
            wp = next_wps[0]
            waypoints.append(wp)
        
        # Extract vehicle and obstacle information
        vehicles = self.world.get_actors().filter('vehicle.*')
        agents_data = []
        
        for vehicle in vehicles:
            if vehicle.id != self.ego_vehicle.id:  # Skip ego vehicle
                v_loc = vehicle.get_location()
                v_vel = vehicle.get_velocity()
                v_rot = vehicle.get_transform().rotation
                v_bbox = vehicle.bounding_box
                
                agents_data.append({
                    'id': vehicle.id,
                    'position': [v_loc.x, v_loc.y, v_loc.z],
                    'velocity': [v_vel.x, v_vel.y, v_vel.z],
                    'rotation': [v_rot.pitch, v_rot.yaw, v_rot.roll],
                    'dimensions': [v_bbox.extent.x*2, v_bbox.extent.y*2, v_bbox.extent.z*2]
                })
        
        # Create map and agents data
        map_data = np.array([[[wp.transform.location.x, wp.transform.location.y] for wp in waypoints]])
        map_mask = np.ones((1, len(waypoints), 1))
        
        # Format agent data for Tree
        agents = np.zeros((1, len(agents_data), 1, 5))  # Format: [batch, agent, time, features]
        agents_mask = np.ones((1, len(agents_data), 1, 1))
        agents_dim = np.zeros((1, len(agents_data), 3))
        
        for i, agent in enumerate(agents_data):
            agents[0, i, 0, 0:2] = agent['position'][0:2]  # x, y position
            agents[0, i, 0, 2] = np.arctan2(agent['velocity'][1], agent['velocity'][0])  # heading
            agents[0, i, 0, 3:5] = agent['velocity'][0:2]  # vx, vy
            agents_dim[0, i] = agent['dimensions']
        
        # Ego vehicle trajectory prediction (simplified)
        ego_pred = np.zeros((1, 10, 2))  # 10 timesteps of predicted trajectory
        for i in range(10):
            ego_pred[0, i, 0] = ego_pos.x + ego_vel.x * 0.1 * i
            ego_pred[0, i, 1] = ego_pos.y + ego_vel.y * 0.1 * i
        
        # Create state dictionary for Tree
        init_state = {
            "map": map_data,
            "map_mask": map_mask,
            "additional_map": map_data,  # Same as primary map for simplicity
            "additional_map_mask": map_mask,
            "agents": agents,
            "agents_mask": agents_mask,
            "agents_dim": agents_dim,
            "ego_pos": np.array([[[ego_pos.x, ego_pos.y]]]),
            "ego_speed": np.array([[[ego_speed]]]),
            "ego_yaw": np.array([[[ego_rot.yaw * np.pi / 180.0]]]),
            "ego_pred": ego_pred,
            "prediction": np.zeros((1, len(agents_data), 3, 20, 2)),  # Default empty predictions
            "static_objects": np.zeros((1, 0, 2)),  # No static objects for simplicity
            "static_objects_dims": np.zeros((1, 0, 3)),
            "pedestrians": np.zeros((1, 0, 2)),  # No pedestrians for simplicity
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
            if len(agents_data) > 0 and len(trajectory_predictions) > 0:
                # Initialize prediction array with shape [batch=1, agent, mode=3, time=20, xy=2]
                predictions = np.zeros((1, len(agents_data), 3, 20, 2))
                
                # Map vehicle IDs to indices in agents_data
                vehicle_id_to_idx = {agent['id']: i for i, agent in enumerate(agents_data)}
                
                # Fill in predictions for agents that have predictions in the belief system
                for vehicle_id, trajectory in trajectory_predictions.items():
                    if vehicle_id in vehicle_id_to_idx:
                        idx = vehicle_id_to_idx[vehicle_id]
                        
                        # Ensure trajectory has the expected number of timesteps
                        traj_len = min(trajectory.shape[0], 20)
                        for mode in range(3):  # 3 modes for CARLA
                            predictions[0, idx, mode, :traj_len, :] = trajectory[:traj_len]
                
                # Replace default predictions with belief-based predictions
                init_state["prediction"] = predictions
        
        return init_state

    def _setup_sensors(self):
        """Setup vehicle sensors (cameras, lidar, etc.)"""
        # Front camera
        cam_bp = self.blueprint_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        
        front_cam = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.ego_vehicle)
        front_cam.listen(lambda image: self._process_camera_image(image, 'front'))
        self.sensors['front_camera'] = front_cam
        
        # Add other sensors as needed (lidar, semantic cameras, etc.)

    def _process_camera_image(self, image, camera_id):
        """Process camera images for use in belief system and MCTS"""
        # Convert Carla raw image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Store image for use by VLM and MCTS
        self.camera_images[camera_id] = array

    def _update_belief_state(self, new_state=None):
        """Update POMDP belief state with new observations"""
        if new_state is None:
            return
            
        # Extract surrounding vehicle states
        vehicle_states = {}
        agent_ids = []
        
        for idx, (location, velocity) in enumerate(new_state.env_state['vehicles']):
            # Skip ego vehicle
            if self._is_ego_vehicle(location):
                continue
                
            # Get lane and speed
            lane = self._calculate_lane(location)
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s to km/h
            
            vehicle_states[idx] = {
                'lane': lane,
                'speed': speed,
                'position': (location.x, location.y, location.z),
                'velocity': (velocity.x, velocity.y, velocity.z)
            }
            agent_ids.append(idx)
        
        # Initialize or update belief agent
        if self.belief_agent is None:
            self.belief = Belief(road_graph=self.road_network, ego_vehicle_id=self.ego_vehicle_id)
            self.belief_agent = OtherVehicleAgent(vehicle_states, agent_ids, belief=self.belief)
        else:
            self.belief_agent.current_states = vehicle_states
            self.belief_agent.agent_ids = agent_ids
            
        # Update belief with new observations
        observation_data = {
            'vehicles': vehicle_states,
            'traffic_signals': self._get_traffic_signal_states(),
            'timestamp': self.timestep
        }
        self.belief.update_from_observation(observation_data)
        
        # Get predicted actions from VLM
        self.belief_agent.get_action()
        
        # Store belief states for MCTS
        self.belief_states = {
            'vehicle_beliefs': deepcopy(self.belief_agent.action_probs),
            'trajectory_predictions': self.belief.sample_vehicle_trajectories(
                prediction_horizon=self.planning_horizon
            ) if hasattr(self.belief, 'sample_vehicle_trajectories') else {}
        }

    def _assess_risk(self):
        """Assess risk based on belief system predictions"""
        if not self.belief or not hasattr(self.belief, 'get_collision_probabilities'):
            return {}
            
        # Get ego vehicle planned trajectory (simplified)
        ego_trajectory = self._get_ego_trajectory()
        
        # Get collision probabilities
        collision_probs = self.belief.get_collision_probabilities(ego_trajectory)
        
        # Get list of risky vehicles
        risky_vehicles = []
        for vehicle_id, data in collision_probs.items():
            if data['probability'] > 0.3:  # Risk threshold
                risky_vehicles.append({
                    'id': vehicle_id,
                    'collision_prob': data['probability'],
                    'time_to_collision': data.get('time_index', 0) * 0.1  # Assuming 0.1s timestep
                })
                
        return {
            'collision_probabilities': collision_probs,
            'risky_vehicles': risky_vehicles
        }

    def _get_ego_trajectory(self, horizon=5.0, dt=0.1):
        """Get predicted ego vehicle trajectory based on current state"""
        # Simple constant velocity model for now
        num_steps = int(horizon / dt)
        trajectory = []
        
        location = self.ego_vehicle.get_location()
        velocity = self.ego_vehicle.get_velocity()
        
        for i in range(num_steps):
            # Project position forward using velocity
            next_x = location.x + velocity.x * dt * i
            next_y = location.y + velocity.y * dt * i
            trajectory.append(np.array([next_x, next_y]))
            
        return np.array(trajectory)

    def _get_navigation_info(self):
        """Get navigation information for planning"""
        # Get current waypoint
        location = self.ego_vehicle.get_location()
        current_wp = self.map.get_waypoint(location)
        
        # Get next waypoints
        next_wps = current_wp.next(10.0)  # Get waypoints 10m ahead
        
        # Extract road option (straight, left, right)
        road_option = 'STRAIGHT'
        if next_wps:
            next_wp = next_wps[0]
            if current_wp.road_id != next_wp.road_id:
                # Junction detected
                angle = np.abs(current_wp.transform.rotation.yaw - next_wp.transform.rotation.yaw)
                if angle > 45 and angle < 135:
                    if current_wp.transform.rotation.yaw > next_wp.transform.rotation.yaw:
                        road_option = 'RIGHT'
                    else:
                        road_option = 'LEFT'
                elif angle >= 135:
                    road_option = 'STRAIGHT'
        
        # Return navigation info
        return {
            'current_waypoint': current_wp,
            'next_waypoints': next_wps,
            'road_option': road_option,
            'target_speed': 30.0,  # km/h, adjust as needed
            'distance_to_goal': self._get_distance_to_goal()
        }

    def copy_for_simulation(self):
        """Create a copy of the environment for MCTS simulation"""
        new_env = deepcopy(self)
        
        # Disconnect from real Carla
        new_env.client = None
        new_env.world = None
        new_env.ego_vehicle = None
        new_env.sensors = {}
        
        # Copy the belief system
        if self.belief_agent is not None:
            new_env.belief_agent = deepcopy(self.belief_agent)
            
        return new_env

    def _get_action_space(self):
        """Define the action space for planning"""
        return {
            'accelerate': {'throttle': 0.8, 'brake': 0.0, 'steer': 0.0},
            'brake': {'throttle': 0.0, 'brake': 0.8, 'steer': 0.0},
            'maintain': {'throttle': 0.5, 'brake': 0.0, 'steer': 0.0},
            'turn_left': {'throttle': 0.5, 'brake': 0.0, 'steer': -0.5},
            'turn_right': {'throttle': 0.5, 'brake': 0.0, 'steer': 0.5},
            'lane_left': {'throttle': 0.5, 'brake': 0.0, 'steer': -0.2},
            'lane_right': {'throttle': 0.5, 'brake': 0.0, 'steer': 0.2},
            'emergency_stop': {'throttle': 0.0, 'brake': 1.0, 'steer': 0.0}
        }

    def _execute_action(self, action):
        """Execute action in Carla simulator"""
        if action in self.action_space:
            control = carla.VehicleControl(
                throttle=self.action_space[action]['throttle'],
                steer=self.action_space[action]['steer'],
                brake=self.action_space[action]['brake'],
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )
            self.ego_vehicle.apply_control(control)
        else:
            # Parse custom action format if not in predefined actions
            try:
                throttle, brake, steer = action
                control = carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(steer),
                    brake=float(brake)
                )
                self.ego_vehicle.apply_control(control)
            except:
                raise ValueError(f"Invalid action format: {action}")

    def _calculate_reward(self, action):
        """Calculate reward based on safety, progress, efficiency, and comfort"""
        # Get current state
        ego_location = self.ego_vehicle.get_location()
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_speed = np.sqrt(ego_velocity.x**2 + ego_velocity.y**2) * 3.6  # km/h
        
        # Safety reward (collision avoidance)
        safety_reward = 0.0
        if self.belief and hasattr(self.belief, 'get_collision_probabilities'):
            collision_probs = self.belief.get_collision_probabilities(self._get_ego_trajectory())
            safety_risk = max([data['probability'] for _, data in collision_probs.items()], default=0.0)
            safety_reward = -10.0 * safety_risk  # Penalize high collision probabilities
        
        # Progress reward
        progress_reward = 0.0
        if len(self.agent_history) > 0:
            prev_location = self.state_graph.nodes[-2].vehicle_state['location']
            distance_traveled = np.sqrt(
                (ego_location.x - prev_location.x)**2 +
                (ego_location.y - prev_location.y)**2
            )
            progress_reward = distance_traveled * 0.5  # Reward for making progress
            
            # Add goal-oriented progress component
            goal_progress = self._get_goal_progress()
            progress_reward += goal_progress * 2.0
        
        # Efficiency reward
        target_speed = 30.0  # km/h
        speed_diff = abs(ego_speed - target_speed)
        efficiency_reward = -0.05 * speed_diff  # Penalize deviation from target speed
        
        # Comfort reward
        comfort_reward = 0.0
        if action in ['emergency_stop', 'brake']:
            comfort_reward = -0.5  # Penalize harsh braking
        elif action in ['turn_left', 'turn_right']:
            comfort_reward = -0.2  # Small penalty for turns
            
        # Combine rewards
        total_reward = (
            self.reward_weights['safety'] * safety_reward +
            self.reward_weights['progress'] * progress_reward +
            self.reward_weights['efficiency'] * efficiency_reward +
            self.reward_weights['comfort'] * comfort_reward
        )
        
        return total_reward

    # Helper methods (simplified implementations)
    def _is_ego_vehicle(self, location):
        if not self.ego_vehicle:
            return False
        ego_loc = self.ego_vehicle.get_location()
        distance = np.sqrt(
            (location.x - ego_loc.x)**2 +
            (location.y - ego_loc.y)**2
        )
        return distance < 0.5  # Within 0.5m is considered ego vehicle
        
    def _calculate_lane(self, location):
        waypoint = self.map.get_waypoint(location)
        return waypoint.lane_id
        
    def _get_traffic_signal_states(self):
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light*')
        return {tl.id: {'state': str(tl.state), 'position': (tl.get_location().x, tl.get_location().y, tl.get_location().z)} 
                for tl in traffic_lights}
                
    def _get_distance_to_goal(self):
        # Simplified - in real implementation, use actual goal location
        return 100.0  # Some default value
        
    def _get_goal_progress(self):
        # Simplified - in real implementation, calculate actual progress towards goal
        return 0.1  # Some default progress value
        
    def _is_action_valid(self, action):
        return action in self.action_space or isinstance(action, tuple) and len(action) == 3
        
    def _get_observation(self):
        """Convert current state to text observation for MCTS"""
        if not self.state_graph.current_node:
            return "Environment not initialized"
            
        ego_state = self.state_graph.current_node.vehicle_state
        speed = np.sqrt(ego_state['velocity'].x**2 + ego_state['velocity'].y**2) * 3.6  # km/h
        
        observation = f"Vehicle at location ({ego_state['location'].x:.1f}, {ego_state['location'].y:.1f}) "
        observation += f"moving at {speed:.1f} km/h with heading {ego_state['heading'].yaw:.1f} degrees. "
        
        # Add surrounding vehicles info
        vehicles = self.state_graph.current_node.env_state['vehicles']
        observation += f"There are {len(vehicles)-1} other vehicles nearby. "  # -1 to exclude ego
        
        # Add traffic light info
        lights = self.state_graph.current_node.env_state.get('traffic_lights', [])
        observation += f"Traffic lights: {', '.join(str(light) for light in lights)}. "
        
        # Add navigation info
        nav_info = self._get_navigation_info()
        observation += f"Next road option: {nav_info['road_option']}. "
        observation += f"Target speed: {nav_info['target_speed']} km/h. "
        
        return observation

