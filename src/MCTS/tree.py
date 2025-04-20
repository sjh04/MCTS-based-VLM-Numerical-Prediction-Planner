from time import perf_counter

import numpy as np
import sys
import os
import gymnasium as gym
import highway_env

from typing import Dict, List, Tuple, Optional, Any, Union

# from src.HighwayEnv.mcts_env import MCTSEnv
from src.MCTS.node import Node
from src.MCTS.utils import (
    check_drivable_area,
    check_ego_collisions_idx,
    get_directional_values,
    split_with_ratio,
    trajectory2action,
    highway_to_numpy_transform,
    highway_lanes_to_path,
    get_vehicle_parameters,
)


class Tree:
    """
    MCTS tree class, used for path planning in Highway Environment
    """

    def __init__(
        self,
        env: gym.Env,
        mcts_env,
        ego_vehicle: Any,
        init_state: Dict,
        policy,
        high_level_action,
        mid_action,
        action: Optional[Tuple[float, float]] = None,
        root: Optional[Node] = None,
        count: int = 0,
    ):
        """
        Initialize MCTS tree for Highway Environment
        
        Args:
            env: Highway Environment object
            ego_vehicle: Highway Environment vehicle object
            init_state: Initial state dictionary
            policy: Policy network for action prediction
            action: Initial action tuple (acceleration, steering)
            root: Root node (if reusing an existing tree)
            count: Simulation count
        """
        self.env = env
        self.mcts_env = mcts_env
        self.ego_vehicle = ego_vehicle
        self.count = count
        self.done = False
        
        # Get vehicle physics parameters from Highway Environment
        self.vehicle_physics = get_vehicle_parameters(ego_vehicle)
        
        # Store sensor data from Highway Environment
        self.camera_image = None
        self.history = []
        self.mid_action = mid_action
        self.high_level_action = high_level_action
        
        # Action space parameters - tuned for Highway Environment vehicle controls
        self.acc_target_range = [-3, 3]  # m/s² (Highway Environment typical range)
        self.steering_target_range = [-np.pi/4, np.pi/4]  # rad (Highway Environment typical range)
        self.acc_values = 13  # Number of discrete acceleration values
        self.steering_values = 13  # Number of discrete steering values
        
        # Calculate conversion coefficients
        self.acc_coef = (self.acc_target_range[1] - self.acc_target_range[0]) / (self.acc_values - 1)
        self.steer_coef = (self.steering_target_range[1] - self.steering_target_range[0]) / (self.steering_values - 1)
        
        # Action space mapping
        self.default_possible_actions = np.arange(self.steering_values * self.acc_values)
        self.default_global2possible = {
            self.default_possible_actions[i]: i for i in range(len(self.default_possible_actions))
        }
        
        # Masks and penalties for actions
        self.masks = {}
        self.masks_action = {}
        self.dt = 0.1  # Highway Environment default time step
        
        # Get Highway Environment vehicle parameters
        if self.vehicle_physics:
            self.wheel_base = self.vehicle_physics.get('wheelbase', 3.5)  # Default if not specified
            length, width, _ = self.vehicle_physics.get('dimensions', (5.0, 2.0, 1.5))
            self.rear2center = length / 4  # Estimate for center of mass position from rear axle
        else:
            self.wheel_base = 3.5  # Default wheelbase for Highway Environment
            self.rear2center = 1.25  # Default center of mass position
        
        self.max_speed = init_state["max_speed"]  # Maximum allowed speed (m/s)
        
        # Adjust vehicle center of mass offset if needed
        offset_vector = np.array([self.rear2center, 0])[None, None]
        init_state["ego_pos"] = init_state["ego_pos"] + offset_vector
        if "map" in init_state:
            init_state["map"] = init_state["map"] + offset_vector
        if "additional_map" in init_state:
            init_state["additional_map"] = init_state["additional_map"] + offset_vector
        
        # Tree parameters
        self.n_nodes = 1
        self.Ts = np.zeros(81)
        self.nodes = {}
        self.mask_init = None
        self.penalty_init = None
        
        # Initialize action masks if initial action is provided
        if action is not None:
            a, st = action
            a = np.fix(a / self.acc_coef) + 6  # Convert to discrete index
            st = np.fix(st / self.steer_coef) + 6  # Convert to discrete index
            action = (a, st)

            # Create masks based on initial action
            acc_vals = np.arange(self.acc_values)
            st_vals = np.arange(self.steering_values)
            acc_dif = np.abs(acc_vals - a)
            st_diff = np.abs(st_vals - st)
            speed = init_state["ego_speed"][0, -1][0]
            
            # Allow larger deviation at low speeds
            mask_a = (acc_dif < 5) | (speed < 0.1) * (a < self.acc_values // 2) * (
                acc_vals < (self.acc_values // 2 + 3)
            )
            mask_st = st_diff < 5
            self.mask_init = mask_a[:, None] * mask_st[None, :]
            self.penalty_init = np.zeros_like(self.mask_init)

            # Get normal action masks
            mask_normal, penalty, _ = self.get_action_masks(None, speed)
            self.mask_init = self.mask_init.flatten() * mask_normal
            self.penalty_init = self.penalty_init.flatten() + penalty
        
        # Filter out vehicles that are too close (likely causing collision at start)
        if "ego_pos" in init_state and "ego_yaw" in init_state and "agents" in init_state:
            predict_traj, predict_yaw = init_state["ego_pos"][:, -1], init_state["ego_yaw"][:, -1:]
            predicted_xy = predict_traj[:, :]
            predicted_yaw = predict_yaw[:, :]
            agents_xy = init_state["agents"][0, :, -1:, :2]
            agents_yaw = init_state["agents"][0, :, -1:, 2:3]
            mask_agents = init_state["agents_mask"][0, :, -1, 0]
            dim_agents = init_state.get("agents_dim", None)
            
            # Check for initial collisions if agent dimensions available
        if dim_agents is not None and agents_xy.size > 0:
            is_collision = check_ego_collisions_idx(
                    predicted_xy,
                    predicted_yaw,
                    agents_xy[None],
                    agents_yaw[None],
                    mask_agents[None],
                    margin=[0.1, 0.1],
                    other_dims=dim_agents,
            )
                
            # Remove vehicles that are in collision with ego
            if is_collision.sum():
                print("Collision detected with ego vehicle")
                is_collision = is_collision > 0
                init_state["agents"] = init_state["agents"][0][~is_collision[0]][None]
                init_state["agents_mask"] = init_state["agents_mask"][0][~is_collision[0]][None]
                if "agents_dim" in init_state:
                    init_state["agents_dim"] = init_state["agents_dim"][0][~is_collision[0]][None]
                if "prediction" in init_state:
                    init_state["prediction"] = init_state["prediction"][0][~is_collision[0]][None]
        
        # Store policy
        self.policy = policy
        self.map_encoding = None
        
        # MCTS parameters
        self.max_T = 20  # Maximum planning horizon
        self.frames_history = 10  # Number of frames to keep in history
        self.n_actions = self.acc_values * self.steering_values
        self.prediction = None
        self.eval_frames = 10  # Number of frames for evaluation
        self.action_frames = 30  # Number of frames for action planning
        self.eval_ratio = 5
        
        # PUCT algorithm parameters
        self.c_puct = 2  # Exploration constant
        self.tau = 1  # Temperature for action selection
        self.relu = lambda x: x * (x > 0)  # ReLU activation
        
        # Get map information
        if "map" in init_state and "map_mask" in init_state:
            # print("init Map shape:", init_state["map"].shape, init_state["map_mask"].shape)
            self.map_info = self.compute_map_infos(map=init_state["map"], map_mask=init_state["map_mask"])
        else:
            # Get map information from Highway Environment
            if hasattr(self.env, 'road'):
                # Extract lane information from Highway Environment
                lanes = self.env.road.lanes if hasattr(self.env.road, 'lanes') else []
                positions, headings, masks = highway_lanes_to_path(lanes)
                
                # Create dummy map info structure
                self.map_info = (
                    positions,         # map
                    headings,          # map_yaw
                    masks,             # map_mask
                    np.zeros((1, len(lanes), 2)),  # map_avg_tan (placeholder)
                    np.zeros((1, len(lanes), 2)),  # map_avg_norm (placeholder)
                    np.ones((1, len(lanes))),      # max_lat (placeholder)
                    np.ones((1, len(lanes)))       # max_tan (placeholder)
                )
            else:
                # Create empty map info
                self.map_info = (
                    np.zeros((1, 0, 0, 2)),  # map
                    np.zeros((1, 0, 0, 1)),  # map_yaw
                    np.zeros((1, 0, 0, 1)),  # map_mask
                    np.zeros((1, 0, 2)),     # map_avg_tan
                    np.zeros((1, 0, 2)),     # map_avg_norm
                    np.zeros((1, 0)),        # max_lat
                    np.zeros((1, 0))         # max_tan
                )
        
        # Initialize behind mask
        # if "agents" in init_state and init_state["agents"].size > 0:
        self.behind_mask = init_state["agents"][:, :, -1, 0] > 0
        # else:
        #     self.behind_mask = np.array([[]])
        
        # Compute other vehicle speeds
        if "prediction" in init_state and init_state["prediction"].size > 0:
            agents_pos = init_state["prediction"][0, :, 0, 10, :2] - init_state["prediction"][0, :, 0, 0, :2]
            norm = np.sqrt((agents_pos**2).sum(-1))
            self.other_speeds = norm
        else:
            self.other_speeds = np.array([])
        
        # Compute total map information
        if "additional_map" in init_state and "additional_map_mask" in init_state:
            map = init_state["additional_map"]
            mask = init_state["additional_map_mask"]
            self.map_info_total = self.compute_map_infos(map=map, map_mask=mask)
        else:
            # Use the same map info for total map
            self.map_info_total = self.map_info
        
        # Create buffers for simulation
        if "agents" in init_state and init_state["agents"].size > 0:
            batch_size, n_agents, ntime, n_features = init_state["agents"].shape
            self.zeros = np.zeros((batch_size, 1 + n_agents, self.eval_frames, n_features))
        else:
            self.zeros = np.zeros((1, 1, self.eval_frames, 4))  # Default shape
        
        # Store static objects and pedestrians
        if "static_objects" in init_state:
            static_objects = init_state["static_objects"][:, :, None]
            static_mask = np.ones_like(static_objects)[..., 0]
            static_dims = init_state.get("static_objects_dims", np.ones_like(static_objects)[..., :2])
            self.static_objects = static_objects, static_dims, static_mask
        else:
            # Empty static objects
            self.static_objects = np.zeros((1, 0, 1, 3)), np.zeros((1, 0, 2)), np.zeros((1, 0, 1))
        
        if "pedestrians" in init_state:
            pedestrians = init_state["pedestrians"][:, :, None]
            pedestrian_mask = np.ones_like(pedestrians)[..., 0]
            pedestrian_dims = init_state.get("pedestrians_dims", np.ones_like(pedestrians)[..., :2])
            self.pedestrians = pedestrians, pedestrian_dims, pedestrian_mask
        else:
            # Empty pedestrians
            self.pedestrians = np.zeros((1, 0, 1, 3)), np.zeros((1, 0, 2)), np.zeros((1, 0, 1))
        
        # Check if within drivable area and goal
        self.no_goal = False
        map, map_yaw, map_mask, map_avg_tan, map_avg_norm, max_lat, max_tan = self.map_info
        
        if hasattr(self.env, 'road') and hasattr(self.env.road, 'lanes'):
            # Use Highway Environment's road network to check drivability
            is_in_goal = np.zeros((1, 1))
            drivable = True  # Assume drivable by default
            
            if hasattr(self.env, 'vehicle') and hasattr(self.env.vehicle, 'position'):
                # Check vehicle position against lanes
                position = self.env.vehicle.position
                on_road = self.env.road.on_road(position)
                drivable = on_road
                
                # Check if in goal region if available
                if hasattr(self.env, 'goal'):
                    goal_reached = self.env.goal.is_reached(self.env.vehicle)
                    is_in_goal = np.array([[int(goal_reached)]])
        else:
            # Fall back to MCTS check_drivable_area
            drivable, baseline_dist, time_drive, _, is_in_goal = check_drivable_area(
                np.array([[[self.rear2center, 0]]]),
                map,
                map_mask,
                np.array([[[0,],],]),
                map_yaw,
                map_avg_tan,
                map_avg_norm,
                max_lat,
                max_tan,
            )
            drivable = is_in_goal.min() == 0
        
        self.baseline_dist = 0
        
        # Set goal and drivability flags
        if drivable:
            self.no_goal = True
        else:
            self.baseline_dist = 1000  # Default large distance
        
        self.no_drive = False
        map, map_yaw, map_mask, map_avg_tan, map_avg_norm, max_lat, max_tan = self.map_info_total
        
        if hasattr(self.env, 'road') and hasattr(self.env.road, 'lanes'):
            # Similar check with total map info
            is_in_drive = np.zeros((1, 1))
            drivable = True
            
            if hasattr(self.env, 'vehicle') and hasattr(self.env.vehicle, 'position'):
                position = self.env.vehicle.position
                on_road = self.env.road.on_road(position)
                drivable = on_road
                is_in_drive = np.array([[int(on_road)]])
        else:
            # Fall back to MCTS check_drivable_area
            drivable, baseline_dist, time_drive, _, is_in_drive = check_drivable_area(
                np.array([[[self.rear2center, 0]]]),
                map,
                map_mask,
                np.array([[[0,],],]),
                map_yaw,
                map_avg_tan,
                map_avg_norm,
                max_lat,
                max_tan,
            )
            is_in_drive = is_in_drive + is_in_goal
            drivable = is_in_drive.min() == 0
        
        if drivable:
            self.no_drive = True
        elif self.no_goal:
            self.baseline_dist = 1000  # Default large distance
            
        # Get initial action from trajectory if available
        if "ego_pred" in init_state and init_state["ego_pred"].size > 0:
            self.init_action = trajectory2action(
                trajectory=init_state["ego_pred"], 
                dt=self.dt,
                vehicle_physics=self.vehicle_physics
            )
        else:
            # Default initial action (no acceleration, no steering)
            self.init_action = (0.0, 0.0, None)
        
        # Initialize root node
        if root is not None:
            self.root = root
            root.T = 0
        else:
            self.root = Node(0, None, self, state=init_state, parent_state=init_state, action=action)
        
        # Start time tracking
        self.start_time = init_state.get("start_time", perf_counter())
        self.pred_idx = 0  # Index for predictions
        
        # Store prior if available
        self.prior = None
        if "prior" in init_state and init_state["prior"][0] is not None:
            self.prior = init_state["prior"]
        
        # Display flag for debugging
        self.disp = False
        self.first = True

    def update_sensors(self, camera_image=None, history=None):
        """
        Update sensor data from Highway Environment
        
        Args:
            camera_image: Camera image data dictionary
            history: List of historical actions/states
        """
        if camera_image is not None:
            self.camera_image = camera_image
        
        if history is not None:
            self.history = history

    def update_state(self, new_state):
        """
        Update the tree's state representation with new Highway Environment state
        
        Args:
            new_state: New state dictionary from Highway Environment
        """
        # Apply center of mass offset
        offset_vector = np.array([self.rear2center, 0])[None, None]
        if "ego_pos" in new_state:
            new_state["ego_pos"] = new_state["ego_pos"] + offset_vector
        if "map" in new_state:
            new_state["map"] = new_state["map"] + offset_vector
        if "additional_map" in new_state:
            new_state["additional_map"] = new_state["additional_map"] + offset_vector
        
        # Update map information if available
        if "map" in new_state and "map_mask" in new_state:
            self.map_info = self.compute_map_infos(map=new_state["map"], map_mask=new_state["map_mask"])
        
        if "additional_map" in new_state and "additional_map_mask" in new_state:
            self.map_info_total = self.compute_map_infos(
                map=new_state["additional_map"], 
                map_mask=new_state["additional_map_mask"]
            )
        
        # Update masks if agents information is available
        # if "agents" in new_state and new_state["agents"].size > 0:
        self.behind_mask = new_state["agents"][:, :, -1, 0] > 0
        
        # Update vehicle speeds if prediction information is available
        if "prediction" in new_state and new_state["prediction"].size > 0:
            agents_pos = new_state["prediction"][0, :, 0, 10, :2] - new_state["prediction"][0, :, 0, 0, :2]
            norm = np.sqrt((agents_pos**2).sum(-1))
            self.other_speeds = norm
        
        # Create a new root node with updated state
        self.root = Node(0, None, self, state=new_state, parent_state=new_state, action=None)
        
        # Reset tree parameters
        self.n_nodes = 1
        self.nodes = {}
        self.first = True

    def simulate(self, k=128):
        """
        Execute MCTS simulation to plan a path in Highway Environment
        
        Args:
            k: Number of simulation iterations
            
        Returns:
            acc_seq: Sequence of acceleration actions
            steering_seq: Sequence of steering actions
        """
        start_time = perf_counter()
        self.k = k
        
        # Expand root node
        self.root.expand()
        self.first = True
        best_T = 0
        max_it_time = 0

        # Perform k iterations of MCTS
        for _i in range(k):
            print(f"Iteration {_i + 1}/{k}")
            last_it = perf_counter()
            
            # Execute search
            t, value, failure, success, s_node = self.search()
            
            # Update first flag
            if self.first:
                if failure:
                    self.first = False
                elif t >= self.max_T:
                    self.first = False
                    break
                    
            # Track best time step found
            if s_node and t > best_T:
                best_T = t
                
            # Track timing information
            now = perf_counter()
            max_it_time = max(max_it_time, now - last_it)
            time_elapsed = now - self.start_time
            
            # # Check if running out of time
            # if time_elapsed > 2.0:  # Time limit for planning
            #     break

        # Get action probabilities from the search
        acc_seq, steering_seq = self.get_probas()
        return acc_seq, steering_seq
    
    def search(self):
        success = False
        leaf, value, path = self.root.select([])
        sucess_node = False

        if value is not None:
            value, success = value
            leaf.backup(value, T=leaf.T, success=success)
            failure = False
        else:
            if leaf.parent is not None:
                if leaf.state is None:
                    # print("No state in leaf")
                    leaf.update_state()

                # print("Leaf state agent:", leaf.state["agents"].shape)
                value, success, failure, fail_index, sucess_node = leaf.evaluate()

                if self.max_T - 1 > leaf.T and not failure:
                    leaf.expand()

                if leaf.predicted_score is not None:
                    value += -leaf.predicted_score[1]

                leaf.backup(value, T=leaf.T, success=success, fail_index=fail_index)

        return leaf.T, value, failure, success, sucess_node

    def compute_map_infos(self, map, map_mask, th=0.1):
        """
        Compute map information for Highway Environment network
        
        Args:
            map: Map coordinates
            map_mask: Map mask
            th: Threshold for lane splitting
            
        Returns:
            Map information tuple
        """
        # Check if map has data
        if map.size == 0 or map_mask.size == 0:
            # Return empty map information
            return (
                np.zeros((1, 0, 0, 2)),  # map
                np.zeros((1, 0, 0, 1)),  # map_yaw
                np.zeros((1, 0, 0, 1)),  # map_mask
                np.zeros((1, 0, 2)),     # map_avg_tan
                np.zeros((1, 0, 2)),     # map_avg_norm
                np.zeros((1, 0)),        # max_lat
                np.zeros((1, 0))         # max_tan
            )
        
        # print("frist Map shape:", map.shape, map_mask.shape)
        # Extract nodes and mask
        nodes = map[0]
        mask = map_mask[0]
        # print("Map shape:", nodes.shape, mask.shape)
        # Get directional values
        _, norm, yaw = get_directional_values(nodes=nodes, mask=mask)
        
        # Calculate relative positions
        start = nodes[:, 0]
        rel_map = nodes - start[:, None]
        
        # Calculate normal projections
        self_norm = (norm[:, None, None] @ rel_map[..., None])[..., 0]
        self_norm = self_norm * mask[..., None]
        
        # Calculate maximum lateral distance
        max_lat = np.abs(self_norm[..., 0]).max(-1)

        # Split lanes based on ratios
        lat_ratios = (max_lat // th).astype(np.int32)
        nodes, mask = split_with_ratio(ratios=lat_ratios, nodes=nodes, mask=mask)

        # Filter out points that are too far behind
        max_x = (nodes[..., 0] * mask).max(-1)
        mask_behind = max_x > -5
        if np.any(mask_behind):
            nodes = nodes[mask_behind]
            mask = mask[mask_behind]
        else:
            print("No points remaining after filtering")
            # If no points remain, return empty map info
            return (
                np.zeros((1, 0, 0, 2)),  # map
                np.zeros((1, 0, 0, 1)),  # map_yaw
                np.zeros((1, 0, 0, 1)),  # map_mask
                np.zeros((1, 0, 2)),     # map_avg_tan
                np.zeros((1, 0, 2)),     # map_avg_norm
                np.zeros((1, 0)),        # max_lat
                np.zeros((1, 0))         # max_tan
            )

        # Recalculate directional values
        map_avg_tan, map_avg_norm, yaw = get_directional_values(nodes=nodes, mask=mask)
        
        # Recalculate relative positions
        start = nodes[:, 0]
        rel_map = nodes - start[:, None]
        
        # Calculate normal and tangential projections
        self_norm = (map_avg_norm[:, None, None] @ rel_map[..., None])[..., 0]
        self_norm = self_norm * mask[..., None]
        max_lat = np.abs(self_norm[..., 0]).max(-1)
        
        self_tan = (map_avg_tan[:, None, None] @ rel_map[..., None])[..., 0] * mask[..., None]
        max_tan = self_tan[..., 0].max(-1)

        # Prepare final output
        map = nodes
        map_yaw = yaw
        map_mask = mask[..., None]
        
        # print("====== Map ====")
        # print("Map shape:", map.shape, map_yaw.shape, map_mask.shape)
        # print("Map max lat:", max_lat.shape)
        # print("Map max tan:", max_tan.shape)
        # print("Map avg tan:", map_avg_tan.shape)
        # print("Map avg norm:", map_avg_norm.shape)
        # print("Map mask:", map_mask.shape)
        # print("======")

        return (
            map[None],
            map_yaw[None],
            map_mask[None],
            map_avg_tan[None],
            map_avg_norm[None],
            max_lat[None],
            max_tan[None],
        )

    def ctridx2pos(self, acc, st, dt, initial_pos, initial_speed, initial_yaw):
        """
        Convert discrete control indices to continuous trajectory
        
        Args:
            acc: Acceleration indices
            st: Steering indices
            dt: Time step
            initial_pos: Initial position
            initial_speed: Initial speed
            initial_yaw: Initial heading
            
        Returns:
            Predicted position, speed, and heading
        """
        # Convert discrete actions to continuous values
        discrete_acc = (self.acc_target_range[1] - self.acc_target_range[0]) * acc / (self.acc_values - 1) + self.acc_target_range[0]
        discrete_steer = (self.steering_target_range[1] - self.steering_target_range[0]) * st / (self.steering_values - 1) + self.steering_target_range[0]

        # Calculate speed profile
        pred_speed = initial_speed + np.cumsum(discrete_acc, -1) * dt
        pred_speed = self.relu(pred_speed)  # Ensure non-negative speed

        # Calculate steering angle rate (bicycle model)
        discrete_yr = pred_speed * np.tan(discrete_steer) / self.wheel_base

        # Calculate heading
        pred_yaw = initial_yaw + np.cumsum(discrete_yr, -1) * dt
        
        # Calculate position
        yaw_vec = np.stack([np.cos(pred_yaw), np.sin(pred_yaw)], -1)
        pred_velocity = yaw_vec * pred_speed[..., None]
        pred_pos = initial_pos[:, None] + np.cumsum(pred_velocity, 1) * dt

        return pred_pos, pred_speed[..., None], pred_yaw[..., None]

    def roll_sample(self, sample, pos, speed, yaw):
        """
        Update sample with new position, speed, and yaw
        
        Args:
            sample: Sample to roll
            pos: Position to add
            speed: Speed to add
            yaw: Yaw to add
            
        Returns:
            Updated sample dictionary
        """
        # Check if required keys exist in sample
        if not all(key in sample for key in ["ego_pos", "ego_speed", "ego_yaw", "ego"]):
            print("Sample does not contain required keys")
            print("Sample keys:", sample.keys())
            # Create default sample with basic structure
            new_sample = {
                "ego_pos": pos,
                "ego_speed": speed,
                "ego_yaw": yaw,
                "ego": np.zeros((pos.shape[0], self.frames_history, 4)),
                "agents": np.zeros((pos.shape[0], 0, self.frames_history, 4)),
                "agents_mask": np.zeros((pos.shape[0], 0, self.frames_history, 1)),
                "agents_dim": np.zeros((pos.shape[0], 0, 2)),
                "prediction": np.zeros((pos.shape[0], 0, 1, self.eval_frames, 3))
            }
            return new_sample

        steps = pos.shape[1]
        steps = steps // self.eval_ratio
        new_sample = {}

        ego_features = sample["ego"]
        # print(f"ego features shape: {ego_features.shape}")
        ego_pos = np.concatenate([sample["ego_pos"], pos], 1)
        ego_speed = np.concatenate([sample["ego_speed"], speed], 1)
        ego_yaw = np.concatenate([sample["ego_yaw"], yaw], 1)
        ego_features = np.concatenate([ego_features, self.zeros[:, 0]], 1)

        # print(f"sample prediction shape: {sample['prediction'].shape}")
        # Handle prediction and agents if they exist
        if "prediction" in sample and sample["prediction"].size > 0:
            other_features = sample["prediction"][:, :, self.pred_idx, : self.eval_frames]
        else:
            other_features = np.zeros((ego_pos.shape[0], 0, self.eval_frames, 4))

        if "agents_mask" in sample and sample["agents_mask"].size > 0:
            agents_mask_shape = sample["agents_mask"][:, :, -1:].shape
            agents_mask_shape_0, agents_mask_shape_1 = agents_mask_shape[0], agents_mask_shape[1]
            if len(agents_mask_shape) > 3:
                agents_mask_shape_3 = agents_mask_shape[3]
            # else:
            #     agents_mask_shape_3 = 1
                
            other_mask = np.concatenate(
                [
                    sample["agents_mask"],
                    np.broadcast_to(
                        sample["agents_mask"][:, :, -1:],
                        (agents_mask_shape_0, agents_mask_shape_1, steps, agents_mask_shape_3),
                    ),
                ],
                2,
            )
        else:
            other_mask = np.zeros((ego_pos.shape[0], 0, self.frames_history, 1))

        new_sample["ego_pos"] = ego_pos
        new_sample["ego_speed"] = ego_speed
        new_sample["ego_yaw"] = ego_yaw

        new_sample["ego"] = ego_features[:, -self.frames_history :]

        new_sample["agents"] = other_features[:, :, -self.frames_history :]
        new_sample["agents_mask"] = other_mask[:, :, -self.frames_history :]
        
        # print(f"agents shape: {new_sample['agents'].shape}")
        # print(f"prediction shape: {new_sample['prediction'].shape}")

        if "agents_dim" in sample:
            new_sample["agents_dim"] = sample["agents_dim"]
        else:
            new_sample["agents_dim"] = np.zeros((ego_pos.shape[0], 0, 2))

        if "prediction" in sample and sample["prediction"].size > 0:
            new_sample["prediction"] = sample["prediction"][:, :, :, self.eval_frames :]
        else:
            new_sample["prediction"] = np.zeros((ego_pos.shape[0], 0, 1, self.eval_frames, 4))

        return new_sample

    def get_action_masks(self, action, value, dt=1):
        """
        Get action masks
        
        Args:
            action: Current action
            value: Current speed
            dt: Time step
            
        Returns:
            mask_action: Action mask
            continuity_penalty: Continuity penalty
            total_mask: Total mask
        """
        if action is not None:
            a, st = action
            key = (a, st, value // 0.1)
        else:
            key = value // 0.1

        if key not in self.masks:
            stopped = value <= 0

            mask_a, mask_st, acc_pen, st_pen, lon_jerk = self.get_action_value_mask(action, stopped, dt)
            mask_a, mask_st, acc_pen, st_pen = mask_a.copy(), mask_st.copy(), acc_pen, st_pen.copy()

            acc_values, steering_values = self.acc_values, self.steering_values
            mix_pen = np.zeros(acc_values * steering_values)

            if value <= 0:
                mask_a[: acc_values // 2] = 0
            if value <= 10:
                acc_step = (self.acc_target_range[1] - self.acc_target_range[0]) / (acc_values - 1)
                max_a = int(np.ceil(value / (self.dt * dt) / acc_step))
                mask_a[: acc_values // 2 - max_a] = 0

            st_vals = np.arange(steering_values)
            st_real_vals = (self.steering_target_range[1] - self.steering_target_range[0]) * st_vals / (
                steering_values - 1
            ) + self.steering_target_range[0]
            yr_real_vals = value * np.tan(st_real_vals) / self.wheel_base

            acc_lat = np.abs(yr_real_vals * value)
            mask_st *= acc_lat < 220

            if mask_a.sum() == 0:
                mask_a[acc_values // 2] = 1
            if mask_st.sum() == 0:
                mask_st[steering_values // 2] = 1
            mask_action = mask_a[:, None] * mask_st[None, :]

            mask_action = (mask_action).flatten()

            continuity_penalty = (acc_pen[:, None] + st_pen[None, :]).flatten() + mix_pen

            mask_action = mask_action == 1

            self.masks[key] = (mask_action, continuity_penalty, continuity_penalty[mask_action])

        return self.masks[key]

    def get_action_value_mask(self, action, stopped, dt):
        """
        Get action value mask
        
        Args:
            action: Current action
            stopped: Whether vehicle is stopped
            dt: Time step
            
        Returns:
            mask_a: Acceleration mask
            mask_st: Steering angle mask
            acc_pen: Acceleration penalty
            st_pen: Steering angle penalty
            lon_jerk: Longitudinal jerk
        """
        if action is not None:
            a, st = action
            key = (a, st, stopped)
        else:
            key = stopped
            
        if key not in self.masks_action:
            acc_values, steering_values = self.acc_values, self.steering_values

            mask_a = np.ones(acc_values)
            mask_st = np.ones(steering_values)

            acc_pen = np.zeros(acc_values)
            acc_pen[-2:] = 1
            st_pen = np.zeros(steering_values)
            acc_step = (self.acc_target_range[1] - self.acc_target_range[0]) / (acc_values - 1)

            lon_jerk = np.zeros(acc_values)

            if action is not None:
                acc_vals = np.arange(acc_values)
                acc_real_vals = (self.acc_target_range[1] - self.acc_target_range[0]) * acc_vals / (
                    acc_values - 1
                ) + self.acc_target_range[0]

                st_vals = np.arange(steering_values)
                acc_dif = np.abs(acc_vals - a)

                st_diff = np.abs(st_vals - st)

                lon_jerk = (not stopped) * acc_dif * acc_step / dt
                acc_pen += (lon_jerk > 4.13) * 1
                acc_pen += np.abs(acc_real_vals) / 10

                acc_pen += acc_dif / 5
                st_pen += st_diff / 10000

                mask_a *= acc_dif < 3 + (stopped) * (a < acc_values // 2) * (acc_vals < (acc_values // 2 + 3))
                mask_st *= st_diff < 3

            self.masks_action[key] = (mask_a, mask_st, acc_pen, st_pen, lon_jerk)

        return self.masks_action[key]

    # def get_probas(self):
    #     """
    #     Get action probabilities from the search
        
    #     Returns:
    #         acc_seq: Sequence of acceleration actions
    #         steering_seq: Sequence of steering actions
    #     """
    #     actions = np.zeros((self.action_frames, self.n_actions))
    #     actions_q = np.zeros((self.action_frames, self.n_actions))

    #     # If using multiple trees or a pruned tree, adapt this logic
    #     node = self.root.update_probas_argmax(actions, actions_q)
        
    #     # Get probabilities for each time step
    #     acc_total = []
    #     rot_total = []
        
    #     # Convert action indices to continuous values
    #     for t in range(min(self.action_frames, node.T + 1)):
    #         if actions[t].sum() > 0:
    #             for i in range(self.acc_values):
    #                 for j in range(self.steering_values):
    #                     idx = i * self.steering_values + j
    #                     if idx in self.root.children:
    #                         actions[t, idx] = self.root.children[idx].n

    #             probs = actions[t].reshape(self.acc_values, self.steering_values)
    #             acc_probs = probs.sum(1)
    #             rot_probs = probs.sum(0)
                
    #             # Get maximum probability indices and convert to continuous values
    #             acc_idx = np.argmax(acc_probs)
    #             rot_idx = np.argmax(rot_probs)
                
    #             # Convert to continuous values
    #             acc = (self.acc_target_range[1] - self.acc_target_range[0]) * acc_idx / (self.acc_values - 1) + self.acc_target_range[0]
    #             rot = (self.steering_target_range[1] - self.steering_target_range[0]) * rot_idx / (self.steering_values - 1) + self.steering_target_range[0]
                
    #             acc_total.append(acc)
    #             rot_total.append(rot)
    #         else:
    #             # Default to no acceleration and no steering if no actions available
    #             acc_total.append(0.0)
    #             rot_total.append(0.0)
        
    #     return np.array(acc_total), np.array(rot_total)

    def is_done(self):
        """
        Check if the MCTS search is done
        
        Returns:
            True if search is done, False otherwise
        """
        return self.root.done

    def get_probas(self):
        """
        Returns:
            acc_ns: acceleration value
            yr_ns: yaw rate value
        """

        n_frames = self.action_frames
        actions = np.zeros((20, self.acc_values * self.steering_values))
        actions_q = np.zeros((n_frames, self.acc_values * self.steering_values))
        self.root.update_probas_argmax(actions, actions_q)
        actions = actions.reshape((20, self.acc_values, self.steering_values))

        acc_ns = actions.max(-1)
        yr_ns = actions.max(-2)

        return acc_ns, yr_ns