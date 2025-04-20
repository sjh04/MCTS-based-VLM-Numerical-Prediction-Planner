from typing import TYPE_CHECKING, Dict, List, Tuple
import numpy as np
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# sys.path.append('/home/ubuntu/dockerCarla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')

# import carla
import gymnasium as gym
import highway_env

# Fix: Replace relative import with absolute import
from src.MCTS.utils import check_drivable_area, check_ego_collisions
from src.VLM.policy import AtomicActionPolicy


if TYPE_CHECKING:
    from .tree import Tree


class Node:
    """
    Class describing an MCTS node inside the Tree simulation. A node is made of a state, its values and its potential
    children
    """

    __slots__ = (
        "w",
        "p",
        "n",
        "q",
        "value",
        "children",
        "parent",
        "state",
        "parent_state",
        "next_actions",
        "tree",
        "action",
        "T",
        "children_p",
        "children_w",
        "children_n",
        "children_q",
        "n_perfect",
        "predicted_score",
        "speed",
        "hash_tuple",
        "last_parent",
        "parents",
        "hash_children",
        "failed",
        "drivable",
        "mask_action",
        "continuity_penalty",
        "probas",
        "total_mask",
        "possible_actions",
        "global2possible",
        "policy_generator",
    )

    def __init__(
        self,
        p: float,
        parent: "Node",
        tree: "Tree",
        action: tuple | None = None,
        next_actions: tuple = ([[]], [[]]),
        parent_state=None,
        state: dict | None = None,
        T: int = 0,
    ):
        """
        Args:
            p: prior probability of the node
            parent: parent node
            tree: tree object
            action: action taken to reach this node
            next_actions: next actions to take
            parent_state: state of the parent node
            state: state of the node
            T: time step of the node
        """

        self.w = 0
        self.p = p
        self.n = 0
        self.q = 0
        self.n_perfect = 0
        self.children = {}
        self.parent = parent
        self.last_parent = parent
        self.parents = {(parent, action)}
        self.state = state
        self.hash_children = {}
        self.failed = False
        self.drivable = False
        self.value = None

        self.children_p = None
        self.children_w = None
        self.children_n = None
        self.children_q = None

        if self.state is not None:
            self.speed = self.state["ego_speed"][0, -1][0]
        else:
            acc = action[0]
            discrete_acc = tree.acc_coef * acc + tree.acc_target_range[0]
            self.speed = max(self.parent.speed + discrete_acc * tree.dt, 0)

        self.parent_state = parent_state

        self.next_actions = next_actions
        self.tree = tree

        self.action = action

        self.mask_action, self.continuity_penalty, self.probas, self.total_mask = None, None, None, None

        self.possible_actions = self.tree.default_possible_actions
        self.global2possible = self.tree.default_global2possible

        # Initialize the policy generator if the tree has a policy
        self.policy_generator = None
        if hasattr(tree, 'policy') and tree.policy is not None:
            self.policy_generator = AtomicActionPolicy(tree.policy)

        self.tree.n_nodes += 1
        self.tree.Ts[T] += 1
        self.T = T
        self.predicted_score = None

        if self.parent is not None:
            self.predicted_score = self.parent.predicted_score

    def select(self, path: list | None = None):
        """
        Select a child node according to the PUCT algorithm.
        
        Args:
            path: path to the node
        """

        path.append((self.T, self.action))

        if self.tree.max_T - 1 < self.T:
            return self, None, path

        if self.children_p is None:
            if len(self.next_actions[0][0]) == 0:
                return self, None, path
            else:
                self.expand()

        acc_values, steering_values = self.tree.acc_values, self.tree.steering_values

        if self.T > 0 and self.T % self.tree.eval_frames != 0:
            a, yr = self.action
            if self.speed <= 0:
                a = max(a, 6)

            selected = a * steering_values + yr

        else:
            # PUCT Selection
            if self.T == 0 and self.tree.penalty_init is not None:
                mask_action = self.tree.mask_init
                total_mask = self.tree.penalty_init[mask_action]
            else:
                mask_action, _, total_mask = self.tree.get_action_masks(self.action, self.speed)

            if self.probas is None:
                # PUCT formula: c_puct * P(s,a)
                self.probas = self.tree.c_puct * self.children_p[mask_action]
                self.possible_actions = np.arange(steering_values * acc_values)[mask_action]
                self.global2possible = {self.possible_actions[i]: i for i in range(len(self.possible_actions))}
                self.children_q = self.children_q[mask_action]
                self.children_n = self.children_n[mask_action]
                self.children_w = self.children_w[mask_action]

            probas = self.probas
            # print(f"probas: {probas}")
            children_counts = self.children_n
            summed = children_counts.sum()

            if self.tree.first or summed == 0:
                # For first iterations or unvisited nodes, rely purely on prior probabilities
                children_pucts = probas
            else:
                # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
                root_count_sum = np.sqrt(summed)
                children_pucts = self.children_q + probas * root_count_sum / (1 + children_counts)

            selected = np.argmax(children_pucts - total_mask)
            # print(f"selected_1: {selected}")
            selected = self.possible_actions[selected]
            # print(f"selected_2: {selected}")
            # print(f"children: {self.children}")

        if selected in self.children:
            self.children[selected].last_parent = self
            return self.children[selected].select(path)
        else:
            # print(f"Node not found in children: {selected}")
            a, yr = selected // steering_values, selected % steering_values

            next_acc, next_yr = self.next_actions
            next_action = (next_acc[:, 1:], next_yr[:, 1:])
            new_child = Node(
                self.children_p[selected],
                self,
                self.tree,
                action=(a, yr),
                next_actions=next_action,
                T=self.T + 1,
                parent_state=self.parent_state,
            )
            self.children[selected] = new_child

            return new_child.select(path)

    def expand(self):
        """Expand the node using either stored next actions or policy predictions."""

        if len(self.next_actions[0][0]) == 0:
            self.expand_with_policy()
        
        # Use stored next actions
        next_acc, next_yr = self.next_actions
        pred_acc = next_acc[0, 0]
        pred_yr = next_yr[0, 0]
        # print(f"pred_acc: {pred_acc}")
        # print(f"pred_yr: {pred_yr}")
        # sys.exit(0)
        proba_action = pred_acc[:, None] * pred_yr[None, :]
        # print(f"proba_action: {proba_action}")
        # sys.exit(0)
        self.children_p = proba_action.flatten()
        self.children_w = 0 * self.children_p
        self.children_n = 0 * self.children_p
        self.children_q = 0 * self.children_p

    def expand_with_policy(self):
        """Expand the node using the low-level policy generator for CARLA control."""
        # self.expand_nn()
        # return
        if self.policy_generator is not None and self.state is not None:
            try:
                # Get history and camera data for policy generation
                history = []
                if hasattr(self.tree, 'history') and self.tree.history is not None:
                    history = self.tree.history
                
                image = None

                mid_action = self.tree.mid_action
                print(f"mid_action: {mid_action}")
                state_description = self.tree.mcts_env._get_observation()
                print("============================")
                print(f"state_description: {state_description}")
                print("============================")
                # Generate policy prediction
                policy_action = self.policy_generator.act(state_description, history, mid_action, self.tree.high_level_action)

                # Convert policy action to acceleration and steering
                acceleration = policy_action["acceleration"]
                steering = policy_action["steering"]
                if acceleration == 'done' and steering == 'done':
                    self.tree.done = True
                    return

                # Ensure acceleration and steering are within valid ranges
                acceleration = float(acceleration)
                steering = float(steering)
                acceleration = max(-3, min(acceleration, 3))
                steering = max(-np.pi/4, min(steering, np.pi/4))
                print(f"acceleration: {acceleration}, steering: {steering}")
                # Get action space dimensions
                acc_values, steering_values = self.tree.acc_values, self.tree.steering_values

                # Convert continuous actions to discrete indices
                # Adding 6 centers the action in the middle of the discretized space
                acc_idx = min(max(int(acceleration / self.tree.acc_coef + 6), 0), acc_values - 1)
                steer_idx = min(max(int(steering / self.tree.steer_coef + 6), 0), steering_values - 1)
                print(f"acc_idx: {acc_idx}, steer_idx: {steer_idx}")
                # Create probability distributions that strongly favor the predicted actions
                # Small probability (0.1) for other actions to allow exploration
                acc_distribution = np.ones((1, self.tree.eval_frames, acc_values))
                acc_distribution = acc_distribution * 0.1
                acc_distribution[0, :, acc_idx] = 1.0
                acc_distribution = acc_distribution / acc_distribution.sum(-1, keepdims=True)
                print(f"acc_distribution: {acc_distribution}")

                steer_distribution = np.ones((1, self.tree.eval_frames, steering_values))
                steer_distribution = steer_distribution * 0.1
                steer_distribution[0, :, steer_idx] = 1.0
                steer_distribution = steer_distribution / steer_distribution.sum(-1, keepdims=True)
                print(f"steer_distribution: {steer_distribution}")

                self.predicted_score = [0, 0]

                self.next_actions = (acc_distribution, steer_distribution)
                return
            except Exception as e:
                print(f"HighwayEnv policy generation failed: {e}")
                pass
                
        # Fallback to original expand_nn method
        self.expand_nn()

    def expand_nn(self):
        """Expand the node using heuristic distributions."""

        baseline_acc, baseline_yr = 6, 6
        th = 0.0

        if self.speed < (self.tree.max_speed - th):
            baseline_acc = 8

        temp_acc, temp_yr = 100, 100
        power = 1

        init_acc = np.ones((1, self.tree.eval_frames, 13))
        acc_values = np.abs(np.arange(13) - baseline_acc)[None, None, :]
        acc_values = acc_values * init_acc
        next_acc = np.exp(-(acc_values**power) / temp_acc)
        next_acc[:, :, 11:] = 0

        init_yr = np.ones((1, self.tree.eval_frames, 13))
        yr_values = np.abs(np.arange(13) - baseline_yr)[None, None, :] * init_yr
        next_yr = np.exp(-(yr_values**power) / temp_yr)

        if self.T == 0:
            next_yr = np.zeros((1, self.tree.eval_frames, 13))
            nn_acc, nn_yr, nn_yrs = self.tree.init_action
            nn_acc = np.fix(nn_acc / self.tree.acc_coef) + 6
            nn_yr = np.fix(nn_yr / self.tree.steer_coef) + 6

            temp_acc = 100
            acc_values = np.abs(np.arange(13) - nn_acc)[None, None, :]
            acc_values = acc_values * init_acc

            next_acc += np.exp(-(acc_values**power) / temp_acc) * 0.9

            yr_values = np.abs(np.arange(13) - nn_yr)[None, None, :] * init_yr
            next_yr += np.exp(-(yr_values**power) / temp_yr)

        next_acc = next_acc / next_acc.sum(-1, keepdims=True)

        next_yr = next_yr / next_yr.sum(-1, keepdims=True)
        self.predicted_score = [0, 0]

        self.next_actions = (next_acc, next_yr)

    def evaluate(self):
        """Evaluate the node in CARLA environment."""

        if self.n == 0:
            predict_traj, predict_yaw = (
                self.state["ego_pos"][:, -self.tree.eval_frames :],
                self.state["ego_yaw"][:, -self.tree.eval_frames :],
            )

            # Extract state information for evaluation
            predicted_xy = predict_traj[:, :]
            predicted_yaw = predict_yaw[:, :]
            predicted_state_xy = self.state["agents"][:, :, -self.tree.eval_frames :, :2]
            predicted_state_yaw = self.state["agents"][:, :, -self.tree.eval_frames :, 2:3]
            mask_agents = self.state["agents_mask"][:, :, -self.tree.eval_frames :, 0]
            dim_agents = self.state["agents_dim"]

            margins = None

            # print("==========================")
            # print(f"mask_agents: {mask_agents}")
            # print(f"state: {self.state}")
            # print(f"state agents: {self.state['agents'].shape}")
            print(f"behind_mask: {self.tree.behind_mask.shape}")
            print(f"predicted_state_xy: {predicted_state_xy.shape}")
            # print(f"predicted_state_yaw: {predicted_state_yaw.shape}")
            # print("==========================")
            # Check for collisions with other vehicles in CARLA
            real_collisions = check_ego_collisions(
                predicted_xy,
                predicted_yaw,
                predicted_state_xy[self.tree.behind_mask][None],
                predicted_state_yaw[self.tree.behind_mask][None],
                mask_agents[self.tree.behind_mask][None],
                margin=[0.7, 0.3],  # Adjusted margins for CARLA vehicles
                other_dims=dim_agents[self.tree.behind_mask][None],
                margins=margins,
                speed=self.speed,
                other_speeds=self.tree.other_speeds[self.tree.behind_mask[0]][None],
            )

            # Check for marginal collisions (near misses)
            margin_collision = 0
            if real_collisions.sum():
                new_collisions = check_ego_collisions(
                    predicted_xy,
                    predicted_yaw,
                    predicted_state_xy[self.tree.behind_mask][None],
                    predicted_state_yaw[self.tree.behind_mask][None],
                    mask_agents[self.tree.behind_mask][None],
                    margin=[0.7, 0.3],
                    other_dims=dim_agents[self.tree.behind_mask][None],
                    margins=margins,
                    speed=0,
                )
                if not new_collisions.sum():
                    margin_collision = 1
                    real_collisions = np.array([0])

            # Initialize collision counters
            behind_cols = 0
            static_cols = 0
            pedestrian_cols = 0

            # Check collisions with pedestrians and static objects if no vehicle collisions
            # if not real_collisions.sum():
            #     # Check pedestrian collisions
            #     pedestrians_coords, pedestrians_dims, pedestrian_mask = self.tree.pedestrians
            #     pedestrian_cols = check_ego_collisions(
            #         predicted_xy,
            #         predicted_yaw,
            #         pedestrians_coords[:, :, :, :2],
            #         pedestrians_coords[:, :, :, 2:],
            #         pedestrian_mask,
            #         margin=[0.5, 0.5],  # Adjusted for pedestrians
            #         other_dims=pedestrians_dims,
            #     )
            #     pedestrian_cols = pedestrian_cols.sum()

            #     # Check static object collisions (buildings, barriers, etc.)
            #     static_coords, static_dims, static_mask = self.tree.static_objects
            #     static_cols = check_ego_collisions(
            #         predicted_xy,
            #         predicted_yaw,
            #         static_coords[:, :, :, :2],
            #         static_coords[:, :, :, 2:],
            #         static_mask,
            #         margin=[0.2, 0.2],  # Adjusted for static objects
            #         other_dims=static_dims,
            #     )
            #     static_cols = static_cols.sum()

            # Calculate progress based on speed
            progress = (self.state["ego_speed"][:, -1]).sum() / self.tree.max_speed
            progress = min(progress, 1)

            # Check if vehicle is on drivable area (road) using map data
            map, map_yaw, map_mask, map_avg_tan, map_avg_norm, max_lat, max_tan = self.tree.map_info
            drivable_goal, closest_dist, time_drive, closest_angle, is_in_goal = check_drivable_area(
                predict_traj,
                map,
                map_mask,
                predict_yaw,
                map_yaw,
                map_avg_tan,
                map_avg_norm,
                max_lat,
                max_tan,
            )
            drivable_goal = is_in_goal.min() == 0

            # Check additional drivable areas if needed
            if drivable_goal:
                map, map_yaw, map_mask, map_avg_tan, map_avg_norm, max_lat, max_tan = self.tree.map_info_total
                (
                    drivable,
                    closest_dist_drive,
                    time_drive,
                    closest_angle_drive,
                    is_in_drivable,
                ) = check_drivable_area(
                    predict_traj,
                    map,
                    map_mask,
                    predict_yaw,
                    map_yaw,
                    map_avg_tan,
                    map_avg_norm,
                    max_lat,
                    max_tan,
                )
                is_in_drivable = is_in_drivable + is_in_goal
                closest_angle = min(closest_angle_drive, closest_angle)
                closest_dist = min(closest_dist_drive, closest_dist)
                drivable = is_in_drivable.min() == 0
            else:
                drivable = np.array([0])

            fail_index = None

            # Set failure index if driving off-road
            if drivable:
                fail_index = self.T - 10 + time_drive

            # Calculate composite score based on collisions, drivability, and progress
            collision = real_collisions.sum() > 0
            drivable = drivable.sum()
            both = collision + drivable_goal
            
            # Penalties for goal achievement and drivability
            goal_penality = -0.5 * drivable_goal * (drivable == 0)
            if self.tree.no_goal:
                goal_penality = 0.1 * (1 - drivable_goal)
                both = collision + drivable
                
            drivable_penalty = -drivable
            if self.tree.no_drive:
                drivable_penalty = 0.1 * (1 - drivable)
                both = collision
                
            closest_angle = np.abs(closest_angle)
            
            # Final score calculation - CARLA specific reward weights
            score = (
                -5 * collision           # Heavy penalty for collisions
                - 2 * static_cols        # Penalty for hitting static objects
                - behind_cols * 0.5      # Small penalty for vehicles behind
                - 0.5 * margin_collision # Penalty for near misses
                - 3 * pedestrian_cols    # Penalty for hitting pedestrians
                + drivable_penalty       # Penalty for off-road driving
                + goal_penality          # Penalty for not reaching goal
                + progress * (both == 0) * (closest_angle < 0.20)  # Reward for progress when safe
                + (both == 0) * 0.05 * (closest_angle < 0.35)      # Small reward for good heading
            )
            score = score.sum()

            # Additional distance-based scoring
            if closest_dist < 100 and (both == 0):
                score -= closest_angle / 2 + closest_dist / 2

            # Mark node as failed if collision or off-road
            if both > 0:
                self.failed = True
                if drivable:
                    self.drivable = True
                    
            self.value = score
            return score, (self.tree.max_T == self.T) * 100 * (both == 0), both > 0, fail_index, (drivable) == 0
        else:
            return self.w / self.n, False, False, None, False

    def backup(self, value, T, success=False, fail_index=None):
        """Back propagate the value of the node."""

        value = value + self.value if self.value is not None and self.n > 0 else value

        self.n += 1
        self.w += value
        self.q = self.w / self.n
        if success:
            self.n_perfect += 1

        if self.last_parent:
            steering_values = self.tree.steering_values

            idx = self.action[0] * steering_values + self.action[1]
            for parent, action in self.parents:
                idx = action[0] * steering_values + action[1]
                real_idx = self.parent.global2possible[idx]
                parent.children_n[real_idx] += 1
                parent.children_w[real_idx] += value
                parent.children_q[real_idx] = parent.children_w[real_idx] / parent.children_n[real_idx]
            self.last_parent.backup(value, T, success, fail_index)

    def get_past_action(self, list_acc, list_yr):
        """
        Get the past action of the node.

        Args:
            list_acc: list of accelerations
            list_yr: list of yaw rates

        Returns:
            list_acc: list of accelerations
            list_yr: list of yaw rates
        """

        list_acc.append(self.action[0])
        list_yr.append(self.action[1])

        if len(list_acc) < self.tree.eval_frames:
            return self.parent.get_past_action(list_acc, list_yr)
        else:
            list_acc.reverse()
            list_yr.reverse()
            return list_acc, list_yr

    def update_state(self):
        """Update the state of the node."""

        past_state = self.parent.parent_state
        # print(f"past_state shape: {past_state['agents'].shape}")
        # print(f"past_state prediction:{past_state['prediction'].shape}")
        acc, yr = self.get_past_action([], [])
        # print(f"ego_pos: {past_state['ego_pos']}")
        initial_pos = past_state["ego_pos"][:, -1]
        initial_speed = past_state["ego_speed"][:, -1]
        initial_yaw = past_state["ego_yaw"][:, -1]

        acc = np.array(acc)[None, :]
        yr = np.array(yr)[None, :]

        pred_pos, pred_speed, pred_yaw = self.tree.ctridx2pos(
            acc[:, -self.tree.eval_frames :],
            yr[:, -self.tree.eval_frames :],
            self.tree.dt,
            initial_pos,
            initial_speed,
            initial_yaw,
        )

        self.state = self.tree.roll_sample(sample=past_state, pos=pred_pos, speed=pred_speed, yaw=pred_yaw)
        self.parent_state = self.state

    def update_probas_argmax(self, actions, actions_q):
        """
        Update the probabilities of the node.

        Args:
            actions: actions
            actions_q: actions q values

        Returns:
            self
        """

        acc_values, steering_values = self.tree.acc_values, self.tree.steering_values

        if self.children:
            for a in range(acc_values):
                for yr in range(steering_values):
                    idx = a * steering_values + yr
                    if idx in self.children:
                        actions[self.T, idx] += (
                            self.children[a * self.tree.steering_values + yr].n
                            + 10 * self.children[a * self.tree.steering_values + yr].n_perfect
                        )

            if actions[self.T].sum() > 0 and self.tree.action_frames - 1 > self.T:
                child_max = np.argmax(actions[self.T])
                return self.children[child_max].update_probas_argmax(actions, actions_q)
            else:
                return self
