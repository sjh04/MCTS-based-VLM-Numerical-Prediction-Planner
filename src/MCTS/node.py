from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple
import numpy as np
import carla

from .utils import check_drivable_area, check_ego_collisions


if TYPE_CHECKING:
    from .tree import Tree


class Node:
    """
    Node class in MCTS tree, used for path planning in CARLA environment
    """
    __slots__ = (
        "w",  # cumulative reward
        "p",  # prior probability
        "n",  # visit count
        "q",  # average reward
        "value",  # node value
        "children",  # child nodes
        "parent",  # parent node
        "state",  # current state
        "parent_state",  # parent node state
        "next_actions",  # possible next actions
        "tree",  # MCTS tree object
        "action",  # action to reach this node
        "T",  # time step
        "children_p",  # child nodes prior probabilities
        "children_w",  # child nodes cumulative rewards
        "children_n",  # child nodes visit counts
        "children_q",  # child nodes average rewards
        "n_perfect",  # perfect execution count
        "predicted_score",  # predicted score
        "speed",  # current speed
        "hash_tuple",  # state hash
        "last_parent",  # last visited parent
        "parents",  # all parent nodes
        "hash_children",  # child nodes hash
        "failed",  # whether failed
        "drivable",  # whether drivable
        "mask_action",  # action mask
        "continuity_penalty",  # continuity penalty
        "probas",  # probability distribution
        "total_mask",  # total mask
        "possible_actions",  # possible actions
        "global2possible",  # global to local action mapping
    )

    def __init__(
        self,
        p: float,
        parent: Node,
        tree: Tree,
        action: Tuple[float, float] | None = None,
        next_actions: Tuple[List[List[float]], List[List[float]]] = ([[]], [[]]),
        parent_state=None,
        state: Dict | None = None,
        T: int = 0,
    ):
        """
        Initialize MCTS node
        Args:
            p: node prior probability
            parent: parent node
            tree: MCTS tree object
            action: action to reach this node (acceleration, steering angle)
            next_actions: possible next actions
            parent_state: parent node state
            state: current state
            T: time step
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
        if self.state is not None:
            self.speed = self.state["speed"]
        else:
            acc = action[0]
            self.speed = max(self.parent.speed + acc * tree.dt, 0)

        self.parent_state = parent_state
        self.next_actions = next_actions
        self.tree = tree
        self.action = action

        self.mask_action = None
        self.continuity_penalty = None
        self.probas = None
        self.total_mask = None

        self.possible_actions = self.tree.default_possible_actions
        self.global2possible = self.tree.default_global2possible

        self.tree.n_nodes += 1
        self.tree.Ts[T] += 1
        self.T = T
        self.predicted_score = None

        if self.parent is not None:
            self.predicted_score = self.parent.predicted_score

    def evaluate(self):
        """
        Evaluate node state
        Returns:
            score: evaluation score
            success: whether successful
            failure: whether failed
            fail_index: failure time step
            not_drivable: whether not drivable
        """
        if self.n == 0:
            # get past actions
            predict_traj = self.state["trajectory"]
            predict_yaw = self.state["yaw"]
            
            # check for collisions
            collision = self.check_collision(predict_traj, predict_yaw)
            
            # check for drivable area
            drivable = self.check_drivable_area(predict_traj, predict_yaw)
            
            # calculate progress
            progress = min(self.speed / self.tree.max_speed, 1)
            
            # calculate score
            score = (
                -5 * collision  # collision penalty
                + drivable * 0.5  # drivable reward
                + progress * (not collision) * drivable  # progress reward
            )
            
            self.value = score
            return score, (self.tree.max_T == self.T) * 100 * (not collision), collision, None, not drivable
        else:
            return self.w / self.n, False, False, None, False

    def check_collision(self, trajectory, yaw):
        """
        Check for collisions
        Args:
            trajectory: predicted trajectory
            yaw: heading angle
        Returns:
            bool: whether collision occurred
        """
        # get all vehicles
        vehicles = self.tree.world.get_actors().filter('vehicle.*')
        
        # check for ego vehicle collisions
        for vehicle in vehicles:
            if vehicle.id != self.tree.ego_vehicle.id:
                if self.check_vehicle_collision(trajectory, yaw, vehicle):
                    return True
        return False

    def check_vehicle_collision(self, trajectory, yaw, other_vehicle):
        """
        Check collision with specific vehicle
        Args:
            trajectory: predicted trajectory
            yaw: heading angle
            other_vehicle: other vehicle
        Returns:
            bool: whether collision occurred
        """
        # 获取车辆边界框
        ego_bbox = self.tree.ego_vehicle.bounding_box
        other_bbox = other_vehicle.bounding_box
        
        # 检查轨迹上的每个点是否与车辆发生碰撞
        for point in trajectory:
            ego_transform = carla.Transform(
                carla.Location(x=point[0], y=point[1]),
                carla.Rotation(yaw=yaw)
            )
            ego_bbox.location = ego_transform.location
            ego_bbox.rotation = ego_transform.rotation
            
            if ego_bbox.overlaps(other_bbox):
                return True
        return False

    def check_drivable_area(self, trajectory, yaw):
        """
        Check if in drivable area
        Args:
            trajectory: predicted trajectory
            yaw: heading angle
        Returns:
            bool: whether in drivable area
        """
        # 获取地图信息
        map = self.tree.world.get_map()
        
        # 检查轨迹上的每个点是否在可行驶区域内
        for point in trajectory:
            location = carla.Location(x=point[0], y=point[1])
            waypoint = map.get_waypoint(location)
            
            if not waypoint or not waypoint.is_junction:
                return False
        return True

    def select(self, path: List | None = None):
        """
        Select best child node
        Args:
            path: path record
        Returns:
            selected_node: selected node
            value: node value
            path: updated path
        """
        if path is None:
            path = []
            
        path.append((self.T, self.action))

        if self.tree.max_T - 1 < self.T:
            return self, None, path

        if self.children_p is None:
            if len(self.next_actions[0][0]) == 0:
                return self, None, path
            else:
                self.expand()

        # 使用PUCT算法选择动作
        if self.T > 0 and self.T % self.tree.eval_frames != 0:
            a, yr = self.action
            if self.speed <= 0:
                a = max(a, 0.5)  # 最小加速度

            selected = a * self.tree.steering_values + yr
        else:
            if self.T == 0 and self.tree.penalty_init is not None:
                mask_action = self.tree.mask_init
                total_mask = self.tree.penalty_init[mask_action]
            else:
                mask_action, _, total_mask = self.tree.get_action_masks(self.action, self.speed)

            if self.probas is None:
                self.probas = self.tree.c_puct * self.children_p[mask_action]
                self.possible_actions = np.arange(self.tree.steering_values * self.tree.acc_values)[mask_action]
                self.global2possible = {self.possible_actions[i]: i for i in range(len(self.possible_actions))}
                self.children_q = self.children_q[mask_action]
                self.children_n = self.children_n[mask_action]
                self.children_w = self.children_w[mask_action]

            probas = self.probas
            children_counts = self.children_n
            summed = children_counts.sum()

            if self.tree.first or summed == 0:
                children_pucts = probas
            else:
                root_count_sum = np.sqrt(summed)
                children_pucts = self.children_q + probas * root_count_sum / (1 + children_counts)

            selected = np.argmax(children_pucts - total_mask)
            selected = self.possible_actions[selected]

        if selected in self.children:
            self.children[selected].last_parent = self
            return self.children[selected].select(path)
        else:
            a, yr = selected // self.tree.steering_values, selected % self.tree.steering_values
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
        """
        Expand node
        """
        if len(self.next_actions[0][0]) == 0:
            self.expand_nn()
        else:
            next_acc, next_yr = self.next_actions
            pred_acc = next_acc[0, 0]
            pred_yr = next_yr[0, 0]
            
            # 计算动作概率
            proba_action = pred_acc[:, None] * pred_yr[None, :]
            
            self.children_p = proba_action.flatten()
            self.children_w = np.zeros_like(self.children_p)
            self.children_n = np.zeros_like(self.children_p)
            self.children_q = np.zeros_like(self.children_p)

    def expand_nn(self):
        """
        Expand node using neural network
        """
        # 设置基准动作
        baseline_acc, baseline_yr = 0.5, 0.0
        th = 0.0

        if self.speed < (self.tree.max_speed - th):
            baseline_acc = 1.0

        temp_acc, temp_yr = 100, 100
        power = 1

        # 生成加速度和转向角分布
        init_acc = np.ones((1, self.tree.eval_frames, 13))
        acc_values = np.abs(np.arange(13) - baseline_acc)[None, None, :]
        acc_values = acc_values * init_acc
        next_acc = np.exp(-(acc_values**power) / temp_acc)
        next_acc[:, :, 11:] = 0

        init_yr = np.ones((1, self.tree.eval_frames, 13))
        yr_values = np.abs(np.arange(13) - baseline_yr)[None, None, :] * init_yr
        next_yr = np.exp(-(yr_values**power) / temp_yr)

        # 归一化概率分布
        next_acc = next_acc / next_acc.sum(-1, keepdims=True)
        next_yr = next_yr / next_yr.sum(-1, keepdims=True)

        self.next_actions = (next_acc, next_yr)

    def backup(self, value, T, success=False, fail_index=None):
        """
        Backpropagate node value
        Args:
            value: node value
            T: time step
            success: whether successful
            fail_index: failure time step
        """
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

    def update_state(self):
        """
        Update node state
        """
        past_state = self.parent.parent_state
        acc, yr = self.get_past_action([], [])

        initial_pos = past_state["position"]
        initial_speed = past_state["speed"]
        initial_yaw = past_state["yaw"]

        acc = np.array(acc)[None, :]
        yr = np.array(yr)[None, :]

        # 预测新的状态
        pred_pos, pred_speed, pred_yaw = self.tree.ctridx2pos(
            acc[:, -self.tree.eval_frames:],
            yr[:, -self.tree.eval_frames:],
            self.tree.dt,
            initial_pos,
            initial_speed,
            initial_yaw,
        )

        # 更新状态
        self.state = self.tree.roll_sample(sample=past_state, pos=pred_pos, speed=pred_speed, yaw=pred_yaw)
        self.parent_state = self.state

    def get_past_action(self, list_acc, list_yr):
        """
        Get past actions
        Args:
            list_acc: acceleration list
            list_yr: steering angle list
        Returns:
            list_acc: updated acceleration list
            list_yr: updated steering angle list
        """
        list_acc.append(self.action[0])
        list_yr.append(self.action[1])

        if len(list_acc) < self.tree.eval_frames:
            return self.parent.get_past_action(list_acc, list_yr)
        else:
            list_acc.reverse()
            list_yr.reverse()
            return list_acc, list_yr

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