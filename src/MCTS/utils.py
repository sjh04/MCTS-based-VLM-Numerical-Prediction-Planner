import numpy as np
from shapely.geometry import Polygon
import sys
import os
# sys.path.append('/home/ubuntu/dockerCarla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
# import carla
import gymnasium as gym
import highway_env


def compute_rot_matrix(theta):
    """
    Computes the rotation matrix for a given angle theta

    Args:
        theta: angle in radians

    Returns:
        rotation matrix
    """

    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def compute_vertexes(x, y, heading, vertexes):
    """
    Computes the vertexes of the agent given its center position, heading and dimensions.

    Args:
        x: x coordinate of the center
        y: y coordinate of the center
        heading: heading of the agent
        vertexes: vertexes of the agent

    Returns:
        vertexes of the agent
    """

    agent_center = np.array([[x, y]])
    agent_center = np.transpose(agent_center)
    rot_matrix = compute_rot_matrix(heading)
    vertexes_rot = agent_center + np.dot(rot_matrix, vertexes)

    return np.transpose(vertexes_rot).tolist()


def compute_vertexes_batch(pos, heading, vertexes):
    """
    Computes the vertexes of the agent given its center position, heading and dimensions.

    Args:
        pos: position of the agent
        heading: heading of the agent
        vertexes: vertexes of the agent

    Returns:
        vertexes of the agent
    """

    agent_center = pos
    rot_matrix = compute_rot_matrix(heading)
    rot_matrix = np.transpose(rot_matrix, (2, 3, 0, 1))
    vertexes_rot = agent_center[..., None] + np.dot(rot_matrix, vertexes)
    vertexes_rot = np.transpose(vertexes_rot, (0, 1, 3, 2))
    return vertexes_rot


def compute_corners(length, width, vehicle=None):
    """
    Computes the corners of the agent given its dimensions.

    Args:
        length: length of the agent
        width: width of the agent
        vehicle: Optional Highway Environment vehicle to get dimensions from

    Returns:
        corners of the agent
    """
    # If Highway Environment vehicle is provided, use its dimensions
    if vehicle is not None:
        if hasattr(vehicle, 'LENGTH'):
            length = vehicle.LENGTH
        if hasattr(vehicle, 'WIDTH'):
            width = vehicle.WIDTH

    vertexes = []
    vertexes.append((0 + length / 2, 0 + width / 2))
    vertexes.append((0 + length / 2, 0 - width / 2))
    vertexes.append((0 - length / 2, 0 - width / 2))
    vertexes.append((0 - length / 2, 0 + width / 2))
    vertexes = np.transpose(np.array(vertexes))

    return vertexes


def check_ego_collisions(
    ego_pred,
    ego_heading,
    other_gt,
    other_heading,
    other_mask,
    margin=(0, 0),
    other_dims=None,
    margins=None,
    speed=0,
    other_speeds=None,
    vehicle_dimensions=None
):
    """
    Checks for collisions at every timestep between an ego prediction and the ground truth position of other vehicles.
    Adapted for Highway Environment.

    Args:
        ego_pred: tensor of size (batch, time, 2) indicating the sequence of coordinates of the ego center
        ego_heading: tensor of size (batch, time, 1) indicating the sequence of heading of the ego
        other_gt: tensor of size (batch, agent, time, 2) indicating the sequence of ccordinates for each agent
        other_heading: tensor of size (batch, agent, time, 1) indicating the sequence of heading for each agent
        other_mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent
        margin: margin to add to the dimensions of the agents
        other_dims: tensor of size (batch, agent, 2) indicating the dimensions of each agent
        margins: tensor of size (batch, agent, 2) indicating the margin to add to the dimensions of each agent
        speed: speed of the ego
        other_speeds: tensor of size (batch, agent) indicating the speed of each agent
        vehicle_dimensions: Optional tuple with (length, width) for ego vehicle

    Returns:
        collision: boolean (batch)
    """
    # print(f"ego_pred: {ego_pred.shape}, ego_heading: {ego_heading.shape}, other_gt: {other_gt.shape}, other_heading: {other_heading.shape}, other_mask: {other_mask.shape}")
    diff = other_gt - ego_pred[:, None]
    dists = np.sqrt((diff**2).sum(-1))

    # Use provided dimensions if available, otherwise use Highway Environment defaults
    if vehicle_dimensions is not None:
        length, width = vehicle_dimensions
    else:
        # Default dimensions for Highway Environment
        length, width = 5.0, 2.0  # Typical Highway Environment vehicle dimensions

    # Add margins and speed-based adjustments for more robust collision detection
    length, width = length + margin[0] + speed / 2, width + margin[1]
    L1 = np.sqrt(length**2 + width**2)

    half_length, half_width = length / 2, width / 2
    if other_dims is not None and other_dims[0].shape[0] > 0:
        l_, w_ = other_dims[:, :, 0] / 2 + margin[0] / 2, other_dims[:, :, 1] / 2 + margin[1] / 2
        l_, w_ = l_[..., None], w_[..., None]
    else:
        l_, w_ = length / 2, width / 2
    if margins is not None:
        w_ = w_ + margins[..., None]
    if other_speeds is not None:
        l_ = l_ + other_speeds[..., None] / 4

    L2 = np.sqrt((2 * l_) ** 2 + (2 * w_) ** 2)
    L = (L1 + L2) / 2

    # Printing debug information
    # print(f"width: {width}, length: {length}, L1: {L1}, L2: {L2}, L: {L}")
    # print(f"dists:{dists.shape}, diff: {diff.shape}, other_gt: {other_gt.shape}")
    # print(f"other_mask: {other_mask.shape}, other_heading: {other_heading.shape}")
    # print(f"ego_heading: {ego_heading.shape}")

    radius_small = (dists < width) * other_mask
    radius_check = (dists < L) * other_mask

    if radius_check.sum() == 0:
        return np.array([0])

    # Calculate relative coordinates and orientations for collision detection
    cos1, sin1 = np.cos(ego_heading[..., 0]), np.sin(ego_heading[..., 0])
    vx = np.stack([cos1, sin1], -1)
    vy = np.stack([-sin1, cos1], -1)
    ego_matrix = np.stack([vx, vy], -2)
    rel_diff = ego_matrix[:, None, :] @ diff[..., None]
    rel_diff = rel_diff[..., 0]

    delta_angle = np.abs(((other_heading - ego_heading[:, None]) + np.pi / 2) % np.pi - np.pi / 2)[..., 0]
    cos, sin = np.cos(delta_angle), np.sin(delta_angle)
    rel_diff = rel_diff[..., ::-1]

    # Check if vehicles are within each other's bounding boxes
    enveloppe = (np.abs(rel_diff[..., 0]) <= (half_width + w_ * cos + l_ * sin)) & (
        np.abs(rel_diff[..., 1]) <= (half_length + l_ * cos + w_ * sin)
    )
    enveloppe_small = (np.abs(rel_diff[..., 0]) <= (half_width + half_width)) & (
        np.abs(rel_diff[..., 1]) <= (half_length + half_width)
    )

    total_check = radius_check * enveloppe
    radius_small = enveloppe_small * other_mask

    batch_size, n_other, n_steps = other_gt.shape[:3]
    real_cols = np.array(radius_small.sum((1, 2)) > 0)

    # Use polygon intersection for more accurate collision detection
    vertexes = compute_corners(length, width)
    worth_check = total_check.sum(1)

    for b in range(batch_size):
        if not real_cols[b]:
            batch_vertices_ = compute_vertexes_batch(ego_pred, ego_heading[..., 0], vertexes)
            for t in range(n_steps):
                if worth_check[b, t]:
                    ego_vertice = batch_vertices_[b, t]
                    ego_box = Polygon(ego_vertice)
                    for a in range(n_other):
                        if total_check[b, a, t]:
                            other_corners = compute_corners(2 * l_[b, a, 0], 2 * w_[b, a, 0])
                            other_vertice = compute_vertexes(
                                other_gt[b, a, t, 0],
                                other_gt[b, a, t, 1],
                                other_heading[b, a, t, 0],
                                other_corners,
                            )
                            other_box = Polygon(other_vertice)
                            is_col = ego_box.intersects(other_box)

                            if is_col:
                                real_cols[b] = 1
                                return real_cols

    return real_cols


def check_ego_collisions_idx(
    ego_pred,
    ego_heading,
    other_gt,
    other_heading,
    other_mask,
    margin=(0, 0),
    other_dims=None,
    margins=None,
    speed=0,
    other_speeds=None,
):
    """
    Checks for collisions at every timestep between an ego prediction and the ground truth position of other vehicles.

    Args:
        ego_pred: tensor of size (batch, time, 2) indicating the sequence of coordinates of the ego center
        ego_heading: tensor of size (batch, time, 1) indicating the sequence of heading of the ego
        other_gt: tensor of size (batch, agent, time, 2) indicating the sequence of ccordinates for each agent
        other_heading: tensor of size (batch, agent, time, 1) indicating the sequence of heading for each agent
        other_mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent
        margin: margin to add to the dimensions of the agents
        other_dims: tensor of size (batch, agent, 2) indicating the dimensions of each agent
        margins: tensor of size (batch, agent, 2) indicating the margin to add to the dimensions of each agent
        speed: speed of the ego
        other_speeds: tensor of size (batch, agent) indicating the speed of each agent

    Returns:
        collision: boolean (batch)
    """

    diff = other_gt - ego_pred[:, None]
    dists = np.sqrt((diff**2).sum(-1))

    # Default Highway Environment vehicle dimensions
    length, width = 5.0, 2.0

    length, width = length + margin[0] + speed / 2, width + margin[1]
    L1 = np.sqrt(length**2 + width**2)

    half_length, half_width = length / 2, width / 2
    if other_dims[0].shape[0] > 0:
        l_, w_ = other_dims[:, :, 0] / 2 + margin[0] / 2, other_dims[:, :, 1] / 2 + margin[1] / 2
        l_, w_ = l_[..., None], w_[..., None]
    else:
        l_, w_ = length / 2, width / 2
    if margins is not None:
        w_ = w_ + margins[..., None]
    if other_speeds is not None:
        l_ = l_ + other_speeds[..., None] / 4

    L2 = np.sqrt((2 * l_) ** 2 + (2 * w_) ** 2)
    L = (L1 + L2) / 2

    radius_small = (dists < width) * other_mask
    radius_check = (dists < L) * other_mask

    batch_size, n_other, n_steps = other_gt.shape[:3]

    if radius_check.sum() == 0:
        return radius_check.sum(2)

    cos1, sin1 = np.cos(ego_heading[..., 0]), np.sin(ego_heading[..., 0])
    vx = np.stack([cos1, sin1], -1)
    vy = np.stack([-sin1, cos1], -1)
    ego_matrix = np.stack([vx, vy], -2)
    rel_diff = ego_matrix[:, None, :] @ diff[..., None]
    rel_diff = rel_diff[..., 0]

    delta_angle = np.abs(((other_heading - ego_heading[:, None]) + np.pi / 2) % np.pi - np.pi / 2)[..., 0]
    cos, sin = np.cos(delta_angle), np.sin(delta_angle)
    rel_diff = rel_diff[..., ::-1]

    enveloppe = (np.abs(rel_diff[..., 0]) <= (half_width + w_ * cos + l_ * sin)) & (
        np.abs(rel_diff[..., 1]) <= (half_length + l_ * cos + w_ * sin)
    )
    enveloppe_small = (np.abs(rel_diff[..., 0]) <= (half_width + half_width)) & (
        np.abs(rel_diff[..., 1]) <= (half_length + half_width)
    )

    total_check = radius_check * enveloppe
    radius_small = enveloppe_small * other_mask

    vertexes = compute_corners(length, width)
    worth_check = total_check.sum(1)

    is_collision = radius_small.sum(2)
    for b in range(batch_size):
        batch_vertices_ = compute_vertexes_batch(ego_pred, ego_heading[..., 0], vertexes)
        for t in range(n_steps):
            if worth_check[b, t]:
                ego_vertice = batch_vertices_[b, t]
                ego_box = Polygon(ego_vertice)
                for a in range(n_other):
                    if total_check[b, a, t]:
                        other_corners = compute_corners(2 * l_[b, a, 0], 2 * w_[b, a, 0])
                        other_vertice = compute_vertexes(
                            other_gt[b, a, t, 0],
                            other_gt[b, a, t, 1],
                            other_heading[b, a, t, 0],
                            other_corners,
                        )
                        other_box = Polygon(other_vertice)
                        is_col = ego_box.intersects(other_box)
                        if is_col:
                            is_collision[b, a] = 1

    return is_collision


def check_drivable_area(ego_pred, map, map_mask, ego_yaw, map_yaw, map_avg_tan, map_avg_norm, max_lat, max_tan, 
                        highway_map=None, lanes=None):
    """
    Checks if the ego prediction is in the drivable area.
    Adapted for Highway Environment with optional direct lane checking.

    Args:
        ego_pred: tensor of size (batch, time, 2) indicating the sequence of coordinates of the ego center
        map: tensor of size (batch, time, 2) indicating the sequence of coordinates of the map center
        map_mask: tensor of size (batch, time, 2) indicating the sequence of mask of the map
        ego_yaw: tensor of size (batch, time, 1) indicating the sequence of heading of the ego
        map_yaw: tensor of size (batch, time, 1) indicating the sequence of heading of the map
        map_avg_tan: tensor of size (batch, time, 2) indicating the sequence of average tangent of the map
        map_avg_norm: tensor of size (batch, time, 2) indicating the sequence of average normal of the map
        max_lat: tensor of size (batch, time, 1) indicating the sequence of maximum lateral distance of the map
        max_tan: tensor of size (batch, time, 1) indicating the sequence of maximum tangent of the map
        highway_map: Optional Highway Environment road map
        lanes: Optional list of Highway Environment lanes

    Returns:
        boolean (batch), distance to closest drivable point, time until off-road, angle deviation, and in-goal mask
    """
    # Use direct Highway Environment map check if provided and no map tensor is available
    if highway_map is not None and (map is None or map.shape[1] == 0) and ego_pred is not None:
        # Simplified direct Highway Environment map check for critical cases
        batch_size = ego_pred.shape[0]
        results = np.ones(batch_size)
        distances = np.ones(batch_size) * 1000
        times = np.zeros(batch_size)
        angles = np.zeros(batch_size)
        in_goal_masks = np.zeros((batch_size, ego_pred.shape[1]-1))
        
        for b in range(batch_size):
            for t, pos in enumerate(ego_pred[b]):
                # Check if position is on road using Highway Environment's is_on_road function
                on_road = False
                
                if hasattr(highway_map, 'is_on_road'):
                    on_road = highway_map.is_on_road(pos)
                elif hasattr(highway_map, 'on_lane'):
                    # Alternative method if is_on_road doesn't exist
                    on_road = any(highway_map.on_lane(pos))
                
                if not on_road:
                    results[b] = 0
                    times[b] = t
                    
                    # Find nearest point on road to calculate distance
                    min_dist = float('inf')
                    if hasattr(highway_map, 'lanes'):
                        for lane in highway_map.lanes:
                            for lane_pos in lane.positions:
                                dist = np.sqrt(np.sum((lane_pos - pos)**2))
                                if dist < min_dist:
                                    min_dist = dist
                                    # Calculate angle difference
                                    lane_heading = lane.heading_at(lane.local_coordinates(pos)[0])
                                    ego_heading = ego_yaw[b, t, 0]
                                    angles[b] = abs((lane_heading - ego_heading + np.pi) % (2*np.pi) - np.pi)
                    
                    distances[b] = min_dist
                    break
                
                if t > 0:
                    in_goal_masks[b, t-1] = 1
        
        return results, np.min(distances), np.min(times), np.min(angles), in_goal_masks

    # If lanes are provided but no map tensor
    if lanes is not None and (map is None or map.shape[1] == 0) and ego_pred is not None:
        # Use lanes for drivability check
        batch_size = ego_pred.shape[0]
        results = np.ones(batch_size)
        distances = np.ones(batch_size) * 1000
        times = np.zeros(batch_size)
        angles = np.zeros(batch_size)
        in_goal_masks = np.zeros((batch_size, ego_pred.shape[1]-1))
        
        for b in range(batch_size):
            for t, pos in enumerate(ego_pred[b]):
                # Calculate distances to all lane points
                min_dist = float('inf')
                min_angle = 0
                on_lane = False
                
                for lane in lanes:
                    # Check if position is on lane
                    longitudinal, lateral = lane.local_coordinates(pos)
                    if longitudinal >= 0 and longitudinal <= lane.length and abs(lateral) <= lane.width/2:
                        on_lane = True
                    
                    # Find closest point on lane
                    for i, lane_pos in enumerate(lane.positions):
                        dist = np.sqrt(np.sum((lane_pos - pos)**2))
                        if dist < min_dist:
                            min_dist = dist
                            # Get lane heading at closest point
                            if i < len(lane.positions) - 1:
                                dx = lane.positions[i+1][0] - lane_pos[0]
                                dy = lane.positions[i+1][1] - lane_pos[1]
                                lane_heading = np.arctan2(dy, dx)
                                min_angle = abs((lane_heading - ego_yaw[b, t, 0] + np.pi) % (2*np.pi) - np.pi)
                
                # Consider off-road if not on any lane
                if not on_lane:
                    results[b] = 0
                    times[b] = t
                    distances[b] = min_dist
                    angles[b] = min_angle
                    break
                
                if t > 0:
                    in_goal_masks[b, t-1] = 1
        
        return results, np.min(distances), np.min(times), np.min(angles), in_goal_masks
    
    # Original implementation with map tensor
    batch_size, n_lanes, n_points, d = map.shape

    # Default Highway Environment vehicle dimensions
    length, width = 5.0, 2.0
    # print(f"Map shape: {map.shape}, Map mask shape: {map_mask.shape}, Ego pred shape: {ego_pred.shape}")
    mask_diff = (((map_mask[:, :, 1:].astype(bool)) & (map_mask[:, :, :-1].astype(bool)))[..., 0]).max(2)

    avg_tan = map_avg_tan
    norm = map_avg_norm

    start = map[:, :, 0]
    rel_ego = ego_pred[:, None] - start[:, :, None]

    lat_proj = (norm[:, :, None, None] @ rel_ego[..., None])[..., 0]

    within_norm_time = (np.abs(lat_proj[..., 0]) < (2 + max_lat[..., None] / 2 + length / 2)) * mask_diff[..., None]
    within_norm = within_norm_time.max(2)
    map_mask = map_mask[..., 0]
    batchsize = len(ego_pred)
    num_lanes = within_norm.sum(-1).astype(np.int32)
    max_lanes = int(num_lanes.max())

    if max_lanes == 0:
        return (
            np.ones(1),
            1001,
            0,
            0,
            np.zeros(1),
        )

    new_start = np.zeros((batchsize, max_lanes, 2))
    new_yaw = np.zeros((batchsize, max_lanes, n_points - 1))
    new_tan = np.zeros((batchsize, max_lanes, 2))
    new_mask = np.zeros((batchsize, max_lanes, n_points))
    new_abs_dists = np.zeros((batchsize, max_lanes))
    new_norm = np.zeros((batchsize, max_lanes, 2))
    new_max_lat = np.zeros((batchsize, max_lanes))
    new_max_tan = np.zeros((batchsize, max_lanes))

    for b in range(batchsize):
        new_start[b, : num_lanes[b]] = start[b][within_norm[b]]
        new_tan[b, : num_lanes[b]] = avg_tan[b][within_norm[b]]
        new_mask[b, : num_lanes[b]] = map_mask[b][within_norm[b]]
        new_yaw[b, : num_lanes[b]] = map_yaw[b][within_norm[b]]
        new_norm[b, : num_lanes[b]] = norm[b][within_norm[b]]
        new_max_lat[b, : num_lanes[b]] = max_lat[b][within_norm[b]]
        new_max_tan[b, : num_lanes[b]] = max_tan[b][within_norm[b]]

    vertexes = []
    vertexes.append((0 + length / 2, 0 + width / 2))
    vertexes.append((0 + length / 2, 0 - width / 2))
    vertexes.append((0 - length / 2, 0 - width / 2))
    vertexes.append((0 - length / 2, 0 + width / 2))
    vertexes = np.transpose(np.array(vertexes))
    batch_size, n_steps = ego_pred.shape[:2]

    batch_vertices_ = compute_vertexes_batch(ego_pred, ego_yaw[..., 0], vertexes)
    batch_vertices = batch_vertices_.reshape(batch_size, n_steps * 4, 2)
    batch_vertices = np.concatenate([batch_vertices, ego_pred[:, -1:]], 1)

    mask_diff = (new_mask[:, :, 1:] + new_mask[:, :, :-1]) > 1
    mask_tot = mask_diff.max(2)

    rel_ego = batch_vertices[:, None] - new_start[:, :, None]

    lat_proj = (new_norm[:, :, None, None] @ rel_ego[..., None])[..., 0]

    within_norm_time = (np.abs(lat_proj[..., 0]) < (1.8 + new_max_lat[..., None] / 2)) * mask_tot[..., None]
    new_abs_dists = np.abs(lat_proj[..., 0])[:, :, -1]

    tan_proj = (new_tan[:, :, None, None] @ rel_ego[..., None])[..., 0]
    within_tan = (
        within_norm_time
        * (tan_proj[..., 0] > -1)
        * (tan_proj[..., 0] < new_max_tan[..., None] + 1)
        * mask_tot[..., None]
    )

    selected = within_tan[:, :, -1]
    if selected.sum():
        new_tan = new_tan[selected]
        mask_diff = mask_diff[selected].max(1)

        new_abs_dists = new_abs_dists[selected]
        selected = selected[0]

        yaw_avg = np.arctan2(new_tan[..., 1], new_tan[..., 0])
        within_yaw = (ego_yaw[0, -1, 0] - yaw_avg) + (1 - mask_diff) * 10
        within_yaw = np.abs(within_yaw)

        closest_angle = np.sin(within_yaw) + (1 - mask_diff) * 1000
        angle_mask = (within_yaw > (np.ones_like(within_yaw) * np.pi / 4)).astype(np.float32) + (
            within_yaw > (np.ones_like(within_yaw) * np.pi / 2)
        ).astype(np.float32)
        within_both = (1 - mask_diff) * 1000 + new_abs_dists + angle_mask
        closest_angle = np.sin(np.abs(within_yaw)) + (1 - mask_diff) * 1000

    else:
        closest_angle = np.array([1000])
        within_both = np.array([1000])
    lat = within_tan.max(1)

    return (
        (1 - lat).max(1),
        within_both.min(),
        (1 - lat).argmax(),
        closest_angle.min(),
        lat[:, :-1],
    )


def split_with_ratio(ratios, nodes, mask):
    """
    Splits the nodes and mask with a given ratio.

    Args:
        ratios: tensor of size (batch, agent, time) indicating the ratio of each agent
        nodes: tensor of size (batch, agent, time, 2) indicating the sequence of coordinates for each agent
        mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent

    Returns:
        nodes: tensor of size (batch, agent, time, 2) indicating the sequence of coordinates for each agent
        mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent
    """
    # print(f"Original nodes shape: {nodes.shape}, mask shape: {mask.shape}")
    # print(f"Ratios shape: {ratios.shape}")

    ratio_mask = ratios >= 2
    ratios = ratios[ratio_mask]
    nodes_selected = nodes[ratio_mask]
    mask_selected = mask[ratio_mask]
    new_nodes = []
    new_masks = []
    n_long, n_points = nodes_selected.shape[:2]

    for i, lane in enumerate(nodes_selected):
        ratio = ratios[i]

        if ratio > 1:
            if ratio > 10:
                ratio = 10
            elif ratio > 5:
                ratio = 5
            elif ratio < 4 and ratio > 2:
                ratio = 2

            lanes = np.zeros((ratio, n_points, 2))
            masks = np.zeros((ratio, n_points))
            lanes[:, : n_points // ratio] = lane.reshape(ratio, n_points // ratio, 2)
            masks[:, : n_points // ratio] = mask_selected[i].reshape(ratio, n_points // ratio)

            if ratio < 10:
                next = lanes[1:, 0]
                lanes[:-1, n_points // ratio] = next
                next_mask = masks[1:, 0]
                masks[:-1, n_points // ratio] = next_mask
                new_nodes.append(lanes)
                new_masks.append(masks)
            else:
                additional_lanes = np.zeros((ratio - 1, n_points, 2))
                additional_masks = np.zeros((ratio - 1, n_points))
                additional_lanes[:, : n_points // ratio] = lane[1:-1].reshape(ratio - 1, n_points // ratio, 2)
                additional_masks[:, : n_points // ratio] = mask_selected[i][1:-1].reshape(ratio - 1, n_points // ratio)
                new_nodes.append(lanes)
                new_masks.append(masks)
                new_nodes.append(additional_lanes)
                new_masks.append(additional_masks)

    if new_nodes:
        new_nodes = np.concatenate(new_nodes, 0)
        new_masks = np.concatenate(new_masks, 0)
        nodes = np.concatenate([nodes[~ratio_mask], new_nodes], 0)
        mask = np.concatenate([mask[~ratio_mask], new_masks], 0)
        mask = mask > 0.1

    # print(f"Split nodes shape: {nodes.shape}, mask shape: {mask.shape}")

    return nodes, mask


def get_directional_values(nodes, mask):
    """
    Computes the directional values of the nodes.

    Args:
        nodes: tensor of size (batch, agent, time, 2) indicating the sequence of coordinates for each agent
        mask: tensor of size (batch, agent, time) indicating the sequence of mask for each agent

    Returns:
        avg_tan: tensor of size (batch, agent, 2) indicating the average tangent for each agent
        norm: tensor of size (batch, agent, 2) indicating the average normal for each agent
        yaw: tensor of size (batch, agent, 1) indicating the average yaw for each agent
    """

    p1 = nodes[:, 1:]
    p2 = nodes[:, :-1]
    diff = p2 - p1

    mask_total = (mask[:, 1:].astype(bool)) & (mask[:, :-1].astype(bool))
    
    # Check if mask_total has any valid points
    if mask_total.shape[1] == 0 or np.sum(mask_total) == 0:
        # Return default values when no valid points
        shape = list(nodes.shape)
        shape[-1] = 2  # For tangent and norm vectors
        shape[-2] = 1  # For single value
        
        # Default tangent is (1,0) - forward direction
        avg_tan = np.zeros(shape[:-2] + [2])
        avg_tan[..., 0] = 1.0
        
        # Default norm is (0,1) - perpendicular to tangent
        norm = np.zeros(shape[:-2] + [2])
        norm[..., 1] = 1.0
        
        # Default yaw is 0 - forward direction
        yaw = np.zeros(shape[:-2] + [1])
        
        return avg_tan, norm, yaw

    tangent = diff
    norm_tan = np.sqrt(np.sum(tangent**2, -1, keepdims=True))
    tangent = tangent / np.maximum(norm_tan, 1e-10)
    
    mask_total = mask_total[..., None]
    
    # Handle broadcasting issue - ensure shapes are compatible
    if mask_total.shape[-2] == 0:
        # Return default values when mask has zero dimension
        shape = list(nodes.shape)
        avg_tan = np.zeros(shape[:-2] + [2])
        avg_tan[..., 0] = 1.0
        norm = np.zeros(shape[:-2] + [2])
        norm[..., 1] = 1.0
        yaw = np.zeros(shape[:-2] + [1])
        return avg_tan, norm, yaw
        
    avg_tan = (-tangent * mask_total).sum(-2) / np.maximum(mask_total[..., 0].sum(-1, keepdims=True), 1)
    avg_tan = avg_tan / np.maximum(np.sqrt(np.sum(avg_tan**2, -1, keepdims=True)), 1e-10)
    
    # Calculate normal vector (perpendicular to tangent)
    norm = np.concatenate([avg_tan[..., 1:], -avg_tan[..., :1]], -1)
    
    # Calculate yaw angle
    yaw = np.arctan2(avg_tan[..., 1:], avg_tan[..., :1])
    
    return avg_tan, norm, yaw


def trajectory2action(trajectory, dt=0.5, vehicle_physics=None):
    """
    Computes the action from a trajectory.
    Adapted for Highway Environment vehicle physics.

    Args:
        trajectory: tensor of size (time, 2) indicating the sequence of coordinates
        dt: time step
        vehicle_physics: Optional vehicle physics settings

    Returns:
        acc: acceleration
        st2: steering
    """
    trajectory = trajectory[0]
    traj_diffs = trajectory[1:] - trajectory[:-1]
    speeds = np.sqrt((traj_diffs[..., :2] ** 2).sum(-1)) / dt
    accs = (speeds[1:] - speeds[:-1]) / dt

    yaws = np.arctan2(traj_diffs[:, 1], traj_diffs[:, 0])
    yrs = (yaws[1:] - yaws[:-1]) / dt

    yr = np.mean(yrs[:4])

    yrs_pred = traj_diffs[..., 2] / dt if traj_diffs.shape[-1] > 2 else np.zeros_like(yrs[:2])
    yr_pred = np.mean(yrs_pred[:2]) if len(yrs_pred) > 0 else 0
    yr = yr + yr_pred / 2
    yr = yr_pred if yr_pred != 0 else yr

    speed_avg = (speeds[1:] + speeds[:-1]) / 2
    
    # Use vehicle-specific wheel base if available
    wheel_base = 3.5  # Default Highway Environment vehicle wheelbase
    if vehicle_physics is not None and hasattr(vehicle_physics, 'wheelbase'):
        wheel_base = vehicle_physics.wheelbase
    
    # Calculate steering angle using the bicycle model formula
    steerings2 = np.arctan(yrs * wheel_base / (speed_avg + 1e-6))
    st2 = np.mean(steerings2[:4])

    acc = np.mean(accs[:2])

    return acc, st2, None


def softmax(a, T=1):
    """
    Compute softmax function for a batch of inputs.
    
    Args:
        a: Input array
        T: Temperature parameter
    
    Returns:
        Softmax output
    """
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# Highway Environment-specific utility functions

def highway_to_numpy_transform(position, heading):
    """
    Convert Highway Environment position and heading to numpy arrays
    
    Args:
        position: Highway Environment position
        heading: Highway Environment heading
        
    Returns:
        Tuple of (position, heading) as numpy arrays
    """
    pos = np.array(position)
    rot = np.array([0, 0, heading])  # Only yaw is relevant for 2D environment
    return pos, rot

def highway_lanes_to_path(lanes, length=20):
    """
    Convert a list of Highway Environment lanes to a path representation suitable for MCTS
    
    Args:
        lanes: List of Highway Environment lane objects
        length: Maximum number of waypoints to include
        
    Returns:
        Tuple of (positions, headings, masks) for use in MCTS
    """
    positions = []
    headings = []
    masks = []
    
    # Extract waypoints from lanes
    for lane in lanes:
        if hasattr(lane, 'positions') and len(lane.positions) > 0:
            # Sample points along the lane
            lane_positions = lane.positions
            
            # Calculate headings based on adjacent points
            lane_headings = []
            for i in range(len(lane_positions) - 1):
                dx = lane_positions[i+1][0] - lane_positions[i][0]
                dy = lane_positions[i+1][1] - lane_positions[i][1]
                lane_headings.append(np.arctan2(dy, dx))
            
            # Add the last heading (same as second to last)
            if lane_headings:
                lane_headings.append(lane_headings[-1])
            else:
                lane_headings.append(0)  # Default if no heading available
            
            # Only take up to 'length' points
            max_points = min(len(lane_positions), length)
            positions.extend(lane_positions[:max_points])
            headings.extend(lane_headings[:max_points])
            masks.extend([1] * max_points)  # All points are valid
    
    # Limit to requested length
    if len(positions) > length:
        positions = positions[:length]
        headings = headings[:length]
        masks = masks[:length]
    
    # If no points found, return empty arrays
    if not positions:
        return (np.zeros((1, 0, 2)), np.zeros((1, 0, 1)), np.zeros((1, 0, 1)))
    
    # Reshape for MCTS format
    positions = np.array(positions).reshape(1, len(positions), 2)  # [batch, points, xy]
    headings = np.array(headings).reshape(1, len(headings), 1)     # [batch, points, 1]
    masks = np.array(masks).reshape(1, len(masks), 1)              # [batch, points, 1]
    
    return positions, headings, masks

def get_vehicle_parameters(vehicle):
    """
    Extract important vehicle parameters from Highway Environment vehicle
    
    Args:
        vehicle: Highway Environment vehicle object
        
    Returns:
        Dictionary of vehicle parameters
    """
    if vehicle is None:
        return None
    
    params = {
        'dimensions': (5.0, 2.0, 1.5),  # Default dimensions (length, width, height)
        'mass': 1000.0,
        'wheelbase': 3.5
    }
    
    # Extract vehicle attributes if available
    if hasattr(vehicle, 'LENGTH'):
        params['dimensions'] = (vehicle.LENGTH, vehicle.WIDTH, 1.5)
    
    if hasattr(vehicle, 'MASS'):
        params['mass'] = vehicle.MASS
    
    if hasattr(vehicle, 'wheelbase'):
        params['wheelbase'] = vehicle.wheelbase
    
    return params


def get_simulation_params(action_space=None, planning_horizon=5.0, reward_weights=None):
    """
    Get simulation parameters for MCTS planning
    
    Args:
        action_space: Dictionary or list of available actions
        planning_horizon: Time horizon for planning in seconds
        reward_weights: Dictionary of weights for different reward components
        
    Returns:
        Dict: Parameters for MCTS simulation
    """
    # Default reward weights if none provided
    if reward_weights is None:
        reward_weights = {
            'safety': 10.0,
            'progress': 1.0, 
            'comfort': 0.5,
            'efficiency': 1.0
        }
        
    # Build default action space if none provided
    if action_space is None:
        action_space = {
            "overtaking": {'acceleration': 0.8, 'steering': 0.0},
            "keeping_lane": {'acceleration': 0.3, 'steering': 0.0},
            "turning_left": {'acceleration': 0.2, 'steering': -0.3},
            "turning_right": {'acceleration': 0.2, 'steering': 0.3},
            "left_change": {'acceleration': 0.3, 'steering': -0.2},
            "right_change": {'acceleration': 0.3, 'steering': 0.2},
            "brake": {'acceleration': -0.3, 'steering': 0.0}
        }
    
    # Create simulation parameters
    params = {
        'action_space': action_space,
        'planning_horizon': planning_horizon,
        'reward_weights': reward_weights,
        'dt': 0.1,  # Standard time step
        'simulation_steps': int(planning_horizon / 0.1),  # Number of steps to simulate
        'discount_factor': 0.95,  # Discount factor for future rewards
        'exploration_constant': 2.0,  # UCT exploration constant
        'iterations': 100,  # Default number of MCTS iterations
        'max_tree_depth': 10,  # Maximum tree depth
        'vehicle_model': 'bicycle',  # Vehicle model for simulation
    }
    
    return params