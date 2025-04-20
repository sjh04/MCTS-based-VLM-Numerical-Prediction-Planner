import math
import sys
sys.path.append('/home/ubuntu/dockerCarla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
import carla
from typing import List, Dict, Union


class GoalChecker:
    def __init__(self):
        self.current_goal = None
        self.success_conditions = {}
        self.goal_waypoints = []
        self.goal_location = None
        self.target_lane_id = None
        self.target_speed = None
        self.goal_radius = 5.0  # Default goal radius in meters
        self.heading_tolerance = 20.0  # Degrees
        self.previous_distance = float('inf')
        self.progress_threshold = 0.5  # Minimum distance progress to consider making progress
        self.min_speed_threshold = 1.0  # Minimum speed to consider making progress (m/s)
        
    def update_goal(self, goal_spec):
        """Update the current goal with new specifications"""
        self.current_goal = goal_spec
        self._parse_goal_spec(goal_spec)
        
        # Reset progress tracking
        self.previous_distance = float('inf')
        
    def _parse_goal_spec(self, spec):
        """
        Parse goal specification into concrete success conditions
        
        Supported goal specs:
        - "reach_position": (x, y, z) - Reach specific coordinates
        - "follow_lane": lane_id - Follow a specific lane
        - "maintain_speed": speed (m/s) - Maintain a target speed
        - "navigate_to_waypoint": waypoint - Navigate to a specific waypoint
        """
        self.success_conditions = {}
        
        if isinstance(spec, dict):
            # Process location goal
            if 'location' in spec:
                self.goal_location = carla.Location(
                    x=spec['location'][0],
                    y=spec['location'][1],
                    z=spec['location'][2] if len(spec['location']) > 2 else 0
                )
                self.success_conditions['reach_position'] = True
                
                # Optional goal radius
                if 'radius' in spec:
                    self.goal_radius = float(spec['radius'])
            
            # Process waypoints goal
            if 'waypoints' in spec:
                self.goal_waypoints = spec['waypoints']
                self.success_conditions['follow_waypoints'] = True
            
            # Process lane goal
            if 'lane_id' in spec:
                self.target_lane_id = int(spec['lane_id'])
                self.success_conditions['follow_lane'] = True
            
            # Process speed goal
            if 'speed' in spec:
                self.target_speed = float(spec['speed'])
                self.success_conditions['maintain_speed'] = True
                
            # Process heading goal
            if 'heading' in spec:
                self.target_heading = float(spec['heading'])
                self.success_conditions['maintain_heading'] = True
                
                # Optional heading tolerance
                if 'heading_tolerance' in spec:
                    self.heading_tolerance = float(spec['heading_tolerance'])
        
        elif isinstance(spec, str):
            # Parse string format like "reach_position:100,200,0;follow_lane:3"
            parts = spec.split(';')
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    if key == 'reach_position':
                        coords = [float(x) for x in value.split(',')]
                        self.goal_location = carla.Location(x=coords[0], y=coords[1], z=coords[2] if len(coords) > 2 else 0)
                        self.success_conditions['reach_position'] = True
                    elif key == 'follow_lane':
                        self.target_lane_id = int(value)
                        self.success_conditions['follow_lane'] = True
                    elif key == 'maintain_speed':
                        self.target_speed = float(value)
                        self.success_conditions['maintain_speed'] = True
        else:
            raise ValueError(f"Unsupported goal specification format: {type(spec)}")
        
    def check_goal_reached(self, vehicle, map_instance):
        """Check if the goal has been reached"""
        if not self.current_goal:
            return False
            
        # Check each success condition
        conditions_met = []
        
        # Position check
        if 'reach_position' in self.success_conditions:
            vehicle_loc = vehicle.get_location()
            distance = vehicle_loc.distance(self.goal_location)
            if distance <= self.goal_radius:
                conditions_met.append('reach_position')
        
        # Lane check
        if 'follow_lane' in self.success_conditions:
            vehicle_loc = vehicle.get_location()
            waypoint = map_instance.get_waypoint(vehicle_loc)
            if waypoint.lane_id == self.target_lane_id:
                conditions_met.append('follow_lane')
        
        # Speed check
        if 'maintain_speed' in self.success_conditions:
            velocity = vehicle.get_velocity()
            speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5  # magnitude
            # Consider successful if within 10% of target
            if abs(speed - self.target_speed) <= 0.1 * self.target_speed:
                conditions_met.append('maintain_speed')
                
        # Heading check
        if 'maintain_heading' in self.success_conditions:
            vehicle_transform = vehicle.get_transform()
            vehicle_heading = vehicle_transform.rotation.yaw % 360
            heading_diff = min(
                abs(vehicle_heading - self.target_heading),
                360 - abs(vehicle_heading - self.target_heading)
            )
            if heading_diff <= self.heading_tolerance:
                conditions_met.append('maintain_heading')
                
        # Waypoints check
        if 'follow_waypoints' in self.success_conditions and self.goal_waypoints:
            # Check if we've reached the final waypoint
            if len(self.goal_waypoints) > 0:
                final_wp = self.goal_waypoints[-1]
                vehicle_loc = vehicle.get_location()
                final_wp_loc = final_wp.transform.location
                distance = vehicle_loc.distance(final_wp_loc)
                if distance <= self.goal_radius:
                    conditions_met.append('follow_waypoints')
        
        # For a goal to be considered reached, all conditions must be met
        return len(conditions_met) == len(self.success_conditions)
        
    def check_progress(self, current_state, history=None):
        """
        Check if the vehicle is making progress toward the goal
        Returns: (is_making_progress: bool, progress_metric: float)
        """
        if not self.current_goal or not current_state:
            return False, 0.0
        
        # Get vehicle location
        if hasattr(current_state, 'vehicle_state'):
            vehicle_loc = current_state.vehicle_state.get('location')
        elif hasattr(current_state, 'get_location'):
            vehicle_loc = current_state.get_location()
        else:
            return False, 0.0
            
        # Check progress based on distance to goal
        progress_metric = 0.0
        making_progress = False
        
        # Position-based progress
        if 'reach_position' in self.success_conditions and self.goal_location:
            current_distance = vehicle_loc.distance(self.goal_location)
            distance_progress = self.previous_distance - current_distance
            
            if distance_progress > self.progress_threshold:
                making_progress = True
                progress_metric = distance_progress / self.previous_distance
                
            self.previous_distance = current_distance
        
        # Check if vehicle is moving
        if hasattr(current_state, 'vehicle_state'):
            velocity = current_state.vehicle_state.get('velocity')
            if velocity:
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                if speed < self.min_speed_threshold:
                    making_progress = False
                    progress_metric *= 0.5  # Reduce progress metric if not moving
        
        # Heading-based progress (check if facing toward goal)
        if 'reach_position' in self.success_conditions and self.goal_location:
            if hasattr(current_state, 'vehicle_state'):
                heading = current_state.vehicle_state.get('heading')
                if heading:
                    # Calculate direction vector to goal
                    goal_vector = carla.Vector3D(
                        self.goal_location.x - vehicle_loc.x,
                        self.goal_location.y - vehicle_loc.y,
                        0
                    )
                    
                    # Calculate vehicle forward vector
                    yaw_rad = heading.yaw * 3.14159 / 180.0
                    forward_vector = carla.Vector3D(
                        math.cos(yaw_rad),
                        math.sin(yaw_rad),
                        0
                    )
                    
                    # Normalize vectors
                    goal_vector_length = (goal_vector.x**2 + goal_vector.y**2)**0.5
                    if goal_vector_length > 0:
                        goal_vector.x /= goal_vector_length
                        goal_vector.y /= goal_vector_length
                    
                    # Dot product (alignment metric)
                    alignment = goal_vector.x * forward_vector.x + goal_vector.y * forward_vector.y
                    
                    # If alignment > 0.7 (~45 degrees), we're generally heading toward the goal
                    if alignment > 0.7:
                        making_progress = making_progress or True
                        progress_metric += 0.2 * alignment  # Bonus for heading toward goal
                    elif alignment < 0:  # Heading away from goal
                        making_progress = False
                        progress_metric = 0
        
        return making_progress, progress_metric
    
    def get_goal_distance(self, current_location):
        """Get distance to the goal location"""
        if 'reach_position' in self.success_conditions and self.goal_location:
            return current_location.distance(self.goal_location)
        return float('inf')
        
    def get_next_waypoint(self, current_location, current_waypoint_idx=0):
        """Get the next waypoint to follow from the waypoint list"""
        if 'follow_waypoints' in self.success_conditions and len(self.goal_waypoints) > current_waypoint_idx:
            return self.goal_waypoints[current_waypoint_idx]
        return None