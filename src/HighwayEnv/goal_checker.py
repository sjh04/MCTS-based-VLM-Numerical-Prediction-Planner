import numpy as np
import gymnasium as gym
import highway_env

class GoalChecker:
    """
    Check if the agent has reached a goal in Highway Environment
    """
    def __init__(self, goal_type='lane_end', goal_position=None, goal_lane=None, goal_distance=None):
        """
        Initialize a goal checker for Highway Environment
        
        Args:
            goal_type: Type of goal ('lane_end', 'position', 'lane')
            goal_position: Goal position coordinates (x, y)
            goal_lane: Goal lane index
            goal_distance: Distance to travel to reach goal
        """
        self.goal_type = goal_type
        self.goal_position = goal_position
        self.goal_lane = goal_lane
        self.goal_distance = goal_distance
        
        # Goal location as position
        self.goal_location = None
        
        # Starting position to measure distance traveled
        self.start_position = None
        self.start_lane = None
        self.distance_traveled = 0.0
        
    def set_vehicle(self, vehicle):
        """Set the vehicle to check goals for"""
        if self.start_position is None and vehicle is not None:
            self.start_position = np.array(vehicle.position)
            if hasattr(vehicle, 'lane_index'):
                self.start_lane = vehicle.lane_index
                
    def check_goal(self, vehicle):
        """
        Check if the vehicle has reached the goal
        
        Args:
            vehicle: Highway Environment vehicle
            
        Returns:
            bool: Whether the goal has been reached
        """
        if vehicle is None:
            return False
            
        # Update goal location if needed
        self._update_goal_location(vehicle)
        
        # Check if goal has been reached based on goal type
        if self.goal_type == 'lane_end':
            return self._check_lane_end_goal(vehicle)
        elif self.goal_type == 'position':
            return self._check_position_goal(vehicle)
        elif self.goal_type == 'lane':
            return self._check_lane_goal(vehicle)
        elif self.goal_type == 'distance':
            return self._check_distance_goal(vehicle)
        
        return False
        
    def _update_goal_location(self, vehicle):
        """Update goal location based on goal type"""
        if self.goal_location is not None:
            return
            
        if self.goal_type == 'position' and self.goal_position is not None:
            # Use provided position as goal
            self.goal_location = np.array([
                self.goal_position[0],
                self.goal_position[1],
                0 if len(self.goal_position) <= 2 else self.goal_position[2]
            ])
        elif self.goal_type == 'lane_end' and hasattr(vehicle, 'lane'):
            # Use end of current lane as goal
            lane = vehicle.lane
            if hasattr(lane, 'length'):
                self.goal_location = lane.position(lane.length, 0)
        elif self.goal_type == 'lane' and self.goal_lane is not None and hasattr(vehicle, 'road'):
            # Use position in target lane as goal
            road = vehicle.road
            if hasattr(road, 'network'):
                current_lane_idx = vehicle.lane_index if hasattr(vehicle, 'lane_index') else None
                if current_lane_idx:
                    # Find a lane with the desired index
                    target_lane_idx = (current_lane_idx[0], current_lane_idx[1], self.goal_lane)
                    lane = road.network.get_lane(target_lane_idx)
                    if lane:
                        # Target middle of the lane at same longitudinal position
                        longitudinal, _ = vehicle.lane.local_coordinates(vehicle.position)
                        self.goal_location = lane.position(longitudinal, 0)
        
    def _check_lane_end_goal(self, vehicle):
        """Check if vehicle has reached the end of its lane"""
        if hasattr(vehicle, 'lane') and hasattr(vehicle.lane, 'length'):
            longitudinal, _ = vehicle.lane.local_coordinates(vehicle.position)
            # Consider goal reached if within 5m of lane end
            return longitudinal >= vehicle.lane.length - 5.0
        return False
        
    def _check_position_goal(self, vehicle):
        """Check if vehicle has reached the goal position"""
        if self.goal_location is not None:
            distance = np.linalg.norm(vehicle.position[:2] - self.goal_location[:2])
            # Consider goal reached if within 5m of goal position
            return distance <= 5.0
        return False
        
    def _check_lane_goal(self, vehicle):
        """Check if vehicle has reached the goal lane"""
        if hasattr(vehicle, 'lane_index') and self.goal_lane is not None:
            return vehicle.lane_index[2] == self.goal_lane
        return False
        
    def _check_distance_goal(self, vehicle):
        """Check if vehicle has traveled the goal distance"""
        if self.goal_distance is not None and self.start_position is not None:
            # Calculate total distance traveled
            if hasattr(vehicle, 'lane'):
                # Use lane coordinates for more accurate longitudinal distance
                lane = vehicle.lane
                start_long, _ = lane.local_coordinates(self.start_position)
                current_long, _ = lane.local_coordinates(vehicle.position)
                distance = current_long - start_long
            else:
                # Fallback to direct distance
                distance = np.linalg.norm(vehicle.position[:2] - self.start_position[:2])
                
            return distance >= self.goal_distance
        return False
        
    def get_goal_direction(self, vehicle):
        """
        Get normalized direction vector toward goal
        
        Args:
            vehicle: Highway Environment vehicle
            
        Returns:
            numpy.ndarray: Direction vector toward goal
        """
        if self.goal_location is None:
            self._update_goal_location(vehicle)
            
        if self.goal_location is not None:
            direction = self.goal_location[:2] - vehicle.position[:2]
            norm = np.linalg.norm(direction)
            if norm > 0:
                return direction / norm
                
        # Default: forward direction
        if hasattr(vehicle, 'heading'):
            return np.array([np.cos(vehicle.heading), np.sin(vehicle.heading)])
            
        return np.array([1.0, 0.0])
        
    def get_distance_to_goal(self, vehicle):
        """
        Get distance to goal
        
        Args:
            vehicle: Highway Environment vehicle
            
        Returns:
            float: Distance to goal
        """
        if self.goal_location is None:
            self._update_goal_location(vehicle)
            
        if self.goal_location is not None:
            return np.linalg.norm(vehicle.position[:2] - self.goal_location[:2])
            
        # Fallback: use lane end if available
        if hasattr(vehicle, 'lane') and hasattr(vehicle.lane, 'length'):
            longitudinal, _ = vehicle.lane.local_coordinates(vehicle.position)
            return vehicle.lane.length - longitudinal
            
        return float('inf')  # unknown distance
