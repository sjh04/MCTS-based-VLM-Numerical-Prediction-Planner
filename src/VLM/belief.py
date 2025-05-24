import numpy as np
import random
import scipy.special
import pdb
import sys
import json
import copy

# Fix: Replace relative imports with absolute imports
from src.VLM.policy import BeliefUpdater
from src.VLM.utils import action_to_id

class OtherVehicleAgent:
    def __init__(self, model, action_space, states, agent_ids, belief=None):
        """
        Initialize the agent
        
        Args:
            model: VLM model
            action_space: Available actions
            states: Dictionary of vehicle states
            agent_ids: List of agent IDs
            belief: Optional Belief instance to update
        """
        self.agent_ids = agent_ids # agent id list
        self.current_states = states # {agent_id: state}, state is a dict, {lane: int, speed: int}
        self.action_probs = {} # {agent_id: action}, action is a array
        self.transition_state = None
        
        # Initialize action space - either use provided or default list
        if isinstance(action_space, list):
            self.action_space = action_space
        else:
            self.action_space = ["overtaking", "keeping_lane", "turning_left", "turning_right", 
                                "left_change", "right_change", "brake"]
        
        self.model = model
        
        # Initialize belief updater with action space
        self.generate_belifs = BeliefUpdater(self.model, self.action_space)
        self.belief = belief  # Store reference to Belief instance

    def get_action(self, episode, step):
        """
        Get the action and update belief if available
        """
        json_path = self.generate_belifs.update(episode, step, self.current_states, self.agent_ids)
        
        # print(f"vehicle_id: {self.agent_ids}")
        with open(json_path, 'r') as f:
            data = json.load(f) # list of dict
            # print(f"data: {data}, type: {type(data)}")

        # print(f"data: {data}, type: {type(data)}")

        for item in data:
            vehicle_id = item['vehicle_id']
            action = item['action']
            if vehicle_id in self.agent_ids:
                action_prob = self._calculate_probability(vehicle_id, action)
                self.action_probs[vehicle_id] = action_prob
                
                # Update belief if it exists
                if self.belief is not None:
                    self._update_belief_with_action(vehicle_id, action, action_prob)

        # Uncomment for debugging
            # for line in f:
            #     data = json.loads(line)
            #     print(f"data: {data}, type: {type(data)}")
            #     data2dict = {item['vehicle_id']: item['action'] for item in data}
            #     vehicle_id = data2dict["vehicle_id"]
            #     if vehicle_id in self.agent_ids:
            #         action = data2dict["action"]
            #         action_prob = self._calculate_probability(vehicle_id, action)
            #         self.action_probs[vehicle_id] = action_prob
                    
            #         # Update belief if it exists
            #         if self.belief is not None:
            #             self._update_belief_with_action(vehicle_id, action, action_prob)
    
    def _calculate_probability(self, agent_id, action, trust_level=1.0, aggression=0.1):
        """
        Calculate the probability
        """
        base_probabilities = np.ones(len(self.action_space)) * (trust_level / 2)
        action_id = action_to_id(action, self.action_space)

        base_probabilities[action_id] = 1.0
         # Apply domain rules to modify probabilities
        safety_mask = np.ones(len(self.action_space))
        
        if self.current_states[agent_id]['speed'] > 80:  # High speed
            # Reduce probability of sharp turns
            turn_indices = [i for i, a in enumerate(self.action_space) if 'turn' in a]
            for idx in turn_indices:
                safety_mask[idx] = aggression
        
        # Apply mask and normalize  
        modified_probs = base_probabilities * safety_mask
        if np.sum(modified_probs) > 0:
            modified_probs = modified_probs / np.sum(modified_probs)
        else:
            modified_probs = np.ones(len(self.action_space)) / len(self.action_space)
        
        return modified_probs

    def _update_belief_with_action(self, vehicle_id, action, action_prob):
        """
        Update the belief with action information for a vehicle
        
        Args:
            vehicle_id: ID of the vehicle
            action: Predicted action
            action_prob: Action probability distribution
        """
        # Make sure vehicle exists in belief system
        if vehicle_id not in self.belief.vehicle_beliefs:
            # Create a basic entry if vehicle not in belief system
            self.belief.vehicle_beliefs[vehicle_id] = {
                'position': None,  # Will be updated from observations
                'velocity': None,
                'heading': None,
                'lane_id': self.current_states[vehicle_id].get('lane'),
                'last_observed': 0,
                'predicted_actions': {}
            }
        
        # Update belief with action probabilities
        self.belief.vehicle_beliefs[vehicle_id]['predicted_actions'] = {
            'distribution': action_prob,
            'most_likely': action,
            'confidence': action_prob[action_to_id(action, self.action_space)]
        }
        
        # Update behavioral model in belief based on action
        if 'behavioral_model' not in self.belief.vehicle_beliefs[vehicle_id]:
            self.belief.vehicle_beliefs[vehicle_id]['behavioral_model'] = {}
            
        # Map certain actions to behavioral traits
        if action == "overtaking":
            self.belief.vehicle_beliefs[vehicle_id]['behavioral_model']['aggressiveness'] = 0.8
        elif action == "keeping_lane":
            self.belief.vehicle_beliefs[vehicle_id]['behavioral_model']['cautiousness'] = 0.7
        elif action == "brake":
            self.belief.vehicle_beliefs[vehicle_id]['behavioral_model']['cautiousness'] = 0.9
            
        # Use action to inform trajectory prediction in belief
        self._update_trajectory_prediction(vehicle_id, action)
    
    def _update_trajectory_prediction(self, vehicle_id, action):
        """
        Update trajectory predictions in belief based on predicted action
        
        Args:
            vehicle_id: ID of the vehicle
            action: Predicted action
        """
        # Skip if no position/velocity info available
        vehicle_belief = self.belief.vehicle_beliefs[vehicle_id]
        if not vehicle_belief["position"].any() or not vehicle_belief["velocity"].any():
            return
            
        # Simple trajectory modifications based on predicted actions
        # In a full implementation, this would be more sophisticated
        if action == "turning_left":
            # Adjust trajectory prediction for left turn
            if 'predicted_trajectory_modifier' not in vehicle_belief:
                vehicle_belief['predicted_trajectory_modifier'] = {
                    'heading': -30,  # degrees
                    'acceleration': -0.3  # m/s²
                }
        elif action == "turning_right":
            # Adjust trajectory prediction for right turn
            if 'predicted_trajectory_modifier' not in vehicle_belief:
                vehicle_belief['predicted_trajectory_modifier'] = {
                    'heading': 30,  # degrees
                    'acceleration': -0.3  # m/s²
                }
        elif action == "overtaking":
            # Adjust trajectory prediction for overtaking
            if 'predicted_trajectory_modifier' not in vehicle_belief:
                vehicle_belief['predicted_trajectory_modifier'] = {
                    'lane_shift': 1,  # lanes
                    'acceleration': 0.5  # m/s²
                }

class Belief():
    def __init__(self, road_graph=None, ego_vehicle_id=0, prior=None, forget_rate=0.0, seed=None):
        """
        Initialize belief state for autonomous driving
        
        Args:
            road_graph: Map/graph of the road network
            ego_vehicle_id: ID of the ego vehicle
            prior: Prior belief distribution
            forget_rate: Rate at which beliefs degrade
            seed: Random seed for reproducibility
        """
        # print("Initializing Belief")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # self.vehicle_belief_generator = OtherVehicleAgent()

        self.debug = False
        self.high_prob = 1e5
        self.low_prob = 1e-5
        
        self.ego_vehicle_id = ego_vehicle_id
        
        # Road structure beliefs
        self.lane_beliefs = {}  # Beliefs about lanes and their properties
        self.intersection_beliefs = {}  # Beliefs about intersections
        
        # Traffic participant beliefs
        self.vehicle_beliefs = {}  # Beliefs about other vehicles
        self.pedestrian_beliefs = {}  # Beliefs about pedestrians
        
        # Traffic rule beliefs
        self.traffic_signal_beliefs = {}  # Beliefs about traffic lights/signs
        
        # Initialize the road network from provided graph
        self.road_graph = road_graph
        self.initialize_road_beliefs()
        
        # Keep track of original beliefs for updating
        self.first_beliefs = {}
        self.forget_rate = forget_rate

    def initialize_road_beliefs(self):
        """Initialize beliefs about the road structure"""
        if not self.road_graph:
            return
            
        # Extract lanes, intersections from the road graph
        lanes = self._extract_lanes_from_graph()
        intersections = self._extract_intersections_from_graph()
        
        # Initialize lane beliefs (e.g., drivability, speed limits)
        for lane_id, lane_data in lanes.items():
            self.lane_beliefs[lane_id] = {
                'drivable': 1.0,  # Probability lane is drivable
                'speed_limit': lane_data.get('speed_limit', 50),  # Default 50 km/h
                'width': lane_data.get('width', 3.5),  # Default 3.5m width
                'connecting_lanes': lane_data.get('connecting_lanes', [])
            }
            
        # Initialize intersection beliefs
        for intersection_id, intersection_data in intersections.items():
            self.intersection_beliefs[intersection_id] = {
                'type': intersection_data.get('type', 'unknown'),  # Intersection type
                'incoming_lanes': intersection_data.get('incoming_lanes', []),
                'outgoing_lanes': intersection_data.get('outgoing_lanes', []),
                'has_traffic_light': intersection_data.get('has_traffic_light', False)
            }
        
        # Store first beliefs for reference
        self.first_beliefs = {
            'lanes': copy.deepcopy(self.lane_beliefs),
            'intersections': copy.deepcopy(self.intersection_beliefs)
        }

    def _extract_lanes_from_graph(self):
        """Extract lane information from road graph"""
        if not self.road_graph or not hasattr(self.road_graph, 'lanes'):
            return {}
        return self.road_graph.lanes

    def _extract_intersections_from_graph(self):
        """Extract intersection information from road graph"""
        if not self.road_graph or not hasattr(self.road_graph, 'intersections'):
            return {}
        return self.road_graph.intersections

    def update(self, origin, final):
        """
        Update belief values with forgetting
        
        Args:
            origin: Original belief value
            final: Target belief value
        """
        dist_total = origin - final
        ratio = (1 - np.exp(-self.forget_rate * np.abs(origin - final)))
        return origin - ratio * dist_total

    def update_to_prior(self):
        """Update beliefs back toward prior beliefs"""
        # Update lane beliefs
        for lane_id in self.lane_beliefs:
            if lane_id in self.first_beliefs['lanes']:
                for key in self.lane_beliefs[lane_id]:
                    if key in self.first_beliefs['lanes'][lane_id]:
                        self.lane_beliefs[lane_id][key] = self.update(
                            self.lane_beliefs[lane_id][key],
                            self.first_beliefs['lanes'][lane_id][key]
                        )
        
        # Update intersection beliefs similarly
        # ...

    def update_from_observation(self, observation):
        """
        Update beliefs based on new observations
        
        Args:
            observation: New observation data about road and traffic
        """
        # Update lane beliefs based on observations
        observed_lanes = observation.get('lanes', {})
        for lane_id, lane_data in observed_lanes.items():
            if lane_id in self.lane_beliefs:
                # Update existing lane beliefs
                for key, value in lane_data.items():
                    if key in self.lane_beliefs[lane_id]:
                        # Update with certainty if directly observed
                        self.lane_beliefs[lane_id][key] = value
            else:
                # Add new lane belief
                self.lane_beliefs[lane_id] = lane_data
        
        # Update vehicle beliefs
        observed_vehicles = observation.get('vehicles', {})
        for vehicle_id, vehicle_data in observed_vehicles.items():
            # Skip ego vehicle
            if vehicle_id == self.ego_vehicle_id:
                continue
                
            # Update or add vehicle belief
            self.vehicle_beliefs[vehicle_id] = {
                'position': vehicle_data.get('position'),
                'velocity': vehicle_data.get('velocity'),
                'acceleration': vehicle_data.get('acceleration'),
                'heading': vehicle_data.get('heading'),
                'lane_id': vehicle_data.get('lane_id'),
                'dimensions': vehicle_data.get('dimensions', (4.5, 2.0)),  # Default car dimensions
                'last_observed': observation.get('timestamp', 0)
            }
            
        # Update traffic signal beliefs
        observed_signals = observation.get('traffic_signals', {})
        for signal_id, signal_data in observed_signals.items():
            self.traffic_signal_beliefs[signal_id] = {
                'state': signal_data.get('state'),
                'position': signal_data.get('position'),
                'controlling_lanes': signal_data.get('controlling_lanes', []),
                'last_observed': observation.get('timestamp', 0)
            }

    def sample_vehicle_trajectories(self, prediction_horizon=5.0, dt=0.1):
        """
        Sample predicted trajectories for vehicles based on current beliefs
        
        Args:
            prediction_horizon: How far to predict in seconds
            dt: Time step for prediction in seconds
            
        Returns:
            Dictionary of predicted trajectories by vehicle ID
        """
        trajectories = {}
        num_steps = int(prediction_horizon / dt)
        
        for vehicle_id, vehicle_belief in self.vehicle_beliefs.items():
            # Simple constant velocity prediction as baseline
            position = np.array(vehicle_belief['position'])
            velocity = np.array(vehicle_belief['velocity'])
            heading = vehicle_belief['heading']
            
            trajectory = []
            current_pos = position.copy()
            
            for _ in range(num_steps):
                # Update position based on velocity
                current_pos = current_pos + velocity * dt
                trajectory.append(current_pos.copy())
            
            # For each position in the trajectory, add the corresponding heading 
            # Creating a combined array of [x, y, heading] for each timestep
            full_trajectory = []
            for pos in trajectory:
                full_trajectory.append(np.append(pos, heading))
            trajectories[vehicle_id] = np.array(full_trajectory)
            # print(f"vehicle_id: {vehicle_id}, trajectory: {trajectories[vehicle_id]}")
            # sys.exit(0)

        # print(f"length of trajectories: {len(trajectories)}")  
        # print(f"trajectories keys: {trajectories.keys()}")  
        return trajectories

    def get_collision_probabilities(self, ego_trajectory, other_trajectories=None):
        """
        Calculate collision probabilities with other vehicles
        
        Args:
            ego_trajectory: Predicted trajectory of ego vehicle
            other_trajectories: Trajectories of other vehicles (if None, will sample)
            
        Returns:
            Dictionary of collision probabilities by vehicle ID
        """
        if other_trajectories is None:
            other_trajectories = self.sample_vehicle_trajectories()
            
        collision_probs = {}
        
        # For each vehicle, calculate collision probability
        for vehicle_id, trajectory in other_trajectories.items():
            vehicle_dims = self.vehicle_beliefs[vehicle_id].get('dimensions', (4.5, 2.0))
            
            # Calculate TTC (Time To Collision) for each timestep
            min_distance = float('inf')
            min_time_idx = 0
            
            # Find the minimum distance between trajectories
            for t in range(min(len(ego_trajectory), len(trajectory))):
                dist = np.linalg.norm(ego_trajectory[t] - trajectory[t][:2])
                if dist < min_distance:
                    min_distance = dist
                    min_time_idx = t
            
            # Simplified collision probability based on minimum distance
            # More sophisticated models would use uncertainty in the trajectories
            threshold = 1.0  # Safety threshold in meters
            collision_prob = max(0, 1 - min_distance / (vehicle_dims[0] + threshold))
            
            collision_probs[vehicle_id] = {
                'probability': collision_prob,
                'time_index': min_time_idx
            }
            
        return collision_probs

    def get_risky_vehicles(self, threshold=0.3):
        """
        Identify vehicles with high collision risk
        
        Args:
            threshold: Risk threshold to consider a vehicle risky
            
        Returns:
            List of risky vehicle IDs
        """
        ego_trajectory = self._generate_ego_trajectory()
        collision_probs = self.get_collision_probabilities(ego_trajectory)
        
        risky_vehicles = []
        for vehicle_id, data in collision_probs.items():
            if data['probability'] > threshold:
                risky_vehicles.append(vehicle_id)
                
        return risky_vehicles

    def _generate_ego_trajectory(self):
        """Generate a simple trajectory for the ego vehicle"""
        # Placeholder - in a real implementation, this would use the planned path
        # from the motion planner or a simple prediction model
        return np.zeros((10, 2))  # 10 timesteps of 2D positions
