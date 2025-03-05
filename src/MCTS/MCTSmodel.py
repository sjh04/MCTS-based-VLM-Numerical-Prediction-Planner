import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, cast
import numpy as np
import carla
from .carla_params import CarlaModelParams, CarlaFeatureParams, CarlaTargetParams

class MCTSModel():
    """
    Vector-based model that uses PointNet-based subgraph layers for collating loose collections of vectorized inputs
    into local feature descriptors to be used as input to a global Transformer.
    Adapted for CARLA simulator with the following changes:
    1. Use CARLA features from vehicle sensors and world state
    2. Format model for using pytorch_lightning
    3. Integrate with CARLA's camera system and vehicle dynamics
    """

    def __init__(
        self,
        model_params: CarlaModelParams,
        feature_params: CarlaFeatureParams,
        target_params: CarlaTargetParams,
    ):
        """
        Initialize CARLA MCTS model.
        Args:
            model_params: internal model parameters
            feature_params: agent and map feature parameters
            target_params: target parameters
        """
        # Define feature types for CARLA environment
        agent_features = feature_params.agent_features
        agent_features2 = [*agent_features, "TRAFFIC_CONE", "GENERIC_OBJECT", "PEDESTRIAN"]
        
        super().__init__(
            feature_builders=[
                VectorSetMapFeatureBuilder(
                    map_features=feature_params.map_features,
                    max_elements=feature_params.max_elements,
                    max_points=feature_params.max_points,
                    radius=feature_params.vector_set_map_feature_radius,
                    interpolation_method=feature_params.interpolation_method,
                ),
                GenericAgentsFeatureBuilder(agent_features2, feature_params.past_trajectory_sampling),
            ],
            target_builders=[
                EgoTrajectoryTargetBuilder(target_params.future_trajectory_sampling),
                MultipleAgentsTrajectoryTargetBuilder(
                    future_trajectory_sampling=target_params.future_trajectory_sampling,
                ),
            ],
            future_trajectory_sampling=target_params.future_trajectory_sampling,
        )
        
        self._model_params = model_params
        self._feature_params = feature_params
        self._target_params = target_params
        self.dt = 0.1  # CARLA default time step

        # Feature embedding layers
        self.feature_embedding = nn.Linear(
            self._feature_params.feature_dimension,
            self._model_params.local_embedding_size,
        )
        self.positional_embedding = SinusoidalPositionalEmbedding(self._model_params.local_embedding_size)
        self.type_embedding = TypeEmbedding(
            self._model_params.global_embedding_size,
            self._feature_params.feature_types,
        )
        
        # Local subgraph processing
        self.local_subgraph = LocalSubGraph(
            num_layers=self._model_params.num_subgraph_layers,
            dim_in=self._model_params.local_embedding_size,
        )
        
        # Global feature processing
        if self._model_params.global_embedding_size != self._model_params.local_embedding_size:
            self.global_from_local = nn.Linear(
                self._model_params.local_embedding_size,
                self._model_params.global_embedding_size,
            )
            
        # Multi-head attention layers for CARLA environment
        num_timesteps = self.future_trajectory_sampling.num_poses
        self.global_map = MultiheadAttentionGlobalHeadMulti(
            self._model_params.global_embedding_size,
            num_timesteps,
            self._target_params.num_output_features // num_timesteps,
            dropout=self._model_params.global_head_dropout,
        )
        self.global_head1 = MultiheadAttentionGlobalHeadMulti(
            self._model_params.global_embedding_size,
            num_timesteps,
            self._target_params.num_output_features // num_timesteps,
            dropout=self._model_params.global_head_dropout,
        )
        self.global_head2 = MultiheadAttentionGlobalHeadMulti(
            self._model_params.global_embedding_size,
            num_timesteps,
            self._target_params.num_output_features // num_timesteps,
            dropout=self._model_params.global_head_dropout,
        )
        
        # Final output layer
        self.final_output = nn.Linear(self._model_params.global_embedding_size, self._target_params.num_output_features)
        
        # Initialize counter
        self.count = 0

    def extract_agent_features(
        self,
        ego_agent_features: GenericAgents,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ego and agent features from CARLA environment
        Args:
            ego_agent_features: agent features to be extracted (ego + other agents)
            batch_size: number of samples in batch to extract
        Returns:
            agent_features: Stacked ego, agent, and map features
            agent_avails: Boolean mask for feature availability
        """
        agent_features = []  # List[<torch.FloatTensor: max_agents+1, total_max_points, feature_dimension>: batch_size]
        agent_avails = []  # List[<torch.BoolTensor: max_agents+1, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            # Ego features
            # maintain fixed feature size through trimming/padding
            sample_ego_feature = ego_agent_features.ego[sample_idx][
                ...,
                : min(self._feature_params.ego_dimension, self._feature_params.feature_dimension),
            ].unsqueeze(0)
            if (
                min(self._feature_params.ego_dimension, GenericAgents.ego_state_dim())
                < self._feature_params.feature_dimension
            ):
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.feature_dimension, dim=2)

            sample_ego_avails = torch.ones(
                sample_ego_feature.shape[0],
                sample_ego_feature.shape[1],
                dtype=torch.bool,
                device=sample_ego_feature.device,
            )

            # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
            sample_ego_feature = torch.flip(sample_ego_feature, dims=[1])

            # maintain fixed number of points per polyline
            sample_ego_feature = sample_ego_feature[:, : self._feature_params.total_max_points, ...]
            sample_ego_avails = sample_ego_avails[:, : self._feature_params.total_max_points, ...]
            if sample_ego_feature.shape[1] < self._feature_params.total_max_points:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.total_max_points, dim=1)
                sample_ego_avails = pad_avails(sample_ego_avails, self._feature_params.total_max_points, dim=1)

            sample_features = [sample_ego_feature]
            sample_avails = [sample_ego_avails]

            # Agent features
            for feature_name in self._feature_params.agent_features:
                # if there exist at least one valid agent in the sample
                if ego_agent_features.has_agents(feature_name, sample_idx):
                    # num_frames x num_agents x num_features -> num_agents x num_frames x num_features
                    sample_agent_features = torch.permute(
                        ego_agent_features.agents[feature_name][sample_idx],
                        (1, 0, 2),
                    )
                    # maintain fixed feature size through trimming/padding
                    sample_agent_features = sample_agent_features[
                        ...,
                        : min(self._feature_params.agent_dimension, self._feature_params.feature_dimension),
                    ]
                    if (
                        min(self._feature_params.agent_dimension, GenericAgents.agents_states_dim())
                        < self._feature_params.feature_dimension
                    ):
                        sample_agent_features = pad_polylines(
                            sample_agent_features,
                            self._feature_params.feature_dimension,
                            dim=2,
                        )

                    sample_agent_avails = torch.ones(
                        sample_agent_features.shape[0],
                        sample_agent_features.shape[1],
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                    # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
                    sample_agent_features = torch.flip(sample_agent_features, dims=[1])

                    # maintain fixed number of points per polyline
                    sample_agent_features = sample_agent_features[:, : self._feature_params.total_max_points, ...]
                    sample_agent_avails = sample_agent_avails[:, : self._feature_params.total_max_points, ...]
                    if sample_agent_features.shape[1] < self._feature_params.total_max_points:
                        sample_agent_features = pad_polylines(
                            sample_agent_features,
                            self._feature_params.total_max_points,
                            dim=1,
                        )
                        sample_agent_avails = pad_avails(
                            sample_agent_avails,
                            self._feature_params.total_max_points,
                            dim=1,
                        )

                    # maintained fixed number of agent polylines of each type per sample
                    sample_agent_features = sample_agent_features[: self._feature_params.max_agents, ...]
                    sample_agent_avails = sample_agent_avails[: self._feature_params.max_agents, ...]
                    if sample_agent_features.shape[0] < (self._feature_params.max_agents):
                        sample_agent_features = pad_polylines(
                            sample_agent_features,
                            self._feature_params.max_agents,
                            dim=0,
                        )
                        sample_agent_avails = pad_avails(sample_agent_avails, self._feature_params.max_agents, dim=0)

                else:
                    sample_agent_features = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        self._feature_params.feature_dimension,
                        dtype=torch.float32,
                        device=sample_ego_feature.device,
                    )
                    sample_agent_avails = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                # add features, avails to sample
                sample_features.append(sample_agent_features)
                sample_avails.append(sample_agent_avails)

            sample_features = torch.cat(sample_features, dim=0)
            sample_avails = torch.cat(sample_avails, dim=0)

            agent_features.append(sample_features)
            agent_avails.append(sample_avails)
        agent_features = torch.stack(agent_features)
        agent_avails = torch.stack(agent_avails)

        return agent_features, agent_avails

    def extract_map_features(
        self,
        vector_set_map_data: VectorSetMap,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract map features from CARLA environment
        Args:
            vector_set_map_data: VectorSetMap features to be extracted
            batch_size: number of samples in batch to extract
        Returns:
            map_features: Stacked map features
            map_avails: Boolean mask for feature availability
        """
        map_features = []  # List[<torch.FloatTensor: max_map_features, total_max_points, feature_dim>: batch_size]
        map_avails = []  # List[<torch.BoolTensor: max_map_features, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            sample_map_features = []
            sample_map_avails = []

            for feature_name in self._feature_params.map_features:
                coords = vector_set_map_data.coords[feature_name][sample_idx]
                tl_data = (
                    vector_set_map_data.traffic_light_data[feature_name][sample_idx]
                    if feature_name in vector_set_map_data.traffic_light_data
                    else None
                )
                avails = vector_set_map_data.availabilities[feature_name][sample_idx]

                # add traffic light data if exists for feature
                if tl_data is not None:
                    coords = torch.cat((coords, tl_data), dim=2)

                # maintain fixed number of points per map element (polyline)
                coords = coords[:, : self._feature_params.total_max_points, ...]
                avails = avails[:, : self._feature_params.total_max_points]

                if coords.shape[1] < self._feature_params.total_max_points:
                    coords = pad_polylines(coords, self._feature_params.total_max_points, dim=1)
                    avails = pad_avails(avails, self._feature_params.total_max_points, dim=1)

                # maintain fixed number of features per point
                coords = coords[..., : self._feature_params.feature_dimension]
                if coords.shape[2] < self._feature_params.feature_dimension:
                    coords = pad_polylines(coords, self._feature_params.feature_dimension, dim=2)

                sample_map_features.append(coords)
                sample_map_avails.append(avails)

            map_features.append(torch.cat(sample_map_features))
            map_avails.append(torch.cat(sample_map_avails))

        map_features = torch.stack(map_features)
        map_avails = torch.stack(map_avails)

        return map_features, map_avails

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Forward pass for CARLA environment
        Args:
            features: input features containing
                     {
                         "vector_set_map": VectorSetMap,
                         "generic_agents": GenericAgents,
                         "camera_data": CameraData,  # Added for CARLA
                         "vehicle_state": VehicleState,  # Added for CARLA
                     }
        Returns:
            targets: predictions from network
                    {
                        "trajectory": Trajectory,
                        "control_commands": ControlCommands,  # Added for CARLA
                    }
        """
        # Recover features
        features_init = features
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])
        
        # Process CARLA-specific features
        camera_data = features.get("camera_data")
        vehicle_state = features.get("vehicle_state")
        
        batch_size = ego_agent_features.batch_size

        # Extract features across batch
        agent_features, agent_avails = self.extract_agent_features(ego_agent_features, batch_size)
        map_features, map_avails = self.extract_map_features(vector_set_map_data, batch_size)
        
        # Combine features
        features = torch.cat([agent_features, map_features], dim=1)
        avails = torch.cat([agent_avails, map_avails], dim=1)

        # Process features through network
        feature_embedding = self.feature_embedding(features)
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2)
        
        # Handle invalid features
        invalid_mask = ~avails
        invalid_polys = invalid_mask.all(-1)
        
        # Process through subgraph and attention layers
        embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding)
        if hasattr(self, "global_from_local"):
            embeddings = self.global_from_local(embeddings)
        embeddings = F.normalize(embeddings, dim=-1) * (self._model_params.global_embedding_size**0.5)
        embeddings = embeddings.transpose(0, 1)
        
        # Add type embeddings
        type_embedding = self.type_embedding(
            batch_size,
            self._feature_params.max_agents,
            self._feature_params.agent_features,
            self._feature_params.map_features,
            self._feature_params.max_elements,
            device=features.device,
        ).transpose(0, 1)
        
        # Process through attention layers
        n_agents = agent_features.size()[1]
        agent_embeddings = embeddings[:n_agents]
        map_embeddings = embeddings[n_agents:]
        agent_types = type_embedding[:n_agents]
        map_types = type_embedding[n_agents:]
        agent_polys = invalid_polys[:, :n_agents]
        map_polys = invalid_polys[:, n_agents:]
        
        # Global attention processing
        map_embeddings, _ = self.global_map(map_embeddings, map_embeddings, map_types, map_polys, n_agents)
        map_embeddings = map_embeddings.transpose(0, 1)
        
        agent_embeddings, _ = self.global_head1(agent_embeddings, map_embeddings, map_types, map_polys, n_agents)
        agent_embeddings = agent_embeddings.transpose(0, 1)
        agent_embeddings, _ = self.global_head2(
            agent_embeddings,
            agent_embeddings,
            agent_types,
            agent_polys,
            n_agents,
        )
        
        # Generate outputs
        outputs = self.final_output(agent_embeddings)
        outputs = outputs.view(
            batch_size,
            n_agents,
            self.future_trajectory_sampling.num_poses,
            self._target_params.num_output_features // self.future_trajectory_sampling.num_poses,
        )
        
        # Process ego predictions
        ego_pred = outputs[:, 0]
        outputs = torch.cat((xyh, outputs), dim=-2)
        
        # Interpolate outputs
        outputs = (
            torch.nn.functional.interpolate(
                outputs[0].permute(0, 2, 1),
                size=16 * 5 + 1,
                mode="linear",
                align_corners=True,
            )
            .permute(0, 2, 1)
            .unsqueeze(0)
        )
        outputs = outputs[..., 1:, :]
        
        # Generate control commands for CARLA
        control_commands = self.generate_control_commands(outputs, vehicle_state)
        
        return {
            "trajectory": Trajectory(data=convert_predictions_to_trajectory(outputs)),
            "agents_trajectories": MultipleAgentsTrajectories(outputs),
            "control_commands": control_commands,  # Added for CARLA
        }
        
    def generate_control_commands(self, outputs: torch.Tensor, vehicle_state: Dict) -> Dict:
        """
        Generate control commands for CARLA vehicle
        Args:
            outputs: model outputs
            vehicle_state: current vehicle state
        Returns:
            control_commands: dictionary containing throttle, brake, and steer values
        """
        # Extract trajectory predictions
        trajectory = outputs[0, 0]  # ego vehicle trajectory
        
        # Calculate desired speed and heading
        desired_speed = torch.norm(trajectory[1:] - trajectory[:-1], dim=-1) / self.dt
        desired_heading = torch.atan2(
            trajectory[..., 1] - trajectory[:-1, ..., 1],
            trajectory[..., 0] - trajectory[:-1, ..., 0]
        )
        
        # Calculate control commands
        current_speed = vehicle_state["speed"]
        current_heading = vehicle_state["heading"]
        
        # Throttle and brake
        speed_diff = desired_speed - current_speed
        throttle = torch.clamp(speed_diff, 0, 1)
        brake = torch.clamp(-speed_diff, 0, 1)
        
        # Steering
        heading_diff = desired_heading - current_heading
        steer = torch.clamp(heading_diff / np.pi, -1, 1)
        
        return {
            "throttle": throttle,
            "brake": brake,
            "steer": steer,
        }
