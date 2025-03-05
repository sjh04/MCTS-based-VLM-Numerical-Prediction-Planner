from dataclasses import dataclass
from typing import List

@dataclass
class CarlaModelParams:
    """CARLA model parameters"""
    local_embedding_size: int = 256
    global_embedding_size: int = 256
    num_subgraph_layers: int = 3
    global_head_dropout: float = 0.1

@dataclass
class CarlaFeatureParams:
    """CARLA feature parameters"""
    agent_features: List[str] = None
    map_features: List[str] = None
    max_elements: int = 100
    max_points: int = 50
    vector_set_map_feature_radius: float = 50.0
    interpolation_method: str = "linear"
    past_trajectory_sampling: int = 10
    feature_dimension: int = 8
    ego_dimension: int = 8
    agent_dimension: int = 8
    total_max_points: int = 50
    max_agents: int = 10
    feature_types: List[str] = None

@dataclass
class CarlaTargetParams:
    """CARLA target parameters"""
    future_trajectory_sampling: int = 20
    num_output_features: int = 4 