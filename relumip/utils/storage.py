from dataclasses import dataclass, field
import numpy as np


@dataclass
class AnnParameters:
    """Dataclass for storage of network parameters."""
    n_layers: int
    nodes_per_layer: np.ndarray
    input_dim: int
    output_dim: int
    input_bounds = None
    output_bounds = None
    weights: list = field(default_factory=list)
    bias: list = field(default_factory=list)
    M_plus: list = field(default_factory=list)
    M_minus: list = field(default_factory=list)
    UB: list = field(default_factory=list)
    LB: list = field(default_factory=list)
    redundancy_matrix: list = field(default_factory=list)
