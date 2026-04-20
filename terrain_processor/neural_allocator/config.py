"""
Configuration dataclasses for the Neural Allocator.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class NeuralAllocatorConfig:
    """Network architecture and inference config."""

    # Embedding dimensions
    d_model: int = 128
    n_heads: int = 8
    n_cross_attn_layers: int = 3
    dropout: float = 0.15
    ff_dim: int = 512

    # Feature dimensions
    robot_feat_dim: int = 7      # (x, y, z, vx, vy, vz, current_load)
    waypoint_feat_dim: int = 5   # (x, y, z, terrain_cost, local_q)
    pairwise_feat_dim: int = 2   # (fmm_cost, euclidean_dist)

    # Decoding
    decode_method: str = "hungarian"
    max_robots: int = 20
    max_waypoints: int = 200

    # RL hyperparameters
    gamma: float = 0.99
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    lr: float = 3e-4
    lr_rl: float = 1e-4

    # Regularization
    weight_decay: float = 1e-3
    label_smoothing: float = 0.05

    @classmethod
    def large(cls):
        """Config for 10-20 robots, 100+ waypoints, 50km terrain."""
        return cls(
            d_model=256,
            n_heads=8,
            n_cross_attn_layers=4,
            ff_dim=1024,
            dropout=0.15,
            max_robots=20,
            max_waypoints=200,
        )


@dataclass
class CurriculumStage:
    """One stage in curriculum learning."""
    terrain_size_range: Tuple[int, int]
    n_robots_range: Tuple[int, int]
    n_waypoints_range: Tuple[int, int]
    n_epochs: int
    n_scenarios: int


# Default curriculum: easy → hard
DEFAULT_CURRICULUM: List[CurriculumStage] = [
    CurriculumStage((32, 64),    (2, 4),   (4, 10),   30,  3000),
    CurriculumStage((48, 128),   (3, 8),   (8, 25),   30,  3000),
    CurriculumStage((96, 256),   (4, 12),  (15, 50),  30,  2000),
    CurriculumStage((128, 512),  (6, 20),  (30, 100), 30,  1000),
]


@dataclass
class TrainingConfig:
    """Training pipeline config."""

    # Data generation
    n_train_scenarios: int = 50000
    n_val_scenarios: int = 5000
    terrain_size_range: Tuple[int, int] = (32, 128)
    n_robots_range: Tuple[int, int] = (2, 8)
    n_waypoints_range: Tuple[int, int] = (4, 30)
    terrain_ndim: int = 3

    # Training
    batch_size: int = 64
    n_epochs_imitation: int = 100
    n_epochs_rl: int = 500
    checkpoint_dir: str = "checkpoints/"
    log_interval: int = 10
    val_interval: int = 5

    # Regularization
    early_stopping_patience: int = 15
    n_augments: int = 4

    # RL rollout
    n_rollout_steps: int = 2048
    n_minibatches: int = 16
    gae_lambda: float = 0.95
    rl_batch_size: int = 128
