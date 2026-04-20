"""
Neural Allocator for Multi-Robot Waypoint Assignment

Hybrid Attention + RL approach inspired by SADCHER:
- Cross-Attention network predicts assignment scores
- Phase 1: Imitation learning from optimal solutions
- Phase 2: PPO fine-tuning to minimize makespan
- Supports large-scale (50km+) terrain via hierarchical features
"""

from .config import NeuralAllocatorConfig, TrainingConfig, CurriculumStage, DEFAULT_CURRICULUM
from .allocator import NeuralWaypointAllocator
from .hierarchical_features import HierarchicalFeatureExtractor
from .geotiff_loader import GeoTIFFAllocator

__all__ = [
    'NeuralAllocatorConfig',
    'TrainingConfig',
    'NeuralWaypointAllocator',
    'HierarchicalFeatureExtractor',
    'GeoTIFFAllocator',
]
