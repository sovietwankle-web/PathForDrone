"""
PyTorch Dataset with dynamic padding collate function.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any


class AllocationDataset(Dataset):
    """
    Dataset of (features, optimal_label) pairs.
    Returns raw (unpadded) features; use dynamic_collate_fn for batching.
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            'robot_feats': torch.from_numpy(s['robot_feats'].astype(np.float32)),
            'waypoint_feats': torch.from_numpy(s['waypoint_feats'].astype(np.float32)),
            'pairwise_feats': torch.from_numpy(s['pairwise_feats'].astype(np.float32)),
            'label': torch.from_numpy(s['label'].astype(np.float32)),
            'n_robots': s['n_robots'],
            'n_waypoints': s['n_waypoints'],
        }


def dynamic_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads each batch to the max actual size in that batch,
    not to a global max. Saves significant memory/compute.
    """
    max_r = max(s['n_robots'] for s in batch)
    max_w = max(s['n_waypoints'] for s in batch)
    B = len(batch)

    feat_r_dim = batch[0]['robot_feats'].shape[-1]
    feat_w_dim = batch[0]['waypoint_feats'].shape[-1]
    feat_p_dim = batch[0]['pairwise_feats'].shape[-1]

    robot_feats = torch.zeros(B, max_r, feat_r_dim)
    waypoint_feats = torch.zeros(B, max_w, feat_w_dim)
    pairwise_feats = torch.zeros(B, max_r, max_w, feat_p_dim)
    robot_mask = torch.zeros(B, max_r, dtype=torch.bool)
    waypoint_mask = torch.zeros(B, max_w, dtype=torch.bool)
    label = torch.zeros(B, max_r, max_w)

    for i, s in enumerate(batch):
        nr, nw = s['n_robots'], s['n_waypoints']
        robot_feats[i, :nr] = s['robot_feats'][:nr]
        waypoint_feats[i, :nw] = s['waypoint_feats'][:nw]
        pairwise_feats[i, :nr, :nw] = s['pairwise_feats'][:nr, :nw]
        robot_mask[i, :nr] = True
        waypoint_mask[i, :nw] = True
        label[i, :nr, :nw] = s['label'][:nr, :nw]

    return {
        'robot_feats': robot_feats,
        'waypoint_feats': waypoint_feats,
        'pairwise_feats': pairwise_feats,
        'robot_mask': robot_mask,
        'waypoint_mask': waypoint_mask,
        'label': label,
    }


def prepare_training_sample(robot_feats: np.ndarray,
                            waypoint_feats: np.ndarray,
                            pairwise_feats: np.ndarray,
                            label: np.ndarray,
                            n_robots: int,
                            n_waypoints: int) -> Dict[str, Any]:
    """Create a sample dict for the dataset."""
    return {
        'robot_feats': robot_feats,
        'waypoint_feats': waypoint_feats,
        'pairwise_feats': pairwise_feats,
        'label': label,
        'n_robots': n_robots,
        'n_waypoints': n_waypoints,
    }
