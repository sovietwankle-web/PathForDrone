#!/usr/bin/env python3
"""
Phase 1: Imitation Learning Training Script.

Generates training data with optimal solutions, then trains the
cross-attention network via supervised learning.

Usage:
    python train_imitation.py [--n_train 5000] [--n_val 500] [--epochs 100]
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_allocator.config import NeuralAllocatorConfig, TrainingConfig
from neural_allocator.model import WaypointAllocationNet
from neural_allocator.features import FeatureExtractor
from neural_allocator.training.data_generator import generate_scenario
from neural_allocator.training.optimal_solver import (
    solve_optimal, assignment_to_label_matrix
)
from neural_allocator.training.dataset import AllocationDataset, prepare_training_sample
from neural_allocator.training.imitation_trainer import ImitationTrainer


def build_sample(scenario, feature_extractor):
    """Convert a scenario + optimal solution into a training sample."""
    # Compute cost matrices (fast mode for training data generation)
    cost_matrix = feature_extractor.compute_cost_matrix(
        scenario.robot_starts, scenario.waypoints, fast=True
    )

    # Waypoint-to-waypoint costs (for TSP ordering)
    n_w = scenario.n_waypoints
    wp_to_wp_cost = np.zeros((n_w, n_w), dtype=np.float64)
    # Approximate with Euclidean distance scaled by mean inverse speed
    mean_slowness = 1.0 / np.mean(scenario.speed_field[scenario.speed_field > 0])
    for i in range(n_w):
        for j in range(n_w):
            if i != j:
                dist = sum((a - b) ** 2
                           for a, b in zip(scenario.waypoints[i], scenario.waypoints[j])) ** 0.5
                wp_to_wp_cost[i, j] = dist * mean_slowness

    # Solve optimal assignment
    assignments, makespan = solve_optimal(
        cost_matrix, wp_to_wp_cost,
        scenario.n_robots, scenario.n_waypoints
    )

    # Extract features
    robot_feats = feature_extractor.extract_robot_features(scenario.robot_starts)
    waypoint_feats = feature_extractor.extract_waypoint_features(scenario.waypoints)
    pairwise_feats = feature_extractor.extract_pairwise_features(
        scenario.robot_starts, scenario.waypoints, cost_matrix
    )

    # Normalize
    robot_feats, waypoint_feats, pairwise_feats = \
        feature_extractor.normalize_features(robot_feats, waypoint_feats, pairwise_feats)

    # Label
    label = assignment_to_label_matrix(assignments, scenario.n_robots, scenario.n_waypoints)

    return prepare_training_sample(
        robot_feats, waypoint_feats, pairwise_feats,
        label, scenario.n_robots, scenario.n_waypoints
    )


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Imitation Learning')
    parser.add_argument('--n_train', type=int, default=5000,
                        help='Number of training scenarios')
    parser.add_argument('--n_val', type=int, default=500,
                        help='Number of validation scenarios')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--max_robots', type=int, default=8,
                        help='Max robots in generated scenarios')
    parser.add_argument('--max_waypoints', type=int, default=20,
                        help='Max waypoints in generated scenarios')
    parser.add_argument('--terrain_size', type=int, default=48,
                        help='Max terrain grid size')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Config
    model_config = NeuralAllocatorConfig(
        max_robots=args.max_robots,
        max_waypoints=args.max_waypoints,
        lr=args.lr,
    )
    train_config = TrainingConfig(
        n_train_scenarios=args.n_train,
        n_val_scenarios=args.n_val,
        n_epochs_imitation=args.epochs,
        batch_size=args.batch_size,
        terrain_size_range=(32, args.terrain_size),
        n_robots_range=(2, min(args.max_robots, 6)),
        n_waypoints_range=(4, min(args.max_waypoints, 15)),
        checkpoint_dir=args.checkpoint_dir,
    )

    # Generate training data
    print(f"Generating {args.n_train} training scenarios...")
    t0 = time.time()
    train_samples = []
    for i in range(args.n_train):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.n_train}")
        try:
            scenario = generate_scenario(
                terrain_size_range=train_config.terrain_size_range,
                n_robots_range=train_config.n_robots_range,
                n_waypoints_range=train_config.n_waypoints_range,
            )
            fe = FeatureExtractor(scenario.speed_field, scenario.spacing)
            sample = build_sample(scenario, fe)
            train_samples.append(sample)
        except Exception as e:
            print(f"  Scenario {i} failed: {e}")
            continue

    print(f"Generated {len(train_samples)} training samples in {time.time()-t0:.1f}s")

    # Generate validation data
    print(f"Generating {args.n_val} validation scenarios...")
    val_samples = []
    for i in range(args.n_val):
        try:
            scenario = generate_scenario(
                terrain_size_range=train_config.terrain_size_range,
                n_robots_range=train_config.n_robots_range,
                n_waypoints_range=train_config.n_waypoints_range,
            )
            fe = FeatureExtractor(scenario.speed_field, scenario.spacing)
            sample = build_sample(scenario, fe)
            val_samples.append(sample)
        except Exception:
            continue

    print(f"Generated {len(val_samples)} validation samples")

    # Create datasets
    train_dataset = AllocationDataset(
        train_samples,
        max_robots=args.max_robots,
        max_waypoints=args.max_waypoints,
    )
    val_dataset = AllocationDataset(
        val_samples,
        max_robots=args.max_robots,
        max_waypoints=args.max_waypoints,
    )

    # Create model and trainer
    model = WaypointAllocationNet(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    trainer = ImitationTrainer(model, train_config, device=device)
    history = trainer.train(train_dataset, val_dataset,
                            checkpoint_dir=args.checkpoint_dir)

    print("\nTraining complete!")
    print(f"Final train accuracy: {history['accuracy'][-1]:.3f}")
    if history['val_accuracy']:
        print(f"Best val accuracy: {max(history['val_accuracy']):.3f}")


if __name__ == '__main__':
    main()
