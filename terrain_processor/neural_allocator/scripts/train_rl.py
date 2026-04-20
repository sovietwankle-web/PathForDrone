#!/usr/bin/env python3
"""
Phase 2: PPO Reinforcement Learning Fine-tuning Script.

Loads a pre-trained model from Phase 1 and fine-tunes with PPO
to minimize makespan.

Usage:
    python train_rl.py [--pretrained checkpoints/best_imitation.pt] [--epochs 500]
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_allocator.config import NeuralAllocatorConfig, TrainingConfig
from neural_allocator.model import WaypointAllocationNet
from neural_allocator.features import FeatureExtractor
from neural_allocator.training.data_generator import generate_scenario
from neural_allocator.training.rl_trainer import PPOTrainer


def build_rl_scenario(scenario, feature_extractor):
    """Build an RL scenario dict with cost matrices."""
    cost_matrix = feature_extractor.compute_cost_matrix(
        scenario.robot_starts, scenario.waypoints, fast=True
    )

    n_w = scenario.n_waypoints
    wp_to_wp_cost = np.zeros((n_w, n_w), dtype=np.float64)
    mean_slowness = 1.0 / np.mean(scenario.speed_field[scenario.speed_field > 0])
    for i in range(n_w):
        for j in range(n_w):
            if i != j:
                dist = sum((a - b) ** 2
                           for a, b in zip(scenario.waypoints[i], scenario.waypoints[j])) ** 0.5
                wp_to_wp_cost[i, j] = dist * mean_slowness

    robot_feats = feature_extractor.extract_robot_features(scenario.robot_starts)
    waypoint_feats = feature_extractor.extract_waypoint_features(scenario.waypoints)
    pairwise_feats = feature_extractor.extract_pairwise_features(
        scenario.robot_starts, scenario.waypoints, cost_matrix
    )
    robot_feats, waypoint_feats, pairwise_feats = \
        feature_extractor.normalize_features(robot_feats, waypoint_feats, pairwise_feats)

    return {
        'robot_feats': robot_feats,
        'waypoint_feats': waypoint_feats,
        'pairwise_feats': pairwise_feats,
        'cost_matrix': cost_matrix,
        'wp_to_wp_cost': wp_to_wp_cost,
        'n_robots': scenario.n_robots,
        'n_waypoints': scenario.n_waypoints,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2: PPO RL Fine-tuning')
    parser.add_argument('--pretrained', type=str, default='checkpoints/best_imitation.pt',
                        help='Path to Phase 1 pretrained model')
    parser.add_argument('--n_scenarios', type=int, default=1000,
                        help='Number of RL training scenarios')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--max_robots', type=int, default=8)
    parser.add_argument('--max_waypoints', type=int, default=20)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model_config = NeuralAllocatorConfig(
        max_robots=args.max_robots,
        max_waypoints=args.max_waypoints,
    )
    train_config = TrainingConfig(
        n_epochs_rl=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Load pre-trained model
    model = WaypointAllocationNet(model_config)
    if os.path.exists(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained model from {args.pretrained}")
    else:
        print(f"WARNING: Pretrained model not found at {args.pretrained}, training from scratch")

    # Generate RL scenarios
    print(f"Generating {args.n_scenarios} RL scenarios...")
    t0 = time.time()
    scenarios = []
    for i in range(args.n_scenarios):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.n_scenarios}")
        try:
            scenario = generate_scenario(
                terrain_size_range=(32, 48),
                n_robots_range=(2, min(args.max_robots, 6)),
                n_waypoints_range=(4, min(args.max_waypoints, 15)),
            )
            fe = FeatureExtractor(scenario.speed_field, scenario.spacing)
            rl_scenario = build_rl_scenario(scenario, fe)
            scenarios.append(rl_scenario)
        except Exception as e:
            print(f"  Scenario {i} failed: {e}")
    print(f"Generated {len(scenarios)} scenarios in {time.time()-t0:.1f}s")

    # Train
    trainer = PPOTrainer(model, train_config, model_config, device=device)
    history = trainer.train(scenarios, checkpoint_dir=args.checkpoint_dir,
                            n_epochs=args.epochs)

    print("\nRL fine-tuning complete!")
    print(f"Best reward: {max(history['reward']):.4f}")


if __name__ == '__main__':
    main()
