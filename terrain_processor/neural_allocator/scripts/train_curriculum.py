#!/usr/bin/env python3
"""
Curriculum Training Script — trains through progressively harder stages.

Usage:
    python train_curriculum.py [--config large] [--checkpoint_dir checkpoints/]
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_allocator.config import (
    NeuralAllocatorConfig, TrainingConfig, DEFAULT_CURRICULUM, CurriculumStage
)
from neural_allocator.model import WaypointAllocationNet
from neural_allocator.features import FeatureExtractor
from neural_allocator.training.data_generator import generate_scenario
from neural_allocator.training.augmentation import augment_scenario
from neural_allocator.training.optimal_solver import solve_optimal, assignment_to_label_matrix
from neural_allocator.training.dataset import AllocationDataset, prepare_training_sample
from neural_allocator.training.imitation_trainer import ImitationTrainer


def build_sample(scenario, n_augments=0):
    """Convert scenario + optimal solution into training samples."""
    fe = FeatureExtractor(scenario.speed_field, scenario.spacing)
    cost_matrix = fe.compute_cost_matrix(scenario.robot_starts, scenario.waypoints, fast=True)

    n_w = scenario.n_waypoints
    wp_to_wp_cost = np.zeros((n_w, n_w), dtype=np.float64)
    for i in range(n_w):
        for j in range(n_w):
            if i != j:
                wp_to_wp_cost[i, j] = fe._line_integral_cost(
                    tuple(scenario.waypoints[i]), tuple(scenario.waypoints[j]))

    assignments, _ = solve_optimal(cost_matrix, wp_to_wp_cost, scenario.n_robots, n_w)

    robot_feats = fe.extract_robot_features(scenario.robot_starts)
    waypoint_feats = fe.extract_waypoint_features(scenario.waypoints)
    pairwise_feats = fe.extract_pairwise_features(
        scenario.robot_starts, scenario.waypoints, cost_matrix)
    _, _, pairwise_feats = fe.normalize_features(robot_feats, waypoint_feats, pairwise_feats)
    label = assignment_to_label_matrix(assignments, scenario.n_robots, n_w)

    samples = [prepare_training_sample(
        robot_feats, waypoint_feats, pairwise_feats, label,
        scenario.n_robots, n_w)]

    # Augmentations
    if n_augments > 0:
        for aug_scenario in augment_scenario(scenario, n_augments):
            try:
                fe_aug = FeatureExtractor(aug_scenario.speed_field, aug_scenario.spacing)
                cm = fe_aug.compute_cost_matrix(
                    aug_scenario.robot_starts, aug_scenario.waypoints, fast=True)
                n_w_a = aug_scenario.n_waypoints
                wpc = np.zeros((n_w_a, n_w_a), dtype=np.float64)
                for i in range(n_w_a):
                    for j in range(n_w_a):
                        if i != j:
                            wpc[i, j] = fe_aug._line_integral_cost(
                                tuple(aug_scenario.waypoints[i]),
                                tuple(aug_scenario.waypoints[j]))
                asgn, _ = solve_optimal(cm, wpc, aug_scenario.n_robots, n_w_a)
                rf = fe_aug.extract_robot_features(aug_scenario.robot_starts)
                wf = fe_aug.extract_waypoint_features(aug_scenario.waypoints)
                pf = fe_aug.extract_pairwise_features(
                    aug_scenario.robot_starts, aug_scenario.waypoints, cm)
                _, _, pf = fe_aug.normalize_features(rf, wf, pf)
                lb = assignment_to_label_matrix(asgn, aug_scenario.n_robots, n_w_a)
                samples.append(prepare_training_sample(rf, wf, pf, lb,
                                                        aug_scenario.n_robots, n_w_a))
            except Exception:
                pass

    return samples


def generate_stage_data(stage: CurriculumStage, n_augments: int = 2):
    """Generate training data for a curriculum stage."""
    samples = []
    n_target = stage.n_scenarios

    for i in range(n_target):
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_target}")
        try:
            scenario = generate_scenario(
                terrain_size_range=stage.terrain_size_range,
                n_robots_range=stage.n_robots_range,
                n_waypoints_range=stage.n_waypoints_range,
                ndim=2,
            )
            new_samples = build_sample(scenario, n_augments=n_augments)
            samples.extend(new_samples)
        except Exception:
            pass

    return samples


def main():
    parser = argparse.ArgumentParser(description='Curriculum Training')
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'large'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.config == 'large':
        model_config = NeuralAllocatorConfig.large()
    else:
        model_config = NeuralAllocatorConfig()

    model = WaypointAllocationNet(model_config)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: d_model={model_config.d_model}, layers={model_config.n_cross_attn_layers}, "
          f"params={n_params:,}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for stage_idx, stage in enumerate(DEFAULT_CURRICULUM):
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx+1}/{len(DEFAULT_CURRICULUM)}")
        print(f"  terrain={stage.terrain_size_range}, robots={stage.n_robots_range}, "
              f"waypoints={stage.n_waypoints_range}")
        print(f"  epochs={stage.n_epochs}, scenarios={stage.n_scenarios}")
        print(f"{'='*60}")

        # Generate data for this stage
        print(f"  Generating training data...")
        t0 = time.time()
        train_samples = generate_stage_data(stage, n_augments=2)

        # 20% for validation
        np.random.shuffle(train_samples)
        n_val = max(len(train_samples) // 5, 10)
        val_samples = train_samples[:n_val]
        train_samples = train_samples[n_val:]
        print(f"  Data: {len(train_samples)} train + {len(val_samples)} val "
              f"in {time.time()-t0:.0f}s")

        train_ds = AllocationDataset(train_samples)
        val_ds = AllocationDataset(val_samples)

        # Train this stage
        train_config = TrainingConfig(
            n_epochs_imitation=stage.n_epochs,
            batch_size=min(64, len(train_samples)),
            log_interval=5,
            val_interval=5,
            early_stopping_patience=10,
        )

        trainer = ImitationTrainer(model, train_config, device=device)
        history = trainer.train(
            train_ds, val_ds,
            checkpoint_dir=args.checkpoint_dir,
            n_epochs=stage.n_epochs,
        )

        # Save stage checkpoint
        stage_path = os.path.join(args.checkpoint_dir, f'stage{stage_idx+1}.pt')
        torch.save(model.state_dict(), stage_path)
        print(f"  Saved: {stage_path}")

        if history['val_accuracy']:
            print(f"  Best val accuracy: {max(history['val_accuracy']):.3f}")

    print(f"\nCurriculum training complete!")
    final_path = os.path.join(args.checkpoint_dir, 'curriculum_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Final model: {final_path}")


if __name__ == '__main__':
    main()
