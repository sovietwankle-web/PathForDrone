#!/usr/bin/env python3
"""
Evaluation script: compare Neural Allocator vs CompetitiveFMM.

Generates test scenarios and compares makespan, load balance,
and inference time between the two methods.

Usage:
    python evaluate.py [--model checkpoints/best_rl.pt] [--n_test 100]
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from viscous_hierarchical_fmm import (
    ViscosityField, ViscosityParams, GasDiffusionParams,
    MultiRobotWaypointAllocator, CompetitiveFMM,
)
from neural_allocator.config import NeuralAllocatorConfig
from neural_allocator.allocator import NeuralWaypointAllocator
from neural_allocator.training.data_generator import generate_scenario


def compute_makespan_from_result(result):
    """Compute makespan from allocation result."""
    max_time = 0.0
    for r, times in result.arrival_times.items():
        if times:
            max_time = max(max_time, max(times))
    return max_time


def compute_load_balance(result, n_robots):
    """Compute load balance: std of waypoint counts per robot."""
    counts = [len(result.assignments.get(r, [])) for r in range(n_robots)]
    return float(np.std(counts))


def main():
    parser = argparse.ArgumentParser(description='Evaluate Neural vs FMM Allocator')
    parser.add_argument('--model', type=str, default='checkpoints/best_rl.pt')
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--max_robots', type=int, default=8)
    parser.add_argument('--max_waypoints', type=int, default=20)
    args = parser.parse_args()

    print(f"Evaluating {args.n_test} test scenarios...")
    print(f"Neural model: {args.model}")
    print(f"Model exists: {os.path.exists(args.model)}")
    print()

    neural_config = NeuralAllocatorConfig(
        max_robots=args.max_robots,
        max_waypoints=args.max_waypoints,
    )

    fmm_makespans = []
    neural_makespans = []
    fmm_balances = []
    neural_balances = []
    fmm_times = []
    neural_times = []

    for i in range(args.n_test):
        if (i + 1) % 10 == 0:
            print(f"  Test {i+1}/{args.n_test}")

        try:
            scenario = generate_scenario(
                terrain_size_range=(32, 48),
                n_robots_range=(2, min(args.max_robots, 6)),
                n_waypoints_range=(4, min(args.max_waypoints, 15)),
            )
        except Exception:
            continue

        gas_params = GasDiffusionParams(
            n_robots=scenario.n_robots,
            robot_starts=scenario.robot_starts,
            waypoints=scenario.waypoints,
        )

        # FMM allocation
        try:
            t0 = time.time()
            fmm_alloc = MultiRobotWaypointAllocator(
                terrain=scenario.terrain, fog_data=scenario.fog_data,
                gas_params=gas_params, spacing=scenario.spacing,
            )
            fmm_result = fmm_alloc.allocate()
            fmm_time = time.time() - t0

            fmm_makespans.append(compute_makespan_from_result(fmm_result))
            fmm_balances.append(compute_load_balance(fmm_result, scenario.n_robots))
            fmm_times.append(fmm_time)
        except Exception as e:
            print(f"  FMM failed on scenario {i}: {e}")
            continue

        # Neural allocation
        try:
            t0 = time.time()
            neural_alloc = NeuralWaypointAllocator(
                terrain=scenario.terrain, fog_data=scenario.fog_data,
                gas_params=gas_params, spacing=scenario.spacing,
                model_path=args.model, config=neural_config,
                fallback_to_fmm=False,
            )
            neural_result = neural_alloc.allocate()
            neural_time = time.time() - t0

            neural_makespans.append(compute_makespan_from_result(neural_result))
            neural_balances.append(compute_load_balance(neural_result, scenario.n_robots))
            neural_times.append(neural_time)
        except Exception as e:
            print(f"  Neural failed on scenario {i}: {e}")
            # Still count FMM result but skip neural for this scenario
            if fmm_makespans and len(fmm_makespans) > len(neural_makespans):
                fmm_makespans.pop()
                fmm_balances.pop()
                fmm_times.pop()

    # Report
    n_valid = min(len(fmm_makespans), len(neural_makespans))
    if n_valid == 0:
        print("No valid comparisons. Check model path and data generation.")
        return

    print(f"\n{'='*60}")
    print(f"RESULTS ({n_valid} scenarios)")
    print(f"{'='*60}")

    fmm_ms = np.array(fmm_makespans[:n_valid])
    neural_ms = np.array(neural_makespans[:n_valid])
    improvement = (fmm_ms - neural_ms) / fmm_ms * 100

    print(f"\n--- Makespan ---")
    print(f"  FMM mean:    {fmm_ms.mean():.2f} +/- {fmm_ms.std():.2f}")
    print(f"  Neural mean: {neural_ms.mean():.2f} +/- {neural_ms.std():.2f}")
    print(f"  Improvement: {improvement.mean():.1f}% +/- {improvement.std():.1f}%")
    print(f"  Neural wins: {(neural_ms < fmm_ms).sum()}/{n_valid}")

    fmm_bal = np.array(fmm_balances[:n_valid])
    neural_bal = np.array(neural_balances[:n_valid])
    print(f"\n--- Load Balance (std of waypoint counts, lower is better) ---")
    print(f"  FMM mean:    {fmm_bal.mean():.2f}")
    print(f"  Neural mean: {neural_bal.mean():.2f}")

    fmm_t = np.array(fmm_times[:n_valid])
    neural_t = np.array(neural_times[:n_valid])
    print(f"\n--- Inference Time ---")
    print(f"  FMM mean:    {fmm_t.mean():.4f}s")
    print(f"  Neural mean: {neural_t.mean():.4f}s")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
