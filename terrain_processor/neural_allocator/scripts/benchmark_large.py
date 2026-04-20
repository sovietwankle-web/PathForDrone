#!/usr/bin/env python3
"""
Benchmark: test neural allocator on 1000x1000 terrain (50km @ 50m).

Usage:
    python benchmark_large.py [--model checkpoints/best_imitation.pt] [--size 1000]
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_allocator.config import NeuralAllocatorConfig
from neural_allocator.hierarchical_features import HierarchicalFeatureExtractor
from neural_allocator.features import FeatureExtractor
from viscous_hierarchical_fmm import ViscosityField, ViscosityParams


def generate_large_terrain(size=1000):
    """Generate a 2D terrain of given size."""
    y, x = np.meshgrid(
        np.linspace(0, 4 * np.pi, size),
        np.linspace(0, 4 * np.pi, size),
        indexing='ij'
    )
    terrain = np.zeros((size, size), dtype=np.float64)

    # Multiple frequency hills
    for _ in range(8):
        freq = np.random.uniform(0.3, 1.5)
        phase = np.random.uniform(0, 2 * np.pi, 2)
        amp = np.random.uniform(0.5, 2.0)
        terrain += amp * np.sin(freq * x + phase[0]) * np.cos(freq * y + phase[1])

    # Large obstacles
    for _ in range(5):
        cx, cy = np.random.randint(size // 4, 3 * size // 4, 2)
        r = np.random.randint(20, size // 8)
        yy, xx = np.ogrid[:size, :size]
        dist = ((yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5
        terrain += 5.0 * np.exp(-dist ** 2 / (2 * r ** 2))

    return terrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--n_robots', type=int, default=15)
    parser.add_argument('--n_waypoints', type=int, default=80)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--n_levels', type=int, default=4)
    args = parser.parse_args()

    print(f"=== Large-Scale Benchmark ===")
    print(f"Terrain: {args.size}×{args.size} (≈{args.size*50/1000:.0f}km @ 50m spacing)")
    print(f"Robots: {args.n_robots}, Waypoints: {args.n_waypoints}")
    print(f"Pyramid levels: {args.n_levels}")
    print()

    # Generate terrain
    print("[1] Generating terrain...")
    t0 = time.time()
    terrain = generate_large_terrain(args.size)
    spacing = (50.0, 50.0)  # 50m resolution
    print(f"    Done in {time.time()-t0:.1f}s, shape={terrain.shape}")

    # Compute speed field
    print("[2] Computing speed field...")
    t0 = time.time()
    vf = ViscosityField(ViscosityParams())
    speed_field = vf.get_speed_field(terrain, None, spacing=spacing)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Place robots and waypoints randomly
    np.random.seed(42)
    valid = speed_field > speed_field.mean() * 0.3
    valid_coords = np.argwhere(valid)
    idx = np.random.choice(len(valid_coords), args.n_robots + args.n_waypoints, replace=False)
    all_points = [tuple(valid_coords[i].tolist()) for i in idx]
    robot_starts = all_points[:args.n_robots]
    waypoints = all_points[args.n_robots:]

    print(f"    Robots placed at: {robot_starts[:3]}...")
    print(f"    Waypoints: {len(waypoints)} placed")

    # Benchmark hierarchical feature extraction
    print("\n[3] Hierarchical feature extraction...")
    t0 = time.time()
    hfe = HierarchicalFeatureExtractor(
        terrain=terrain, n_levels=args.n_levels,
        spacing=spacing,
    )
    build_time = time.time() - t0
    print(f"    Pyramid build: {build_time:.2f}s")

    t0 = time.time()
    cost_matrix = hfe.compute_cost_matrix(robot_starts, waypoints)
    cost_time = time.time() - t0
    print(f"    Cost matrix ({args.n_robots}×{args.n_waypoints}): {cost_time*1000:.1f}ms")

    t0 = time.time()
    robot_feats = hfe.extract_robot_features(robot_starts)
    wp_feats = hfe.extract_waypoint_features(waypoints)
    pairwise_feats = hfe.extract_pairwise_features(robot_starts, waypoints, cost_matrix)
    _, _, pairwise_feats = hfe.normalize_features(robot_feats, wp_feats, pairwise_feats)
    feat_time = time.time() - t0
    print(f"    All features: {feat_time*1000:.1f}ms")

    # Benchmark neural network inference
    print("\n[4] Neural network inference...")
    import torch
    from neural_allocator.model import WaypointAllocationNet

    cfg = NeuralAllocatorConfig.large()
    model = WaypointAllocationNet(cfg)

    if args.model and os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
        print(f"    Loaded model from {args.model}")

    model.eval()

    n_r, n_w = args.n_robots, args.n_waypoints
    r_f = torch.from_numpy(robot_feats).unsqueeze(0)
    w_f = torch.from_numpy(wp_feats).unsqueeze(0)
    p_f = torch.from_numpy(pairwise_feats).unsqueeze(0)
    r_m = torch.ones(1, n_r, dtype=torch.bool)
    w_m = torch.ones(1, n_w, dtype=torch.bool)

    with torch.no_grad():
        t0 = time.time()
        scores = model(r_f, w_f, p_f, r_m, w_m)
        infer_time = time.time() - t0
    print(f"    Inference: {infer_time*1000:.1f}ms, scores: {scores.shape}")

    # Decode assignment
    from neural_allocator.decoder import hungarian_decode
    t0 = time.time()
    assignments = hungarian_decode(scores[0].numpy(), n_w, n_r)
    decode_time = time.time() - t0
    print(f"    Hungarian decode: {decode_time*1000:.1f}ms")

    print("\n[5] Assignment summary:")
    counts = [len(assignments[r]) for r in range(n_r)]
    print(f"    Load per robot: min={min(counts)}, max={max(counts)}, "
          f"mean={np.mean(counts):.1f}, std={np.std(counts):.1f}")

    total_time = build_time + cost_time + feat_time + infer_time + decode_time
    print(f"\n    TOTAL (excl. path planning): {total_time:.2f}s")
    print(f"    Breakdown: pyramid={build_time:.2f}s, cost={cost_time*1000:.0f}ms, "
          f"features={feat_time*1000:.0f}ms, inference={infer_time*1000:.0f}ms")

    # Memory usage estimate
    terrain_mb = terrain.nbytes / 1024 / 1024
    speed_mb = speed_field.nbytes / 1024 / 1024
    print(f"\n    Terrain memory: {terrain_mb:.0f} MB, Speed field: {speed_mb:.0f} MB")


if __name__ == '__main__':
    main()
