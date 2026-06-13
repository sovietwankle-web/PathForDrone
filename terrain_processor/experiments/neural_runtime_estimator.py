"""
Fast runtime estimator for constrained neural/RL UAV allocators.

This script avoids full training. It measures a few synthetic small-model
training steps plus several target-size warmup steps, then extrapolates the
total imitation/RL training time and per-scenario inference latency.

The synthetic batch includes two cheap constraints:
  - range: robot-waypoint distance must be under max_range_ratio * grid diagonal
  - no-fly: a fraction of robot-waypoint pairs is masked as forbidden

Usage:
  python -m experiments.neural_runtime_estimator --quick
  python -m experiments.neural_runtime_estimator --target_robots 20 --target_waypoints 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from neural_allocator.config import NeuralAllocatorConfig
from neural_allocator.model import WaypointAllocationNet


@dataclass
class BenchSpec:
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    robots: int
    waypoints: int
    batch_size: int
    steps: int


def _make_config(spec: BenchSpec, lr: float) -> NeuralAllocatorConfig:
    ff_dim = max(spec.d_model * 4, 64)
    return NeuralAllocatorConfig(
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        n_cross_attn_layers=spec.n_layers,
        ff_dim=ff_dim,
        dropout=0.0,
        max_robots=spec.robots,
        max_waypoints=spec.waypoints,
        lr=lr,
        lr_rl=lr,
    )


def _safe_heads(d_model: int, preferred: int) -> int:
    for h in range(min(preferred, d_model), 0, -1):
        if d_model % h == 0:
            return h
    return 1


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _synthetic_batch(
    batch_size: int,
    n_robots: int,
    n_waypoints: int,
    device: torch.device,
    no_fly_ratio: float,
    max_range_ratio: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    robot_pos = rng.random((batch_size, n_robots, 3), dtype=np.float32)
    waypoint_pos = rng.random((batch_size, n_waypoints, 3), dtype=np.float32)

    diff = waypoint_pos[:, None, :, :] - robot_pos[:, :, None, :]
    dist = np.linalg.norm(diff, axis=-1).astype(np.float32)
    max_range = max_range_ratio * math.sqrt(3.0)
    feasible = dist <= max_range

    if no_fly_ratio > 0:
        feasible &= rng.random(feasible.shape) >= no_fly_ratio

    # Keep each waypoint feasible for at least one robot.
    nearest = dist.argmin(axis=1)
    for b in range(batch_size):
        feasible[b, nearest[b], np.arange(n_waypoints)] = True

    terrain_cost = rng.random((batch_size, n_waypoints, 1), dtype=np.float32)
    local_q = rng.random((batch_size, n_waypoints, 1), dtype=np.float32)
    waypoint_feats = np.concatenate([waypoint_pos, terrain_cost, local_q], axis=-1)

    velocities = np.zeros((batch_size, n_robots, 3), dtype=np.float32)
    loads = rng.random((batch_size, n_robots, 1), dtype=np.float32)
    robot_feats = np.concatenate([robot_pos, velocities, loads], axis=-1)

    cost = dist * (1.0 + terrain_cost.transpose(0, 2, 1))
    pairwise = np.stack([cost, dist], axis=-1).astype(np.float32)
    mean = pairwise.mean(axis=(1, 2), keepdims=True)
    std = pairwise.std(axis=(1, 2), keepdims=True) + 1e-6
    pairwise = (pairwise - mean) / std

    targets = np.zeros((batch_size, n_waypoints), dtype=np.int64)
    for b in range(batch_size):
        masked = np.where(feasible[b], cost[b], np.inf)
        targets[b] = masked.argmin(axis=0)

    return {
        "robot_feats": torch.from_numpy(robot_feats).to(device),
        "waypoint_feats": torch.from_numpy(waypoint_feats).to(device),
        "pairwise_feats": torch.from_numpy(pairwise).to(device),
        "robot_mask": torch.ones(batch_size, n_robots, dtype=torch.bool, device=device),
        "waypoint_mask": torch.ones(batch_size, n_waypoints, dtype=torch.bool, device=device),
        "pair_feasible": torch.from_numpy(feasible).to(device),
        "targets": torch.from_numpy(targets).to(device),
    }


def _constraint_loss(scores: torch.Tensor,
                     pair_feasible: torch.Tensor,
                     targets: torch.Tensor) -> torch.Tensor:
    # scores: (B, R, W), targets: (B, W)
    masked = scores.masked_fill(~pair_feasible, -1e9)
    logits = masked.permute(0, 2, 1).reshape(-1, masked.shape[1])
    labels = targets.reshape(-1)
    return F.cross_entropy(logits, labels)


def _run_imitation_probe(
    spec: BenchSpec,
    lr: float,
    device: torch.device,
    no_fly_ratio: float,
    max_range_ratio: float,
    total_scheduler_steps: int,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    cfg = _make_config(spec, lr)
    model = WaypointAllocationNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max(total_scheduler_steps, spec.steps + 2),
        pct_start=0.1,
    )
    batch = _synthetic_batch(
        spec.batch_size, spec.robots, spec.waypoints, device,
        no_fly_ratio, max_range_ratio, seed,
    )

    # Warmup is not counted.
    scores = model(
        batch["robot_feats"], batch["waypoint_feats"], batch["pairwise_feats"],
        batch["robot_mask"], batch["waypoint_mask"],
    )
    loss = _constraint_loss(scores, batch["pair_feasible"], batch["targets"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    step_times: List[float] = []
    lrs: List[float] = []
    losses: List[float] = []
    for _ in range(spec.steps):
        t0 = time.perf_counter()
        scores = model(
            batch["robot_feats"], batch["waypoint_feats"], batch["pairwise_feats"],
            batch["robot_mask"], batch["waypoint_mask"],
        )
        loss = _constraint_loss(scores, batch["pair_feasible"], batch["targets"])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        lrs.append(float(optimizer.param_groups[0]["lr"]))
        losses.append(float(loss.detach().cpu()))

    infer_times: List[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(max(2, spec.steps)):
            t0 = time.perf_counter()
            scores = model(
                batch["robot_feats"], batch["waypoint_feats"], batch["pairwise_feats"],
                batch["robot_mask"], batch["waypoint_mask"],
            )
            scores = scores.masked_fill(~batch["pair_feasible"], -1e9)
            _ = scores.argmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_times.append(time.perf_counter() - t0)

    return {
        "name": spec.name,
        "mode": "imitation",
        "d_model": spec.d_model,
        "n_heads": spec.n_heads,
        "n_layers": spec.n_layers,
        "robots": spec.robots,
        "waypoints": spec.waypoints,
        "batch_size": spec.batch_size,
        "params": _param_count(model),
        "mean_step_s": statistics.fmean(step_times),
        "median_step_s": statistics.median(step_times),
        "mean_infer_s": statistics.fmean(infer_times) / spec.batch_size,
        "lr_first_steps": lrs,
        "loss_first_steps": losses,
    }


def _run_rl_probe(
    spec: BenchSpec,
    lr: float,
    device: torch.device,
    no_fly_ratio: float,
    max_range_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed + 1009)
    cfg = _make_config(spec, lr)
    model = WaypointAllocationNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    batch = _synthetic_batch(
        spec.batch_size, spec.robots, spec.waypoints, device,
        no_fly_ratio, max_range_ratio, seed + 1009,
    )
    assigned = torch.zeros(
        spec.batch_size, spec.waypoints, dtype=torch.bool, device=device
    )

    step_times: List[float] = []
    losses: List[float] = []
    rollout_steps = min(spec.steps, spec.waypoints)
    for k in range(rollout_steps):
        t0 = time.perf_counter()
        scores, value = model.forward_rl(
            batch["robot_feats"], batch["waypoint_feats"], batch["pairwise_feats"],
            assigned, batch["robot_mask"], batch["waypoint_mask"],
        )
        feasible = batch["pair_feasible"] & ~assigned[:, None, :]
        scores = scores.masked_fill(~feasible, -1e9)
        logits = scores.permute(0, 2, 1).reshape(-1, spec.robots)
        targets = batch["targets"].reshape(-1)
        policy_loss = F.cross_entropy(logits, targets)
        value_loss = (value.squeeze(-1) ** 2).mean()
        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        assigned[:, k] = True
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        losses.append(float(loss.detach().cpu()))

    return {
        "name": spec.name,
        "mode": "rl",
        "d_model": spec.d_model,
        "n_heads": spec.n_heads,
        "n_layers": spec.n_layers,
        "robots": spec.robots,
        "waypoints": spec.waypoints,
        "batch_size": spec.batch_size,
        "params": _param_count(model),
        "mean_step_s": statistics.fmean(step_times),
        "median_step_s": statistics.median(step_times),
        "mean_infer_s": math.nan,
        "lr_first_steps": [lr] * len(step_times),
        "loss_first_steps": losses,
    }


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24.0:.1f}d"


def _estimate(rows: List[Dict[str, Any]],
              target_name: str,
              n_train: int,
              n_val: int,
              epochs: int,
              batch_size: int,
              val_interval: int,
              rl_epochs: int,
              rl_rollout_steps: int) -> Dict[str, Any]:
    target = next(r for r in rows if r["name"] == target_name and r["mode"] == "imitation")
    target_rl = next(r for r in rows if r["name"] == target_name and r["mode"] == "rl")
    steps_per_epoch = math.ceil(n_train / batch_size)
    val_steps = math.ceil(n_val / batch_size)
    n_val_runs = epochs // max(val_interval, 1)
    imitation_train_s = target["mean_step_s"] * steps_per_epoch * epochs
    imitation_val_s = target["mean_infer_s"] * batch_size * val_steps * n_val_runs
    rl_train_s = target_rl["mean_step_s"] * rl_rollout_steps * rl_epochs
    return {
        "target": target_name,
        "steps_per_epoch": steps_per_epoch,
        "target_step_s": target["mean_step_s"],
        "target_infer_s_per_case": target["mean_infer_s"],
        "imitation_train_s": imitation_train_s,
        "imitation_val_s": imitation_val_s,
        "imitation_total_s": imitation_train_s + imitation_val_s,
        "rl_train_s": rl_train_s,
        "combined_s": imitation_train_s + imitation_val_s + rl_train_s,
        "human": {
            "imitation_total": _format_seconds(imitation_train_s + imitation_val_s),
            "rl_train": _format_seconds(rl_train_s),
            "combined": _format_seconds(imitation_train_s + imitation_val_s + rl_train_s),
            "inference_per_case": _format_seconds(target["mean_infer_s"]),
        },
        "lr_first_steps": target["lr_first_steps"],
    }


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "name", "mode", "d_model", "n_heads", "n_layers", "robots", "waypoints",
        "batch_size", "params", "mean_step_s", "median_step_s", "mean_infer_s",
        "lr_first_steps", "loss_first_steps",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use a tiny target probe for a sub-minute smoke run.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--small_steps", type=int, default=2)
    parser.add_argument("--large_steps", type=int, default=3)
    parser.add_argument("--target_robots", type=int, default=12)
    parser.add_argument("--target_waypoints", type=int, default=80)
    parser.add_argument("--target_d_model", type=int, default=128)
    parser.add_argument("--target_layers", type=int, default=3)
    parser.add_argument("--target_heads", type=int, default=8)
    parser.add_argument("--target_batch_size", type=int, default=8)
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--rl_epochs", type=int, default=200)
    parser.add_argument("--rl_rollout_steps", type=int, default=2048)
    parser.add_argument("--no_fly_ratio", type=float, default=0.12)
    parser.add_argument("--max_range_ratio", type=float, default=0.72)
    parser.add_argument("--out", default="experiments_data/neural_runtime_estimate.csv")
    parser.add_argument("--json_out", default="experiments_data/neural_runtime_estimate.json")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.quick:
        args.target_robots = min(args.target_robots, 6)
        args.target_waypoints = min(args.target_waypoints, 32)
        args.target_d_model = min(args.target_d_model, 64)
        args.target_layers = min(args.target_layers, 2)
        args.target_batch_size = min(args.target_batch_size, 4)
        args.n_train = min(args.n_train, 5000)
        args.n_val = min(args.n_val, 500)
        args.epochs = min(args.epochs, 30)
        args.rl_epochs = min(args.rl_epochs, 30)
        args.rl_rollout_steps = min(args.rl_rollout_steps, 256)

    small_specs = [
        BenchSpec("small_2r8w", 32, _safe_heads(32, 4), 1, 2, 8, 8, args.small_steps),
        BenchSpec("small_4r16w", 48, _safe_heads(48, 4), 2, 4, 16, 6, args.small_steps),
        BenchSpec("small_6r32w", 64, _safe_heads(64, 4), 2, 6, 32, 4, args.small_steps),
    ]
    target_spec = BenchSpec(
        "target_probe",
        args.target_d_model,
        _safe_heads(args.target_d_model, args.target_heads),
        args.target_layers,
        args.target_robots,
        args.target_waypoints,
        args.target_batch_size,
        args.large_steps,
    )
    specs = small_specs + [target_spec]

    total_scheduler_steps = math.ceil(args.n_train / args.target_batch_size) * args.epochs
    print(f"Device: {device}")
    print("Running short constrained imitation/RL probes...")
    rows: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        print(
            f"  -> {spec.name}: R={spec.robots} W={spec.waypoints} "
            f"d={spec.d_model} L={spec.n_layers} B={spec.batch_size}"
        )
        rows.append(_run_imitation_probe(
            spec, args.lr, device, args.no_fly_ratio, args.max_range_ratio,
            total_scheduler_steps, args.seed + idx,
        ))
        rows.append(_run_rl_probe(
            spec, args.lr, device, args.no_fly_ratio, args.max_range_ratio,
            args.seed + idx,
        ))

    estimate = _estimate(
        rows, "target_probe", args.n_train, args.n_val, args.epochs,
        args.target_batch_size, args.val_interval, args.rl_epochs,
        args.rl_rollout_steps,
    )

    _write_csv(rows, args.out)
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "estimate": estimate}, f, indent=2)

    print("\nEstimate")
    print(f"  target step: {estimate['target_step_s']:.4f}s")
    print(f"  target inference/case: {estimate['target_infer_s_per_case']:.5f}s")
    print(f"  imitation total: {estimate['human']['imitation_total']}")
    print(f"  rl train: {estimate['human']['rl_train']}")
    print(f"  combined: {estimate['human']['combined']}")
    print(f"  first target learning rates: {estimate['lr_first_steps']}")
    print(f"\nsaved to {args.out}")
    print(f"saved to {args.json_out}")


if __name__ == "__main__":
    main()
