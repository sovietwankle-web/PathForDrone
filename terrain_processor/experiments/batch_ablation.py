"""
Batch round-trip ablation experiments.

评估分批次出发-返回任务分配的几个关键模块：
  A0_full            DP切分 + 跨机器人局部搜索 + 批同步槽对齐
  A1_no_local        关闭局部搜索
  A2_greedy          用贪心分批替代DP切分
  A3_no_capacity     关闭每批容量约束
  A4_no_cross_refine 局部搜索只允许机器人内部搬移
  A5_no_sync_align   不对齐同步批次槽
  A6_independent_obj 用独立机器人完成时间替代同步批次目标
  A7_single_batch    单批往返基线
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from viscous_hierarchical_fmm import (
    GasDiffusionParams,
    MultiRobotWaypointAllocator,
    ViscosityParams,
)
from experiments.datasets import (
    ScenarioRecord,
    generate_dataset,
)


@dataclass
class BatchAblationConfig:
    name: str
    n_batches: int
    max_waypoints_per_batch: int | None
    local_search_iterations: int = 2
    use_dp_partition: bool = True
    allow_cross_robot_refine: bool = True
    align_sync_slots: bool = True
    objective: str = "sync"


def _valid_point(terrain: np.ndarray, rng: np.random.Generator) -> Tuple[int, ...]:
    shape = terrain.shape
    max_t = float(np.max(terrain))
    for _ in range(500):
        point = tuple(int(rng.integers(1, s - 1)) for s in shape)
        if terrain[point] < max_t * 0.65:
            return point
    return tuple(int(s // 2) for s in shape)


def make_multi_robot_task(scn: ScenarioRecord,
                          n_robots: int,
                          n_waypoints: int) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """从单起终点场景派生多机器人/多路径点任务，保证可复现。"""
    rng = np.random.default_rng(scn.seed + 9173)
    starts = [scn.start]
    while len(starts) < n_robots:
        p = _valid_point(scn.terrain, rng)
        if min(math.dist(p, q) for q in starts) >= 3.0:
            starts.append(p)

    waypoints: List[Tuple[int, ...]] = []
    while len(waypoints) < n_waypoints:
        p = _valid_point(scn.terrain, rng)
        if min(math.dist(p, q) for q in starts) < 4.0:
            continue
        if waypoints and min(math.dist(p, q) for q in waypoints) < 2.0:
            continue
        waypoints.append(p)

    return starts, waypoints


def run_one(scn: ScenarioRecord,
            cfg: BatchAblationConfig,
            n_robots: int,
            n_waypoints: int,
            n_levels: int) -> Dict[str, Any]:
    starts, waypoints = make_multi_robot_task(scn, n_robots, n_waypoints)
    params = GasDiffusionParams(
        n_robots=n_robots,
        robot_starts=starts,
        waypoints=waypoints,
        n_batches=cfg.n_batches,
        return_to_start=True,
        max_waypoints_per_batch=cfg.max_waypoints_per_batch,
        batch_local_search_iterations=cfg.local_search_iterations,
        batch_use_dp_partition=cfg.use_dp_partition,
        batch_allow_cross_robot_refine=cfg.allow_cross_robot_refine,
        batch_align_sync_slots=cfg.align_sync_slots,
        batch_objective=cfg.objective,
    )

    t0 = time.time()
    result = MultiRobotWaypointAllocator(
        terrain=scn.terrain.astype(np.float64),
        fog_data=scn.fog.astype(np.float64),
        n_levels=n_levels,
        viscosity_params=ViscosityParams(),
        gas_params=params,
        spacing=scn.spacing,
    ).allocate()
    elapsed = time.time() - t0
    stats = result.stats

    return {
        "name": cfg.name,
        "scenario_id": scn.scenario_id,
        "success": bool(stats.get("success", False)),
        "assigned": int(stats.get("n_waypoints_assigned", 0)),
        "total_waypoints": int(stats.get("n_waypoints_total", 0)),
        "mission_time": float(stats.get("independent_robot_mission_time", math.inf)),
        "sync_time": float(stats.get("synchronized_batch_mission_time", math.inf)),
        "sum_batch_time": float(stats.get("sum_batch_time", math.inf)),
        "allocation_time": float(stats.get("allocation_time", 0.0)),
        "batch_allocation_time": float(stats.get("batch_allocation_time", 0.0)),
        "path_planning_time": float(stats.get("path_planning_time", 0.0)),
        "total_time": float(stats.get("total_time", elapsed)),
        "wall_time": elapsed,
        "capacity_relaxed": bool(stats.get("capacity_relaxed", False)),
    }


def aggregate(name: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in runs if r["success"]]
    mission = [r["mission_time"] for r in ok if math.isfinite(r["mission_time"])]
    sync = [r["sync_time"] for r in ok if math.isfinite(r["sync_time"])]
    total_t = [r["total_time"] for r in runs]
    batch_t = [r["batch_allocation_time"] for r in runs]
    return {
        "name": name,
        "n_success": len(ok),
        "success_rate": len(ok) / max(1, len(runs)),
        "mission_mean": statistics.fmean(mission) if mission else math.inf,
        "sync_mean": statistics.fmean(sync) if sync else math.inf,
        "total_time_mean": statistics.fmean(total_t) if total_t else math.inf,
        "batch_alloc_mean": statistics.fmean(batch_t) if batch_t else math.inf,
        "assigned_mean": statistics.fmean(r["assigned"] for r in runs) if runs else 0.0,
        "capacity_relaxed": sum(1 for r in runs if r["capacity_relaxed"]),
    }


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = list(rows[0].keys())
    print(" | ".join(f"{h:>18}" for h in headers))
    print("-" * (21 * len(headers)))
    for row in rows:
        cells = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                cells.append(f"{v:>18.4f}")
            else:
                cells.append(f"{str(v):>18}")
        print(" | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--shape", type=str, default="8,12,12")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--waypoints", type=int, default=9)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--out", type=str, default="experiments_data/batch_ablation.csv")
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    records = generate_dataset(n_scenarios=args.n, shape=shape, seed=args.seed)
    configs = [
        BatchAblationConfig("A0_full", args.batches, args.capacity),
        BatchAblationConfig("A1_no_local", args.batches, args.capacity,
                            local_search_iterations=0),
        BatchAblationConfig("A2_greedy", args.batches, args.capacity,
                            use_dp_partition=False),
        BatchAblationConfig("A3_no_capacity", args.batches, None),
        BatchAblationConfig("A4_no_cross_refine", args.batches, args.capacity,
                            allow_cross_robot_refine=False),
        BatchAblationConfig("A5_no_sync_align", args.batches, args.capacity,
                            align_sync_slots=False),
        BatchAblationConfig("A6_independent_obj", args.batches, args.capacity,
                            objective="independent"),
        BatchAblationConfig("A7_single_batch", 1, None,
                            local_search_iterations=0),
    ]

    print(
        f"=== Batch Ablation: n={args.n} shape={shape} "
        f"robots={args.robots} waypoints={args.waypoints} batches={args.batches} ==="
    )
    per_cfg: Dict[str, List[Dict[str, Any]]] = {cfg.name: [] for cfg in configs}
    for cfg in configs:
        print(f"  -> {cfg.name}", flush=True)
        for scn in records:
            per_cfg[cfg.name].append(
                run_one(scn, cfg, args.robots, args.waypoints, args.levels)
            )

    rows = [aggregate(name, runs) for name, runs in per_cfg.items()]
    print()
    print_table(rows)
    write_csv(rows, args.out)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
