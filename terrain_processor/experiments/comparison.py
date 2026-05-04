"""
Comparison experiments
======================

把多个基线在同一数据集上跑一遍，输出汇总表 (CSV + 文本)。

  python -m experiments.comparison --n 8 --shape 12,18,18 --out results/cmp.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from typing import Dict, List

import numpy as np

from experiments.datasets import generate_dataset, ScenarioRecord
from experiments.baselines import (
    BaselineResult,
    GridAStarBaseline,
    HierarchicalFMMBaseline,
    SingleResolutionFMMBaseline,
    UAVHierarchicalFMMBaseline,
)
from uav_constraints import UAVKinematics, check_path_feasibility


def aggregate(name: str, results: List[BaselineResult]) -> Dict[str, float]:
    succ = [r for r in results if r.success]
    sr = len(succ) / max(1, len(results))
    if not succ:
        return {"name": name, "success_rate": sr, "n_total": len(results)}
    cost = [r.cost for r in succ]
    plan_t = [r.plan_time_s for r in succ]
    pts = [len(r.path) for r in succ]
    viol = [r.feasibility.get("n_violations", 0) for r in succ]
    R_min = [r.feasibility.get("min_curvature_radius_m", math.inf) for r in succ]
    R_min_finite = [v for v in R_min if math.isfinite(v)]
    return {
        "name": name,
        "n_total": len(results),
        "n_success": len(succ),
        "success_rate": sr,
        "cost_mean": statistics.fmean(cost),
        "cost_std": statistics.pstdev(cost) if len(cost) > 1 else 0.0,
        "plan_time_mean": statistics.fmean(plan_t),
        "plan_time_p95": float(np.percentile(plan_t, 95)) if len(plan_t) > 1 else plan_t[0],
        "path_points_mean": statistics.fmean(pts),
        "viol_mean": statistics.fmean(viol) if viol else 0.0,
        "min_R_mean_m": statistics.fmean(R_min_finite) if R_min_finite else math.inf,
    }


def run_comparison(records: List[ScenarioRecord],
                   uav: UAVKinematics) -> Dict[str, List[BaselineResult]]:
    baselines = [
        GridAStarBaseline(),
        SingleResolutionFMMBaseline(),
        HierarchicalFMMBaseline(n_levels=3),
        UAVHierarchicalFMMBaseline(n_levels=3, kinematics=uav, do_smoothing=True),
    ]
    out: Dict[str, List[BaselineResult]] = {b.name: [] for b in baselines}
    for r in records:
        for b in baselines:
            try:
                res = b.run(r)
            except Exception as e:
                res = BaselineResult(b.name, False, [], math.inf, 0.0,
                                      extras={"error": repr(e)})
            # 统一在末尾对所有路径计算可行性指标 (用于公平评比)
            if res.success and not res.feasibility:
                try:
                    res.feasibility = check_path_feasibility(
                        res.path, uav, r.spacing
                    )
                except Exception:
                    res.feasibility = {}
            out[b.name].append(res)
    return out


def write_csv(rows: List[Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_table(rows: List[Dict[str, float]]) -> None:
    cols = ["name", "n_success", "success_rate", "cost_mean",
            "plan_time_mean", "plan_time_p95", "path_points_mean",
            "viol_mean", "min_R_mean_m"]
    header = " | ".join(f"{c:>17}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "-")
            if isinstance(v, float):
                if math.isinf(v):
                    cells.append("    inf")
                else:
                    cells.append(f"{v:>17.3f}")
            else:
                cells.append(f"{str(v):>17}")
        print(" | ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--shape", type=str, default="10,15,15")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="experiments_data/comparison.csv")
    parser.add_argument("--turn-radius", type=float, default=4.0)
    parser.add_argument("--vmax", type=float, default=15.0)
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    print(f"=== Generating dataset: n={args.n} shape={shape} seed={args.seed} ===")
    records = generate_dataset(n_scenarios=args.n, shape=shape, seed=args.seed)

    uav = UAVKinematics(min_turn_radius=args.turn_radius,
                        vmax_horizontal=args.vmax)
    print(f"=== Running baselines (UAV: R_min={uav.min_turn_radius}m,"
          f" vmax={uav.vmax_horizontal}m/s) ===")
    results = run_comparison(records, uav)
    rows = [aggregate(name, res) for name, res in results.items()]
    print()
    print_table(rows)
    write_csv(rows, args.out)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
