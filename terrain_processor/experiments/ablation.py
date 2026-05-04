"""
Ablation experiments
====================

固定基线 = ``UAVHierarchicalFMMBaseline``（含全部新约束/重规划），
逐个关闭以下因子并对比代价与可行性指标：

  A0  full                — 全开 (基线)
  A1  no_slope            — slope_weight = 0
  A2  no_roughness        — roughness_weight = 0
  A3  no_fog              — fog_weight = 0
  A4  no_uav              — 不应用 UAV 速度修饰
  A5  no_smooth           — 不做转弯半径平滑
  A6  no_anisotropy       — UAV 修饰中关闭 conservative_anisotropy
  A7  no_alt_decay        — 海拔衰减率置 0
  A8  online              — 用 OnlineReplanner 替换静态规划 (只跑一个静态雾场)

输出 CSV + 文本表。

用法：
  python -m experiments.ablation --n 6 --shape 10,15,15
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from viscous_hierarchical_fmm import (
    HierarchicalPathPlanner,
    ViscosityField,
    ViscosityParams,
)
from uav_constraints import (
    UAVKinematics,
    apply_uav_speed_modifier,
    check_path_feasibility,
    enforce_turn_radius,
)
from online_replanner import (
    OnlineReplanner,
    ReplanTrigger,
    StaticWeatherSource,
)
from experiments.datasets import generate_dataset, ScenarioRecord
from experiments.comparison import write_csv, print_table


@dataclass
class AblationConfig:
    name: str
    use_uav: bool = True
    use_smoothing: bool = True
    use_anisotropy: bool = True
    use_altitude_decay: bool = True
    slope_weight: float = 0.3
    roughness_weight: float = 0.2
    fog_weight: float = 0.5
    use_online: bool = False


ABLATIONS: List[AblationConfig] = [
    AblationConfig("A0_full"),
    AblationConfig("A1_no_slope", slope_weight=0.0),
    AblationConfig("A2_no_roughness", roughness_weight=0.0),
    AblationConfig("A3_no_fog", fog_weight=0.0),
    AblationConfig("A4_no_uav", use_uav=False, use_smoothing=False),
    AblationConfig("A5_no_smooth", use_smoothing=False),
    AblationConfig("A6_no_anisotropy", use_anisotropy=False),
    AblationConfig("A7_no_alt_decay", use_altitude_decay=False),
    AblationConfig("A8_online", use_online=True),
]


def run_one(scn: ScenarioRecord,
            cfg: AblationConfig,
            kin_base: UAVKinematics) -> Dict[str, Any]:
    vp = ViscosityParams(slope_weight=cfg.slope_weight,
                         roughness_weight=cfg.roughness_weight,
                         fog_weight=cfg.fog_weight)
    # 派生 UAV (按 cfg 调整 anisotropy / altitude_decay)
    kin = UAVKinematics(
        vmax_horizontal=kin_base.vmax_horizontal,
        vmax_climb=kin_base.vmax_climb,
        vmax_descent=kin_base.vmax_descent,
        vmin=kin_base.vmin,
        min_turn_radius=kin_base.min_turn_radius,
        max_pitch_angle=kin_base.max_pitch_angle,
        angle_decay_exponent=kin_base.angle_decay_exponent if cfg.use_anisotropy else 0.0,
        altitude_decay_per_m=(kin_base.altitude_decay_per_m
                              if cfg.use_altitude_decay else 0.0),
        altitude_ref=kin_base.altitude_ref,
        altitude_ceiling=kin_base.altitude_ceiling,
        max_slope_angle=kin_base.max_slope_angle,
    )

    if cfg.use_online:
        weather = StaticWeatherSource(scn.fog.astype(np.float64))
        rep = OnlineReplanner(
            terrain=scn.terrain.astype(np.float64),
            weather_source=weather,
            kinematics=kin if cfg.use_uav else UAVKinematics(),
            viscosity_params=vp,
            trigger=ReplanTrigger(periodic_interval=10.0),
            spacing=scn.spacing,
            n_levels=3,
            dt=1.0,
        )
        t0 = time.time()
        trace = rep.run(scn.start, scn.goal, t_max=60.0)
        elapsed = time.time() - t0
        feas = trace.feasibility
        return {
            "name": cfg.name,
            "success": trace.success,
            "cost": (feas.get("total_distance_m", math.inf) if feas
                     else math.inf),
            "plan_time_s": trace.total_replan_time_s,
            "wall_time_s": elapsed,
            "n_replans": len(trace.replan_events),
            "n_points": len(trace.waypoints),
            "viol": feas.get("n_violations", 0) if feas else 0,
            "min_R_m": feas.get("min_curvature_radius_m", math.inf) if feas else math.inf,
        }

    planner = HierarchicalPathPlanner(
        terrain=scn.terrain.astype(np.float64),
        fog_data=scn.fog.astype(np.float64),
        n_levels=3,
        viscosity_params=vp,
        spacing=scn.spacing,
    )
    if cfg.use_uav:
        for level in range(len(planner.pyramid.speed_levels)):
            sp = planner.pyramid.spacings[level]
            planner.pyramid.speed_levels[level] = apply_uav_speed_modifier(
                planner.pyramid.speed_levels[level],
                planner.pyramid.levels[level],
                kin,
                spacing=sp,
                conservative_anisotropy=cfg.use_anisotropy,
            )

    t0 = time.time()
    path, stats = planner.plan(scn.start, scn.goal)
    if path and cfg.use_smoothing and cfg.use_uav:
        path = enforce_turn_radius(path, kin, scn.spacing)
    plan_time = time.time() - t0

    ok = stats.get("success", False) and len(path) > 0
    cost = stats.get("path_cost", math.inf)
    feas = check_path_feasibility(path, kin, scn.spacing) if path else {}
    return {
        "name": cfg.name,
        "success": ok,
        "cost": cost,
        "plan_time_s": plan_time,
        "wall_time_s": plan_time,
        "n_replans": 0,
        "n_points": len(path),
        "viol": feas.get("n_violations", 0) if feas else 0,
        "min_R_m": feas.get("min_curvature_radius_m", math.inf) if feas else math.inf,
    }


def aggregate(name: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    succ = [r for r in runs if r["success"]]
    sr = len(succ) / max(1, len(runs))
    if not succ:
        return {"name": name, "success_rate": sr, "n_total": len(runs)}
    cost = [r["cost"] for r in succ]
    plan_t = [r["plan_time_s"] for r in succ]
    n_pts = [r["n_points"] for r in succ]
    viol = [r["viol"] for r in succ]
    Rs = [r["min_R_m"] for r in succ if math.isfinite(r["min_R_m"])]
    return {
        "name": name,
        "n_total": len(runs),
        "n_success": len(succ),
        "success_rate": sr,
        "cost_mean": statistics.fmean(cost),
        "plan_time_mean": statistics.fmean(plan_t),
        "plan_time_p95": float(np.percentile(plan_t, 95)) if len(plan_t) > 1 else plan_t[0],
        "path_points_mean": statistics.fmean(n_pts),
        "viol_mean": statistics.fmean(viol) if viol else 0.0,
        "min_R_mean_m": statistics.fmean(Rs) if Rs else math.inf,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--shape", type=str, default="10,15,15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments_data/ablation.csv")
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    records = generate_dataset(n_scenarios=args.n, shape=shape, seed=args.seed)
    kin_base = UAVKinematics(min_turn_radius=4.0, vmax_horizontal=15.0)

    print(f"=== Ablation: n={args.n} shape={shape} ===")
    rows = []
    per_cfg: Dict[str, List[Dict[str, Any]]] = {a.name: [] for a in ABLATIONS}
    for cfg in ABLATIONS:
        print(f"  -> {cfg.name}", flush=True)
        for scn in records:
            res = run_one(scn, cfg, kin_base)
            per_cfg[cfg.name].append(res)
        rows.append(aggregate(cfg.name, per_cfg[cfg.name]))

    print()
    print_table(rows)
    write_csv(rows, args.out)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
