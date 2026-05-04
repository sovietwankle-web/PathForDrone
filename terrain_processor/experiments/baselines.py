"""
Baselines for path-planning comparison
=======================================

提供三组基线，全部接受同一 ``ScenarioRecord`` 并返回 ``BaselineResult``：

  1. ``GridAStarBaseline``           — 26-邻域 A* on speed_field (各向同性, 无UAV约束)
  2. ``SingleResolutionFMMBaseline`` — 现有 FMM 单分辨率求解
  3. ``HierarchicalFMMBaseline``     — 分层 FMM (无 UAV 约束)
  4. ``UAVHierarchicalFMMBaseline``  — 分层 FMM + UAV 约束 (本次工作)

便于 comparison.py / ablation.py 调用。
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from viscous_hierarchical_fmm import (
    HierarchicalPathPlanner,
    NarrowBandFMM,
    ViscosityField,
    ViscosityParams,
)
from uav_constraints import (
    UAVKinematics,
    apply_uav_speed_modifier,
    attach_uav_constraints,
    check_path_feasibility,
    enforce_turn_radius,
)
from experiments.datasets import ScenarioRecord


@dataclass
class BaselineResult:
    name: str
    success: bool
    path: List[Tuple[float, ...]] = field(default_factory=list)
    cost: float = math.inf
    plan_time_s: float = 0.0
    feasibility: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


def _path_cost(path, speed_field, spacing) -> float:
    if len(path) < 2:
        return math.inf
    spc = np.array(spacing[: len(path[0])], dtype=np.float64)
    cost = 0.0
    shape = speed_field.shape
    for i in range(len(path) - 1):
        p1, p2 = np.array(path[i]), np.array(path[i + 1])
        d = float(np.linalg.norm((p2 - p1) * spc))
        mid = ((p1 + p2) / 2.0).astype(int)
        mid = tuple(max(0, min(int(m), s - 1)) for m, s in zip(mid, shape))
        v = float(speed_field[mid])
        cost += d / max(v, 1e-3)
    return cost


# ------------------------------------------------------------
# 1. Grid A*
# ------------------------------------------------------------
class GridAStarBaseline:
    name = "AStar26"

    def run(self, scenario: ScenarioRecord,
            viscosity_params: Optional[ViscosityParams] = None,
            **kwargs) -> BaselineResult:
        vf = ViscosityField(viscosity_params or ViscosityParams())
        speed = vf.get_speed_field(scenario.terrain.astype(np.float64),
                                   scenario.fog.astype(np.float64),
                                   spacing=scenario.spacing)
        t0 = time.time()
        path = self._astar(speed, scenario.start, scenario.goal, scenario.spacing)
        t = time.time() - t0
        if not path:
            return BaselineResult(self.name, False, [], math.inf, t)
        cost = _path_cost(path, speed, scenario.spacing)
        return BaselineResult(self.name, True, path, cost, t,
                               feasibility={}, extras={"n_points": len(path)})

    @staticmethod
    def _astar(speed, start, goal, spacing) -> List[Tuple[float, ...]]:
        ndim = len(start)
        shape = speed.shape
        spc = np.array(spacing[:ndim], dtype=np.float64)

        # 26-neighborhood (3D) or 8 (2D)
        deltas: List[Tuple[int, ...]] = []
        if ndim == 3:
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dz == dy == dx == 0:
                            continue
                        deltas.append((dz, dy, dx))
        else:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0:
                        continue
                    deltas.append((dy, dx))

        def h(p):
            return float(np.linalg.norm((np.array(p) - np.array(goal)) * spc))

        open_h = [(h(start), 0.0, start, None)]
        came_from: Dict[Tuple[int, ...], Optional[Tuple[int, ...]]] = {start: None}
        gscore: Dict[Tuple[int, ...], float] = {start: 0.0}

        while open_h:
            f, g, current, _ = heapq.heappop(open_h)
            if current == goal:
                # reconstruct
                path = [current]
                while came_from[path[-1]] is not None:
                    path.append(came_from[path[-1]])  # type: ignore
                path.reverse()
                return [tuple(float(c) for c in p) for p in path]

            for d in deltas:
                nb = tuple(c + dd for c, dd in zip(current, d))
                if any(n < 0 or n >= s for n, s in zip(nb, shape)):
                    continue
                step = float(np.linalg.norm(np.array(d) * spc))
                v = float(speed[nb])
                if v <= 1e-3:
                    continue
                cost = step / v
                tentative = g + cost
                if tentative < gscore.get(nb, math.inf):
                    came_from[nb] = current
                    gscore[nb] = tentative
                    heapq.heappush(open_h, (tentative + h(nb), tentative, nb, current))
        return []


# ------------------------------------------------------------
# 2. Single-resolution FMM
# ------------------------------------------------------------
class SingleResolutionFMMBaseline:
    name = "FMM_single"

    def run(self, scenario: ScenarioRecord,
            viscosity_params: Optional[ViscosityParams] = None,
            **kwargs) -> BaselineResult:
        planner = HierarchicalPathPlanner(
            terrain=scenario.terrain.astype(np.float64),
            fog_data=scenario.fog.astype(np.float64),
            n_levels=1,
            viscosity_params=viscosity_params,
            spacing=scenario.spacing,
        )
        t0 = time.time()
        path, stats = planner.plan(scenario.start, scenario.goal)
        t = time.time() - t0
        ok = stats.get("success", False) and len(path) > 0
        cost = stats.get("path_cost", math.inf) if ok else math.inf
        return BaselineResult(self.name, ok, path, cost, t,
                               extras={"stats": {k: stats.get(k) for k in
                                                 ("path_length", "n_levels")}})


# ------------------------------------------------------------
# 3. Hierarchical FMM (no UAV)
# ------------------------------------------------------------
class HierarchicalFMMBaseline:
    name = "FMM_hier"

    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels

    def run(self, scenario: ScenarioRecord,
            viscosity_params: Optional[ViscosityParams] = None,
            **kwargs) -> BaselineResult:
        planner = HierarchicalPathPlanner(
            terrain=scenario.terrain.astype(np.float64),
            fog_data=scenario.fog.astype(np.float64),
            n_levels=self.n_levels,
            viscosity_params=viscosity_params,
            spacing=scenario.spacing,
        )
        t0 = time.time()
        path, stats = planner.plan(scenario.start, scenario.goal)
        t = time.time() - t0
        ok = stats.get("success", False) and len(path) > 0
        cost = stats.get("path_cost", math.inf) if ok else math.inf
        return BaselineResult(self.name, ok, path, cost, t)


# ------------------------------------------------------------
# 4. UAV-aware Hierarchical FMM
# ------------------------------------------------------------
class UAVHierarchicalFMMBaseline:
    name = "FMM_hier_UAV"

    def __init__(self,
                 n_levels: int = 3,
                 kinematics: Optional[UAVKinematics] = None,
                 do_smoothing: bool = True):
        self.n_levels = n_levels
        self.kin = kinematics or UAVKinematics()
        self.do_smoothing = do_smoothing

    def run(self, scenario: ScenarioRecord,
            viscosity_params: Optional[ViscosityParams] = None,
            **kwargs) -> BaselineResult:
        planner = HierarchicalPathPlanner(
            terrain=scenario.terrain.astype(np.float64),
            fog_data=scenario.fog.astype(np.float64),
            n_levels=self.n_levels,
            viscosity_params=viscosity_params,
            spacing=scenario.spacing,
        )
        attach_uav_constraints(planner, self.kin, scenario.spacing)
        t0 = time.time()
        path, stats = planner.plan(scenario.start, scenario.goal)
        if path and self.do_smoothing:
            path = enforce_turn_radius(path, self.kin, scenario.spacing)
        t = time.time() - t0
        ok = stats.get("success", False) and len(path) > 0
        cost = stats.get("path_cost", math.inf) if ok else math.inf
        feas = check_path_feasibility(path, self.kin, scenario.spacing) if path else {}
        return BaselineResult(self.name, ok, path, cost, t, feasibility=feas)
