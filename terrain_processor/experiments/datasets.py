"""
Dataset Generation for Path-Planning Experiments
=================================================

为对比/消融实验和神经网络训练提供一组可复现的场景：
  * 随机起伏地形 + 高斯雾团
  * 起点/终点 (保证最低粘滞速度阈值)
  * 可选时变天气源 (用于在线重规划评估)

每个场景被序列化为 ``ScenarioRecord``。
"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------
# 1. 数据结构
# ------------------------------------------------------------
@dataclass
class ScenarioRecord:
    scenario_id: int
    terrain: np.ndarray
    fog: np.ndarray
    start: Tuple[int, ...]
    goal: Tuple[int, ...]
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    seed: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# 2. 地形 / 雾气合成
# ------------------------------------------------------------
def generate_terrain_3d(shape: Tuple[int, int, int],
                        rng: np.random.Generator,
                        n_peaks: int = 5,
                        peak_amp: float = 8.0,
                        peak_sigma: Tuple[float, float] = (3.0, 7.0)) -> np.ndarray:
    """3D 体素地形：在体积内放若干高斯山峰，作为障碍密度。"""
    terrain = np.zeros(shape, dtype=np.float64)
    nz, ny, nx = shape
    coords = np.indices(shape)
    for _ in range(n_peaks):
        cz = rng.uniform(0.1 * nz, 0.9 * nz)
        cy = rng.uniform(0.1 * ny, 0.9 * ny)
        cx = rng.uniform(0.1 * nx, 0.9 * nx)
        s = rng.uniform(*peak_sigma)
        amp = rng.uniform(0.4, 1.0) * peak_amp
        d2 = ((coords[0] - cz) ** 2 + (coords[1] - cy) ** 2
              + (coords[2] - cx) ** 2) / (s * s)
        terrain += amp * np.exp(-d2 / 2)
    # 略微平滑
    return terrain


def generate_fog_3d(shape: Tuple[int, int, int],
                    rng: np.random.Generator,
                    n_clouds: int = 3,
                    cloud_amp: float = 1.0,
                    sigma: Tuple[float, float] = (3.0, 6.0)) -> np.ndarray:
    fog = np.zeros(shape, dtype=np.float64)
    coords = np.indices(shape)
    for _ in range(n_clouds):
        c = [rng.uniform(0.1 * s, 0.9 * s) for s in shape]
        s = rng.uniform(*sigma)
        amp = rng.uniform(0.5, 1.0) * cloud_amp
        d2 = sum((coords[d] - c[d]) ** 2 for d in range(3)) / (s * s)
        fog += amp * np.exp(-d2 / 2)
    return fog


def _is_start_goal_ok(terrain: np.ndarray, p: Tuple[int, ...],
                      max_terrain: float) -> bool:
    """起点/终点不应位于山峰内部 (terrain[p] 太大)。"""
    return terrain[p] < max_terrain * 0.4


def sample_start_goal(terrain: np.ndarray,
                      rng: np.random.Generator,
                      min_distance: float = 10.0,
                      max_attempts: int = 200,
                      ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    shape = terrain.shape
    max_t = float(terrain.max())
    for _ in range(max_attempts):
        s = tuple(int(rng.integers(2, sh - 2)) for sh in shape)
        g = tuple(int(rng.integers(2, sh - 2)) for sh in shape)
        if not _is_start_goal_ok(terrain, s, max_t):
            continue
        if not _is_start_goal_ok(terrain, g, max_t):
            continue
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(s, g)))
        if d >= min_distance:
            return s, g
    # fallback：直接使用网格对角
    return ((1,) * len(shape),
            tuple(s - 2 for s in shape))


# ------------------------------------------------------------
# 3. 数据集生成
# ------------------------------------------------------------
def generate_dataset(n_scenarios: int = 20,
                     shape: Tuple[int, int, int] = (12, 18, 18),
                     seed: int = 0,
                     min_distance_ratio: float = 0.5,
                     ) -> List[ScenarioRecord]:
    """生成 ``n_scenarios`` 个场景。"""
    records: List[ScenarioRecord] = []
    base_rng = np.random.default_rng(seed)
    diag = math.sqrt(sum(s * s for s in shape))
    min_dist = diag * min_distance_ratio
    for i in range(n_scenarios):
        sub_seed = int(base_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)
        terrain = generate_terrain_3d(shape, rng,
                                      n_peaks=int(rng.integers(3, 8)))
        fog = generate_fog_3d(shape, rng,
                              n_clouds=int(rng.integers(2, 5)))
        start, goal = sample_start_goal(terrain, rng, min_distance=min_dist)
        records.append(ScenarioRecord(
            scenario_id=i,
            terrain=terrain.astype(np.float32),
            fog=fog.astype(np.float32),
            start=start, goal=goal,
            spacing=(1.0, 1.0, 1.0),
            seed=sub_seed,
            meta={"shape": shape, "min_distance": min_dist},
        ))
    return records


def save_dataset(records: List[ScenarioRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(records, f)


def load_dataset(path: str) -> List[ScenarioRecord]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------
# 4. CLI
# ------------------------------------------------------------
def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate path-planning dataset")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--shape", type=str, default="12,18,18")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="experiments_data/dataset.pkl")
    args = parser.parse_args()
    shape = tuple(int(x) for x in args.shape.split(","))
    records = generate_dataset(n_scenarios=args.n, shape=shape, seed=args.seed)
    save_dataset(records, args.out)
    print(f"saved {len(records)} scenarios to {args.out}")


if __name__ == "__main__":
    _cli()
