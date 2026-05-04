"""
UAV Kinematic Constraints for Viscous Hierarchical FMM
======================================================

提供针对无人机（UAV/Robot）的较完整运动学约束并以"速度场修饰函数 +
路径后处理 + 可行性检查"三段式接入到现有 HierarchicalPathPlanner / FMM 求解器。

约束维度：
  1. 水平 / 爬升 / 下降 速度上限     (vmax_horizontal / vmax_climb / vmax_descent)
  2. 最小转弯半径                    (min_turn_radius)  -> 路径后处理 + 可行性指标
  3. 下一刻相对当前飞行方向的最大速度  (角度相关速度场)
  4. 速度与海拔(高度)的关系          (altitude_speed_factor)
  5. 地形坡度可行性                  (max_slope_angle)
  6. 最大俯仰角                       (max_pitch_angle)

接入策略：
  * FMM 求解器是各向同性的，无法直接利用方向相关速度。
    我们选择"保守化"做法：把方向因子的最坏情形上界吸收进各向同性速度场，
    这样 FMM 仍能在保守可行集合中搜索。
  * 转弯半径和路径平滑由 ``enforce_turn_radius`` 在路径产生后处理。
  * 可行性指标由 ``check_path_feasibility`` 给出，便于实验对比。

接口最小：
  >>> kin = UAVKinematics()
  >>> mod_speed = apply_uav_speed_modifier(speed_field, terrain, kin)
  >>> planner.pyramid.speed_levels[0] = mod_speed   # 接入 HierarchicalPathPlanner
  >>> path, _ = planner.plan(start, goal)
  >>> path = enforce_turn_radius(path, kin, spacing=(1.0,1.0,1.0))
  >>> metrics = check_path_feasibility(path, kin, spacing=(1.0,1.0,1.0))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------
# 1. Kinematic configuration
# ------------------------------------------------------------
@dataclass
class UAVKinematics:
    """无人机运动学/动力学约束集合（默认值粗略对应中型多旋翼）。"""

    # --- 速度上限 ---
    vmax_horizontal: float = 20.0       # m/s
    vmax_climb: float = 5.0             # m/s, 仅指 +z 方向
    vmax_descent: float = 8.0           # m/s, 仅指 -z 方向, 取正值
    vmin: float = 0.5                   # m/s, 防止零速度（FMM 中除零）

    # --- 转弯/角速率 ---
    min_turn_radius: float = 6.0        # m, 水平最小转弯半径
    max_yaw_rate: float = math.radians(120.0)   # rad/s, 偏航
    max_pitch_angle: float = math.radians(30.0) # rad

    # --- 角度-速度衰减 ---
    # 给定方向变化角 θ ∈ [0, π]，下一刻最大速度 = vmax * angle_speed_factor(θ)
    # 默认采用平滑余弦衰减： cos(θ/2)^k
    angle_decay_exponent: float = 2.0

    # 当方向变化大于 angle_speed_breakpoint 时直接被视为"几乎停车"
    angle_speed_breakpoint: float = math.radians(120.0)

    # --- 海拔-速度衰减 ---
    # 简单线性衰减： factor = max(eps, 1 - altitude_decay_per_m * (z - altitude_ref))
    # 高于 altitude_ceiling 时视为不可飞
    altitude_ref: float = 0.0           # m, 起始衰减高度
    altitude_decay_per_m: float = 5e-4  # 每米衰减比例
    altitude_ceiling: float = 4000.0    # m, 飞行升限
    altitude_floor_eps: float = 0.1     # 高海拔最小保留速度系数

    # --- 地形坡度（决定可飞voxel）---
    max_slope_angle: float = math.radians(75.0)  # rad, 大于此角度的坡度视为禁飞

    # --- 数值参数 ---
    smoothing_passes: int = 2           # enforce_turn_radius 平滑迭代次数
    feasibility_dt: float = 1.0         # check_path_feasibility 中默认两点时间间隔

    # ----- 派生量 -----
    @property
    def max_curvature(self) -> float:
        """最大曲率 = 1 / R_min。"""
        return 1.0 / max(self.min_turn_radius, 1e-6)


# ------------------------------------------------------------
# 2. Single-value primitives  (用于 cost / 后处理 / 评估)
# ------------------------------------------------------------
def angle_speed_factor(theta_rad: float, kin: UAVKinematics) -> float:
    """方向变化角 -> 速度系数 ∈ [eps, 1].

    采用平滑曲线：
        f(θ) = cos(θ/2) ** k          (θ < θ_break)
        f(θ) = eps                    (θ >= θ_break)
    """
    if theta_rad >= kin.angle_speed_breakpoint:
        return 0.05
    base = math.cos(min(theta_rad, math.pi) / 2.0)
    return max(0.05, base ** kin.angle_decay_exponent)


def altitude_speed_factor(z_meters: float, kin: UAVKinematics) -> float:
    """高度 -> 速度系数 ∈ [eps, 1]."""
    if z_meters >= kin.altitude_ceiling:
        return kin.altitude_floor_eps
    delta = max(0.0, z_meters - kin.altitude_ref)
    return max(kin.altitude_floor_eps, 1.0 - kin.altitude_decay_per_m * delta)


def slope_feasibility_factor(slope_rad: float, kin: UAVKinematics) -> float:
    """坡度 -> 速度系数 ∈ [0, 1]，越接近 max_slope_angle 衰减越剧烈."""
    if slope_rad >= kin.max_slope_angle:
        return 0.0
    ratio = slope_rad / kin.max_slope_angle
    return max(0.0, 1.0 - ratio * ratio)


def vertical_speed_cap(direction_vec: Sequence[float], kin: UAVKinematics) -> float:
    """给定单位方向向量 (dz, dy, dx) 时，整体速度上限 = min(vmax_horizontal,
    爬升/下降约束派生的整体速度).

    若 dz > 0（爬升）：v <= vmax_climb / dz  （限制爬升分量）
    若 dz < 0（下降）：v <= vmax_descent / |dz|
    """
    if abs(direction_vec[0]) < 1e-9:
        return kin.vmax_horizontal

    dz = direction_vec[0]
    if dz > 0:
        return min(kin.vmax_horizontal, kin.vmax_climb / dz)
    return min(kin.vmax_horizontal, kin.vmax_descent / abs(dz))


# ------------------------------------------------------------
# 3. Speed field modifier  (FMM hook)
# ------------------------------------------------------------
def apply_uav_speed_modifier(
    speed_field: np.ndarray,
    terrain: Optional[np.ndarray],
    kin: UAVKinematics,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    altitude_axis: int = 0,
    use_terrain_as_altitude: bool = False,
    conservative_anisotropy: bool = True,
) -> np.ndarray:
    """把 UAV 约束吸收进各向同性速度场。

    Args:
        speed_field: 已有的（粘滞）速度场，形状 (nz,ny,nx) 或 (ny,nx)。
        terrain: 用于 ``use_terrain_as_altitude`` 时取每点高度（2D网格），可为 None。
        kin: ``UAVKinematics`` 实例。
        spacing: 网格间距，米/cell。
        altitude_axis: 体素网格中表示"高度"的轴索引（默认 z 轴=0）。
        use_terrain_as_altitude: 如果 terrain 是 2D（DEM），则用 terrain[y,x] 作为高度。
        conservative_anisotropy: 是否乘以保守的方向最坏情形系数（向心方向相对最大可能转角）。

    Returns:
        ``speed_modified``：与 speed_field 同形状，已乘上若干因子。
    """
    sf = np.asarray(speed_field, dtype=np.float64).copy()
    ndim = sf.ndim

    # 3.1 海拔速度因子
    if ndim == 3:
        nz = sf.shape[altitude_axis]
        z_indices = np.arange(nz)
        z_meters = z_indices * spacing[altitude_axis]
        alt_factor = np.array([altitude_speed_factor(z, kin) for z in z_meters])
        # broadcast 到 (nz,1,1) 等
        shape = [1, 1, 1]
        shape[altitude_axis] = nz
        sf *= alt_factor.reshape(shape)
    elif ndim == 2 and use_terrain_as_altitude and terrain is not None:
        # terrain 用作 DEM, terrain[y,x] = altitude
        alt_factor = np.vectorize(lambda z: altitude_speed_factor(z, kin))(terrain)
        sf *= alt_factor

    # 3.2 坡度可行性因子（基于 terrain 梯度）
    if terrain is not None:
        slope_rad = _terrain_slope_radians(terrain, spacing)
        # 把 slope 广播到 sf 形状
        if slope_rad.ndim == ndim:
            slope_factor = np.vectorize(
                lambda s: slope_feasibility_factor(s, kin)
            )(slope_rad)
            sf *= slope_factor
        elif slope_rad.ndim == ndim - 1:
            # terrain 是 DEM (2D)，speed_field 是 3D — 广播到所有 z
            slope_factor = np.vectorize(
                lambda s: slope_feasibility_factor(s, kin)
            )(slope_rad)
            shape = list(sf.shape)
            shape[altitude_axis] = 1
            sf *= slope_factor.reshape(shape)

    # 3.3 方向最坏情形保守系数（cos(60°)^k = 0.5^k 默认 k=2 -> 0.25）
    if conservative_anisotropy:
        # 假设 FMM 中每步可能出现最大 60° 转向（六邻域最大单步角度变化）
        worst_theta = math.radians(60.0)
        worst_factor = angle_speed_factor(worst_theta, kin)
        sf *= worst_factor

    # 3.4 vmin 下限
    sf = np.maximum(sf, kin.vmin / max(kin.vmax_horizontal, 1e-6))

    # 3.5 整体水平速度上限缩放：把粘滞 speed_field 解释为 [0,1] 的因子，
    #     乘上 vmax_horizontal 得到真实物理速度（仅在 speed_field<=1 时有效）。
    if sf.max() <= 1.0 + 1e-6:
        sf *= kin.vmax_horizontal

    return sf


def _terrain_slope_radians(terrain: np.ndarray,
                           spacing: Tuple[float, ...]) -> np.ndarray:
    """坡度（相对水平面的角度），返回与 terrain 同形状的弧度数组。"""
    arr = terrain.astype(np.float64)
    spacings = list(spacing[-arr.ndim:])
    if arr.ndim == 1:
        grad_mag = np.abs(np.gradient(arr, spacings[0]))
    else:
        grads = np.gradient(arr, *spacings)
        if not isinstance(grads, (list, tuple)):
            grads = (grads,)
        grad_mag = np.sqrt(sum(g ** 2 for g in grads))
    return np.arctan(grad_mag)


# ------------------------------------------------------------
# 4. 简易 "modifier" : 一行调用接入 FMM
# ------------------------------------------------------------
def make_fmm_speed_modifier(kin: UAVKinematics,
                            spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
                            altitude_axis: int = 0,
                            use_terrain_as_altitude: bool = False,
                            ) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """生成一个 ``modifier(speed_field, terrain) -> speed_field`` 闭包。

    这是给 ``HierarchicalPathPlanner`` / ``MultiPathPlanner`` 用的最简接入：
    在构造规划器后逐层调用 modifier 即可。
    """
    def modifier(speed_field: np.ndarray,
                 terrain: Optional[np.ndarray] = None) -> np.ndarray:
        return apply_uav_speed_modifier(
            speed_field, terrain, kin,
            spacing=spacing,
            altitude_axis=altitude_axis,
            use_terrain_as_altitude=use_terrain_as_altitude,
            conservative_anisotropy=True,
        )
    return modifier


def attach_uav_constraints(planner,
                           kin: UAVKinematics,
                           spacing: Optional[Tuple[float, ...]] = None,
                           altitude_axis: int = 0) -> None:
    """把 UAV 约束**就地**接入到 HierarchicalPathPlanner 或 MultiPathPlanner。

    会把每个金字塔层级的 speed_field 用 ``apply_uav_speed_modifier`` 修饰一次。
    """
    pyramid = getattr(planner, "pyramid", None)
    if pyramid is None and hasattr(planner, "planner"):
        pyramid = planner.planner.pyramid
    if pyramid is None:
        raise ValueError("planner 上找不到 pyramid 属性")

    base_spacing = spacing or pyramid.spacings[0]
    for level in range(len(pyramid.speed_levels)):
        sp = pyramid.spacings[level]
        terr = pyramid.levels[level]
        pyramid.speed_levels[level] = apply_uav_speed_modifier(
            pyramid.speed_levels[level],
            terr,
            kin,
            spacing=sp,
            altitude_axis=altitude_axis,
            conservative_anisotropy=True,
        )

    # 同步 MultiPathPlanner 内部缓存（如果存在）
    if hasattr(planner, "_original_speed_levels"):
        planner._original_speed_levels = [
            level.copy() for level in pyramid.speed_levels
        ]


# ------------------------------------------------------------
# 5. 路径后处理：转弯半径平滑 + 爬升率限速
# ------------------------------------------------------------
def enforce_turn_radius(path: Sequence[Sequence[float]],
                        kin: UAVKinematics,
                        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
                        resample_step_m: Optional[float] = None,
                        ) -> List[Tuple[float, ...]]:
    """通过弧长重采样 + 移动平均平滑限制路径曲率。

    步骤:
      1. 把原始路径按弧长重采样（默认每 R_min/2 米一个点），避免密集采样导致
         曲率估计噪声过大。
      2. 多轮平滑：对每个内部点，若其曲率 > 1/R_min，则把该点向其前后两邻点
         的中点方向拉拢，直到收敛或达到最大轮数。
      3. 端点保留（起终点固定）。

    Args:
        path: [(z,y,x), ...] 浮点路径点列表（原始分辨率坐标）
        kin: 运动学
        spacing: 网格间距 -> 米
        resample_step_m: 重采样步长（米），默认 R_min/2

    Returns:
        平滑后的路径
    """
    if len(path) < 3:
        return [tuple(map(float, p)) for p in path]

    pts = np.array(path, dtype=np.float64)
    spc = np.array(spacing[: pts.shape[1]], dtype=np.float64)

    R_min = max(kin.min_turn_radius, 1e-3)
    step_m = resample_step_m if resample_step_m is not None else max(R_min / 2.0, 0.5)

    # ---- 1. 弧长重采样 ----
    pts_m = pts * spc
    seg_lens = np.linalg.norm(np.diff(pts_m, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(arc[-1])
    if total < step_m * 2:
        return [tuple(map(float, p)) for p in pts]

    n_samples = max(3, int(round(total / step_m)) + 1)
    sample_arc = np.linspace(0.0, total, n_samples)
    resampled_m = np.empty((n_samples, pts.shape[1]), dtype=np.float64)
    for d in range(pts.shape[1]):
        resampled_m[:, d] = np.interp(sample_arc, arc, pts_m[:, d])

    # ---- 2. 多轮收敛平滑 ----
    max_passes = max(20, kin.smoothing_passes * 20)
    inv_R = 1.0 / R_min
    for _ in range(max_passes):
        worst = 0.0
        for i in range(1, n_samples - 1):
            a, b, c = resampled_m[i - 1], resampled_m[i], resampled_m[i + 1]
            kappa = _three_point_curvature(a, b, c)
            worst = max(worst, kappa)
            if kappa > inv_R:
                target = (a + c) * 0.5
                excess = (kappa - inv_R) * R_min
                t = min(0.6, 0.3 + 0.4 * excess)
                resampled_m[i] = (1.0 - t) * b + t * target
        if worst <= inv_R * 1.05:
            break

    # ---- 3. 转回 voxel 坐标 ----
    resampled = resampled_m / spc
    return [tuple(map(float, p)) for p in resampled]


def _three_point_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """三点圆弧曲率近似 = 4 * Area(triangle) / (|ab|*|bc|*|ca|)."""
    ab = b - a
    bc = c - b
    ca = a - c
    # |ab x bc| 在 3D 中是叉乘模长
    if a.shape[0] == 3:
        cross = np.cross(ab, bc)
        area2 = np.linalg.norm(cross)
    else:
        area2 = abs(ab[0] * bc[1] - ab[1] * bc[0])
    denom = np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ca)
    if denom < 1e-9:
        return 0.0
    return 2.0 * area2 / denom


# ------------------------------------------------------------
# 6. Feasibility metrics
# ------------------------------------------------------------
def check_path_feasibility(path: Sequence[Sequence[float]],
                           kin: UAVKinematics,
                           spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
                           dt: Optional[float] = None,
                           ) -> Dict[str, float]:
    """给出路径相对 UAV 约束的可行性指标。

    所有距离使用 ``spacing`` 转换为米；速度假设两点之间用时 ``dt`` 秒。
    若 ``dt`` 为 None，默认 ``kin.feasibility_dt``。

    Returns:
        dict 包含：
          n_points, total_distance_m, total_time_s,
          max_speed_mps, max_climb_rate_mps, max_descent_rate_mps,
          min_curvature_radius_m, max_pitch_rad,
          n_violations(汇总), per_violation: dict
    """
    if dt is None:
        dt = kin.feasibility_dt

    if len(path) < 2:
        return {"n_points": len(path), "total_distance_m": 0.0,
                "total_time_s": 0.0, "n_violations": 0}

    pts = np.array(path, dtype=np.float64)
    spc = np.array(spacing[: pts.shape[1]], dtype=np.float64)
    seg = np.diff(pts * spc, axis=0)  # 米
    seg_len = np.linalg.norm(seg, axis=1)
    total_distance = float(seg_len.sum())
    total_time = total_distance / max(kin.vmax_horizontal, 1e-6)

    # 速度：假设两点间用最快可行速度 -> v = seg_len / dt
    speeds = seg_len / max(dt, 1e-6)
    max_speed = float(speeds.max())

    # 爬升率：seg[:,0] 是 z 方向米
    climb = seg[:, 0] / max(dt, 1e-6) if pts.shape[1] >= 3 else np.zeros(len(seg))
    max_climb = float(np.maximum(0, climb).max(initial=0.0))
    max_descent = float(-np.minimum(0, climb).min(initial=0.0))

    # 俯仰角：atan2(dz, sqrt(dy^2+dx^2))
    if pts.shape[1] == 3:
        horiz = np.sqrt(seg[:, 1] ** 2 + seg[:, 2] ** 2)
        pitch = np.arctan2(np.abs(seg[:, 0]), np.maximum(horiz, 1e-9))
        max_pitch = float(pitch.max(initial=0.0))
    else:
        max_pitch = 0.0

    # 曲率
    curvs: List[float] = []
    for i in range(1, len(pts) - 1):
        a = pts[i - 1] * spc
        b = pts[i] * spc
        c = pts[i + 1] * spc
        curvs.append(_three_point_curvature(a, b, c))
    max_curv = max(curvs) if curvs else 0.0
    min_radius = (1.0 / max_curv) if max_curv > 1e-9 else float("inf")

    # 违例统计
    viol = {
        "speed": int(np.sum(speeds > kin.vmax_horizontal * 1.01)),
        "climb_rate": int(np.sum(climb > kin.vmax_climb * 1.01)),
        "descent_rate": int(np.sum(-climb > kin.vmax_descent * 1.01)),
        "pitch_angle": int(max_pitch > kin.max_pitch_angle * 1.01),
        "turn_radius": int(min_radius < kin.min_turn_radius * 0.99),
    }
    n_viol = sum(viol.values())

    return {
        "n_points": len(path),
        "total_distance_m": total_distance,
        "total_time_s": total_time,
        "max_speed_mps": max_speed,
        "max_climb_rate_mps": max_climb,
        "max_descent_rate_mps": max_descent,
        "min_curvature_radius_m": min_radius,
        "max_pitch_rad": max_pitch,
        "n_violations": int(n_viol),
        "per_violation": viol,
    }


# ------------------------------------------------------------
# 7. self-test (smoke)
# ------------------------------------------------------------
def _smoke_test() -> None:
    """运行最小烟雾测试。"""
    print("[uav_constraints] smoke test ...")
    kin = UAVKinematics()
    # angle factor sanity
    assert angle_speed_factor(0.0, kin) > 0.99
    assert angle_speed_factor(math.pi, kin) < 0.1
    # altitude factor sanity
    assert altitude_speed_factor(0.0, kin) > 0.99
    assert altitude_speed_factor(kin.altitude_ceiling + 100, kin) < 0.5
    # speed field modifier
    sf = np.ones((10, 20, 20))
    terr = np.zeros((20, 20))
    mod = apply_uav_speed_modifier(sf, terr, kin, spacing=(1.0, 1.0, 1.0))
    assert mod.shape == sf.shape
    # path: a sharp 90-degree turn
    path = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0),
            (0.0, 1.0, 2.0), (0.0, 2.0, 2.0)]
    smoothed = enforce_turn_radius(path, kin, spacing=(1.0, 1.0, 1.0))
    assert len(smoothed) == len(path)
    metrics = check_path_feasibility(smoothed, kin, spacing=(1.0, 1.0, 1.0))
    print("  metrics:", {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in metrics.items() if k != "per_violation"})
    print("[uav_constraints] OK")


if __name__ == "__main__":
    _smoke_test()
