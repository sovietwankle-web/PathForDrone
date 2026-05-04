"""
Online Replanning with Real-Time Weather Updates
================================================

针对 ``HierarchicalPathPlanner`` 的闭环在线重规划框架。

设计目标：
  * 把动态环境 (天气/雾/风) 视为时变速度场扰动；
  * 在 UAV 沿当前规划轨迹前进时持续轮询新天气；
  * 当扰动超过阈值，或前方走廊速度被压低到危险水平时触发重规划；
  * 重规划复用粘滞分层金字塔的地形部分，仅更新雾气/速度场并从当前位置出发，
    避免冷启动。

核心组件：
  * ``WeatherSource``      抽象基类，提供 ``get_fog(t)`` 等
  * ``SyntheticWeatherSource`` 时变高斯雾团 + 风扰动的合成实现
  * ``ReplanTrigger``      触发判据（差异阈值/cost 增长/距离）
  * ``OnlineReplanner``    闭环执行器，返回完整轨迹与重规划日志

使用：
    >>> source = SyntheticWeatherSource(shape=terrain.shape, seed=0)
    >>> replanner = OnlineReplanner(terrain=terrain, weather_source=source,
    ...                             kinematics=kin)
    >>> trace = replanner.run(start, goal, t_max=120.0)
    >>> print(trace.summary())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# ------------------------------------------------------------
# 1. Weather sources
# ------------------------------------------------------------
class WeatherSource:
    """实时天气数据源抽象。"""

    def get_fog(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def get_wind(self, t: float) -> Optional[np.ndarray]:
        """可选：风速场（同形状或形状 + 向量分量）"""
        return None


class SyntheticWeatherSource(WeatherSource):
    """合成时变天气：若干随机游走的雾团叠加。

    每个雾团是高斯椭球，中心按布朗运动漂移，幅值按正弦呼吸。
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 n_clouds: int = 4,
                 seed: int = 0,
                 cloud_radius: Tuple[float, float] = (3.0, 8.0),
                 drift_speed: float = 0.5,
                 amplitude: float = 1.0):
        self.shape = shape
        self.rng = np.random.default_rng(seed)
        self.amplitude = amplitude
        self.drift_speed = drift_speed
        # 初始化云团
        self._centers0 = self.rng.uniform(
            low=[s * 0.1 for s in shape],
            high=[s * 0.9 for s in shape],
            size=(n_clouds, len(shape))
        )
        self._radii = self.rng.uniform(*cloud_radius, size=n_clouds)
        self._phases = self.rng.uniform(0, 2 * math.pi, size=n_clouds)
        self._drift = self.rng.normal(0, 1, size=(n_clouds, len(shape)))
        # 记忆已生成的雾场，用于增量更新
        self._cache: Dict[float, np.ndarray] = {}
        self._coords = np.indices(shape)  # (ndim, *shape)

    def _centers_at(self, t: float) -> np.ndarray:
        # 缓慢漂移 + 周期扰动
        c = self._centers0.copy()
        c += self._drift * self.drift_speed * t
        # 边界反射
        for d, s in enumerate(self.shape):
            c[:, d] = np.clip(c[:, d], 0, s - 1)
        return c

    def get_fog(self, t: float) -> np.ndarray:
        if t in self._cache:
            return self._cache[t]
        centers = self._centers_at(t)
        fog = np.zeros(self.shape, dtype=np.float64)
        for k, center in enumerate(centers):
            r = self._radii[k]
            amp = 0.5 + 0.5 * math.sin(0.3 * t + self._phases[k])
            sq = sum((self._coords[d] - center[d]) ** 2
                     for d in range(len(self.shape)))
            fog += self.amplitude * amp * np.exp(-sq / (2 * r * r))
        self._cache[t] = fog
        # 限制缓存大小
        if len(self._cache) > 16:
            self._cache.pop(next(iter(self._cache)))
        return fog


class StaticWeatherSource(WeatherSource):
    """常量天气，用作对照组。"""

    def __init__(self, fog: np.ndarray):
        self._fog = fog

    def get_fog(self, t: float) -> np.ndarray:
        return self._fog


# ------------------------------------------------------------
# 2. Replan trigger
# ------------------------------------------------------------
@dataclass
class ReplanTrigger:
    """重规划触发判据。"""

    # 雾场相对变化（L2 范数）阈值
    fog_delta_threshold: float = 0.15
    # 距上次规划的最小时间间隔（避免过频）
    min_replan_interval: float = 2.0
    # 若前方走廊平均速度下降比例超过此值则触发
    speed_drop_threshold: float = 0.3
    # 强制周期（即使无显著变化也偶尔重规划）
    periodic_interval: Optional[float] = 30.0

    def should_replan(self,
                      t: float,
                      t_last: float,
                      fog_delta_l2: float,
                      speed_drop_ratio: float) -> Tuple[bool, str]:
        if t - t_last < self.min_replan_interval:
            return False, ""
        if fog_delta_l2 >= self.fog_delta_threshold:
            return True, f"fog_delta={fog_delta_l2:.3f}"
        if speed_drop_ratio >= self.speed_drop_threshold:
            return True, f"speed_drop={speed_drop_ratio:.3f}"
        if (self.periodic_interval is not None
                and t - t_last >= self.periodic_interval):
            return True, f"periodic({self.periodic_interval}s)"
        return False, ""


# ------------------------------------------------------------
# 3. Trajectory record
# ------------------------------------------------------------
@dataclass
class ReplanEvent:
    t: float
    position: Tuple[float, ...]
    reason: str
    new_path_length: int
    plan_time_s: float


@dataclass
class TrajectoryRecord:
    waypoints: List[Tuple[float, ...]] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    replan_events: List[ReplanEvent] = field(default_factory=list)
    success: bool = False
    final_distance_to_goal: float = math.inf
    total_replan_time_s: float = 0.0
    feasibility: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "n_waypoints": len(self.waypoints),
            "n_replans": len(self.replan_events),
            "total_replan_time_s": self.total_replan_time_s,
            "final_distance_to_goal": self.final_distance_to_goal,
            "duration_s": self.times[-1] if self.times else 0.0,
            "feasibility": self.feasibility,
        }


# ------------------------------------------------------------
# 4. Online replanner
# ------------------------------------------------------------
class OnlineReplanner:
    """带重规划的闭环执行器。

    核心逻辑：
      while not at_goal and t < t_max:
          fog_now = source.get_fog(t)
          if trigger.should_replan(...):  replan from current position
          step along path (~ kin.vmax_horizontal * dt)
    """

    def __init__(self,
                 terrain: np.ndarray,
                 weather_source: WeatherSource,
                 kinematics: Optional[UAVKinematics] = None,
                 viscosity_params: Optional[ViscosityParams] = None,
                 trigger: Optional[ReplanTrigger] = None,
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
                 n_levels: int = 3,
                 dt: float = 1.0):
        self.terrain = terrain
        self.weather = weather_source
        self.kin = kinematics or UAVKinematics()
        self.visc_params = viscosity_params or ViscosityParams()
        self.trigger = trigger or ReplanTrigger()
        self.spacing = spacing
        self.n_levels = n_levels
        self.dt = dt
        self._last_fog: Optional[np.ndarray] = None

        # 一次性构造带无雾基准的 planner，并缓存 UAV-修饰后的"静态"速度场。
        # 重规划时只需把 fog 注入再 plan，不需要重新构造金字塔。
        self._planner: HierarchicalPathPlanner = HierarchicalPathPlanner(
            terrain=terrain,
            fog_data=None,
            n_levels=n_levels,
            viscosity_params=self.visc_params,
            spacing=spacing,
        )
        self._base_speed_levels: List[np.ndarray] = []
        self._fog_pyramid_shapes: List[Tuple[int, ...]] = []
        for level in range(n_levels):
            sp = self._planner.pyramid.spacings[level]
            base = apply_uav_speed_modifier(
                self._planner.pyramid.speed_levels[level],
                self._planner.pyramid.levels[level],
                self.kin,
                spacing=sp,
                conservative_anisotropy=True,
            )
            self._base_speed_levels.append(base)
            self._fog_pyramid_shapes.append(self._planner.pyramid.shapes[level])

    # --- internal helpers ---
    def _inject_fog(self, fog: np.ndarray) -> None:
        """在缓存的静态速度场上叠加雾气惩罚，写入 planner 的金字塔。"""
        from scipy.ndimage import zoom
        fog_max = fog.max() + 1e-10
        fog_norm = fog / fog_max
        gamma = self.visc_params.fog_weight
        for level in range(self.n_levels):
            target_shape = self._fog_pyramid_shapes[level]
            if level == 0:
                fog_l = fog_norm
            else:
                factors = tuple(t / s for t, s in zip(target_shape, fog_norm.shape))
                fog_l = zoom(fog_norm, factors, order=1)
                # 数值稳健性
                fog_l = np.clip(fog_l, 0.0, 1.0)
            penalty = 1.0 / (1.0 + gamma * fog_l)
            self._planner.pyramid.speed_levels[level] = (
                self._base_speed_levels[level] * penalty
            )

    def _plan_from(self, current: Tuple[float, ...],
                   goal: Tuple[int, ...],
                   fog: np.ndarray) -> Tuple[List[Tuple[float, ...]], float]:
        cur_int = tuple(int(round(c)) for c in current)
        cur_int = tuple(max(0, min(c, s - 1))
                        for c, s in zip(cur_int, self.terrain.shape))
        t0 = time.time()
        self._inject_fog(fog)
        path, _ = self._planner.plan(cur_int, goal)
        plan_time = time.time() - t0
        if path:
            path = enforce_turn_radius(path, self.kin, self.spacing)
        return path, plan_time

    def _eval_speed_drop(self, path: List[Tuple[float, ...]],
                         fog_new: np.ndarray) -> float:
        """估计当前规划路径走廊上的平均雾气增长（替代直接重算速度场，避免开销）."""
        if not path or self._last_fog is None:
            return 0.0
        head = path[: max(2, len(path) // 2)]
        idxs = [tuple(max(0, min(int(round(c)), s - 1))
                       for c, s in zip(p, self.terrain.shape))
                for p in head]
        if self._last_fog.max() < 1e-6 and fog_new.max() < 1e-6:
            return 0.0
        old = np.array([self._last_fog[i] for i in idxs])
        new = np.array([fog_new[i] for i in idxs])
        # 雾气均值上升 -> 等价速度下降；归一到 [0,1]
        denom = max(old.mean(), new.mean(), 1e-3)
        return float(max(0.0, (new.mean() - old.mean()) / denom))

    # --- public API ---
    def run(self,
            start: Tuple[int, ...],
            goal: Tuple[int, ...],
            t_max: float = 120.0,
            max_iterations: int = 5000,
            stuck_distance: float = 0.05,
            stuck_window: int = 10,
            on_step: Optional[Callable[[float, Tuple[float, ...]], None]] = None,
            ) -> TrajectoryRecord:
        """执行闭环仿真。

        终止条件:
          * 到达 goal (距离 < 1.5 voxel)
          * t >= t_max
          * 迭代数 >= max_iterations
          * 检测到"卡住"：最近 stuck_window 步累积位移 < stuck_distance
        """
        rec = TrajectoryRecord()
        t = 0.0
        position = tuple(float(s) for s in start)
        rec.waypoints.append(position)
        rec.times.append(t)

        # 初始规划
        fog0 = self.weather.get_fog(0.0)
        self._last_fog = fog0.copy()
        path, plan_time = self._plan_from(position, goal, fog0)
        rec.total_replan_time_s += plan_time
        rec.replan_events.append(ReplanEvent(
            t=0.0, position=position, reason="initial",
            new_path_length=len(path), plan_time_s=plan_time,
        ))
        if not path:
            rec.success = False
            rec.final_distance_to_goal = _euclid(position, goal)
            return rec

        last_replan_t = 0.0
        path_idx = 0
        iteration = 0
        max_replans = 12  # 上限：避免无限循环

        while t < t_max and iteration < max_iterations:
            iteration += 1
            if path_idx >= len(path) - 1:
                if _euclid(position, goal) < 1.5:
                    rec.success = True
                    break
                if len(rec.replan_events) >= max_replans:
                    break
                fog_now = self.weather.get_fog(t)
                path, plan_time = self._plan_from(position, goal, fog_now)
                rec.total_replan_time_s += plan_time
                rec.replan_events.append(ReplanEvent(
                    t=t, position=position, reason="path_exhausted",
                    new_path_length=len(path), plan_time_s=plan_time,
                ))
                last_replan_t = t
                self._last_fog = fog_now
                path_idx = 0
                if not path:
                    break
                continue

            target = np.array(path[path_idx + 1])
            cur = np.array(position)
            seg = target - cur
            seg_m = seg * np.array(self.spacing[: len(seg)])
            dist_m = float(np.linalg.norm(seg_m))
            step_m = self.kin.vmax_horizontal * self.dt
            if dist_m < step_m:
                position = tuple(target)
                path_idx += 1
                t += dist_m / max(self.kin.vmax_horizontal, 1e-6)
            else:
                cur = cur + seg * (step_m / dist_m)
                position = tuple(cur)
                t += self.dt

            rec.waypoints.append(position)
            rec.times.append(t)
            if on_step is not None:
                on_step(t, position)

            if _euclid(position, goal) < 1.5:
                rec.success = True
                break

            # 卡住检测
            if len(rec.waypoints) > stuck_window:
                tail = rec.waypoints[-stuck_window:]
                disp = float(np.linalg.norm(
                    np.array(tail[-1]) - np.array(tail[0])))
                if disp < stuck_distance:
                    rec.replan_events.append(ReplanEvent(
                        t=t, position=position, reason="stuck",
                        new_path_length=0, plan_time_s=0.0,
                    ))
                    break

            # 检测天气变化
            fog_now = self.weather.get_fog(t)
            denom = float(np.linalg.norm(self._last_fog) + 1e-9)
            fog_delta = float(np.linalg.norm(fog_now - self._last_fog) / denom)
            speed_drop = self._eval_speed_drop(path[path_idx:], fog_now)
            should, reason = self.trigger.should_replan(
                t, last_replan_t, fog_delta, speed_drop
            )
            if should and len(rec.replan_events) < max_replans:
                new_path, plan_time = self._plan_from(position, goal, fog_now)
                rec.total_replan_time_s += plan_time
                rec.replan_events.append(ReplanEvent(
                    t=t, position=position, reason=reason,
                    new_path_length=len(new_path), plan_time_s=plan_time,
                ))
                last_replan_t = t
                self._last_fog = fog_now
                if new_path:
                    path = new_path
                    path_idx = 0

        rec.final_distance_to_goal = _euclid(position, goal)
        rec.feasibility = check_path_feasibility(
            rec.waypoints, self.kin, self.spacing
        )
        return rec


def _euclid(a, b) -> float:
    return float(np.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


# ------------------------------------------------------------
# 5. smoke test
# ------------------------------------------------------------
def _smoke_test() -> None:
    from viscous_hierarchical_fmm import create_test_terrain_3d
    print("[online_replanner] smoke test ...", flush=True)
    shape = (10, 15, 15)
    terrain = create_test_terrain_3d(shape)
    weather = SyntheticWeatherSource(shape=shape, n_clouds=2, seed=1,
                                     drift_speed=0.3)
    kin = UAVKinematics(min_turn_radius=3.0, vmax_horizontal=10.0)
    trigger = ReplanTrigger(fog_delta_threshold=0.05,
                            speed_drop_threshold=0.2,
                            min_replan_interval=2.0,
                            periodic_interval=15.0)
    replanner = OnlineReplanner(terrain=terrain, weather_source=weather,
                                 kinematics=kin, trigger=trigger,
                                 spacing=(1.0, 1.0, 1.0), n_levels=2, dt=1.0)
    print("  init done; running...", flush=True)
    trace = replanner.run((1, 2, 2), (8, 12, 12), t_max=30.0)
    s = trace.summary()
    print("  summary:", s, flush=True)
    print("  events:", [(round(e.t, 2), e.reason, round(e.plan_time_s, 3))
                         for e in trace.replan_events], flush=True)
    print("[online_replanner] OK", flush=True)


if __name__ == "__main__":
    _smoke_test()
