"""
Viscous Hierarchical Fast Marching Method

优化的路径规划算法实现，包含：
1. ViscosityField - 粘滞系数场计算（坡度、粗糙度、环境因子）
2. TerrainPyramid - 多分辨率地形金字塔
3. NarrowBandFMM - 内存优化的窄带快速行进法
4. HierarchicalPathPlanner - 分层路径规划主类

Author: Optimized Path Planning Implementation
"""

import numpy as np
import heapq
from typing import Tuple, List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from math import exp
import time


@dataclass
class ViscosityParams:
    """粘滞系数参数配置"""
    slope_weight: float = 0.3       # 坡度影响权重
    roughness_weight: float = 0.2   # 粗糙度影响权重
    fog_weight: float = 0.5         # 雾气/环境影响权重
    roughness_window: int = 3       # 粗糙度计算窗口大小
    min_viscosity: float = 0.01     # 最小粘滞系数（防止除零）


@dataclass
class MultiPathParams:
    """多路径规划参数"""
    n_paths: int = 3                    # 需要的路径数量 K
    penalty_weight: float = 0.5         # 惩罚强度 (0=无惩罚, 1=完全阻断)
    penalty_radius: int = 5             # 惩罚影响半径（原始分辨率像素）
    min_path_separation: float = 10.0   # 最小路径间距（欧氏距离）
    penalty_decay: str = 'gaussian'     # 衰减模式: 'gaussian' 或 'linear'
    max_cost_ratio: float = 3.0         # 最大允许代价比（相对首条路径）


@dataclass
class MultiPathResult:
    """多路径规划结果"""
    paths: List[List[Tuple[float, ...]]]            # K条路径
    costs: List[float]                               # 每条路径的代价
    stats: Dict[str, Any] = field(default_factory=dict)          # 汇总统计
    per_path_stats: List[Dict[str, Any]] = field(default_factory=list)  # 每条路径详细统计
    diversity_matrix: Optional[np.ndarray] = None    # K×K 路径间距矩阵


@dataclass
class GasDiffusionParams:
    """多机器人路径点分配参数（不互溶气体扩散模型）"""
    n_robots: int = 3                                       # 机器人/气体数量
    robot_starts: List[Tuple[int, ...]] = field(default_factory=list)  # 各机器人起点
    waypoints: List[Tuple[int, ...]] = field(default_factory=list)     # 待分配路径点
    max_waypoints_per_robot: Optional[int] = None           # 每机器人最大路径点数（None=不限）


@dataclass
class WaypointAllocationResult:
    """路径点分配结果"""
    assignments: Dict[int, List[Tuple[int, ...]]] = field(default_factory=dict)
    # robot_id → 按到达时间排序的路径点列表
    assignment_order: List[Tuple[int, int]] = field(default_factory=list)
    # (robot_id, waypoint_idx) 按全局时间排序
    robot_paths: Dict[int, List[List[Tuple[float, ...]]]] = field(default_factory=dict)
    # robot_id → 路径段列表 [start→wp1, wp1→wp2, ...]
    arrival_times: Dict[int, List[float]] = field(default_factory=dict)
    # robot_id → 各路径点到达时间
    territory_map: Optional[np.ndarray] = None
    # 领土地图：每个像素属于哪个机器人 (-1=未占领)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QValueMultiPathParams:
    """Q值多路径规划参数

    在目标区域内存在一个 Q 值场（得分场），多路径规划不仅考虑路径代价，
    还考虑终点在目标区域内的 Q 值。最终综合得分:
        score = q_weight × Q(goal) - cost_weight × path_cost（归一化后）
    选择 score 最大的路径作为最优路径。
    """
    n_paths: int = 5                    # 候选路径数量
    q_weight: float = 1.0              # Q值权重
    cost_weight: float = 0.5           # 路径代价权重
    penalty_weight: float = 0.5        # 多路径惩罚强度
    penalty_radius: int = 5            # 惩罚半径
    min_path_separation: float = 8.0   # 最小路径间距
    penalty_decay: str = 'gaussian'    # 衰减模式
    max_cost_ratio: float = 5.0        # 最大代价比（宽松，因为高Q值可能需要更远的路径）


@dataclass
class QValueMultiPathResult:
    """Q值多路径规划结果"""
    best_path: List[Tuple[float, ...]] = field(default_factory=list)
    # 最优路径（综合得分最高）
    best_goal: Tuple[int, ...] = field(default_factory=tuple)
    # 最优终点
    best_score: float = 0.0
    # 最优综合得分
    best_q_value: float = 0.0
    # 最优终点的 Q 值
    best_cost: float = 0.0
    # 最优路径的代价
    all_paths: List[List[Tuple[float, ...]]] = field(default_factory=list)
    # 所有候选路径
    all_goals: List[Tuple[int, ...]] = field(default_factory=list)
    # 所有候选终点
    all_scores: List[float] = field(default_factory=list)
    # 所有候选得分
    all_q_values: List[float] = field(default_factory=list)
    # 所有候选 Q 值
    all_costs: List[float] = field(default_factory=list)
    # 所有候选代价
    stats: Dict[str, Any] = field(default_factory=dict)


class ViscosityField:
    """
    粘滞系数场计算

    代价公式: C(x) = 1/f(x)
    其中 f(x) = base_speed × viscosity_factor(x)

    viscosity_factor = 1 / (1 + α×slope + β×roughness + γ×fog + ...)
    """

    def __init__(self, params: Optional[ViscosityParams] = None):
        """
        初始化粘滞系数场计算器

        Args:
            params: 粘滞系数参数，None则使用默认值
        """
        self.params = params or ViscosityParams()

    def compute_slope(self, terrain: np.ndarray,
                      spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        计算坡度场（梯度幅值）

        Args:
            terrain: 地形高程数据 (nz, ny, nx) 或 (ny, nx)
            spacing: 网格间距

        Returns:
            坡度场，与输入形状相同
        """
        ndim = terrain.ndim
        slope = np.zeros_like(terrain, dtype=np.float64)

        if ndim == 3:
            dz, dy, dx = spacing
            # Z方向梯度
            grad_z = np.zeros_like(terrain)
            grad_z[1:-1, :, :] = (terrain[2:, :, :] - terrain[:-2, :, :]) / (2 * dz)
            grad_z[0, :, :] = (terrain[1, :, :] - terrain[0, :, :]) / dz
            grad_z[-1, :, :] = (terrain[-1, :, :] - terrain[-2, :, :]) / dz

            # Y方向梯度
            grad_y = np.zeros_like(terrain)
            grad_y[:, 1:-1, :] = (terrain[:, 2:, :] - terrain[:, :-2, :]) / (2 * dy)
            grad_y[:, 0, :] = (terrain[:, 1, :] - terrain[:, 0, :]) / dy
            grad_y[:, -1, :] = (terrain[:, -1, :] - terrain[:, -2, :]) / dy

            # X方向梯度
            grad_x = np.zeros_like(terrain)
            grad_x[:, :, 1:-1] = (terrain[:, :, 2:] - terrain[:, :, :-2]) / (2 * dx)
            grad_x[:, :, 0] = (terrain[:, :, 1] - terrain[:, :, 0]) / dx
            grad_x[:, :, -1] = (terrain[:, :, -1] - terrain[:, :, -2]) / dx

            slope = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)

        elif ndim == 2:
            dy, dx = spacing[-2:]
            # Y方向梯度
            grad_y = np.zeros_like(terrain)
            grad_y[1:-1, :] = (terrain[2:, :] - terrain[:-2, :]) / (2 * dy)
            grad_y[0, :] = (terrain[1, :] - terrain[0, :]) / dy
            grad_y[-1, :] = (terrain[-1, :] - terrain[-2, :]) / dy

            # X方向梯度
            grad_x = np.zeros_like(terrain)
            grad_x[:, 1:-1] = (terrain[:, 2:] - terrain[:, :-2]) / (2 * dx)
            grad_x[:, 0] = (terrain[:, 1] - terrain[:, 0]) / dx
            grad_x[:, -1] = (terrain[:, -1] - terrain[:, -2]) / dx

            slope = np.sqrt(grad_y**2 + grad_x**2)

        return slope

    def compute_roughness(self, terrain: np.ndarray) -> np.ndarray:
        """
        计算地形粗糙度（局部高程方差）

        Args:
            terrain: 地形高程数据

        Returns:
            粗糙度场
        """
        from scipy.ndimage import uniform_filter

        window = self.params.roughness_window

        # 局部均值
        local_mean = uniform_filter(terrain.astype(np.float64), size=window, mode='nearest')

        # 局部方差 = E[X^2] - E[X]^2
        local_sq_mean = uniform_filter(terrain.astype(np.float64)**2, size=window, mode='nearest')
        roughness = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))

        return roughness

    def compute_viscosity(self, terrain: np.ndarray,
                         fog_data: Optional[np.ndarray] = None,
                         spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        计算完整粘滞系数场

        viscosity = 1.0 / (1.0 + slope_cost + roughness_cost + fog_cost)

        Args:
            terrain: 地形高程数据
            fog_data: 雾气/环境数据，None则忽略
            spacing: 网格间距

        Returns:
            粘滞系数场（速度因子），值域 (0, 1]
        """
        # 计算坡度
        slope = self.compute_slope(terrain, spacing)
        slope_normalized = slope / (slope.max() + 1e-10)  # 归一化到 [0, 1]
        slope_cost = self.params.slope_weight * slope_normalized

        # 计算粗糙度
        roughness = self.compute_roughness(terrain)
        roughness_normalized = roughness / (roughness.max() + 1e-10)
        roughness_cost = self.params.roughness_weight * roughness_normalized

        # 雾气/环境因子
        fog_cost = np.zeros_like(terrain, dtype=np.float64)
        if fog_data is not None:
            fog_normalized = fog_data / (fog_data.max() + 1e-10)
            fog_cost = self.params.fog_weight * fog_normalized

        # 合成粘滞系数（速度因子）
        viscosity = 1.0 / (1.0 + slope_cost + roughness_cost + fog_cost)
        viscosity = np.maximum(viscosity, self.params.min_viscosity)

        return viscosity

    def get_speed_field(self, terrain: np.ndarray,
                        fog_data: Optional[np.ndarray] = None,
                        base_speed: float = 1.0,
                        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        获取FMM使用的速度场

        Args:
            terrain: 地形高程数据
            fog_data: 雾气数据
            base_speed: 基础速度
            spacing: 网格间距

        Returns:
            速度场 = base_speed × viscosity_factor
        """
        viscosity = self.compute_viscosity(terrain, fog_data, spacing)
        return base_speed * viscosity


class TerrainPyramid:
    """
    多分辨率地形金字塔

    Level 0: 原始分辨率 (仅在起点/终点附近)
    Level 1: 2x下采样 (中等距离)
    Level 2: 4x下采样 (远距离)
    ...
    """

    def __init__(self, terrain: np.ndarray, n_levels: int = 3,
                 viscosity_field: Optional[ViscosityField] = None,
                 fog_data: Optional[np.ndarray] = None,
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        """
        构建多分辨率金字塔

        Args:
            terrain: 原始地形数据
            n_levels: 金字塔层级数
            viscosity_field: 粘滞系数计算器
            fog_data: 雾气数据
            spacing: 原始网格间距
        """
        self.n_levels = n_levels
        self.levels: List[np.ndarray] = []
        self.speed_levels: List[np.ndarray] = []
        self.spacings: List[Tuple[float, ...]] = []
        self.shapes: List[Tuple[int, ...]] = []

        self.viscosity_field = viscosity_field or ViscosityField()

        # 构建金字塔
        self._build_pyramid(terrain, fog_data, spacing)

    def _build_pyramid(self, terrain: np.ndarray,
                       fog_data: Optional[np.ndarray],
                       spacing: Tuple[float, ...]):
        """构建多分辨率金字塔"""
        from scipy.ndimage import zoom

        current_terrain = terrain.astype(np.float64)
        current_fog = fog_data.astype(np.float64) if fog_data is not None else None
        current_spacing = spacing

        for level in range(self.n_levels):
            self.levels.append(current_terrain)
            self.spacings.append(current_spacing)
            self.shapes.append(current_terrain.shape)

            # 计算当前层级的速度场
            speed = self.viscosity_field.get_speed_field(
                current_terrain, current_fog, spacing=current_spacing
            )
            self.speed_levels.append(speed)

            # 下采样准备下一层级
            if level < self.n_levels - 1:
                # 2x下采样
                scale_factor = 0.5
                ndim = current_terrain.ndim

                if ndim == 3:
                    zoom_factors = (scale_factor, scale_factor, scale_factor)
                else:
                    zoom_factors = (scale_factor, scale_factor)

                current_terrain = zoom(current_terrain, zoom_factors, order=1)
                if current_fog is not None:
                    current_fog = zoom(current_fog, zoom_factors, order=1)

                # 更新间距
                current_spacing = tuple(s * 2 for s in current_spacing)

    def get_level(self, level: int) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
        """
        获取指定层级的数据

        Args:
            level: 层级索引 (0=最高分辨率)

        Returns:
            (地形数据, 速度场, 间距)
        """
        level = max(0, min(level, self.n_levels - 1))
        return self.levels[level], self.speed_levels[level], self.spacings[level]

    def get_level_for_distance(self, distance: float,
                               near_threshold: float = 20.0,
                               mid_threshold: float = 50.0) -> int:
        """
        根据距离选择分辨率层级

        Args:
            distance: 到目标点的距离
            near_threshold: 近距离阈值
            mid_threshold: 中距离阈值

        Returns:
            推荐的层级索引
        """
        if distance < near_threshold:
            return 0  # 高分辨率
        elif distance < mid_threshold:
            return min(1, self.n_levels - 1)  # 中分辨率
        else:
            return self.n_levels - 1  # 低分辨率

    def coord_to_level(self, coord: Tuple[int, ...],
                       from_level: int, to_level: int) -> Tuple[int, ...]:
        """
        跨层级坐标映射

        Args:
            coord: 原始坐标
            from_level: 源层级
            to_level: 目标层级

        Returns:
            目标层级的坐标
        """
        scale = 2 ** (to_level - from_level)
        return tuple(int(c / scale) for c in coord)

    def coord_from_level(self, coord: Tuple[int, ...],
                         from_level: int, to_level: int) -> Tuple[int, ...]:
        """
        从低分辨率到高分辨率的坐标映射

        Args:
            coord: 低分辨率坐标
            from_level: 源层级
            to_level: 目标层级

        Returns:
            高分辨率坐标
        """
        scale = 2 ** (from_level - to_level)
        return tuple(int(c * scale) for c in coord)


# 状态常量
FAR = 0
CONSIDERED = 1
ACCEPTED = 2


class NarrowBandFMM:
    """
    内存优化的窄带快速行进法

    优化策略:
    - 稀疏存储: 只存储活跃点的phi和status
    - 势值阈值剪枝: phi > threshold 的点不再扩展
    - 可达性掩码: 限制扩展区域（corridor）
    - 内存释放: 定期释放远离当前前沿的点
    """

    def __init__(self, shape: Tuple[int, ...],
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        """
        初始化窄带FMM求解器

        Args:
            shape: 网格形状
            spacing: 网格间距
        """
        self.shape = shape
        self.spacing = spacing
        self.ndim = len(shape)

        # 稀疏存储
        self.phi: Dict[Tuple[int, ...], float] = {}
        self.status: Dict[Tuple[int, ...], int] = {}
        self.heap: List[Tuple[float, Tuple[int, ...]]] = []

        # 速度场（密集存储，因为经常访问）
        self.speed_field: Optional[np.ndarray] = None

        # 剪枝参数
        self.phi_threshold: float = float('inf')
        self.phi_threshold_factor: float = 1.5

        # 可选的有效区域掩码
        self.valid_mask: Optional[np.ndarray] = None

        # 统计
        self.stats = {
            'pruned_count': 0,
            'accepted_count': 0,
            'released_count': 0,
            'max_heap_size': 0
        }

    def set_speed_field(self, speed_field: np.ndarray):
        """设置速度场"""
        self.speed_field = speed_field.astype(np.float64)

    def set_valid_region(self, mask: np.ndarray):
        """
        设置有效区域掩码（corridor限制）

        Args:
            mask: 布尔数组，True表示可扩展区域
        """
        self.valid_mask = mask.astype(bool)

    def set_phi_threshold(self, threshold: float):
        """设置势值阈值"""
        self.phi_threshold = threshold

    def reset(self):
        """重置求解器状态"""
        self.phi.clear()
        self.status.clear()
        self.heap.clear()
        self.stats = {
            'pruned_count': 0,
            'accepted_count': 0,
            'released_count': 0,
            'max_heap_size': 0
        }

    def _is_valid_point(self, point: Tuple[int, ...]) -> bool:
        """检查点是否在有效范围内"""
        # 边界检查
        for i, (p, s) in enumerate(zip(point, self.shape)):
            if p < 0 or p >= s:
                return False

        # 掩码检查
        if self.valid_mask is not None:
            if not self.valid_mask[point]:
                return False

        return True

    def _get_neighbors(self, point: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """获取有效邻居点"""
        neighbors = []
        for dim in range(self.ndim):
            for delta in [-1, 1]:
                neighbor = list(point)
                neighbor[dim] += delta
                neighbor = tuple(neighbor)
                if self._is_valid_point(neighbor):
                    neighbors.append(neighbor)
        return neighbors

    def _get_phi(self, point: Tuple[int, ...]) -> float:
        """获取点的phi值，未初始化返回inf"""
        return self.phi.get(point, float('inf'))

    def _get_status(self, point: Tuple[int, ...]) -> int:
        """获取点的状态，未初始化返回FAR"""
        return self.status.get(point, FAR)

    def _solve_eikonal_at_point(self, point: Tuple[int, ...]) -> float:
        """
        使用Godunov方案求解单点Eikonal方程

        ||∇Φ|| = 1/f(x)
        """
        speed = self.speed_field[point]
        if speed <= 0:
            return float('inf')

        slowness = 1.0 / speed

        # 收集各维度的单边差分
        phi_vals = []
        for dim in range(self.ndim):
            h = self.spacing[dim]

            # 获取该维度两侧的phi值
            point_minus = list(point)
            point_minus[dim] -= 1
            point_plus = list(point)
            point_plus[dim] += 1

            phi_minus = self._get_phi(tuple(point_minus)) if point_minus[dim] >= 0 else float('inf')
            phi_plus = self._get_phi(tuple(point_plus)) if point_plus[dim] < self.shape[dim] else float('inf')

            # Godunov: 选择较小的
            phi_min = min(phi_minus, phi_plus)
            if phi_min < float('inf'):
                phi_vals.append((phi_min, h))

        if not phi_vals:
            return float('inf')

        # 按phi值排序
        phi_vals.sort(key=lambda x: x[0])

        # 逐维度求解
        phi_new = float('inf')

        # 一维情况
        if len(phi_vals) >= 1:
            phi_a, h_a = phi_vals[0]
            phi_1d = phi_a + slowness * h_a
            if phi_1d < phi_new:
                phi_new = phi_1d

        # 二维情况
        if len(phi_vals) >= 2:
            phi_a, h_a = phi_vals[0]
            phi_b, h_b = phi_vals[1]

            # 求解二次方程
            # (phi - phi_a)^2/h_a^2 + (phi - phi_b)^2/h_b^2 = slowness^2
            a = 1.0/h_a**2 + 1.0/h_b**2
            b = -2.0 * (phi_a/h_a**2 + phi_b/h_b**2)
            c = phi_a**2/h_a**2 + phi_b**2/h_b**2 - slowness**2

            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                phi_2d = (-b + np.sqrt(discriminant)) / (2*a)
                if phi_2d >= max(phi_a, phi_b) and phi_2d < phi_new:
                    phi_new = phi_2d

        # 三维情况
        if len(phi_vals) >= 3:
            phi_a, h_a = phi_vals[0]
            phi_b, h_b = phi_vals[1]
            phi_c, h_c = phi_vals[2]

            a = 1.0/h_a**2 + 1.0/h_b**2 + 1.0/h_c**2
            b = -2.0 * (phi_a/h_a**2 + phi_b/h_b**2 + phi_c/h_c**2)
            c = phi_a**2/h_a**2 + phi_b**2/h_b**2 + phi_c**2/h_c**2 - slowness**2

            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                phi_3d = (-b + np.sqrt(discriminant)) / (2*a)
                if phi_3d >= max(phi_a, phi_b, phi_c) and phi_3d < phi_new:
                    phi_new = phi_3d

        return phi_new

    def solve(self, source: Tuple[int, ...],
              goal: Optional[Tuple[int, ...]] = None,
              max_iterations: int = 1000000) -> bool:
        """
        带剪枝的FMM求解

        Args:
            source: 源点（phi=0）
            goal: 目标点（可选，到达后更新阈值）
            max_iterations: 最大迭代次数

        Returns:
            是否成功求解
        """
        if self.speed_field is None:
            raise ValueError("Speed field not set")

        # 初始化源点
        self.phi[source] = 0.0
        self.status[source] = CONSIDERED
        heapq.heappush(self.heap, (0.0, source))

        iteration = 0
        goal_reached = False

        while self.heap and iteration < max_iterations:
            iteration += 1

            # 更新统计
            self.stats['max_heap_size'] = max(self.stats['max_heap_size'], len(self.heap))

            # 弹出最小phi点
            phi_val, current = heapq.heappop(self.heap)

            # 跳过已接受的点（堆中可能有重复）
            if self._get_status(current) == ACCEPTED:
                continue

            # 剪枝：势值超过阈值
            if phi_val > self.phi_threshold:
                self.stats['pruned_count'] += 1
                continue

            # 标记为已接受
            self.status[current] = ACCEPTED
            self.stats['accepted_count'] += 1

            # 到达目标后更新阈值
            if goal is not None and current == goal:
                goal_reached = True
                self.phi_threshold = phi_val * self.phi_threshold_factor

            # 更新邻居
            for neighbor in self._get_neighbors(current):
                if self._get_status(neighbor) == ACCEPTED:
                    continue

                # 计算新的phi值
                phi_new = self._solve_eikonal_at_point(neighbor)

                # 更新如果更优
                if phi_new < self._get_phi(neighbor):
                    self.phi[neighbor] = phi_new
                    self.status[neighbor] = CONSIDERED
                    heapq.heappush(self.heap, (phi_new, neighbor))

        return goal_reached or (goal is None)

    def get_phi_value(self, point: Tuple[int, ...]) -> float:
        """获取指定点的phi值"""
        return self._get_phi(point)

    def get_phi_array(self) -> np.ndarray:
        """
        将稀疏phi转换为密集数组

        Returns:
            完整的phi数组，未访问点为inf
        """
        phi_array = np.full(self.shape, np.inf, dtype=np.float64)
        for point, value in self.phi.items():
            phi_array[point] = value
        return phi_array

    def release_far_points(self, center: Tuple[int, ...], radius: float):
        """
        释放距离中心过远的点以节省内存

        Args:
            center: 中心点
            radius: 保留半径
        """
        to_remove = []
        for point in self.phi:
            dist = np.sqrt(sum((p - c)**2 for p, c in zip(point, center)))
            if dist > radius:
                to_remove.append(point)

        for p in to_remove:
            del self.phi[p]
            if p in self.status:
                del self.status[p]

        self.stats['released_count'] += len(to_remove)

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存使用统计"""
        return {
            'phi_points': len(self.phi),
            'status_points': len(self.status),
            'heap_size': len(self.heap),
            **self.stats
        }


class HierarchicalPathPlanner:
    """
    分层路径规划主类

    实现粗到细的多分辨率求解策略：
    1. 粗分辨率快速求解，获得大致路径
    2. 提取路径corridor
    3. 在corridor内逐层细化求解
    """

    def __init__(self, terrain: np.ndarray,
                 fog_data: Optional[np.ndarray] = None,
                 n_levels: int = 3,
                 viscosity_params: Optional[ViscosityParams] = None,
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        """
        初始化分层路径规划器

        Args:
            terrain: 地形数据
            fog_data: 雾气数据
            n_levels: 金字塔层级数
            viscosity_params: 粘滞系数参数
            spacing: 网格间距
        """
        self.viscosity_field = ViscosityField(viscosity_params)
        self.pyramid = TerrainPyramid(
            terrain, n_levels, self.viscosity_field, fog_data, spacing
        )
        self.n_levels = n_levels
        self.base_spacing = spacing

        # 规划参数
        self.corridor_width = 10  # corridor宽度（像素）
        self.phi_threshold_factor = 1.5
        self.memory_release_radius = 100

        # 统计
        self.stats: Dict[str, Any] = {}

    def _create_corridor_mask(self, path: List[Tuple[int, ...]],
                              shape: Tuple[int, ...],
                              width: int) -> np.ndarray:
        """
        根据路径创建corridor掩码

        Args:
            path: 路径点列表
            shape: 目标形状
            width: corridor宽度

        Returns:
            布尔掩码数组
        """
        mask = np.zeros(shape, dtype=bool)
        ndim = len(shape)

        for point in path:
            # 在每个路径点周围标记corridor
            if ndim == 3:
                z, y, x = point
                for dz in range(-width, width + 1):
                    for dy in range(-width, width + 1):
                        for dx in range(-width, width + 1):
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if (0 <= nz < shape[0] and
                                0 <= ny < shape[1] and
                                0 <= nx < shape[2]):
                                # 球形corridor
                                if dz**2 + dy**2 + dx**2 <= width**2:
                                    mask[nz, ny, nx] = True
            else:  # 2D
                y, x = point
                for dy in range(-width, width + 1):
                    for dx in range(-width, width + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                            if dy**2 + dx**2 <= width**2:
                                mask[ny, nx] = True

        return mask

    def _extract_path_from_phi(self, phi: Dict[Tuple[int, ...], float],
                               start: Tuple[int, ...],
                               goal: Tuple[int, ...],
                               shape: Tuple[int, ...],
                               spacing: Tuple[float, ...],
                               max_steps: int = 10000,
                               step_size: float = 0.5) -> List[Tuple[float, ...]]:
        """
        从phi场提取路径（梯度下降）

        Args:
            phi: 稀疏phi字典
            start: 起点
            goal: 终点
            shape: 网格形状
            spacing: 网格间距
            max_steps: 最大步数
            step_size: 步长

        Returns:
            路径点列表
        """
        # 转换为密集数组用于梯度计算
        phi_array = np.full(shape, np.inf, dtype=np.float64)
        for point, value in phi.items():
            phi_array[point] = value

        path = [tuple(float(c) for c in start)]
        current = np.array(start, dtype=np.float64)
        ndim = len(shape)

        for _ in range(max_steps):
            # 检查是否到达目标
            dist_to_goal = np.sqrt(sum((current[i] - goal[i])**2 for i in range(ndim)))
            if dist_to_goal < 1.5:
                path.append(tuple(float(c) for c in goal))
                break

            # 计算梯度
            idx = tuple(int(round(c)) for c in current)
            idx = tuple(max(0, min(idx[i], shape[i] - 1)) for i in range(ndim))

            gradient = np.zeros(ndim)
            for dim in range(ndim):
                h = spacing[dim]
                idx_minus = list(idx)
                idx_minus[dim] = max(0, idx[dim] - 1)
                idx_plus = list(idx)
                idx_plus[dim] = min(shape[dim] - 1, idx[dim] + 1)

                phi_minus = phi_array[tuple(idx_minus)]
                phi_plus = phi_array[tuple(idx_plus)]

                if phi_minus < float('inf') and phi_plus < float('inf'):
                    gradient[dim] = (phi_plus - phi_minus) / (2 * h)
                elif phi_minus < float('inf'):
                    gradient[dim] = (phi_array[idx] - phi_minus) / h
                elif phi_plus < float('inf'):
                    gradient[dim] = (phi_plus - phi_array[idx]) / h

            # 归一化并前进
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < 1e-10:
                break

            direction = -gradient / grad_norm
            current = current + step_size * direction

            # 边界限制
            for i in range(ndim):
                current[i] = max(0, min(current[i], shape[i] - 1))

            path.append(tuple(float(c) for c in current))

        return path

    def _solve_at_level(self, level: int,
                        start: Tuple[int, ...],
                        goal: Tuple[int, ...],
                        corridor_mask: Optional[np.ndarray] = None) -> Tuple[List[Tuple[int, ...]], Dict]:
        """
        在指定层级求解

        Args:
            level: 金字塔层级
            start: 起点（该层级坐标）
            goal: 终点（该层级坐标）
            corridor_mask: 可选的corridor掩码

        Returns:
            (离散路径点列表, 统计信息)
        """
        terrain, speed_field, spacing = self.pyramid.get_level(level)
        shape = terrain.shape

        # 确保坐标在有效范围内
        start = tuple(min(max(0, s), shape[i] - 1) for i, s in enumerate(start))
        goal = tuple(min(max(0, g), shape[i] - 1) for i, g in enumerate(goal))

        # 创建求解器
        fmm = NarrowBandFMM(shape, spacing)
        fmm.set_speed_field(speed_field)
        fmm.phi_threshold_factor = self.phi_threshold_factor

        if corridor_mask is not None:
            fmm.set_valid_region(corridor_mask)

        # 求解（从goal到start，因为FMM从源点向外传播）
        start_time = time.time()
        success = fmm.solve(goal, start)
        solve_time = time.time() - start_time

        # 提取路径
        if success or fmm.get_phi_value(start) < float('inf'):
            path = self._extract_path_from_phi(
                fmm.phi, start, goal, shape, spacing
            )
            # 转换为整数坐标
            discrete_path = [tuple(int(round(c)) for c in p) for p in path]
        else:
            discrete_path = []

        stats = {
            'level': level,
            'shape': shape,
            'solve_time': solve_time,
            'success': success,
            'path_length': len(discrete_path),
            **fmm.get_memory_stats()
        }

        return discrete_path, stats

    def plan(self, start: Tuple[int, ...],
             goal: Tuple[int, ...]) -> Tuple[List[Tuple[float, ...]], Dict[str, Any]]:
        """
        分层路径规划主函数

        Args:
            start: 起点（原始分辨率坐标）
            goal: 终点（原始分辨率坐标）

        Returns:
            (路径点列表, 统计信息)
        """
        total_start_time = time.time()
        level_stats = []

        # Step 1: 粗分辨率快速求解
        coarse_level = self.n_levels - 1
        coarse_start = self.pyramid.coord_to_level(start, 0, coarse_level)
        coarse_goal = self.pyramid.coord_to_level(goal, 0, coarse_level)

        coarse_path, coarse_stats = self._solve_at_level(
            coarse_level, coarse_start, coarse_goal
        )
        level_stats.append(coarse_stats)

        if not coarse_path:
            return [], {'success': False, 'error': 'Coarse level failed'}

        # Step 2-N: 逐层细化
        current_path = coarse_path

        for level in range(coarse_level - 1, -1, -1):
            # 将路径坐标转换到当前层级
            scale = 2 ** (coarse_level - level)
            scaled_path = [tuple(int(c * scale) for c in p) for p in current_path]

            # 获取当前层级形状
            _, _, _ = self.pyramid.get_level(level)
            current_shape = self.pyramid.shapes[level]

            # 创建corridor掩码
            corridor_width = self.corridor_width * (level + 1)  # 高层级宽度更大
            corridor_mask = self._create_corridor_mask(
                scaled_path, current_shape, corridor_width
            )

            # 当前层级坐标
            level_start = self.pyramid.coord_to_level(start, 0, level)
            level_goal = self.pyramid.coord_to_level(goal, 0, level)

            # 在corridor内求解
            refined_path, refined_stats = self._solve_at_level(
                level, level_start, level_goal, corridor_mask
            )
            level_stats.append(refined_stats)

            if refined_path:
                current_path = refined_path
            else:
                # 如果细化失败，使用上一层的结果
                current_path = scaled_path

        # 转换最终路径为浮点坐标
        final_path = [tuple(float(c) for c in p) for p in current_path]

        # 计算路径代价
        path_cost = self._compute_path_cost(final_path)

        total_time = time.time() - total_start_time

        self.stats = {
            'success': len(final_path) > 0,
            'total_time': total_time,
            'path_length': len(final_path),
            'path_cost': path_cost,
            'level_stats': level_stats,
            'n_levels': self.n_levels
        }

        return final_path, self.stats

    def _compute_path_cost(self, path: List[Tuple[float, ...]]) -> float:
        """计算路径总代价"""
        if len(path) < 2:
            return 0.0

        _, speed_field, spacing = self.pyramid.get_level(0)
        total_cost = 0.0

        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]

            # 欧氏距离
            dist = np.sqrt(sum((a - b)**2 * s**2
                              for a, b, s in zip(p1, p2, spacing)))

            # 获取中点的速度
            mid = tuple(int(round((a + b) / 2)) for a, b in zip(p1, p2))
            mid = tuple(max(0, min(m, speed_field.shape[i] - 1))
                       for i, m in enumerate(mid))
            speed = speed_field[mid]

            # 代价 = 距离 / 速度
            if speed > 0:
                total_cost += dist / speed
            else:
                total_cost += dist * 1000  # 惩罚

        return total_cost

    def plan_with_single_resolution(self, start: Tuple[int, ...],
                                    goal: Tuple[int, ...]) -> Tuple[List[Tuple[float, ...]], Dict]:
        """
        单分辨率求解（用于对比）

        Args:
            start: 起点
            goal: 终点

        Returns:
            (路径, 统计)
        """
        return self._solve_at_level(0, start, goal)


class MultiPathPlanner:
    """
    多路径规划器

    使用迭代惩罚方法在粘滞分层FMM框架下找到K条多样化的替代路径。

    算法:
    1. 使用原始速度场找到最优路径 P_1
    2. 在所有金字塔层级的速度场上施加惩罚（降低 P_1 附近的速度）
    3. 使用惩罚后的速度场找到次优路径 P_2
    4. 重复直到找到K条路径或无法找到更多满足条件的路径
    """

    def __init__(self, terrain: np.ndarray,
                 fog_data: Optional[np.ndarray] = None,
                 n_levels: int = 3,
                 viscosity_params: Optional[ViscosityParams] = None,
                 multi_path_params: Optional[MultiPathParams] = None,
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        """
        初始化多路径规划器

        Args:
            terrain: 地形数据
            fog_data: 雾气数据
            n_levels: 金字塔层级数
            viscosity_params: 粘滞系数参数
            multi_path_params: 多路径参数
            spacing: 网格间距
        """
        self.multi_params = multi_path_params or MultiPathParams()
        self.n_levels = n_levels

        # 创建分层规划器
        self.planner = HierarchicalPathPlanner(
            terrain=terrain,
            fog_data=fog_data,
            n_levels=n_levels,
            viscosity_params=viscosity_params,
            spacing=spacing
        )

        # 保存原始速度场的深拷贝（用于每次迭代前恢复）
        self._original_speed_levels = [
            level.copy() for level in self.planner.pyramid.speed_levels
        ]

    def plan_multi_path(self, start: Tuple[int, ...],
                        goal: Tuple[int, ...]) -> MultiPathResult:
        """
        多路径规划主函数

        Args:
            start: 起点（原始分辨率坐标）
            goal: 终点（原始分辨率坐标）

        Returns:
            MultiPathResult 包含多条路径及统计信息
        """
        total_start_time = time.time()

        found_paths: List[List[Tuple[float, ...]]] = []
        found_costs: List[float] = []
        per_path_stats: List[Dict[str, Any]] = []

        # 累积惩罚掩码（每层级一个，初始全1.0）
        penalty_masks = [
            np.ones_like(speed, dtype=np.float64)
            for speed in self._original_speed_levels
        ]

        # 允许额外尝试次数（路径太相似时跳过但继续尝试）
        max_attempts = self.multi_params.n_paths * 2

        for attempt in range(max_attempts):
            if len(found_paths) >= self.multi_params.n_paths:
                break

            # 将惩罚后的速度场写入金字塔
            for level in range(self.n_levels):
                self.planner.pyramid.speed_levels[level] = (
                    self._original_speed_levels[level] * penalty_masks[level]
                )

            # 求解
            path_k, stats_k = self.planner.plan(start, goal)

            if not path_k:
                break

            cost_k = stats_k.get('path_cost', float('inf'))

            # 代价超标检查
            if found_costs and cost_k > found_costs[0] * self.multi_params.max_cost_ratio:
                break

            # 多样性检查
            if found_paths:
                min_sep = min(
                    self._compute_path_separation(path_k, existing)
                    for existing in found_paths
                )
                if min_sep < self.multi_params.min_path_separation:
                    # 路径太相似，仍施加惩罚但不保存
                    self._apply_penalty_to_speed_levels(path_k, penalty_masks)
                    continue

            # 保存路径
            found_paths.append(path_k)
            found_costs.append(cost_k)
            per_path_stats.append(stats_k)

            # 施加惩罚
            self._apply_penalty_to_speed_levels(path_k, penalty_masks)

        # 恢复原始速度场
        for level in range(self.n_levels):
            self.planner.pyramid.speed_levels[level] = (
                self._original_speed_levels[level].copy()
            )

        # 计算多样性矩阵
        diversity_matrix = self._compute_diversity_matrix(found_paths)

        total_time = time.time() - total_start_time

        stats = {
            'success': len(found_paths) > 0,
            'n_paths_found': len(found_paths),
            'n_paths_requested': self.multi_params.n_paths,
            'total_time': total_time,
        }

        return MultiPathResult(
            paths=found_paths,
            costs=found_costs,
            stats=stats,
            per_path_stats=per_path_stats,
            diversity_matrix=diversity_matrix
        )

    def _apply_penalty_to_speed_levels(self, path: List[Tuple[float, ...]],
                                        penalty_masks: List[np.ndarray]) -> None:
        """
        在所有金字塔层级的惩罚掩码上施加路径惩罚

        Args:
            path: 路径点列表（原始分辨率坐标）
            penalty_masks: 各层级惩罚掩码列表（会被就地修改）
        """
        for level in range(self.n_levels):
            shape = self.planner.pyramid.shapes[level]
            scale = 2 ** level
            ndim = len(shape)

            # 缩放路径坐标到当前层级
            scaled_radius = max(1, self.multi_params.penalty_radius // scale)

            # 对路径进行子采样（每隔 scale 个点取一个，减少计算量）
            step = max(1, scale)
            for point in path[::step]:
                idx = tuple(
                    max(0, min(int(round(c / scale)), shape[d] - 1))
                    for d, c in enumerate(point)
                )

                # 在半径范围内施加惩罚
                for neighbor in self._neighborhood_iterator(idx, scaled_radius, shape):
                    dist = np.sqrt(sum(
                        (a - b) ** 2 for a, b in zip(idx, neighbor)
                    ))
                    if dist > scaled_radius:
                        continue

                    # 计算衰减因子
                    if self.multi_params.penalty_decay == 'gaussian':
                        sigma = scaled_radius / 2.0
                        decay = exp(-(dist ** 2) / (2 * sigma ** 2))
                    else:  # linear
                        decay = 1.0 - dist / scaled_radius if scaled_radius > 0 else 1.0

                    penalty = 1.0 - self.multi_params.penalty_weight * decay
                    # 累积惩罚（取最小值，只会降低不会恢复）
                    penalty_masks[level][neighbor] = min(
                        penalty_masks[level][neighbor], penalty
                    )

    @staticmethod
    def _neighborhood_iterator(center: Tuple[int, ...],
                                radius: int,
                                shape: Tuple[int, ...]):
        """
        生成中心点周围给定半径内的所有有效整数坐标

        Args:
            center: 中心点坐标
            radius: 邻域半径
            shape: 网格形状（用于边界检查）

        Yields:
            有效邻域坐标元组
        """
        ndim = len(shape)
        # 生成各维度的范围
        ranges = []
        for d in range(ndim):
            lo = max(0, center[d] - radius)
            hi = min(shape[d] - 1, center[d] + radius)
            ranges.append(range(lo, hi + 1))

        if ndim == 3:
            for z in ranges[0]:
                for y in ranges[1]:
                    for x in ranges[2]:
                        yield (z, y, x)
        elif ndim == 2:
            for y in ranges[0]:
                for x in ranges[1]:
                    yield (y, x)

    def _compute_path_separation(self, path_a: List[Tuple[float, ...]],
                                  path_b: List[Tuple[float, ...]]) -> float:
        """
        计算两条路径之间的平均最小距离

        Args:
            path_a: 路径A
            path_b: 路径B

        Returns:
            平均最小距离
        """
        if not path_a or not path_b:
            return 0.0

        # 子采样到最多100个点
        step_a = max(1, len(path_a) // 100)
        step_b = max(1, len(path_b) // 100)
        sampled_a = path_a[::step_a]
        sampled_b = path_b[::step_b]

        arr_a = np.array(sampled_a)
        arr_b = np.array(sampled_b)

        # 对 path_a 中每个点，找到 path_b 中最近点的距离
        min_dists = []
        for pa in arr_a:
            dists = np.sqrt(np.sum((arr_b - pa) ** 2, axis=1))
            min_dists.append(np.min(dists))

        return float(np.mean(min_dists))

    def _compute_diversity_matrix(self,
                                   paths: List[List[Tuple[float, ...]]]) -> np.ndarray:
        """
        计算路径间多样性矩阵

        Args:
            paths: 路径列表

        Returns:
            K×K 对称距离矩阵
        """
        k = len(paths)
        matrix = np.zeros((k, k), dtype=np.float64)

        for i in range(k):
            for j in range(i + 1, k):
                sep = self._compute_path_separation(paths[i], paths[j])
                matrix[i, j] = sep
                matrix[j, i] = sep

        return matrix


class CompetitiveFMM:
    """
    竞争性快速行进法（不互溶气体扩散模型）

    多个气体（机器人）同时从各自起点扩散，竞争领土。
    - 共享优先队列，按 phi 值（到达时间）排序
    - 不互溶性：一旦某个气体占领某个像素，其他气体无法进入
    - 惠更斯原理：气体到达路径点后，从该路径点继续扩散（累积 phi）

    基于 NarrowBandFMM 的 Godunov 格式求解 Eikonal 方程。
    """

    FAR = 0
    CONSIDERED = 1
    ACCEPTED = 2

    def __init__(self, speed_field: np.ndarray, spacing: Tuple[float, ...]):
        """
        Args:
            speed_field: 速度场 (与地形同形状)
            spacing: 网格间距
        """
        self.speed_field = speed_field
        self.spacing = spacing
        self.shape = speed_field.shape
        self.ndim = speed_field.ndim

    def solve(self, robot_starts: List[Tuple[int, ...]],
              waypoints: List[Tuple[int, ...]],
              max_per_robot: Optional[int] = None
              ) -> Tuple[Dict[int, List[Tuple[int, ...]]],
                         np.ndarray,
                         Dict[int, List[float]],
                         List[Tuple[int, int]]]:
        """
        执行竞争性 FMM 扩散

        Args:
            robot_starts: 各机器人起点列表
            waypoints: 待分配路径点列表
            max_per_robot: 每机器人最大路径点数

        Returns:
            assignments: robot_id → 路径点列表（按到达顺序）
            territory_map: 领土地图 (-1=未占领, 0..n-1=机器人编号)
            arrival_times: robot_id → 各路径点到达时间列表
            assignment_order: (robot_id, waypoint_idx) 全局到达顺序
        """
        n_robots = len(robot_starts)

        # 状态存储（稀疏）
        phi = {}           # point → phi value
        status = {}        # point → FAR/CONSIDERED/ACCEPTED
        owner = {}         # point → robot_id

        # 领土地图（稠密，用于可视化）
        territory_map = np.full(self.shape, -1, dtype=np.int32)

        # 路径点集合（用于快速查找）
        waypoint_set = set(waypoints)
        remaining_waypoints = set(waypoints)

        # 分配结果
        assignments = {r: [] for r in range(n_robots)}
        arrival_times = {r: [] for r in range(n_robots)}
        assignment_order = []
        robot_wp_count = {r: 0 for r in range(n_robots)}

        # 优先队列：(phi_value, counter, point, robot_id)
        heap = []
        counter = 0

        # 初始化：各机器人起点
        for r, start in enumerate(robot_starts):
            start = tuple(start)
            phi[start] = 0.0
            status[start] = self.ACCEPTED
            owner[start] = r
            territory_map[start] = r

            # 将起点邻居加入队列
            for neighbor in self._get_neighbors(start):
                if neighbor not in status or status[neighbor] == self.FAR:
                    new_phi = self._solve_eikonal_at_point(neighbor, phi, owner, r)
                    if new_phi < float('inf'):
                        phi[neighbor] = new_phi
                        status[neighbor] = self.CONSIDERED
                        owner[neighbor] = r
                        heapq.heappush(heap, (new_phi, counter, neighbor, r))
                        counter += 1

        # 主循环
        max_iterations = self.shape[0] * self.shape[1]
        if self.ndim == 3:
            max_iterations *= self.shape[2]

        iteration = 0
        while heap and remaining_waypoints and iteration < max_iterations:
            iteration += 1

            phi_val, _, point, robot_id = heapq.heappop(heap)

            # 已被接受则跳过
            if point in status and status[point] == self.ACCEPTED:
                continue

            # 该点已被其他气体占领（不互溶）
            if point in owner and owner[point] != robot_id and status.get(point) == self.ACCEPTED:
                continue

            # 如果该点已有更好的 phi 值，跳过过时条目
            if point in phi and phi_val > phi[point] * 1.01:
                continue

            # 标记为 ACCEPTED
            status[point] = self.ACCEPTED
            phi[point] = phi_val
            owner[point] = robot_id
            territory_map[point] = robot_id

            # 检查是否到达路径点
            if point in remaining_waypoints:
                # 容量检查
                if max_per_robot is None or robot_wp_count[robot_id] < max_per_robot:
                    wp_idx = waypoints.index(point)
                    assignments[robot_id].append(point)
                    arrival_times[robot_id].append(phi_val)
                    assignment_order.append((robot_id, wp_idx))
                    robot_wp_count[robot_id] += 1
                    remaining_waypoints.discard(point)

                    # 惠更斯原理：从该路径点重新开始扩散
                    # 使用累积 phi（不重置），作为新的波源
                    for neighbor in self._get_neighbors(point):
                        if neighbor in status and status[neighbor] == self.ACCEPTED:
                            continue
                        # 如果邻居已被其他气体占领，跳过（不互溶）
                        if (neighbor in owner and owner[neighbor] != robot_id
                                and neighbor in status and status[neighbor] == self.ACCEPTED):
                            continue

                        new_phi = self._solve_eikonal_at_point(neighbor, phi, owner, robot_id)
                        if new_phi < float('inf'):
                            if neighbor not in phi or new_phi < phi[neighbor]:
                                phi[neighbor] = new_phi
                                status[neighbor] = self.CONSIDERED
                                owner[neighbor] = robot_id
                                heapq.heappush(heap, (new_phi, counter, neighbor, robot_id))
                                counter += 1

            # 扩展邻居
            for neighbor in self._get_neighbors(point):
                if neighbor in status and status[neighbor] == self.ACCEPTED:
                    continue

                new_phi = self._solve_eikonal_at_point(neighbor, phi, owner, robot_id)
                if new_phi < float('inf'):
                    if neighbor not in phi or new_phi < phi[neighbor]:
                        phi[neighbor] = new_phi
                        status[neighbor] = self.CONSIDERED
                        owner[neighbor] = robot_id
                        heapq.heappush(heap, (new_phi, counter, neighbor, robot_id))
                        counter += 1

        return assignments, territory_map, arrival_times, assignment_order

    def _get_neighbors(self, point: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """获取有效邻居点（6邻域/4邻域）"""
        neighbors = []
        for d in range(self.ndim):
            for delta in (-1, 1):
                neighbor = list(point)
                neighbor[d] += delta
                if 0 <= neighbor[d] < self.shape[d]:
                    neighbors.append(tuple(neighbor))
        return neighbors

    def _solve_eikonal_at_point(self, point: Tuple[int, ...],
                                 phi: Dict, owner: Dict,
                                 robot_id: int) -> float:
        """
        在给定点求解 Eikonal 方程（Godunov 格式）
        只使用同一气体（机器人）已接受的邻居值。

        Args:
            point: 待求解点
            phi: 全局 phi 字典
            owner: 全局 owner 字典
            robot_id: 当前气体编号

        Returns:
            该点的 phi 值
        """
        speed = self.speed_field[point]
        if speed <= 0:
            return float('inf')

        # 收集各维度的最小已接受邻居 phi
        phi_neighbors = []
        for d in range(self.ndim):
            candidates = []
            for delta in (-1, 1):
                neighbor = list(point)
                neighbor[d] += delta
                if not (0 <= neighbor[d] < self.shape[d]):
                    continue
                neighbor = tuple(neighbor)
                if (neighbor in phi and
                    owner.get(neighbor) == robot_id):
                    candidates.append(phi[neighbor])

            if candidates:
                phi_neighbors.append((min(candidates), self.spacing[d] if d < len(self.spacing) else 1.0))

        if not phi_neighbors:
            return float('inf')

        # 按 phi 值排序
        phi_neighbors.sort(key=lambda x: x[0])

        slowness = 1.0 / speed

        # 尝试从高维到低维求解
        for n_dims in range(len(phi_neighbors), 0, -1):
            subset = phi_neighbors[:n_dims]
            result = self._solve_quadratic(subset, slowness)
            if result is not None and result >= subset[-1][0]:
                return result

        # 1D fallback
        phi_min, h = phi_neighbors[0]
        return phi_min + h * slowness

    @staticmethod
    def _solve_quadratic(phi_h_pairs: List[Tuple[float, float]],
                          slowness: float) -> Optional[float]:
        """
        求解多维 Eikonal 二次方程

        sum_d ((phi - phi_d) / h_d)^2 = slowness^2
        """
        # a * phi^2 - 2b * phi + c = 0
        a = 0.0
        b = 0.0
        c = -slowness * slowness

        for phi_d, h_d in phi_h_pairs:
            inv_h2 = 1.0 / (h_d * h_d)
            a += inv_h2
            b += phi_d * inv_h2
            c += phi_d * phi_d * inv_h2

        discriminant = b * b - a * c
        if discriminant < 0:
            return None

        return (b + discriminant ** 0.5) / a


class MultiRobotWaypointAllocator:
    """
    多机器人路径点分配器

    使用不互溶气体扩散模型（竞争性 FMM + 惠更斯原理）将路径点分配给多个机器人，
    然后为每个机器人规划路径段。

    工作流程：
    1. 竞争性 FMM：多种气体同时扩散，先到先得
    2. 惠更斯原理：气体到达路径点后，从该点继续扩散
    3. 路径规划：为每个机器人规划起点 → 路径点1 → 路径点2 → ... 的路径
    """

    def __init__(self, terrain: np.ndarray,
                 fog_data: Optional[np.ndarray] = None,
                 n_levels: int = 3,
                 viscosity_params: Optional[ViscosityParams] = None,
                 gas_params: Optional[GasDiffusionParams] = None,
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
                 use_neural: bool = False,
                 neural_model_path: Optional[str] = None):
        """
        Args:
            terrain: 地形数据
            fog_data: 雾气数据
            n_levels: 金字塔层级数（用于路径规划阶段）
            viscosity_params: 粘滞系数参数
            gas_params: 气体扩散分配参数
            spacing: 网格间距
            use_neural: 是否使用神经网络分配器（替代 CompetitiveFMM）
            neural_model_path: 神经网络模型路径（.pt 文件）
        """
        self.use_neural = use_neural
        self.neural_model_path = neural_model_path
        self.terrain = terrain
        self.gas_params = gas_params or GasDiffusionParams()
        self.viscosity_params = viscosity_params or ViscosityParams()
        self.spacing = spacing
        self.n_levels = n_levels
        self.fog_data = fog_data

        # 计算速度场（原始分辨率，用于竞争性 FMM）
        vf = ViscosityField(self.viscosity_params)
        self.speed_field = vf.get_speed_field(terrain, fog_data, spacing=spacing)

        # 路径规划器（用于分配后的路径段规划）
        self.planner = HierarchicalPathPlanner(
            terrain=terrain,
            fog_data=fog_data,
            n_levels=n_levels,
            viscosity_params=viscosity_params,
            spacing=spacing
        )

    def allocate(self) -> WaypointAllocationResult:
        """
        执行路径点分配与路径规划

        Returns:
            WaypointAllocationResult 包含分配、路径、领土地图和统计信息
        """
        # 如果启用神经网络分配器，委托给 NeuralWaypointAllocator
        if self.use_neural:
            from neural_allocator import NeuralWaypointAllocator
            neural = NeuralWaypointAllocator(
                terrain=self.terrain, fog_data=self.fog_data,
                n_levels=self.n_levels, viscosity_params=self.viscosity_params,
                gas_params=self.gas_params, spacing=self.spacing,
                model_path=self.neural_model_path, fallback_to_fmm=True,
            )
            return neural.allocate()

        total_start = time.time()

        # Step 1: 竞争性 FMM 分配路径点
        print("   [Gas Diffusion] Running competitive FMM...")
        cfmm = CompetitiveFMM(self.speed_field, self.spacing)

        alloc_start = time.time()
        assignments, territory_map, arrival_times, assignment_order = cfmm.solve(
            robot_starts=self.gas_params.robot_starts,
            waypoints=self.gas_params.waypoints,
            max_per_robot=self.gas_params.max_waypoints_per_robot
        )
        alloc_time = time.time() - alloc_start

        print(f"   [Gas Diffusion] Allocation done in {alloc_time:.4f}s")
        for r in range(self.gas_params.n_robots):
            print(f"     Robot {r}: {len(assignments[r])} waypoints assigned")

        # Step 2: 为每个机器人规划路径段
        print("   [Path Planning] Planning path segments...")
        plan_start = time.time()
        robot_paths = {}

        for robot_id, wp_list in assignments.items():
            if not wp_list:
                robot_paths[robot_id] = []
                continue

            # 构建路径链: start → wp1 → wp2 → ...
            chain = [self.gas_params.robot_starts[robot_id]] + wp_list
            segments = []

            for i in range(len(chain) - 1):
                seg_start = tuple(chain[i])
                seg_goal = tuple(chain[i + 1])
                try:
                    path, _ = self.planner.plan(seg_start, seg_goal)
                    if path:
                        segments.append(path)
                    else:
                        segments.append([seg_start, seg_goal])
                except Exception:
                    segments.append([seg_start, seg_goal])

            robot_paths[robot_id] = segments

        plan_time = time.time() - plan_start
        total_time = time.time() - total_start

        # 统计信息
        stats = {
            'success': sum(len(v) for v in assignments.values()) > 0,
            'n_robots': self.gas_params.n_robots,
            'n_waypoints_total': len(self.gas_params.waypoints),
            'n_waypoints_assigned': sum(len(v) for v in assignments.values()),
            'allocation_time': alloc_time,
            'path_planning_time': plan_time,
            'total_time': total_time,
            'per_robot': {
                r: {
                    'n_waypoints': len(assignments[r]),
                    'n_segments': len(robot_paths.get(r, [])),
                    'arrival_times': arrival_times[r],
                }
                for r in range(self.gas_params.n_robots)
            }
        }

        return WaypointAllocationResult(
            assignments=assignments,
            assignment_order=assignment_order,
            robot_paths=robot_paths,
            arrival_times=arrival_times,
            territory_map=territory_map,
            stats=stats
        )


class QValueMultiPathPlanner:
    """
    Q值引导的多路径规划器

    核心思想：目标不是一个精确点，而是一个目标区域。区域内每个像素有一个 Q 值（得分）。
    规划器生成多条到达目标区域不同位置的候选路径，然后选择综合得分最高的路径。

    综合得分公式（归一化后）:
        score(path_k) = q_weight × Q_norm(goal_k) - cost_weight × cost_norm(path_k)

    其中:
        Q_norm = (Q - Q_min) / (Q_max - Q_min)        归一化到 [0, 1]
        cost_norm = (cost - cost_min) / (cost_max - cost_min)  归一化到 [0, 1]

    工作流程：
    1. 在目标区域内按 Q 值排序，选出多个候选终点
    2. 对每个候选终点，使用 FMM 规划路径
    3. 在路径之间施加惩罚以保证多样性
    4. 计算综合得分，选择最优路径
    """

    def __init__(self, terrain: np.ndarray,
                 fog_data: Optional[np.ndarray] = None,
                 n_levels: int = 3,
                 viscosity_params: Optional[ViscosityParams] = None,
                 q_params: Optional[QValueMultiPathParams] = None,
                 spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        """
        Args:
            terrain: 地形数据
            fog_data: 雾气数据
            n_levels: 金字塔层级数
            viscosity_params: 粘滞系数参数
            q_params: Q值多路径参数
            spacing: 网格间距
        """
        self.q_params = q_params or QValueMultiPathParams()
        self.n_levels = n_levels
        self.spacing = spacing
        self.terrain = terrain

        # 创建内部多路径参数
        self._multi_params = MultiPathParams(
            n_paths=self.q_params.n_paths,
            penalty_weight=self.q_params.penalty_weight,
            penalty_radius=self.q_params.penalty_radius,
            min_path_separation=self.q_params.min_path_separation,
            penalty_decay=self.q_params.penalty_decay,
            max_cost_ratio=self.q_params.max_cost_ratio
        )

        # 创建分层规划器
        self.planner = HierarchicalPathPlanner(
            terrain=terrain,
            fog_data=fog_data,
            n_levels=n_levels,
            viscosity_params=viscosity_params,
            spacing=spacing
        )

        # 保存原始速度场
        self._original_speed_levels = [
            level.copy() for level in self.planner.pyramid.speed_levels
        ]

    def plan(self, start: Tuple[int, ...],
             q_field: np.ndarray,
             target_mask: np.ndarray) -> QValueMultiPathResult:
        """
        Q值引导的多路径规划

        Args:
            start: 起点坐标
            q_field: Q值场，与地形同形状，每个像素的得分
            target_mask: 目标区域掩码（bool），True 表示目标区域内的有效终点

        Returns:
            QValueMultiPathResult
        """
        total_start = time.time()

        # Step 1: 在目标区域内选出候选终点（按 Q 值降序）
        candidate_goals = self._select_candidate_goals(q_field, target_mask)

        if not candidate_goals:
            return QValueMultiPathResult(stats={'success': False, 'error': 'No valid goals in target region'})

        print(f"   [Q-Value Planner] {len(candidate_goals)} candidate goals selected")

        # Step 2: 为每个候选终点规划路径（带惩罚保证多样性）
        found_paths = []
        found_goals = []
        found_costs = []
        found_q_values = []

        penalty_masks = [
            np.ones_like(speed, dtype=np.float64)
            for speed in self._original_speed_levels
        ]

        max_attempts = self.q_params.n_paths * 2

        goal_idx = 0
        for attempt in range(max_attempts):
            if len(found_paths) >= self.q_params.n_paths:
                break
            if goal_idx >= len(candidate_goals):
                break

            goal_point, q_val = candidate_goals[goal_idx]
            goal_idx += 1

            # 应用惩罚后的速度场
            for level in range(self.n_levels):
                self.planner.pyramid.speed_levels[level] = (
                    self._original_speed_levels[level] * penalty_masks[level]
                )

            # 规划路径
            path_k, stats_k = self.planner.plan(start, goal_point)

            if not path_k:
                continue

            cost_k = stats_k.get('path_cost', float('inf'))

            # 代价超标检查（相对第一条路径）
            if found_costs and cost_k > found_costs[0] * self.q_params.max_cost_ratio:
                continue

            # 多样性检查
            if found_paths:
                min_sep = min(
                    self._compute_path_separation(path_k, existing)
                    for existing in found_paths
                )
                if min_sep < self.q_params.min_path_separation:
                    self._apply_penalty(path_k, penalty_masks)
                    continue

            # 保存
            found_paths.append(path_k)
            found_goals.append(goal_point)
            found_costs.append(cost_k)
            found_q_values.append(q_val)

            # 施加惩罚
            self._apply_penalty(path_k, penalty_masks)

        # 恢复原始速度场
        for level in range(self.n_levels):
            self.planner.pyramid.speed_levels[level] = (
                self._original_speed_levels[level].copy()
            )

        if not found_paths:
            return QValueMultiPathResult(stats={'success': False, 'error': 'No valid paths found'})

        # Step 3: 计算综合得分
        scores = self._compute_scores(found_q_values, found_costs)

        # Step 4: 选择最优路径
        best_idx = int(np.argmax(scores))

        total_time = time.time() - total_start

        stats = {
            'success': True,
            'n_candidates': len(candidate_goals),
            'n_paths_found': len(found_paths),
            'best_index': best_idx,
            'total_time': total_time,
            'q_weight': self.q_params.q_weight,
            'cost_weight': self.q_params.cost_weight,
        }

        return QValueMultiPathResult(
            best_path=found_paths[best_idx],
            best_goal=found_goals[best_idx],
            best_score=scores[best_idx],
            best_q_value=found_q_values[best_idx],
            best_cost=found_costs[best_idx],
            all_paths=found_paths,
            all_goals=found_goals,
            all_scores=scores,
            all_q_values=found_q_values,
            all_costs=found_costs,
            stats=stats
        )

    def _select_candidate_goals(self, q_field: np.ndarray,
                                 target_mask: np.ndarray
                                 ) -> List[Tuple[Tuple[int, ...], float]]:
        """
        在目标区域内按 Q 值降序选出候选终点。

        为避免候选点聚集在同一高 Q 值区域，使用空间稀疏化：
        候选点之间至少相隔 penalty_radius 个像素。

        Returns:
            [(goal_coord, q_value), ...] 按 Q 值降序排列
        """
        # 获取目标区域内所有有效点及其 Q 值
        valid_indices = np.argwhere(target_mask)
        if len(valid_indices) == 0:
            return []

        q_values = np.array([q_field[tuple(idx)] for idx in valid_indices])

        # 按 Q 值降序排序
        sorted_order = np.argsort(-q_values)

        # 空间稀疏化
        min_dist = self.q_params.penalty_radius
        candidates = []
        selected_coords = []

        max_candidates = self.q_params.n_paths * 3  # 多选一些备用

        for order_idx in sorted_order:
            if len(candidates) >= max_candidates:
                break

            coord = tuple(valid_indices[order_idx].tolist())
            q_val = float(q_values[order_idx])

            # 检查与已选候选点的距离
            too_close = False
            for sel in selected_coords:
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(coord, sel)))
                if dist < min_dist:
                    too_close = True
                    break

            if not too_close:
                candidates.append((coord, q_val))
                selected_coords.append(coord)

        return candidates

    def _compute_scores(self, q_values: List[float],
                         costs: List[float]) -> List[float]:
        """
        计算归一化综合得分

        score = q_weight × Q_norm - cost_weight × cost_norm

        Q_norm 和 cost_norm 各自归一化到 [0, 1]
        """
        q_arr = np.array(q_values)
        cost_arr = np.array(costs)

        # Q 值归一化
        q_range = q_arr.max() - q_arr.min()
        if q_range > 1e-10:
            q_norm = (q_arr - q_arr.min()) / q_range
        else:
            q_norm = np.ones_like(q_arr)

        # 代价归一化
        cost_range = cost_arr.max() - cost_arr.min()
        if cost_range > 1e-10:
            cost_norm = (cost_arr - cost_arr.min()) / cost_range
        else:
            cost_norm = np.zeros_like(cost_arr)

        scores = self.q_params.q_weight * q_norm - self.q_params.cost_weight * cost_norm
        return scores.tolist()

    def _apply_penalty(self, path: List[Tuple[float, ...]],
                        penalty_masks: List[np.ndarray]) -> None:
        """在所有金字塔层级施加路径惩罚"""
        for level in range(self.n_levels):
            shape = self.planner.pyramid.shapes[level]
            scale = 2 ** level
            scaled_radius = max(1, self.q_params.penalty_radius // scale)
            step = max(1, scale)

            for point in path[::step]:
                idx = tuple(
                    max(0, min(int(round(c / scale)), shape[d] - 1))
                    for d, c in enumerate(point)
                )
                for neighbor in self._neighborhood_iterator(idx, scaled_radius, shape):
                    dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(idx, neighbor)))
                    if dist > scaled_radius:
                        continue
                    if self.q_params.penalty_decay == 'gaussian':
                        sigma = scaled_radius / 2.0
                        decay = exp(-(dist ** 2) / (2 * sigma ** 2))
                    else:
                        decay = 1.0 - dist / scaled_radius if scaled_radius > 0 else 1.0
                    penalty = 1.0 - self.q_params.penalty_weight * decay
                    penalty_masks[level][neighbor] = min(
                        penalty_masks[level][neighbor], penalty
                    )

    @staticmethod
    def _neighborhood_iterator(center, radius, shape):
        """生成邻域坐标"""
        ndim = len(shape)
        ranges = []
        for d in range(ndim):
            lo = max(0, center[d] - radius)
            hi = min(shape[d] - 1, center[d] + radius)
            ranges.append(range(lo, hi + 1))
        if ndim == 3:
            for z in ranges[0]:
                for y in ranges[1]:
                    for x in ranges[2]:
                        yield (z, y, x)
        elif ndim == 2:
            for y in ranges[0]:
                for x in ranges[1]:
                    yield (y, x)

    @staticmethod
    def _compute_path_separation(path_a, path_b) -> float:
        """计算两条路径平均最小距离"""
        if not path_a or not path_b:
            return 0.0
        step_a = max(1, len(path_a) // 100)
        step_b = max(1, len(path_b) // 100)
        arr_a = np.array(path_a[::step_a])
        arr_b = np.array(path_b[::step_b])
        min_dists = []
        for pa in arr_a:
            dists = np.sqrt(np.sum((arr_b - pa) ** 2, axis=1))
            min_dists.append(np.min(dists))
        return float(np.mean(min_dists))


def create_test_terrain_3d(shape: Tuple[int, int, int] = (30, 50, 50)) -> np.ndarray:
    """创建测试用3D地形"""
    nz, ny, nx = shape
    terrain = np.zeros(shape, dtype=np.float64)

    # 基础地形：起伏的山丘
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                terrain[z, y, x] = (
                    10 * np.sin(x / 10) * np.cos(y / 10) +
                    5 * np.sin(z / 5) +
                    z * 0.5  # 高度渐变
                )

    # 添加障碍物（高粘滞区域）
    terrain[10:20, 20:30, 20:30] += 50  # 中央障碍

    return terrain


def create_test_fog_3d(shape: Tuple[int, int, int] = (30, 50, 50)) -> np.ndarray:
    """创建测试用3D雾气场"""
    nz, ny, nx = shape
    fog = np.zeros(shape, dtype=np.float64)

    # 雾气区域
    center = (nz // 2, ny // 2, nx // 2)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                dist = np.sqrt((z - center[0])**2 +
                              (y - center[1])**2 +
                              (x - center[2])**2)
                fog[z, y, x] = max(0, 1 - dist / 20)

    return fog


def visualize_results(terrain: np.ndarray,
                      path: List[Tuple[float, ...]],
                      viscosity: np.ndarray,
                      stats: Dict[str, Any]):
    """可视化结果"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available, skipping visualization")
        return

    fig = plt.figure(figsize=(15, 10))

    ndim = terrain.ndim

    if ndim == 3:
        # 3D可视化
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')

        # 绘制路径
        if path:
            path_array = np.array(path)
            ax1.plot3D(path_array[:, 2], path_array[:, 1], path_array[:, 0],
                      'r-', linewidth=2, label='Path')
            ax1.scatter([path[0][2]], [path[0][1]], [path[0][0]],
                       c='g', s=100, marker='o', label='Start')
            ax1.scatter([path[-1][2]], [path[-1][1]], [path[-1][0]],
                       c='b', s=100, marker='*', label='Goal')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Path')
        ax1.legend()

        # 粘滞系数切片
        ax2 = fig.add_subplot(2, 2, 2)
        mid_z = terrain.shape[0] // 2
        im = ax2.imshow(viscosity[mid_z], cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax2, label='Viscosity')

        if path:
            # 绘制路径在该切片上的投影
            path_array = np.array(path)
            z_indices = np.abs(path_array[:, 0] - mid_z) < 3
            ax2.plot(path_array[z_indices, 2], path_array[z_indices, 1],
                    'r-', linewidth=2)

        ax2.set_title(f'Viscosity Field (Z={mid_z} slice)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # 地形切片
        ax3 = fig.add_subplot(2, 2, 3)
        im = ax3.imshow(terrain[mid_z], cmap='terrain', origin='lower')
        plt.colorbar(im, ax=ax3, label='Elevation')
        ax3.set_title(f'Terrain (Z={mid_z} slice)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

    else:
        # 2D可视化
        ax1 = fig.add_subplot(2, 2, 1)
        im = ax1.imshow(terrain, cmap='terrain', origin='lower')
        plt.colorbar(im, ax=ax1, label='Elevation')

        if path:
            path_array = np.array(path)
            ax1.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
            ax1.scatter([path[0][1]], [path[0][0]], c='g', s=100, marker='o')
            ax1.scatter([path[-1][1]], [path[-1][0]], c='b', s=100, marker='*')

        ax1.set_title('Terrain with Path')

        ax2 = fig.add_subplot(2, 2, 2)
        im = ax2.imshow(viscosity, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax2, label='Viscosity')
        ax2.set_title('Viscosity Field')

    # 统计信息
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    stats_text = "Statistics:\n"
    stats_text += f"Success: {stats.get('success', 'N/A')}\n"
    stats_text += f"Total Time: {stats.get('total_time', 0):.4f}s\n"
    stats_text += f"Path Length: {stats.get('path_length', 0)} points\n"
    stats_text += f"Path Cost: {stats.get('path_cost', 0):.2f}\n"
    stats_text += f"Pyramid Levels: {stats.get('n_levels', 0)}\n"

    if 'level_stats' in stats:
        stats_text += "\nPer-Level Stats:\n"
        for ls in stats['level_stats']:
            stats_text += f"  Level {ls['level']}: {ls['solve_time']:.4f}s, "
            stats_text += f"{ls['accepted_count']} accepted\n"

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax4.set_title('Statistics')

    plt.tight_layout()
    plt.savefig('hierarchical_fmm_result.png', dpi=150)
    plt.show()
    print("Result saved to hierarchical_fmm_result.png")


def visualize_multi_path_results(terrain: np.ndarray,
                                  result: MultiPathResult,
                                  viscosity: np.ndarray):
    """多路径可视化结果"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available, skipping visualization")
        return

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    fig = plt.figure(figsize=(16, 12))
    ndim = terrain.ndim
    paths = result.paths

    if ndim == 3:
        # 3D路径视图
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        for i, path in enumerate(paths):
            if path:
                path_array = np.array(path)
                c = colors[i % len(colors)]
                label = f'Path {i+1} (cost={result.costs[i]:.1f})'
                ax1.plot3D(path_array[:, 2], path_array[:, 1], path_array[:, 0],
                          '-', color=c, linewidth=2, label=label)
        if paths:
            ax1.scatter([paths[0][0][2]], [paths[0][0][1]], [paths[0][0][0]],
                       c='lime', s=100, marker='o', label='Start', zorder=5)
            ax1.scatter([paths[0][-1][2]], [paths[0][-1][1]], [paths[0][-1][0]],
                       c='black', s=100, marker='*', label='Goal', zorder=5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Multi-Path 3D View')
        ax1.legend(fontsize=8)

        # 粘滞系数切片 + 多路径投影
        ax2 = fig.add_subplot(2, 2, 2)
        mid_z = terrain.shape[0] // 2
        im = ax2.imshow(viscosity[mid_z], cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax2, label='Viscosity')
        for i, path in enumerate(paths):
            if path:
                path_array = np.array(path)
                z_indices = np.abs(path_array[:, 0] - mid_z) < 3
                if np.any(z_indices):
                    ax2.plot(path_array[z_indices, 2], path_array[z_indices, 1],
                            '-', color=colors[i % len(colors)], linewidth=2,
                            label=f'Path {i+1}')
        ax2.set_title(f'Viscosity + Paths (Z={mid_z} slice)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend(fontsize=8)

        # 地形切片 + 多路径
        ax3 = fig.add_subplot(2, 2, 3)
        im = ax3.imshow(terrain[mid_z], cmap='terrain', origin='lower')
        plt.colorbar(im, ax=ax3, label='Elevation')
        for i, path in enumerate(paths):
            if path:
                path_array = np.array(path)
                z_indices = np.abs(path_array[:, 0] - mid_z) < 3
                if np.any(z_indices):
                    ax3.plot(path_array[z_indices, 2], path_array[z_indices, 1],
                            '-', color=colors[i % len(colors)], linewidth=2)
        ax3.set_title(f'Terrain + Paths (Z={mid_z} slice)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

    else:
        # 2D可视化
        ax1 = fig.add_subplot(2, 2, 1)
        im = ax1.imshow(terrain, cmap='terrain', origin='lower')
        plt.colorbar(im, ax=ax1, label='Elevation')
        for i, path in enumerate(paths):
            if path:
                path_array = np.array(path)
                c = colors[i % len(colors)]
                ax1.plot(path_array[:, 1], path_array[:, 0], '-', color=c,
                        linewidth=2, label=f'Path {i+1}')
        if paths:
            ax1.scatter([paths[0][0][1]], [paths[0][0][0]], c='lime', s=100, marker='o')
            ax1.scatter([paths[0][-1][1]], [paths[0][-1][0]], c='black', s=100, marker='*')
        ax1.set_title('Terrain with Multi-Paths')
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(2, 2, 2)
        im = ax2.imshow(viscosity, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax2, label='Viscosity')
        for i, path in enumerate(paths):
            if path:
                path_array = np.array(path)
                ax2.plot(path_array[:, 1], path_array[:, 0], '-',
                        color=colors[i % len(colors)], linewidth=2)
        ax2.set_title('Viscosity + Paths')

        ax3 = fig.add_subplot(2, 2, 3)

    # 统计信息面板
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    stats_text = "Multi-Path Statistics:\n"
    stats_text += f"Paths found: {result.stats.get('n_paths_found', 0)}"
    stats_text += f" / {result.stats.get('n_paths_requested', 0)}\n"
    stats_text += f"Total time: {result.stats.get('total_time', 0):.4f}s\n\n"

    for i, (path, cost) in enumerate(zip(result.paths, result.costs)):
        ratio = cost / result.costs[0] if result.costs[0] > 0 else 0
        tag = " (optimal)" if i == 0 else f" (+{(ratio - 1) * 100:.0f}%)"
        stats_text += f"Path {i+1}: cost={cost:.2f}, points={len(path)}{tag}\n"

    if result.diversity_matrix is not None and len(result.paths) > 1:
        stats_text += "\nDiversity (avg min dist):\n"
        k = len(result.paths)
        header = "      " + "".join(f"  P{j+1:d}" for j in range(k))
        stats_text += header + "\n"
        for i in range(k):
            row = f"  P{i+1:d} "
            for j in range(k):
                if i == j:
                    row += "    - "
                else:
                    row += f" {result.diversity_matrix[i, j]:5.1f}"
            stats_text += row + "\n"

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    ax4.set_title('Statistics')

    plt.tight_layout()
    plt.savefig('multi_path_fmm_result.png', dpi=150)
    plt.show()
    print("Result saved to multi_path_fmm_result.png")


def visualize_waypoint_allocation(terrain: np.ndarray,
                                   result: WaypointAllocationResult,
                                   gas_params: GasDiffusionParams,
                                   viscosity: np.ndarray):
    """多机器人路径点分配结果可视化"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available, skipping visualization")
        return

    robot_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan',
                    'magenta', 'yellow']
    n_robots = gas_params.n_robots
    ndim = terrain.ndim

    fig = plt.figure(figsize=(18, 12))

    if ndim == 3:
        mid_z = terrain.shape[0] // 2

        # 1. 领土地图切片
        ax1 = fig.add_subplot(2, 3, 1)
        territory_slice = result.territory_map[mid_z]
        # 自定义颜色映射: -1=白色, 0=red, 1=blue, ...
        cmap_colors = ['white'] + robot_colors[:n_robots]
        cmap = ListedColormap(cmap_colors)
        im = ax1.imshow(territory_slice, cmap=cmap, origin='lower',
                        vmin=-1, vmax=n_robots - 1, alpha=0.6)
        # 标注路径点
        for r in range(n_robots):
            for wp in result.assignments.get(r, []):
                if abs(wp[0] - mid_z) < 3:
                    ax1.scatter(wp[2], wp[1], c=robot_colors[r % len(robot_colors)],
                               s=120, marker='D', edgecolors='black', linewidth=1.5, zorder=5)
        # 标注起点
        for r, start in enumerate(gas_params.robot_starts):
            if abs(start[0] - mid_z) < 3:
                ax1.scatter(start[2], start[1], c=robot_colors[r % len(robot_colors)],
                           s=150, marker='o', edgecolors='black', linewidth=2, zorder=6)
        ax1.set_title(f'Territory Map (Z={mid_z} slice)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # 2. 地形 + 路径段
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(terrain[mid_z], cmap='terrain', origin='lower')
        plt.colorbar(im2, ax=ax2, label='Elevation')
        for r in range(n_robots):
            c = robot_colors[r % len(robot_colors)]
            for seg_idx, seg in enumerate(result.robot_paths.get(r, [])):
                if seg:
                    path_array = np.array(seg)
                    z_mask = np.abs(path_array[:, 0] - mid_z) < 3
                    if np.any(z_mask):
                        label = f'Robot {r}' if seg_idx == 0 else None
                        ax2.plot(path_array[z_mask, 2], path_array[z_mask, 1],
                                '-', color=c, linewidth=2, label=label)
        ax2.legend(fontsize=7)
        ax2.set_title(f'Terrain + Paths (Z={mid_z} slice)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # 3. 3D路径视图
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        for r in range(n_robots):
            c = robot_colors[r % len(robot_colors)]
            for seg_idx, seg in enumerate(result.robot_paths.get(r, [])):
                if seg and len(seg) > 1:
                    path_array = np.array(seg)
                    label = f'Robot {r}' if seg_idx == 0 else None
                    ax3.plot3D(path_array[:, 2], path_array[:, 1], path_array[:, 0],
                              '-', color=c, linewidth=2, label=label)
            # 标注起点
            s = gas_params.robot_starts[r]
            ax3.scatter([s[2]], [s[1]], [s[0]], c=c, s=100, marker='o',
                       edgecolors='black', zorder=5)
            # 标注路径点
            for wp in result.assignments.get(r, []):
                ax3.scatter([wp[2]], [wp[1]], [wp[0]], c=c, s=80, marker='D',
                           edgecolors='black', zorder=5)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('3D Multi-Robot Paths')
        ax3.legend(fontsize=7)

        # 4. 粘滞系数 + 领土边界
        ax4 = fig.add_subplot(2, 3, 4)
        im4 = ax4.imshow(viscosity[mid_z], cmap='viridis', origin='lower')
        plt.colorbar(im4, ax=ax4, label='Viscosity')
        # 领土边界叠加
        territory_slice = result.territory_map[mid_z]
        # 简单边界检测
        boundary = np.zeros_like(territory_slice, dtype=bool)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(territory_slice, dy, axis=0), dx, axis=1)
            boundary |= (territory_slice != shifted) & (territory_slice >= 0)
        ax4.contour(boundary.astype(float), levels=[0.5], colors='white',
                    linewidths=1.5)
        ax4.set_title(f'Viscosity + Territory Boundary (Z={mid_z})')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')

    else:
        # 2D 可视化
        # 1. 领土地图
        ax1 = fig.add_subplot(2, 3, 1)
        cmap_colors = ['white'] + robot_colors[:n_robots]
        cmap = ListedColormap(cmap_colors)
        ax1.imshow(result.territory_map, cmap=cmap, origin='lower',
                   vmin=-1, vmax=n_robots - 1, alpha=0.6)
        for r in range(n_robots):
            for wp in result.assignments.get(r, []):
                ax1.scatter(wp[1], wp[0], c=robot_colors[r % len(robot_colors)],
                           s=120, marker='D', edgecolors='black', linewidth=1.5, zorder=5)
        for r, start in enumerate(gas_params.robot_starts):
            ax1.scatter(start[1], start[0], c=robot_colors[r % len(robot_colors)],
                       s=150, marker='o', edgecolors='black', linewidth=2, zorder=6)
        ax1.set_title('Territory Map')

        # 2. 地形 + 路径段
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(terrain, cmap='terrain', origin='lower')
        plt.colorbar(im2, ax=ax2, label='Elevation')
        for r in range(n_robots):
            c = robot_colors[r % len(robot_colors)]
            for seg_idx, seg in enumerate(result.robot_paths.get(r, [])):
                if seg:
                    path_array = np.array(seg)
                    label = f'Robot {r}' if seg_idx == 0 else None
                    ax2.plot(path_array[:, 1], path_array[:, 0],
                            '-', color=c, linewidth=2, label=label)
        ax2.legend(fontsize=7)
        ax2.set_title('Terrain + Paths')

        # 3. 粘滞系数
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(viscosity, cmap='viridis', origin='lower')
        plt.colorbar(im3, ax=ax3, label='Viscosity')
        ax3.set_title('Viscosity Field')

        # 4. 领土边界
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(terrain, cmap='terrain', origin='lower', alpha=0.5)
        boundary = np.zeros_like(result.territory_map, dtype=bool)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(result.territory_map, dy, axis=0), dx, axis=1)
            boundary |= (result.territory_map != shifted) & (result.territory_map >= 0)
        ax4.contour(boundary.astype(float), levels=[0.5], colors='red', linewidths=1.5)
        ax4.set_title('Territory Boundary')

    # 5. 时间线（各路径点到达时间）
    ax5 = fig.add_subplot(2, 3, 5)
    for r in range(n_robots):
        times = result.arrival_times.get(r, [])
        if times:
            ax5.barh([f'R{r}-WP{i}' for i in range(len(times))],
                     times, color=robot_colors[r % len(robot_colors)], alpha=0.8)
    ax5.set_xlabel('Arrival Time (phi)')
    ax5.set_title('Waypoint Arrival Timeline')

    # 6. 统计面板
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = "Waypoint Allocation Statistics:\n"
    stats_text += f"Robots: {result.stats.get('n_robots', 0)}\n"
    stats_text += f"Waypoints: {result.stats.get('n_waypoints_assigned', 0)}"
    stats_text += f" / {result.stats.get('n_waypoints_total', 0)}\n"
    stats_text += f"Alloc time: {result.stats.get('allocation_time', 0):.4f}s\n"
    stats_text += f"Path plan time: {result.stats.get('path_planning_time', 0):.4f}s\n"
    stats_text += f"Total time: {result.stats.get('total_time', 0):.4f}s\n"
    stats_text += "\nPer-Robot Details:\n"

    for r in range(n_robots):
        per = result.stats.get('per_robot', {}).get(r, {})
        n_wp = per.get('n_waypoints', 0)
        n_seg = per.get('n_segments', 0)
        times_str = ', '.join(f'{t:.1f}' for t in per.get('arrival_times', []))
        stats_text += f"  Robot {r}: {n_wp} WPs, {n_seg} segs\n"
        if times_str:
            stats_text += f"    t=[{times_str}]\n"

    stats_text += "\nAssignment Order:\n"
    for robot_id, wp_idx in result.assignment_order:
        wp = gas_params.waypoints[wp_idx]
        stats_text += f"  R{robot_id} -> WP{wp_idx} {wp}\n"

    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace')
    ax6.set_title('Statistics')

    plt.suptitle('Multi-Robot Waypoint Allocation (Immiscible Gas Diffusion + Huygens)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('waypoint_allocation_result.png', dpi=150)
    plt.show()
    print("Result saved to waypoint_allocation_result.png")


    plt.savefig('waypoint_allocation_result.png', dpi=150)
    plt.show()
    print("Result saved to waypoint_allocation_result.png")


def visualize_q_value_results(terrain: np.ndarray,
                                q_field: np.ndarray,
                                target_mask: np.ndarray,
                                result: QValueMultiPathResult):
    """Q值多路径规划结果可视化"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available, skipping visualization")
        return

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    ndim = terrain.ndim
    fig = plt.figure(figsize=(18, 12))

    if ndim == 3:
        mid_z = terrain.shape[0] // 2

        # 1. Q值场 + 目标区域 + 候选路径
        ax1 = fig.add_subplot(2, 3, 1)
        q_slice = q_field[mid_z].copy()
        mask_slice = target_mask[mid_z]
        # Q值场只在目标区域内显示
        q_display = np.full_like(q_slice, np.nan)
        q_display[mask_slice] = q_slice[mask_slice]
        im1 = ax1.imshow(q_display, cmap='hot', origin='lower', alpha=0.8)
        plt.colorbar(im1, ax=ax1, label='Q Value')
        # 目标区域边界
        ax1.contour(mask_slice.astype(float), levels=[0.5], colors='lime',
                    linewidths=2, linestyles='--')
        # 候选终点
        for i, goal in enumerate(result.all_goals):
            if abs(goal[0] - mid_z) < 3:
                is_best = (i == result.stats.get('best_index', -1))
                marker = '*' if is_best else 'o'
                size = 200 if is_best else 80
                ax1.scatter(goal[2], goal[1], c=colors[i % len(colors)],
                           s=size, marker=marker, edgecolors='black', linewidth=1.5, zorder=5)
        ax1.set_title(f'Q-Value Field + Goals (Z={mid_z})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # 2. 地形 + 所有候选路径（最优路径加粗）
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(terrain[mid_z], cmap='terrain', origin='lower')
        plt.colorbar(im2, ax=ax2, label='Elevation')
        best_idx = result.stats.get('best_index', 0)
        for i, path in enumerate(result.all_paths):
            if path:
                path_array = np.array(path)
                z_mask = np.abs(path_array[:, 0] - mid_z) < 3
                if np.any(z_mask):
                    is_best = (i == best_idx)
                    lw = 3 if is_best else 1.5
                    alpha = 1.0 if is_best else 0.5
                    label = f'Path {i} (Q={result.all_q_values[i]:.1f}, score={result.all_scores[i]:.2f})'
                    if is_best:
                        label += ' [BEST]'
                    ax2.plot(path_array[z_mask, 2], path_array[z_mask, 1],
                            '-', color=colors[i % len(colors)],
                            linewidth=lw, alpha=alpha, label=label)
        # 起点
        if result.best_path:
            sp = result.best_path[0]
            ax2.scatter(sp[2], sp[1], c='lime', s=150, marker='o',
                       edgecolors='black', linewidth=2, zorder=6, label='Start')
        ax2.legend(fontsize=6, loc='upper left')
        ax2.set_title(f'Terrain + Candidate Paths (Z={mid_z})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # 3. 3D 最优路径
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        if result.best_path:
            path_array = np.array(result.best_path)
            ax3.plot3D(path_array[:, 2], path_array[:, 1], path_array[:, 0],
                      'r-', linewidth=3, label='Best Path')
            ax3.scatter([path_array[0, 2]], [path_array[0, 1]], [path_array[0, 0]],
                       c='lime', s=150, marker='o', zorder=5, label='Start')
            bg = result.best_goal
            ax3.scatter([bg[2]], [bg[1]], [bg[0]],
                       c='gold', s=200, marker='*', zorder=5,
                       label=f'Best Goal (Q={result.best_q_value:.1f})')
        # 其他路径（透明）
        for i, path in enumerate(result.all_paths):
            if i != best_idx and path:
                pa = np.array(path)
                ax3.plot3D(pa[:, 2], pa[:, 1], pa[:, 0],
                          '-', color=colors[i % len(colors)], linewidth=1, alpha=0.4)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('3D Best Path')
        ax3.legend(fontsize=7)

        # 4. 得分对比柱状图
        ax4 = fig.add_subplot(2, 3, 4)
        n_paths = len(result.all_paths)
        x_pos = np.arange(n_paths)
        bar_width = 0.35
        bars_q = ax4.bar(x_pos - bar_width / 2, result.all_q_values, bar_width,
                         label='Q Value (norm)', color='steelblue', alpha=0.8)
        # 归一化代价（反转：低代价 = 高条形）
        if result.all_costs:
            max_cost = max(result.all_costs)
            inv_costs = [1.0 - c / max_cost if max_cost > 0 else 0 for c in result.all_costs]
        else:
            inv_costs = []
        bars_c = ax4.bar(x_pos + bar_width / 2, inv_costs, bar_width,
                         label='1 - Cost (norm)', color='coral', alpha=0.8)
        # 标注最优
        if 0 <= best_idx < n_paths:
            ax4.annotate('BEST', (x_pos[best_idx], max(result.all_q_values[best_idx],
                         inv_costs[best_idx]) + 0.05),
                        ha='center', fontweight='bold', color='red')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'P{i}' for i in range(n_paths)])
        ax4.set_ylabel('Value')
        ax4.set_title('Q-Value vs Cost Tradeoff')
        ax4.legend(fontsize=8)

    else:
        # 2D 可视化
        ax1 = fig.add_subplot(2, 3, 1)
        q_display = np.full_like(q_field, np.nan, dtype=float)
        q_display[target_mask] = q_field[target_mask]
        im1 = ax1.imshow(q_display, cmap='hot', origin='lower', alpha=0.8)
        plt.colorbar(im1, ax=ax1, label='Q Value')
        ax1.contour(target_mask.astype(float), levels=[0.5], colors='lime', linewidths=2)
        for i, goal in enumerate(result.all_goals):
            is_best = (i == result.stats.get('best_index', -1))
            marker = '*' if is_best else 'o'
            size = 200 if is_best else 80
            ax1.scatter(goal[1], goal[0], c=colors[i % len(colors)],
                       s=size, marker=marker, edgecolors='black', zorder=5)
        ax1.set_title('Q-Value Field + Goals')

        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(terrain, cmap='terrain', origin='lower')
        plt.colorbar(im2, ax=ax2, label='Elevation')
        best_idx = result.stats.get('best_index', 0)
        for i, path in enumerate(result.all_paths):
            if path:
                pa = np.array(path)
                is_best = (i == best_idx)
                lw = 3 if is_best else 1.5
                alpha = 1.0 if is_best else 0.5
                label = f'P{i} (Q={result.all_q_values[i]:.1f})'
                if is_best:
                    label += ' [BEST]'
                ax2.plot(pa[:, 1], pa[:, 0], '-', color=colors[i % len(colors)],
                        linewidth=lw, alpha=alpha, label=label)
        ax2.legend(fontsize=6)
        ax2.set_title('Terrain + Candidate Paths')

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(terrain, cmap='terrain', origin='lower', alpha=0.5)
        ax3.set_title('Viscosity Field')

        ax4 = fig.add_subplot(2, 3, 4)
        n_paths = len(result.all_paths)
        x_pos = np.arange(n_paths)
        ax4.bar(x_pos, result.all_scores, color=[colors[i % len(colors)] for i in range(n_paths)])
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'P{i}' for i in range(n_paths)])
        ax4.set_title('Composite Scores')

    # 5. 综合得分排名
    ax5 = fig.add_subplot(2, 3, 5)
    if result.all_scores:
        sorted_indices = np.argsort(result.all_scores)[::-1]
        y_pos = np.arange(len(sorted_indices))
        bar_colors = [colors[i % len(colors)] for i in sorted_indices]
        bars = ax5.barh(y_pos, [result.all_scores[i] for i in sorted_indices],
                        color=bar_colors, alpha=0.8)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([f'Path {i}' for i in sorted_indices])
        ax5.set_xlabel('Composite Score')
        ax5.set_title('Score Ranking')
        ax5.invert_yaxis()

    # 6. 统计面板
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = "Q-Value Multi-Path Statistics:\n"
    stats_text += f"Candidates evaluated: {result.stats.get('n_candidates', 0)}\n"
    stats_text += f"Paths found: {result.stats.get('n_paths_found', 0)}\n"
    stats_text += f"Q weight: {result.stats.get('q_weight', 0)}\n"
    stats_text += f"Cost weight: {result.stats.get('cost_weight', 0)}\n"
    stats_text += f"Total time: {result.stats.get('total_time', 0):.4f}s\n"
    stats_text += f"\nBest Path (#{result.stats.get('best_index', 0)}):\n"
    stats_text += f"  Goal: {result.best_goal}\n"
    stats_text += f"  Q Value: {result.best_q_value:.2f}\n"
    stats_text += f"  Path Cost: {result.best_cost:.2f}\n"
    stats_text += f"  Score: {result.best_score:.3f}\n"
    stats_text += f"\nAll Paths:\n"
    for i in range(len(result.all_paths)):
        tag = " <-- BEST" if i == result.stats.get('best_index', -1) else ""
        stats_text += f"  P{i}: Q={result.all_q_values[i]:.1f}"
        stats_text += f", cost={result.all_costs[i]:.1f}"
        stats_text += f", score={result.all_scores[i]:.2f}{tag}\n"

    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace')
    ax6.set_title('Statistics')

    plt.suptitle('Q-Value Guided Multi-Path Planning (max Q with cost tradeoff)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('q_value_multipath_result.png', dpi=150)
    plt.show()
    print("Result saved to q_value_multipath_result.png")


def main():
    """主函数：测试分层路径规划"""
    print("=" * 60)
    print("Viscous Hierarchical Fast Marching Method - Demo")
    print("=" * 60)

    # 创建测试数据
    print("\n1. Creating test terrain and fog data...")
    shape = (30, 50, 50)
    terrain = create_test_terrain_3d(shape)
    fog_data = create_test_fog_3d(shape)

    print(f"   Terrain shape: {terrain.shape}")
    print(f"   Terrain range: [{terrain.min():.2f}, {terrain.max():.2f}]")
    print(f"   Fog range: [{fog_data.min():.2f}, {fog_data.max():.2f}]")

    # 配置参数
    print("\n2. Configuring planner...")
    params = ViscosityParams(
        slope_weight=0.3,
        roughness_weight=0.2,
        fog_weight=0.5
    )

    # 创建规划器
    planner = HierarchicalPathPlanner(
        terrain=terrain,
        fog_data=fog_data,
        n_levels=3,
        viscosity_params=params,
        spacing=(1.0, 1.0, 1.0)
    )

    print(f"   Pyramid levels: {planner.n_levels}")
    for i, shape in enumerate(planner.pyramid.shapes):
        print(f"   Level {i}: {shape}")

    # 定义起点终点
    start = (2, 5, 5)
    goal = (25, 45, 45)
    print(f"\n3. Planning path from {start} to {goal}...")

    # 分层规划
    print("\n   Running hierarchical planning...")
    path_hierarchical, stats_hierarchical = planner.plan(start, goal)

    print(f"   Success: {stats_hierarchical['success']}")
    print(f"   Total time: {stats_hierarchical['total_time']:.4f}s")
    print(f"   Path points: {stats_hierarchical['path_length']}")
    print(f"   Path cost: {stats_hierarchical['path_cost']:.2f}")

    # 单分辨率规划（对比）
    print("\n   Running single-resolution planning for comparison...")
    path_single, stats_single = planner.plan_with_single_resolution(start, goal)

    print(f"   Success: {stats_single.get('success', False)}")
    print(f"   Solve time: {stats_single.get('solve_time', 0):.4f}s")
    print(f"   Accepted points: {stats_single.get('accepted_count', 0)}")

    # 内存使用对比
    print("\n4. Memory usage comparison:")

    total_hierarchical = sum(
        ls.get('phi_points', 0) for ls in stats_hierarchical.get('level_stats', [])
    )
    single_resolution = stats_single.get('phi_points', 0)

    print(f"   Hierarchical total phi points: {total_hierarchical}")
    print(f"   Single resolution phi points: {single_resolution}")

    if single_resolution > 0:
        reduction = (1 - total_hierarchical / single_resolution) * 100
        print(f"   Memory reduction: {reduction:.1f}%")

    # 可视化
    print("\n5. Generating visualization...")
    viscosity = planner.viscosity_field.compute_viscosity(terrain, fog_data)

    try:
        visualize_results(terrain, path_hierarchical, viscosity, stats_hierarchical)
    except Exception as e:
        print(f"   Visualization skipped: {e}")

    print("\n" + "=" * 60)
    print("Single-Path Demo completed!")
    print("=" * 60)

    # ======== 多路径规划 Demo ========
    print("\n" + "=" * 60)
    print("Multi-Path Planning Demo")
    print("=" * 60)

    multi_params = MultiPathParams(
        n_paths=3,
        penalty_weight=0.5,
        penalty_radius=5,
        min_path_separation=8.0,
        max_cost_ratio=3.0
    )

    multi_planner = MultiPathPlanner(
        terrain=terrain,
        fog_data=fog_data,
        n_levels=3,
        viscosity_params=params,
        multi_path_params=multi_params,
        spacing=(1.0, 1.0, 1.0)
    )

    print(f"\n6. Finding {multi_params.n_paths} diverse paths from {start} to {goal}...")
    result = multi_planner.plan_multi_path(start, goal)

    print(f"   Paths found: {result.stats.get('n_paths_found', 0)}")
    print(f"   Total time: {result.stats.get('total_time', 0):.4f}s")

    for i, (p, cost) in enumerate(zip(result.paths, result.costs)):
        ratio = cost / result.costs[0] if result.costs[0] > 0 else 0
        tag = "(optimal)" if i == 0 else f"(+{(ratio - 1) * 100:.0f}%)"
        print(f"   Path {i+1}: cost={cost:.2f} {tag}, points={len(p)}")

    if result.diversity_matrix is not None and len(result.paths) > 1:
        print(f"\n   Diversity matrix (avg min distance):")
        k = len(result.paths)
        header = "         " + "".join(f"  P{j+1:d}" for j in range(k))
        print(header)
        for i in range(k):
            row = f"      P{i+1:d} "
            for j in range(k):
                if i == j:
                    row += "    - "
                else:
                    row += f" {result.diversity_matrix[i, j]:5.1f}"
            print(row)

    print("\n7. Generating multi-path visualization...")
    try:
        visualize_multi_path_results(terrain, result, viscosity)
    except Exception as e:
        print(f"   Visualization skipped: {e}")

    # ======== 多机器人路径点分配 Demo ========
    print("\n" + "=" * 60)
    print("Multi-Robot Waypoint Allocation Demo")
    print("(Immiscible Gas Diffusion + Huygens Principle)")
    print("=" * 60)

    # 定义 3 个机器人起点和 6 个路径点
    robot_starts = [
        (2, 5, 5),      # Robot 0: 左下前方
        (2, 45, 5),     # Robot 1: 左上前方
        (2, 25, 45),    # Robot 2: 右中前方
    ]
    waypoints = [
        (10, 15, 15),   # WP0
        (10, 35, 15),   # WP1
        (10, 25, 25),   # WP2
        (20, 10, 40),   # WP3
        (20, 40, 40),   # WP4
        (15, 25, 10),   # WP5
    ]

    gas_params = GasDiffusionParams(
        n_robots=3,
        robot_starts=robot_starts,
        waypoints=waypoints,
        max_waypoints_per_robot=None  # 不限制
    )

    print(f"\n8. Allocating {len(waypoints)} waypoints to {gas_params.n_robots} robots...")
    print(f"   Robot starts: {robot_starts}")
    print(f"   Waypoints: {waypoints}")

    allocator = MultiRobotWaypointAllocator(
        terrain=terrain,
        fog_data=fog_data,
        n_levels=3,
        viscosity_params=params,
        gas_params=gas_params,
        spacing=(1.0, 1.0, 1.0)
    )

    alloc_result = allocator.allocate()

    print(f"\n   Allocation results:")
    print(f"   Total time: {alloc_result.stats.get('total_time', 0):.4f}s")
    print(f"   Waypoints assigned: {alloc_result.stats.get('n_waypoints_assigned', 0)}"
          f" / {alloc_result.stats.get('n_waypoints_total', 0)}")

    for r in range(gas_params.n_robots):
        wps = alloc_result.assignments.get(r, [])
        times = alloc_result.arrival_times.get(r, [])
        print(f"   Robot {r}: {len(wps)} waypoints")
        for i, (wp, t) in enumerate(zip(wps, times)):
            print(f"     WP {wp} @ t={t:.2f}")

    print(f"\n   Assignment order (global):")
    for robot_id, wp_idx in alloc_result.assignment_order:
        wp = waypoints[wp_idx]
        t = alloc_result.arrival_times[robot_id][
            alloc_result.assignments[robot_id].index(wp)]
        print(f"     Robot {robot_id} -> WP{wp_idx} {wp} @ t={t:.2f}")

    print("\n9. Generating waypoint allocation visualization...")
    try:
        visualize_waypoint_allocation(terrain, alloc_result, gas_params, viscosity)
    except Exception as e:
        print(f"   Visualization skipped: {e}")

    # ======== Q值多路径规划 Demo ========
    print("\n" + "=" * 60)
    print("Q-Value Guided Multi-Path Planning Demo")
    print("(Find highest-scoring goal in target region)")
    print("=" * 60)

    # 创建 Q 值场：在目标区域内模拟一个得分分布
    # 目标区域：终点附近的一个区域
    print("\n10. Creating Q-value field and target region...")
    terrain_shape = terrain.shape
    q_field = np.zeros(terrain_shape, dtype=np.float64)
    target_mask = np.zeros(terrain_shape, dtype=bool)

    # 目标区域：z=20-28, y=35-48, x=35-48 的立方体区域
    target_mask[20:28, 35:48, 35:48] = True

    # Q值场：在目标区域内模拟多个高分点（比如多个感兴趣目标）
    # 高分点1：(24, 40, 40) Q=10.0 — 最高分但���离起点
    # 高分点2：(22, 38, 38) Q=7.0 — 中等分
    # 高分点3：(21, 42, 45) Q=9.0 — 高分，位置不同
    for z in range(terrain_shape[0]):
        for y in range(terrain_shape[1]):
            for x in range(terrain_shape[2]):
                if target_mask[z, y, x]:
                    # 基础 Q 值：距离目标区域中心的函数
                    d1 = np.sqrt((z - 24)**2 + (y - 40)**2 + (x - 40)**2)
                    d2 = np.sqrt((z - 22)**2 + (y - 38)**2 + (x - 38)**2)
                    d3 = np.sqrt((z - 21)**2 + (y - 42)**2 + (x - 45)**2)
                    q_field[z, y, x] = (
                        10.0 * np.exp(-d1**2 / 18.0) +
                        7.0 * np.exp(-d2**2 / 12.0) +
                        9.0 * np.exp(-d3**2 / 15.0) +
                        np.random.uniform(0, 0.5)  # 微小随机扰动
                    )

    print(f"   Target region size: {target_mask.sum()} voxels")
    print(f"   Q-value range in target: [{q_field[target_mask].min():.2f}, {q_field[target_mask].max():.2f}]")

    q_params = QValueMultiPathParams(
        n_paths=5,
        q_weight=1.0,
        cost_weight=0.5,
        penalty_weight=0.4,
        penalty_radius=5,
        min_path_separation=6.0,
        max_cost_ratio=5.0
    )

    q_planner = QValueMultiPathPlanner(
        terrain=terrain,
        fog_data=fog_data,
        n_levels=3,
        viscosity_params=params,
        q_params=q_params,
        spacing=(1.0, 1.0, 1.0)
    )

    print(f"\n11. Planning paths from {start} to target region (max Q)...")
    q_result = q_planner.plan(start, q_field, target_mask)

    if q_result.stats.get('success'):
        print(f"   Paths found: {q_result.stats.get('n_paths_found', 0)}")
        print(f"   Total time: {q_result.stats.get('total_time', 0):.4f}s")
        print(f"\n   Best path:")
        print(f"     Goal: {q_result.best_goal}")
        print(f"     Q value: {q_result.best_q_value:.2f}")
        print(f"     Path cost: {q_result.best_cost:.2f}")
        print(f"     Composite score: {q_result.best_score:.3f}")
        print(f"\n   All candidates:")
        for i in range(len(q_result.all_paths)):
            tag = " <-- BEST" if i == q_result.stats.get('best_index') else ""
            print(f"     P{i}: goal={q_result.all_goals[i]}, "
                  f"Q={q_result.all_q_values[i]:.2f}, "
                  f"cost={q_result.all_costs[i]:.2f}, "
                  f"score={q_result.all_scores[i]:.3f}{tag}")
    else:
        print(f"   Failed: {q_result.stats.get('error', 'unknown')}")

    print("\n12. Generating Q-value multi-path visualization...")
    try:
        visualize_q_value_results(terrain, q_field, target_mask, q_result)
    except Exception as e:
        print(f"   Visualization skipped: {e}")

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)

    return q_result


if __name__ == "__main__":
    main()
