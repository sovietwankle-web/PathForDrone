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
from dataclasses import dataclass
import time


@dataclass
class ViscosityParams:
    """粘滞系数参数配置"""
    slope_weight: float = 0.3       # 坡度影响权重
    roughness_weight: float = 0.2   # 粗糙度影响权重
    fog_weight: float = 0.5         # 雾气/环境影响权重
    roughness_window: int = 3       # 粗糙度计算窗口大小
    min_viscosity: float = 0.01     # 最小粘滞系数（防止除零）


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
    print("Demo completed successfully!")
    print("=" * 60)

    return path_hierarchical, stats_hierarchical


if __name__ == "__main__":
    main()
