"""
基于Eikonal方程的动态雾气环境路径规划系统

结合动态雾气障碍物和快速行进法(FMM)的四维路径规划
支持时间变化的水汽/雾气障碍物环境

数学理论基础:
1. 四维Eikonal方程: ||∇Φ(x,t)|| = 1/f(x,t)
2. 动态速度场: f(x,t) = σ(x,t) 考虑地形和雾气的时变影响
3. 时空梯度下降: 在(x,y,z,t)四维空间中寻找最优路径

作者: Kilo Code
日期: 2025-08-26
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
from scipy.ndimage import gaussian_filter
import time
from typing import Tuple, List, Optional, Dict
import os
import csv
import rasterio
from pathlib import Path

# 导入之前的动态雾气环境
from dynamic_fog_path_planning_optimized import OptimizedDynamicFogEnvironment


class DynamicEikonalSolver:
    """
    四维动态Eikonal方程求解器
    
    求解方程: ||∇Φ(x,t)|| = 1/f(x,t)
    其中 Φ(x,t) 是时空势函数，f(x,t) 是动态速度函数
    """
    
    def __init__(self, shape: Tuple[int, int, int], time_steps: int,
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 dt: float = 1.0):
        """
        初始化动态求解器
        
        Args:
            shape: 三维空间网格形状 (nz, ny, nx)
            time_steps: 时间步数
            spacing: 空间网格间距 (dz, dy, dx)
            dt: 时间步长
        """
        self.shape = shape
        self.nz, self.ny, self.nx = shape
        self.time_steps = time_steps
        self.dz, self.dy, self.dx = spacing
        self.dt = dt
        
        # 四维势场数组 (t, z, y, x)
        self.phi = np.full((time_steps,) + shape, np.inf, dtype=np.float64)
        
        # 状态标记: 0=Far, 1=Considered, 2=Accepted
        self.status = np.zeros((time_steps,) + shape, dtype=np.int32)
        
        # 动态速度场 (t, z, y, x)
        self.speed_field = np.ones((time_steps,) + shape, dtype=np.float64)
        
        # 优先队列 (phi_value, (t, z, y, x))
        self.heap = []
        
    def set_dynamic_speed_field(self, speed_field: np.ndarray):
        """设置动态速度场 f(x,t)"""
        self.speed_field = np.maximum(speed_field, 1e-10)  # 避免除零
        
    def set_boundary_conditions(self, goal_points: List[Tuple[int, int, int, int]]):
        """
        设置边界条件 Φ(x_goal, t_goal) = 0
        
        Args:
            goal_points: 目标点列表 [(t, z, y, x), ...]
        """
        for t, z, y, x in goal_points:
            if (0 <= t < self.time_steps and 0 <= z < self.nz and 
                0 <= y < self.ny and 0 <= x < self.nx):
                self.phi[t, z, y, x] = 0.0
                self.status[t, z, y, x] = 2  # Accepted
                
                # 将邻居加入Considered集合
                self._add_neighbors_to_considered(t, z, y, x)
    
    def _add_neighbors_to_considered(self, t: int, z: int, y: int, x: int):
        """将时空邻居点加入Considered集合"""
        neighbors = [
            # 空间邻居 (同一时间)
            (t, z-1, y, x), (t, z+1, y, x),
            (t, z, y-1, x), (t, z, y+1, x),
            (t, z, y, x-1), (t, z, y, x+1),
            # 时间邻居 (同一位置)
            (t-1, z, y, x), (t+1, z, y, x)
        ]
        
        for nt, nz, ny, nx in neighbors:
            if (0 <= nt < self.time_steps and 0 <= nz < self.nz and 
                0 <= ny < self.ny and 0 <= nx < self.nx and 
                self.status[nt, nz, ny, nx] == 0):  # Far点
                
                # 计算tentative值
                phi_new = self._solve_eikonal_at_point(nt, nz, ny, nx)
                self.phi[nt, nz, ny, nx] = phi_new
                self.status[nt, nz, ny, nx] = 1  # Considered
                
                # 加入优先队列
                heapq.heappush(self.heap, (phi_new, (nt, nz, ny, nx)))
    
    def _solve_eikonal_at_point(self, t: int, z: int, y: int, x: int) -> float:
        """
        在时空点(t,z,y,x)处求解四维Eikonal方程
        
        使用扩展的Godunov数值格式处理时空维度
        """
        # 获取邻居点的势值
        phi_neighbors = self._get_neighbor_phi_values(t, z, y, x)
        
        # 计算空间方向的有限差分
        dx_minus = max(0, (self.phi[t, z, y, x] - phi_neighbors['x_minus']) / self.dx) if phi_neighbors['x_minus'] < np.inf else 0
        dy_minus = max(0, (self.phi[t, z, y, x] - phi_neighbors['y_minus']) / self.dy) if phi_neighbors['y_minus'] < np.inf else 0
        dz_minus = max(0, (self.phi[t, z, y, x] - phi_neighbors['z_minus']) / self.dz) if phi_neighbors['z_minus'] < np.inf else 0
        
        # 计算时间方向的有限差分
        dt_minus = max(0, (self.phi[t, z, y, x] - phi_neighbors['t_minus']) / self.dt) if phi_neighbors['t_minus'] < np.inf else 0
        
        # 四维Godunov格式
        spatial_grad_sq = dx_minus**2 + dy_minus**2 + dz_minus**2
        temporal_grad_sq = dt_minus**2
        
        # 求解四维Eikonal方程
        # ||∇Φ||² = (∂Φ/∂x)² + (∂Φ/∂y)² + (∂Φ/∂z)² + (∂Φ/∂t)² = 1/f²
        
        speed = self.speed_field[t, z, y, x]
        target_grad_sq = 1.0 / (speed**2)
        
        # 使用迭代方法求解
        phi_old = self.phi[t, z, y, x] if self.phi[t, z, y, x] < np.inf else 0
        
        # 简化求解：优先考虑空间梯度
        if spatial_grad_sq > 0:
            # 基于空间邻居的最小值
            min_spatial = min([
                phi_neighbors['x_minus'] + self.dx/speed,
                phi_neighbors['y_minus'] + self.dy/speed,
                phi_neighbors['z_minus'] + self.dz/speed
            ])
            
            # 考虑时间维度
            if phi_neighbors['t_minus'] < np.inf:
                min_temporal = phi_neighbors['t_minus'] + self.dt/speed
                phi_new = min(min_spatial, min_temporal)
            else:
                phi_new = min_spatial
        else:
            # 基于最近邻居
            valid_neighbors = [v for v in phi_neighbors.values() if v < np.inf]
            if valid_neighbors:
                phi_new = min(valid_neighbors) + 1.0/speed
            else:
                phi_new = 1.0/speed
        
        return max(phi_new, 0)
    
    def _get_neighbor_phi_values(self, t: int, z: int, y: int, x: int) -> Dict[str, float]:
        """获取时空邻居点的势值"""
        neighbors = {}
        
        # 空间邻居
        neighbors['x_minus'] = self.phi[t, z, y, x-1] if x > 0 else np.inf
        neighbors['x_plus'] = self.phi[t, z, y, x+1] if x < self.nx-1 else np.inf
        neighbors['y_minus'] = self.phi[t, z, y-1, x] if y > 0 else np.inf
        neighbors['y_plus'] = self.phi[t, z, y+1, x] if y < self.ny-1 else np.inf
        neighbors['z_minus'] = self.phi[t, z-1, y, x] if z > 0 else np.inf
        neighbors['z_plus'] = self.phi[t, z+1, y, x] if z < self.nz-1 else np.inf
        
        # 时间邻居
        neighbors['t_minus'] = self.phi[t-1, z, y, x] if t > 0 else np.inf
        neighbors['t_plus'] = self.phi[t+1, z, y, x] if t < self.time_steps-1 else np.inf
        
        return neighbors
    
    def solve(self, max_iterations: int = 2000000) -> bool:
        """
        使用快速行进法求解四维动态Eikonal方程
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            是否成功求解
        """
        iteration = 0
        
        while self.heap and iteration < max_iterations:
            # 从优先队列中取出phi值最小的点
            phi_val, (t, z, y, x) = heapq.heappop(self.heap)
            
            # 跳过已经被处理的点
            if self.status[t, z, y, x] == 2:  # Already Accepted
                continue
                
            # 标记为Accepted
            self.status[t, z, y, x] = 2
            
            # 更新邻居点
            self._update_neighbors(t, z, y, x)
            
            iteration += 1
            
            if iteration % 100000 == 0:
                print(f"迭代进度: {iteration}/{max_iterations}")
        
        print(f"求解完成，总迭代次数: {iteration}")
        return iteration < max_iterations
    
    def _update_neighbors(self, t: int, z: int, y: int, x: int):
        """更新时空邻居点的势值"""
        neighbors = [
            # 空间邻居
            (t, z-1, y, x), (t, z+1, y, x),
            (t, z, y-1, x), (t, z, y+1, x),
            (t, z, y, x-1), (t, z, y, x+1),
            # 时间邻居
            (t-1, z, y, x), (t+1, z, y, x)
        ]
        
        for nt, nz, ny, nx in neighbors:
            if (0 <= nt < self.time_steps and 0 <= nz < self.nz and 
                0 <= ny < self.ny and 0 <= nx < self.nx and 
                self.status[nt, nz, ny, nx] != 2):  # 不是Accepted点
                
                # 计算新的phi值
                phi_new = self._solve_eikonal_at_point(nt, nz, ny, nx)
                
                if phi_new < self.phi[nt, nz, ny, nx]:
                    self.phi[nt, nz, ny, nx] = phi_new
                    
                    if self.status[nt, nz, ny, nx] == 0:  # Far点
                        self.status[nt, nz, ny, nx] = 1  # 标记为Considered
                    
                    # 加入优先队列
                    heapq.heappush(self.heap, (phi_new, (nt, nz, ny, nx)))


class DynamicGradientPathGenerator:
    """
    基于四维梯度下降的动态路径生成器
    """
    
    def __init__(self, phi_field: np.ndarray, 
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 dt: float = 1.0):
        """
        初始化动态路径生成器
        
        Args:
            phi_field: 四维势场 Φ(t,z,y,x)
            spacing: 空间网格间距 (dz, dy, dx)
            dt: 时间步长
        """
        self.phi = phi_field
        self.shape = phi_field.shape
        self.time_steps, self.nz, self.ny, self.nx = self.shape
        self.dz, self.dy, self.dx = spacing
        self.dt = dt
        
        # 计算四维梯度场
        self.gradient = self._compute_4d_gradient_field()
    
    def _compute_4d_gradient_field(self) -> np.ndarray:
        """
        计算四维势场的梯度 ∇Φ(x,t)
        
        Returns:
            四维梯度数组 (t, z, y, x, 4) 最后一维为 [∂Φ/∂t, ∂Φ/∂z, ∂Φ/∂y, ∂Φ/∂x]
        """
        grad_t = np.zeros_like(self.phi)
        grad_z = np.zeros_like(self.phi)
        grad_y = np.zeros_like(self.phi)
        grad_x = np.zeros_like(self.phi)
        
        # 时间方向梯度
        grad_t[1:-1, :, :, :] = (self.phi[2:, :, :, :] - self.phi[:-2, :, :, :]) / (2 * self.dt)
        grad_t[0, :, :, :] = (self.phi[1, :, :, :] - self.phi[0, :, :, :]) / self.dt
        grad_t[-1, :, :, :] = (self.phi[-1, :, :, :] - self.phi[-2, :, :, :]) / self.dt
        
        # Z方向梯度
        grad_z[:, 1:-1, :, :] = (self.phi[:, 2:, :, :] - self.phi[:, :-2, :, :]) / (2 * self.dz)
        grad_z[:, 0, :, :] = (self.phi[:, 1, :, :] - self.phi[:, 0, :, :]) / self.dz
        grad_z[:, -1, :, :] = (self.phi[:, -1, :, :] - self.phi[:, -2, :, :]) / self.dz
        
        # Y方向梯度
        grad_y[:, :, 1:-1, :] = (self.phi[:, :, 2:, :] - self.phi[:, :, :-2, :]) / (2 * self.dy)
        grad_y[:, :, 0, :] = (self.phi[:, :, 1, :] - self.phi[:, :, 0, :]) / self.dy
        grad_y[:, :, -1, :] = (self.phi[:, :, -1, :] - self.phi[:, :, -2, :]) / self.dy
        
        # X方向梯度
        grad_x[:, :, :, 1:-1] = (self.phi[:, :, :, 2:] - self.phi[:, :, :, :-2]) / (2 * self.dx)
        grad_x[:, :, :, 0] = (self.phi[:, :, :, 1] - self.phi[:, :, :, 0]) / self.dx
        grad_x[:, :, :, -1] = (self.phi[:, :, :, -1] - self.phi[:, :, :, -2]) / self.dx
        
        return np.stack([grad_t, grad_z, grad_y, grad_x], axis=-1)
    
    def generate_path(self, start_point: Tuple[float, float, float, float], 
                     goal_point: Tuple[float, float, float, float],
                     max_steps: int = 10000,
                     step_size: float = 0.3,
                     tolerance: float = 1.0) -> List[Tuple[float, float, float, float]]:
        """
        使用四维梯度下降生成动态路径
        
        Args:
            start_point: 起点 (t, z, y, x)
            goal_point: 终点 (t, z, y, x)
            max_steps: 最大步数
            step_size: 步长
            tolerance: 收敛容差
            
        Returns:
            路径点列表 [(t, z, y, x), ...]
        """
        path = [start_point]
        current = np.array(start_point, dtype=np.float64)
        goal = np.array(goal_point, dtype=np.float64)
        
        for step in range(max_steps):
            # 检查是否到达目标
            spatial_dist = np.linalg.norm(current[1:] - goal[1:])  # 只考虑空间距离
            if spatial_dist < tolerance:
                path.append(tuple(goal))
                break
            
            # 获取当前位置的梯度
            gradient = self._interpolate_4d_gradient(current)
            
            if np.linalg.norm(gradient) < 1e-10:
                print(f"梯度过小，在步骤 {step} 处停止")
                break
            
            # 四维梯度下降步骤
            # 对时间维度使用较小的权重，主要沿空间梯度移动
            gradient_weighted = gradient.copy()
            gradient_weighted[0] *= 0.1  # 时间维度权重较小
            
            direction = -gradient_weighted / np.linalg.norm(gradient_weighted)
            
            # 自适应步长
            adaptive_step = min(step_size, 0.5 / np.linalg.norm(gradient_weighted))
            
            # 更新位置
            next_pos = current + adaptive_step * direction
            
            # 边界检查
            next_pos[0] = np.clip(next_pos[0], 0, self.time_steps - 1)
            next_pos[1] = np.clip(next_pos[1], 0, self.nz - 1)
            next_pos[2] = np.clip(next_pos[2], 0, self.ny - 1)
            next_pos[3] = np.clip(next_pos[3], 0, self.nx - 1)
            
            current = next_pos
            path.append(tuple(current))
        
        return path
    
    def _interpolate_4d_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        在给定四维位置插值梯度值
        
        使用四线性插值
        """
        t, z, y, x = position
        
        # 获取整数坐标
        t0 = int(np.floor(t))
        z0 = int(np.floor(z))
        y0 = int(np.floor(y))
        x0 = int(np.floor(x))
        
        t1 = min(t0 + 1, self.time_steps - 1)
        z1 = min(z0 + 1, self.nz - 1)
        y1 = min(y0 + 1, self.ny - 1)
        x1 = min(x0 + 1, self.nx - 1)
        
        # 获取插值权重
        wt = t - t0 if t1 > t0 else 0
        wz = z - z0 if z1 > z0 else 0
        wy = y - y0 if y1 > y0 else 0
        wx = x - x0 if x1 > x0 else 0
        
        # 四线性插值（简化版本，使用最近邻）
        gradient = self.gradient[t0, z0, y0, x0]
        
        return gradient


class DynamicEikonalPathPlanner:
    """
    基于动态Eikonal方程的完整路径规划系统
    """
    
    def __init__(self, terrain_data: np.ndarray, fog_data: np.ndarray,
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 dt: float = 1.0):
        """
        初始化动态路径规划器
        
        Args:
            terrain_data: 地形数据 (nz, ny, nx)
            fog_data: 动态雾气数据 (time_steps, nz, ny, nx)
            spacing: 空间网格间距 (dz, dy, dx)
            dt: 时间步长
        """
        self.terrain = terrain_data
        self.fog_data = fog_data
        self.shape = terrain_data.shape
        self.time_steps = fog_data.shape[0]
        self.spacing = spacing
        self.dt = dt
        
        # 创建动态速度场
        self.speed_field = self._create_dynamic_speed_field()
        
        # 初始化求解器
        self.solver = DynamicEikonalSolver(self.shape, self.time_steps, spacing, dt)
        self.solver.set_dynamic_speed_field(self.speed_field)
        
        # 路径生成器
        self.path_generator = None
    
    def _create_dynamic_speed_field(self) -> np.ndarray:
        """
        创建动态速度场 f(x,t)
        
        速度场定义了在每个时空位置的传播速度
        """
        speed = np.ones((self.time_steps,) + self.shape, dtype=np.float64)
        
        for t in range(self.time_steps):
            # 地形影响：高度差越大，速度越慢
            grad_z = np.gradient(self.terrain, axis=0)
            grad_y = np.gradient(self.terrain, axis=1)
            grad_x = np.gradient(self.terrain, axis=2)
            terrain_gradient = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
            
            # 速度与地形梯度成反比
            speed[t] *= 1.0 / (1.0 + 0.1 * terrain_gradient)
            
            # 雾气影响：雾气密度越大，速度越慢
            fog_factor = 1.0 / (1.0 + 20.0 * self.fog_data[t])
            speed[t] *= fog_factor
        
        # 确保速度为正
        speed = np.maximum(speed, 0.01)
        
        return speed
    
    def plan_path(self, start: Tuple[int, int, int, int], 
                 goal: Tuple[int, int, int, int]) -> Tuple[List[Tuple[float, float, float, float]], Dict]:
        """
        规划动态路径
        
        Args:
            start: 起点 (t, z, y, x)
            goal: 终点 (t, z, y, x)
            
        Returns:
            (路径点列表, 统计信息)
        """
        start_time = time.time()
        
        # 设置边界条件
        self.solver.set_boundary_conditions([goal])
        
        # 求解动态Eikonal方程
        print("正在求解四维动态Eikonal方程...")
        success = self.solver.solve()
        
        if not success:
            return [], {"success": False, "error": "动态Eikonal求解失败"}
        
        solve_time = time.time() - start_time
        
        # 生成路径
        print("正在生成动态路径...")
        self.path_generator = DynamicGradientPathGenerator(self.solver.phi, self.spacing, self.dt)
        path = self.path_generator.generate_path(start, goal)
        
        total_time = time.time() - start_time
        
        # 计算路径统计
        path_length = len(path)
        path_cost = self.solver.phi[start] if self.solver.phi[start] < np.inf else np.inf
        
        stats = {
            "success": True,
            "path_length": path_length,
            "path_cost": path_cost,
            "solve_time": solve_time,
            "total_time": total_time,
            "phi_min": np.min(self.solver.phi[self.solver.phi < np.inf]),
            "phi_max": np.max(self.solver.phi[self.solver.phi < np.inf])
        }
        
        return path, stats
    
    def visualize_dynamic_path(self, path: List[Tuple[float, float, float, float]], 
                              start: Tuple[int, int, int, int], 
                              goal: Tuple[int, int, int, int],
                              save_path: str = None):
        """
        可视化动态路径
        
        Args:
            path: 四维路径点列表
            start: 起点
            goal: 终点
            save_path: 保存路径
        """
        if not path:
            print("路径为空，无法可视化")
            return
        
        fig = plt.figure(figsize=(20, 15))
        
        # 3D空间路径视图
        ax1 = fig.add_subplot(231, projection='3d')
        
        # 绘制地形（下采样）
        step = max(1, min(self.shape) // 15)
        z_terrain, y_terrain, x_terrain = np.meshgrid(
            np.arange(0, self.shape[0], step),
            np.arange(0, self.shape[1], step),
            np.arange(0, self.shape[2], step),
            indexing='ij'
        )
        
        terrain_sample = self.terrain[::step, ::step, ::step]
        ax1.scatter(x_terrain.flatten(), y_terrain.flatten(), z_terrain.flatten(), 
                   c=terrain_sample.flatten(), cmap='terrain', alpha=0.1, s=1)
        
        # 绘制路径
        if len(path) > 1:
            path_array = np.array(path)
            ax1.plot(path_array[:, 3], path_array[:, 2], path_array[:, 1], 
                    'r-', linewidth=3, label='Dynamic FMM Path')
        
        # 标记起点和终点
        ax1.scatter([start[3]], [start[2]], [start[1]], c='green', s=100, label='Start')
        ax1.scatter([goal[3]], [goal[2]], [goal[1]], c='red', s=100, label='Goal')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Spatial Path')
        ax1.legend()
        
        # 时间-高度图
        ax2 = fig.add_subplot(232)
        if len(path) > 1:
            path_array = np.array(path)
            ax2.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Height vs Time')
        ax2.scatter([start[0]], [start[1]], c='green', s=100, label='Start')
        ax2.scatter([goal[0]], [goal[1]], c='red', s=100, label='Goal')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Height (Z)')
        ax2.set_title('Height Evolution')
        ax2.legend()
        ax2.grid(True)
        
        # XY投影
        ax3 = fig.add_subplot(233)
        if len(path) > 1:
            ax3.plot(path_array[:, 3], path_array[:, 2], 'r-', linewidth=2, label='Path Projection')
        ax3.scatter([start[3]], [start[2]], c='green', s=100, label='Start')
        ax3.scatter([goal[3]], [goal[2]], c='red', s=100, label='Goal')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('XY Plane Projection')
        ax3.legend()
        ax3.grid(True)
        
        # 时间-X图
        ax4 = fig.add_subplot(234)
        if len(path) > 1:
            ax4.plot(path_array[:, 0], path_array[:, 3], 'g-', linewidth=2, label='X vs Time')
        ax4.scatter([start[0]], [start[3]], c='green', s=100, label='Start')
        ax4.scatter([goal[0]], [goal[3]], c='red', s=100, label='Goal')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('X Position')
        ax4.set_title('X Movement Over Time')
        ax4.legend()
        ax4.grid(True)
        
        # 时间-Y图
        ax5 = fig.add_subplot(235)
        if len(path) > 1:
            ax5.plot(path_array[:, 0], path_array[:, 2], 'm-', linewidth=2, label='Y vs Time')
        ax5.scatter([start[0]], [start[2]], c='green', s=100, label='Start')
        ax5.scatter([goal[0]], [goal[2]], c='red', s=100, label='Goal')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Y Position')
        ax5.set_title('Y Movement Over Time')
        ax5.legend()
        ax5.grid(True)
        
        # 路径长度累积图
        ax6 = fig.add_subplot(236)
        if len(path) > 1:
            # 计算累积距离
            cumulative_dist = [0]
            for i in range(1, len(path)):
                dist = np.linalg.norm(np.array(path[i][1:]) - np.array(path[i-1][1:]))
                cumulative_dist.append(cumulative_dist[-1] + dist)
            
            ax6.plot(path_array[:, 0], cumulative_dist, 'c-', linewidth=2, label='Cumulative Distance')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Cumulative Distance')
        ax6.set_title('Distance Traveled Over Time')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dynamic path visualization saved to: {save_path}")
        
        plt.show()


def demo_dynamic_eikonal_planning():
    """
    演示动态Eikonal路径规划
    """
    print("=== Dynamic Eikonal Path Planning Demo ===")
    
    # 读取TIF文件
    tif_folder = Path("tif")
    if tif_folder.exists():
        tif_files = list(tif_folder.glob("*.tif"))
        if tif_files:
            print(f"Found TIF file: {tif_files[0]}")
            with rasterio.open(tif_files[0]) as src:
                terrain_full = src.read(1)
                print(f"Original terrain shape: {terrain_full.shape}")
                
                # 下采样以减少计算量
                downsample_factor = 8
                terrain_2d = terrain_full[::downsample_factor, ::downsample_factor]
                print(f"Downsampled terrain shape: {terrain_2d.shape}")
        else:
            print("No TIF files found, using synthetic terrain")
            terrain_2d = np.random.rand(30, 40) * 100
    else:
        print("TIF folder not found, using synthetic terrain")
        terrain_2d = np.random.rand(30, 40) * 100
    
    # 创建动态雾气环境（使用二维地形数据）
    print("Creating dynamic fog environment...")
    fog_env = OptimizedDynamicFogEnvironment(
        terrain_2d,
        time_steps=10
    )
    
    fog_data = fog_env.fog_data
    print(f"Dynamic fog data shape: {fog_data.shape}")
    
    # 创建三维地形用于路径规划
    nz = fog_data.shape[1]  # 使用雾气数据的高度层数
    terrain_3d = np.zeros((nz, terrain_2d.shape[0], terrain_2d.shape[1]))
    for i in range(nz):
        terrain_3d[i] = terrain_2d + i * 2  # 每层高度增加2米
    # 创建动态Eikonal路径规划器
    planner = DynamicEikonalPathPlanner(
        terrain_3d,
        fog_data,
        spacing=(1.0, 1.0, 1.0),
        dt=1.0
    )
    
    # 设置起点和终点
    start = (0, 2, 2, 2)  # (t, z, y, x)
    goal = (8, nz-3, terrain_2d.shape[0]-5, terrain_2d.shape[1]-5)
    
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    
    # 规划路径
    path, stats = planner.plan_path(start, goal)
    
    # 打印结果
    print(f"\n=== Dynamic Path Planning Results ===")
    print(f"Success: {stats['success']}")
    if stats['success']:
        print(f"Path length: {stats['path_length']} steps")
        print(f"Path cost: {stats['path_cost']:.2f}")
        print(f"Solve time: {stats['solve_time']:.2f} seconds")
        print(f"Total time: {stats['total_time']:.2f} seconds")
        print(f"Potential field range: [{stats['phi_min']:.2f}, {stats['phi_max']:.2f}]")
    
    # 创建输出目录
    output_dir = "output/dynamic_eikonal_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化路径
    if path:
        planner.visualize_dynamic_path(
            path, start, goal,
            save_path=f"{output_dir}/dynamic_eikonal_path.png"
        )
        
        # 保存结果到CSV
        with open(f"{output_dir}/dynamic_eikonal_results.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Algorithm', 'Success', 'Path_Length', 'Path_Cost', 'Solve_Time', 'Total_Time'])
            writer.writerow(['Dynamic_FMM', stats['success'], stats['path_length'], 
                           stats['path_cost'], stats['solve_time'], stats['total_time']])
    
    return path, stats


if __name__ == "__main__":
    # 运行演示
    path, stats = demo_dynamic_eikonal_planning()