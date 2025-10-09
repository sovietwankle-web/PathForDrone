"""
基于Eikonal方程和快速行进法(FMM)的三维路径规划系统

数学理论基础:
1. Eikonal方程: ||∇Φ(x)|| = 1/f(x), x ∈ Ω_free
2. 快速行进法(FMM): 高效求解Eikonal方程的数值方法
3. 梯度下降: 沿势场梯度生成最优路径

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


class EikonalSolver3D:
    """
    三维Eikonal方程求解器
    
    求解方程: ||∇Φ(x)|| = 1/f(x)
    其中 Φ(x) 是势函数，f(x) 是速度函数
    """
    
    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        初始化求解器
        
        Args:
            shape: 三维网格形状 (nz, ny, nx)
            spacing: 网格间距 (dz, dy, dx)
        """
        self.shape = shape
        self.nz, self.ny, self.nx = shape
        self.dz, self.dy, self.dx = spacing
        
        # 势场数组
        self.phi = np.full(shape, np.inf, dtype=np.float64)
        
        # 状态标记: 0=Far, 1=Considered, 2=Accepted
        self.status = np.zeros(shape, dtype=np.int32)
        
        # 速度场
        self.speed = np.ones(shape, dtype=np.float64)
        
        # 优先队列 (phi_value, (z, y, x))
        self.heap = []
        
    def set_speed_field(self, speed_field: np.ndarray):
        """设置速度场 f(x)"""
        self.speed = np.maximum(speed_field, 1e-10)  # 避免除零
        
    def set_boundary_conditions(self, goal_points: List[Tuple[int, int, int]]):
        """
        设置边界条件 Φ(x_goal) = 0
        
        Args:
            goal_points: 目标点列表 [(z, y, x), ...]
        """
        for z, y, x in goal_points:
            if 0 <= z < self.nz and 0 <= y < self.ny and 0 <= x < self.nx:
                self.phi[z, y, x] = 0.0
                self.status[z, y, x] = 2  # Accepted
                
                # 将邻居加入Considered集合
                self._add_neighbors_to_considered(z, y, x)
    
    def _add_neighbors_to_considered(self, z: int, y: int, x: int):
        """将邻居点加入Considered集合"""
        neighbors = [
            (z-1, y, x), (z+1, y, x),
            (z, y-1, x), (z, y+1, x),
            (z, y, x-1), (z, y, x+1)
        ]
        
        for nz, ny, nx in neighbors:
            if (0 <= nz < self.nz and 0 <= ny < self.ny and 0 <= nx < self.nx and 
                self.status[nz, ny, nx] == 0):  # Far点
                
                # 计算tentative值
                phi_new = self._solve_eikonal_at_point(nz, ny, nx)
                self.phi[nz, ny, nx] = phi_new
                self.status[nz, ny, nx] = 1  # Considered
                
                # 加入优先队列
                heapq.heappush(self.heap, (phi_new, (nz, ny, nx)))
    
    def _solve_eikonal_at_point(self, z: int, y: int, x: int) -> float:
        """
        在点(z,y,x)处求解Eikonal方程
        
        使用Godunov数值格式:
        max(D-x Φ, -D+x Φ, 0)² + max(D-y Φ, -D+y Φ, 0)² + max(D-z Φ, -D+z Φ, 0)² = 1/f²
        """
        # 获取邻居点的势值
        phi_neighbors = self._get_neighbor_phi_values(z, y, x)
        
        # 计算各方向的有限差分
        dx_minus = (self.phi[z, y, x] - phi_neighbors['x_minus']) / self.dx if phi_neighbors['x_minus'] < np.inf else 0
        dx_plus = (phi_neighbors['x_plus'] - self.phi[z, y, x]) / self.dx if phi_neighbors['x_plus'] < np.inf else 0
        
        dy_minus = (self.phi[z, y, x] - phi_neighbors['y_minus']) / self.dy if phi_neighbors['y_minus'] < np.inf else 0
        dy_plus = (phi_neighbors['y_plus'] - self.phi[z, y, x]) / self.dy if phi_neighbors['y_plus'] < np.inf else 0
        
        dz_minus = (self.phi[z, y, x] - phi_neighbors['z_minus']) / self.dz if phi_neighbors['z_minus'] < np.inf else 0
        dz_plus = (phi_neighbors['z_plus'] - self.phi[z, y, x]) / self.dz if phi_neighbors['z_plus'] < np.inf else 0
        
        # Godunov格式
        dx_godunov = max(dx_minus, -dx_plus, 0)
        dy_godunov = max(dy_minus, -dy_plus, 0)
        dz_godunov = max(dz_minus, -dz_plus, 0)
        
        # 求解二次方程
        a = (dx_godunov**2 + dy_godunov**2 + dz_godunov**2)
        b = 1.0 / (self.speed[z, y, x]**2)
        
        if a == 0:
            return min(phi_neighbors['x_minus'] + self.dx/self.speed[z, y, x],
                      phi_neighbors['y_minus'] + self.dy/self.speed[z, y, x],
                      phi_neighbors['z_minus'] + self.dz/self.speed[z, y, x])
        
        # 使用迭代方法求解
        phi_old = self.phi[z, y, x] if self.phi[z, y, x] < np.inf else 0
        
        for _ in range(10):  # 最多迭代10次
            # 更新各方向差分
            dx_minus = max(0, (phi_old - phi_neighbors['x_minus']) / self.dx) if phi_neighbors['x_minus'] < np.inf else 0
            dy_minus = max(0, (phi_old - phi_neighbors['y_minus']) / self.dy) if phi_neighbors['y_minus'] < np.inf else 0
            dz_minus = max(0, (phi_old - phi_neighbors['z_minus']) / self.dz) if phi_neighbors['z_minus'] < np.inf else 0
            
            # 计算新的phi值
            sum_sq = dx_minus**2 + dy_minus**2 + dz_minus**2
            
            if sum_sq > 0:
                phi_new = phi_old + (1.0/self.speed[z, y, x] - np.sqrt(sum_sq)) / sum_sq
            else:
                phi_new = phi_old + 1.0/self.speed[z, y, x]
            
            if abs(phi_new - phi_old) < 1e-6:
                break
            phi_old = phi_new
        
        return max(phi_new, 0)
    
    def _get_neighbor_phi_values(self, z: int, y: int, x: int) -> Dict[str, float]:
        """获取邻居点的势值"""
        neighbors = {}
        
        # X方向邻居
        neighbors['x_minus'] = self.phi[z, y, x-1] if x > 0 else np.inf
        neighbors['x_plus'] = self.phi[z, y, x+1] if x < self.nx-1 else np.inf
        
        # Y方向邻居
        neighbors['y_minus'] = self.phi[z, y-1, x] if y > 0 else np.inf
        neighbors['y_plus'] = self.phi[z, y+1, x] if y < self.ny-1 else np.inf
        
        # Z方向邻居
        neighbors['z_minus'] = self.phi[z-1, y, x] if z > 0 else np.inf
        neighbors['z_plus'] = self.phi[z+1, y, x] if z < self.nz-1 else np.inf
        
        return neighbors
    
    def solve(self, max_iterations: int = 1000000) -> bool:
        """
        使用快速行进法求解Eikonal方程
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            是否成功求解
        """
        iteration = 0
        
        while self.heap and iteration < max_iterations:
            # 从优先队列中取出phi值最小的点
            phi_val, (z, y, x) = heapq.heappop(self.heap)
            
            # 跳过已经被处理的点
            if self.status[z, y, x] == 2:  # Already Accepted
                continue
                
            # 标记为Accepted
            self.status[z, y, x] = 2
            
            # 更新邻居点
            self._update_neighbors(z, y, x)
            
            iteration += 1
        
        return iteration < max_iterations
    
    def _update_neighbors(self, z: int, y: int, x: int):
        """更新邻居点的势值"""
        neighbors = [
            (z-1, y, x), (z+1, y, x),
            (z, y-1, x), (z, y+1, x),
            (z, y, x-1), (z, y, x+1)
        ]
        
        for nz, ny, nx in neighbors:
            if (0 <= nz < self.nz and 0 <= ny < self.ny and 0 <= nx < self.nx and 
                self.status[nz, ny, nx] != 2):  # 不是Accepted点
                
                # 计算新的phi值
                phi_new = self._solve_eikonal_at_point(nz, ny, nx)
                
                if phi_new < self.phi[nz, ny, nx]:
                    self.phi[nz, ny, nx] = phi_new
                    
                    if self.status[nz, ny, nx] == 0:  # Far点
                        self.status[nz, ny, nx] = 1  # 标记为Considered
                    
                    # 加入优先队列
                    heapq.heappush(self.heap, (phi_new, (nz, ny, nx)))


class GradientPathGenerator:
    """
    基于梯度下降的路径生成器
    """
    
    def __init__(self, phi_field: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        初始化路径生成器
        
        Args:
            phi_field: 势场 Φ(x)
            spacing: 网格间距 (dz, dy, dx)
        """
        self.phi = phi_field
        self.shape = phi_field.shape
        self.nz, self.ny, self.nx = self.shape
        self.dz, self.dy, self.dx = spacing
        
        # 计算梯度场
        self.gradient = self._compute_gradient_field()
    
    def _compute_gradient_field(self) -> np.ndarray:
        """
        计算势场的梯度 ∇Φ(x)
        
        使用中心差分法:
        ∂Φ/∂x ≈ (Φ[i+1] - Φ[i-1]) / (2Δx)
        """
        grad_z = np.zeros_like(self.phi)
        grad_y = np.zeros_like(self.phi)
        grad_x = np.zeros_like(self.phi)
        
        # Z方向梯度
        grad_z[1:-1, :, :] = (self.phi[2:, :, :] - self.phi[:-2, :, :]) / (2 * self.dz)
        grad_z[0, :, :] = (self.phi[1, :, :] - self.phi[0, :, :]) / self.dz
        grad_z[-1, :, :] = (self.phi[-1, :, :] - self.phi[-2, :, :]) / self.dz
        
        # Y方向梯度
        grad_y[:, 1:-1, :] = (self.phi[:, 2:, :] - self.phi[:, :-2, :]) / (2 * self.dy)
        grad_y[:, 0, :] = (self.phi[:, 1, :] - self.phi[:, 0, :]) / self.dy
        grad_y[:, -1, :] = (self.phi[:, -1, :] - self.phi[:, -2, :]) / self.dy
        
        # X方向梯度
        grad_x[:, :, 1:-1] = (self.phi[:, :, 2:] - self.phi[:, :, :-2]) / (2 * self.dx)
        grad_x[:, :, 0] = (self.phi[:, :, 1] - self.phi[:, :, 0]) / self.dx
        grad_x[:, :, -1] = (self.phi[:, :, -1] - self.phi[:, :, -2]) / self.dx
        
        return np.stack([grad_z, grad_y, grad_x], axis=-1)
    
    def generate_path(self, start_point: Tuple[float, float, float], 
                     goal_point: Tuple[float, float, float],
                     max_steps: int = 10000,
                     step_size: float = 0.5,
                     tolerance: float = 1.0) -> List[Tuple[float, float, float]]:
        """
        使用梯度下降生成路径
        
        Args:
            start_point: 起点 (z, y, x)
            goal_point: 终点 (z, y, x)
            max_steps: 最大步数
            step_size: 步长
            tolerance: 收敛容差
            
        Returns:
            路径点列表
        """
        path = [start_point]
        current = np.array(start_point, dtype=np.float64)
        goal = np.array(goal_point, dtype=np.float64)
        
        for step in range(max_steps):
            # 检查是否到达目标
            if np.linalg.norm(current - goal) < tolerance:
                path.append(tuple(goal))
                break
            
            # 获取当前位置的梯度
            gradient = self._interpolate_gradient(current)
            
            if np.linalg.norm(gradient) < 1e-10:
                print(f"梯度过小，在步骤 {step} 处停止")
                break
            
            # 梯度下降步骤: x_{k+1} = x_k - η * ∇Φ(x_k) / ||∇Φ(x_k)||
            direction = -gradient / np.linalg.norm(gradient)
            
            # 自适应步长
            adaptive_step = min(step_size, 0.1 / np.linalg.norm(gradient))
            
            # 更新位置
            next_pos = current + adaptive_step * direction
            
            # 边界检查
            next_pos[0] = np.clip(next_pos[0], 0, self.nz - 1)
            next_pos[1] = np.clip(next_pos[1], 0, self.ny - 1)
            next_pos[2] = np.clip(next_pos[2], 0, self.nx - 1)
            
            current = next_pos
            path.append(tuple(current))
        
        return path
    
    def _interpolate_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        在给定位置插值梯度值
        
        使用三线性插值
        """
        z, y, x = position
        
        # 获取整数坐标
        z0, y0, x0 = int(np.floor(z)), int(np.floor(y)), int(np.floor(x))
        z1, y1, x1 = min(z0 + 1, self.nz - 1), min(y0 + 1, self.ny - 1), min(x0 + 1, self.nx - 1)
        
        # 获取插值权重
        wz, wy, wx = z - z0, y - y0, x - x0
        
        # 三线性插值
        gradient = (
            (1 - wz) * (1 - wy) * (1 - wx) * self.gradient[z0, y0, x0] +
            (1 - wz) * (1 - wy) * wx * self.gradient[z0, y0, x1] +
            (1 - wz) * wy * (1 - wx) * self.gradient[z0, y1, x0] +
            (1 - wz) * wy * wx * self.gradient[z0, y1, x1] +
            wz * (1 - wy) * (1 - wx) * self.gradient[z1, y0, x0] +
            wz * (1 - wy) * wx * self.gradient[z1, y0, x1] +
            wz * wy * (1 - wx) * self.gradient[z1, y1, x0] +
            wz * wy * wx * self.gradient[z1, y1, x1]
        )
        
        return gradient


class EikonalPathPlanner:
    """
    基于Eikonal方程的完整路径规划系统
    """
    
    def __init__(self, terrain_data: np.ndarray, fog_data: Optional[np.ndarray] = None,
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        初始化路径规划器
        
        Args:
            terrain_data: 地形数据 (nz, ny, nx)
            fog_data: 雾气数据 (nz, ny, nx)，可选
            spacing: 网格间距 (dz, dy, dx)
        """
        self.terrain = terrain_data
        self.fog = fog_data
        self.shape = terrain_data.shape
        self.spacing = spacing
        
        # 创建速度场
        self.speed_field = self._create_speed_field()
        
        # 初始化求解器
        self.solver = EikonalSolver3D(self.shape, spacing)
        self.solver.set_speed_field(self.speed_field)
        
        # 路径生成器
        self.path_generator = None
    
    def _create_speed_field(self) -> np.ndarray:
        """
        创建速度场 f(x)
        
        速度场定义了在每个位置的传播速度，与障碍物密度成反比
        """
        speed = np.ones(self.shape, dtype=np.float64)
        
        # 地形影响：高度差越大，速度越慢
        if self.terrain is not None:
            # 计算地形梯度
            grad_z = np.gradient(self.terrain, axis=0)
            grad_y = np.gradient(self.terrain, axis=1)
            grad_x = np.gradient(self.terrain, axis=2)
            terrain_gradient = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
            
            # 速度与地形梯度成反比
            speed *= 1.0 / (1.0 + 0.1 * terrain_gradient)
        
        # 雾气影响：雾气密度越大，速度越慢
        if self.fog is not None:
            fog_factor = 1.0 / (1.0 + 10.0 * self.fog)
            speed *= fog_factor
        
        # 确保速度为正
        speed = np.maximum(speed, 0.01)
        
        return speed
    
    def plan_path(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> Tuple[List[Tuple[float, float, float]], Dict]:
        """
        规划路径
        
        Args:
            start: 起点 (z, y, x)
            goal: 终点 (z, y, x)
            
        Returns:
            (路径点列表, 统计信息)
        """
        start_time = time.time()
        
        # 设置边界条件
        self.solver.set_boundary_conditions([goal])
        
        # 求解Eikonal方程
        print("正在求解Eikonal方程...")
        success = self.solver.solve()
        
        if not success:
            return [], {"success": False, "error": "Eikonal求解失败"}
        
        solve_time = time.time() - start_time
        
        # 生成路径
        print("正在生成路径...")
        self.path_generator = GradientPathGenerator(self.solver.phi, self.spacing)
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
    
    def visualize_potential_field(self, slice_idx: int = None, save_path: str = None):
        """
        可视化势场
        
        Args:
            slice_idx: 切片索引，如果为None则显示中间切片
            save_path: 保存路径
        """
        if self.solver.phi is None:
            print("请先运行路径规划")
            return
        
        if slice_idx is None:
            slice_idx = self.shape[0] // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 势场切片
        phi_slice = self.solver.phi[slice_idx, :, :]
        phi_slice[phi_slice == np.inf] = np.nan
        
        im1 = axes[0].imshow(phi_slice, cmap='viridis', origin='lower')
        axes[0].set_title(f'势场 Φ(x) - Z切片 {slice_idx}')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0])
        
        # 速度场切片
        speed_slice = self.speed_field[slice_idx, :, :]
        im2 = axes[1].imshow(speed_slice, cmap='plasma', origin='lower')
        axes[1].set_title(f'速度场 f(x) - Z切片 {slice_idx}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"势场可视化已保存到: {save_path}")
        
        plt.show()
    
    def visualize_path_3d(self, path: List[Tuple[float, float, float]], 
                         start: Tuple[int, int, int], goal: Tuple[int, int, int],
                         save_path: str = None):
        """
        三维路径可视化
        
        Args:
            path: 路径点列表
            start: 起点
            goal: 终点
            save_path: 保存路径
        """
        if not path:
            print("路径为空，无法可视化")
            return
        
        fig = plt.figure(figsize=(15, 12))
        
        # 3D视图
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 绘制地形（下采样）
        step = max(1, min(self.shape) // 20)
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
            ax1.plot(path_array[:, 2], path_array[:, 1], path_array[:, 0], 
                    'r-', linewidth=3, label='FMM路径')
        
        # 标记起点和终点
        ax1.scatter([start[2]], [start[1]], [start[0]], c='green', s=100, label='起点')
        ax1.scatter([goal[2]], [goal[1]], [goal[0]], c='red', s=100, label='终点')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('三维路径视图')
        ax1.legend()
        
        # XY投影
        ax2 = fig.add_subplot(222)
        if len(path) > 1:
            path_array = np.array(path)
            ax2.plot(path_array[:, 2], path_array[:, 1], 'r-', linewidth=2, label='路径投影')
        ax2.scatter([start[2]], [start[1]], c='green', s=100, label='起点')
        ax2.scatter([goal[2]], [goal[1]], c='red', s=100, label='终点')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY平面投影')
        ax2.legend()
        ax2.grid(True)
        
        # XZ投影
        ax3 = fig.add_subplot(223)
        if len(path) > 1:
            ax3.plot(path_array[:, 2], path_array[:, 0], 'r-', linewidth=2, label='路径投影')
        ax3.scatter([start[2]], [start[0]], c='green', s=100, label='起点')
        ax3.scatter([goal[2]], [goal[0]], c='red', s=100, label='终点')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ平面投影')
        ax3.legend()
        ax3.grid(True)
        
        # YZ投影
        ax4 = fig.add_subplot(224)
        if len(path) > 1:
            ax4.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2, label='路径投影')
        ax4.scatter([start[1]], [start[0]], c='green', s=100, label='起点')
        ax4.scatter([goal[1]], [goal[0]], c='red', s=100, label='终点')
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Z')
        ax4.set_title('YZ平面投影')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"路径可视化已保存到: {save_path}")
        
        plt.show()


def demo_eikonal_path_planning():
    """
    演示Eikonal路径规划
    """
    print("=== Eikonal方程路径规划演示 ===")
    
    # 创建测试地形
    nz, ny, nx = 30, 40, 50
    terrain = np.zeros((nz, ny, nx))
    
    # 添加一些地形特征
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                # 创建山丘地形
                terrain[i, j, k] = 10 * np.exp(-((j-20)**2 + (k-25)**2) / 100)
    
    # 创建雾气障碍物
    fog = np.zeros((nz, ny, nx))
    
    # 在中间高度添加雾气
    for i in range(10, 20):
        for j in range(15, 35):
            for k in range(10, 40):
                if np.random.random() < 0.3:  # 30%的雾气覆盖率
                    fog[i, j, k] = 0.8
    
    # 创建路径规划器
    planner = EikonalPathPlanner(terrain, fog, spacing=(1.0, 1.0, 1.0))
    
    # 设置起点和终点
    start = (5, 5, 5)
    goal = (25, 35, 45)
    
    print(f"起点: {start}")
    print(f"终点: {goal}")
    
    # 规划路径
    path, stats = planner.plan_path(start, goal)
    
    # 打印结果
    print(f"\n=== 路径规划结果 ===")
    print(f"成功: {stats['success']}")
    if stats['success']:
        print(f"路径长度: {stats['path_length']} 步")
        print(f"路径代价: {stats['path_cost']:.2f}")
        print(f"求解时间: {stats['solve_time']:.2f} 秒")
        print(f"总时间: {stats['total_time']:.2f} 秒")
        print(f"势场范围: [{stats['phi_min']:.2f}, {stats['phi_max']:.2f}]")
    
    # 创建输出目录
    output_dir = "output/eikonal_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化势场
    planner.visualize_potential_field(
        slice_idx=15,
        save_path=f"{output_dir}/potential_field.png"
    )
    
    # 可视化路径
    if path:
        planner.visualize_path_3d(
            path, start, goal,
            save_path=f"{output_dir}/eikonal_path_3d.png"
        )
    
    return path, stats


if __name__ == "__main__":
    # 运行演示
    path, stats = demo_eikonal_path_planning()