#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的动态水汽/雾气障碍物三维路径规划系统
降低雾气密度，增加多种路径规划算法
"""

import os
import sys
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from collections import deque
import heapq
import random
from typing import List, Tuple, Optional, Dict, Any
import warnings
import rasterio
from scipy import ndimage
from scipy.interpolate import interp2d

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

class OptimizedDynamicFogEnvironment:
    """优化的动态雾气环境类"""
    
    def __init__(self, terrain_data: np.ndarray, time_steps: int = 20):
        """
        初始化动态雾气环境
        
        Args:
            terrain_data: 地形高程数据 (height, width)
            time_steps: 时间步数
        """
        self.terrain = terrain_data
        self.height, self.width = terrain_data.shape
        self.time_steps = time_steps
        
        # 处理NaN值
        if np.any(np.isnan(terrain_data)):
            mean_elevation = np.nanmean(terrain_data)
            self.terrain = np.where(np.isnan(terrain_data), mean_elevation, terrain_data)
        
        # 地形统计信息
        self.min_elevation = np.min(self.terrain)
        self.max_elevation = np.max(self.terrain)
        self.elevation_range = self.max_elevation - self.min_elevation
        
        # 优化雾气参数（降低密度）
        self.fog_base_height = self.min_elevation + 0.2 * self.elevation_range  # 雾气基础高度
        self.fog_max_height = self.min_elevation + 0.6 * self.elevation_range   # 雾气最大高度
        self.fog_thickness = self.fog_max_height - self.fog_base_height
        
        # 雾气覆盖率（降低到30%）
        self.fog_coverage = 0.3
        
        # 生成动态雾气数据
        self.fog_data = self._generate_optimized_fog()
        
        print(f"优化动态雾气环境初始化完成:")
        print(f"  地形尺寸: {self.height} x {self.width}")
        print(f"  时间步数: {self.time_steps}")
        print(f"  地形高程范围: {self.min_elevation:.2f} - {self.max_elevation:.2f}")
        print(f"  雾气高度范围: {self.fog_base_height:.2f} - {self.fog_max_height:.2f}")
        print(f"  雾气覆盖率: {self.fog_coverage*100:.1f}%")
    
    def _generate_optimized_fog(self) -> np.ndarray:
        """
        生成优化的动态雾气数据（降低密度）
        
        Returns:
            fog_data: 四维数组 (time, height_levels, y, x)
        """
        print("生成优化的动态雾气数据...")
        
        # 雾气高度层数（减少层数）
        fog_levels = 10
        fog_heights = np.linspace(self.fog_base_height, self.fog_max_height, fog_levels)
        
        # 初始化雾气数据 (time, height_levels, y, x)
        fog_data = np.zeros((self.time_steps, fog_levels, self.height, self.width))
        
        # 生成稀疏的雾气模式
        base_fog_pattern = self._create_sparse_fog_pattern()
        
        for t in range(self.time_steps):
            for level_idx, fog_height in enumerate(fog_heights):
                # 时间变化因子（更平缓的变化）
                time_factor = 0.8 + 0.2 * np.sin(2 * np.pi * t / self.time_steps)
                
                # 高度衰减因子（越高雾气越少）
                height_factor = np.exp(-(fog_height - self.fog_base_height) / (self.fog_thickness * 0.8))
                
                # 地形影响因子（只在特定区域有雾）
                terrain_factor = self._calculate_sparse_terrain_fog_factor(fog_height)
                
                # 随机扰动（更小的变化）
                random_factor = 1.0 + 0.05 * np.sin(2 * np.pi * t / 8 + level_idx) * np.random.random((self.height, self.width)) * 0.3
                
                # 组合所有因子
                fog_intensity = (base_fog_pattern * time_factor * height_factor * 
                               terrain_factor * random_factor)
                
                # 应用更强的平滑滤波
                fog_intensity = ndimage.gaussian_filter(fog_intensity, sigma=2.0)
                
                # 限制雾气强度范围并应用覆盖率
                fog_intensity = np.clip(fog_intensity, 0, 1) * self.fog_coverage
                
                fog_data[t, level_idx, :, :] = fog_intensity
        
        return fog_data
    
    def _create_sparse_fog_pattern(self) -> np.ndarray:
        """创建稀疏的雾气分布模式"""
        # 创建更稀疏的雾气分布
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        
        # 创建几个雾气中心
        fog_centers = [
            (self.width * 0.3, self.height * 0.3),
            (self.width * 0.7, self.height * 0.4),
            (self.width * 0.5, self.height * 0.7)
        ]
        
        base_pattern = np.zeros((self.height, self.width))
        
        for center_x, center_y in fog_centers:
            # 高斯分布的雾气团
            distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            fog_patch = np.exp(-distance**2 / (min(self.width, self.height) * 0.3)**2)
            base_pattern += fog_patch * 0.4
        
        # 添加少量随机噪声
        noise = np.random.random((self.height, self.width)) * 0.1
        base_pattern += noise
        
        return np.clip(base_pattern, 0, 1)
    
    def _calculate_sparse_terrain_fog_factor(self, fog_height: float) -> np.ndarray:
        """计算稀疏的地形雾气影响"""
        # 计算相对高度
        relative_height = fog_height - self.terrain
        
        # 只在低洼地区有雾，且影响更小
        terrain_factor = np.where(relative_height > 0, 
                                np.exp(-relative_height / (self.fog_thickness * 0.5)), 
                                0)
        
        # 进一步降低地形影响
        terrain_factor *= 0.6
        
        return terrain_factor
    
    def get_fog_at_time(self, t: int) -> np.ndarray:
        """获取指定时间的雾气数据"""
        t = int(np.clip(t, 0, self.time_steps - 1))
        return self.fog_data[t]
    
    def is_position_blocked_by_fog(self, x: float, y: float, z: float, t: int, 
                                  fog_threshold: float = 0.3) -> bool:
        """检查位置是否被雾气阻挡（降低阈值）"""
        # 边界检查
        if (x < 0 or x >= self.width or y < 0 or y >= self.height or 
            z < self.fog_base_height or z > self.fog_max_height):
            return False
        
        # 获取当前时间的雾气数据
        fog_slice = self.get_fog_at_time(t)
        
        # 找到对应的高度层
        fog_levels = fog_slice.shape[0]
        fog_heights = np.linspace(self.fog_base_height, self.fog_max_height, fog_levels)
        
        # 找到最接近的高度层
        height_idx = np.argmin(np.abs(fog_heights - z))
        
        # 获取雾气强度（使用双线性插值）
        x_int, y_int = int(x), int(y)
        x_frac, y_frac = x - x_int, y - y_int
        
        # 边界处理
        x_int = min(x_int, self.width - 2)
        y_int = min(y_int, self.height - 2)
        
        # 双线性插值
        fog_intensity = (fog_slice[height_idx, y_int, x_int] * (1 - x_frac) * (1 - y_frac) +
                        fog_slice[height_idx, y_int, x_int + 1] * x_frac * (1 - y_frac) +
                        fog_slice[height_idx, y_int + 1, x_int] * (1 - x_frac) * y_frac +
                        fog_slice[height_idx, y_int + 1, x_int + 1] * x_frac * y_frac)
        
        return fog_intensity > fog_threshold


class OptimizedDynamicPathPlanning:
    """优化的动态环境路径规划类"""
    
    def __init__(self, fog_env: OptimizedDynamicFogEnvironment, max_altitude: float = 50.0):
        """初始化优化的动态路径规划"""
        self.fog_env = fog_env
        self.terrain = fog_env.terrain
        self.height, self.width = fog_env.terrain.shape
        self.max_altitude = max_altitude
        self.time_steps = fog_env.time_steps
        
        # 起点和终点
        self.start = (0, 0, self.terrain[0, 0] + 15)
        self.goal = (self.width - 1, self.height - 1, self.terrain[-1, -1] + 15)
        
        print(f"优化动态路径规划初始化:")
        print(f"  起点: {self.start}")
        print(f"  终点: {self.goal}")
    
    def _get_terrain_height(self, x: float, y: float) -> float:
        """获取地形高度"""
        x_int = int(np.clip(x, 0, self.width - 1))
        y_int = int(np.clip(y, 0, self.height - 1))
        return self.terrain[y_int, x_int]
    
    def _is_valid_position(self, pos: Tuple[float, float, float], t: int) -> bool:
        """检查位置在指定时间是否有效"""
        x, y, z = pos
        
        # 边界检查
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # 地形高度检查
        terrain_height = self._get_terrain_height(x, y)
        if z < terrain_height + 2 or z > terrain_height + self.max_altitude:
            return False
        
        # 雾气阻挡检查
        if self.fog_env.is_position_blocked_by_fog(x, y, z, t):
            return False
        
        return True
    
    def _get_cost_3d(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """计算三维移动代价"""
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        
        # 欧几里得距离
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        # 高度代价
        avg_height = (z1 + z2) / 2
        terrain_height = (self._get_terrain_height(x1, y1) + self._get_terrain_height(x2, y2)) / 2
        altitude_factor = 1.0 + (avg_height - terrain_height) / self.max_altitude * 0.3
        
        return distance * altitude_factor
    
    def _euclidean_distance_3d(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """三维欧几里得距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
    
    def dynamic_a_star(self, start_time: int = 0) -> Tuple[List[Tuple[float, float, float, int]], float]:
        """动态A*算法"""
        print(f"运行动态A*算法，起始时间: {start_time}")
        
        start_state = (*self.start, start_time)
        
        open_set = [(0, start_state)]
        came_from = {}
        g_score = {start_state: 0}
        f_score = {start_state: self._euclidean_distance_3d(self.start, self.goal)}
        
        max_iterations = 5000
        iteration = 0
        
        while open_set and iteration < max_iterations:
            iteration += 1
            current_state = heapq.heappop(open_set)[1]
            current_pos = current_state[:3]
            current_time = current_state[3]
            
            # 检查是否到达目标
            if self._euclidean_distance_3d(current_pos, self.goal) < 4.0:
                # 重构路径
                path = []
                state = current_state
                while state in came_from:
                    path.append(state)
                    state = came_from[state]
                path.append(start_state)
                
                total_cost = g_score[current_state]
                print(f"  找到路径! 迭代次数: {iteration}, 路径长度: {len(path[::-1])}")
                return path[::-1], total_cost
            
            # 扩展邻居节点（使用更大的步长）
            for dx in [-3, 0, 3]:
                for dy in [-3, 0, 3]:
                    for dz in [-3, 0, 3]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        
                        # 时间推进
                        next_time = min(current_time + 1, self.time_steps - 1)
                        next_pos = (current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz)
                        next_state = (*next_pos, next_time)
                        
                        # 检查新位置是否有效
                        if not self._is_valid_position(next_pos, next_time):
                            continue
                        
                        tentative_g = g_score[current_state] + self._get_cost_3d(current_pos, next_pos)
                        
                        if next_state not in g_score or tentative_g < g_score[next_state]:
                            came_from[next_state] = current_state
                            g_score[next_state] = tentative_g
                            f_score[next_state] = tentative_g + self._euclidean_distance_3d(next_pos, self.goal)
                            heapq.heappush(open_set, (f_score[next_state], next_state))
            
            if iteration % 1000 == 0:
                print(f"    动态A* 进度: {iteration}/{max_iterations} 迭代")
        
        print("  未找到路径")
        return [], float('inf')
    
    def dynamic_rrt(self, start_time: int = 0, max_iterations: int = 3000) -> Tuple[List[Tuple[float, float, float, int]], float]:
        """动态RRT算法"""
        print(f"运行动态RRT算法，起始时间: {start_time}")
        
        tree = {(*self.start, start_time): None}
        
        for iteration in range(max_iterations):
            # 随机采样
            if random.random() < 0.6:  # 60%概率采样目标
                rand_pos = self.goal
                rand_time = min(start_time + iteration // 100, self.time_steps - 1)
            else:
                rand_pos = (random.uniform(0, self.width-1), 
                           random.uniform(0, self.height-1),
                           random.uniform(self.start[2] - 10, self.start[2] + 20))
                rand_time = min(start_time + iteration // 100, self.time_steps - 1)
            
            rand_state = (*rand_pos, rand_time)
            
            # 找到最近的树节点
            nearest_state = min(tree.keys(), key=lambda s: self._euclidean_distance_3d(s[:3], rand_pos))
            nearest_pos = nearest_state[:3]
            
            # 向随机点扩展
            direction = np.array(rand_pos) - np.array(nearest_pos)
            length = np.linalg.norm(direction)
            
            if length > 0:
                step_size = min(4.0, length)
                new_pos = tuple(np.array(nearest_pos) + direction * step_size / length)
                new_time = min(nearest_state[3] + 1, self.time_steps - 1)
                new_state = (*new_pos, new_time)
                
                # 检查新点是否有效
                if self._is_valid_position(new_pos, new_time):
                    tree[new_state] = nearest_state
                    
                    # 检查是否到达目标
                    if self._euclidean_distance_3d(new_pos, self.goal) < 5.0:
                        # 重构路径
                        path = []
                        current = new_state
                        cost = 0
                        while current is not None:
                            path.append(current)
                            if tree[current] is not None:
                                cost += self._get_cost_3d(tree[current][:3], current[:3])
                            current = tree[current]
                        
                        print(f"  找到路径! 迭代次数: {iteration+1}, 路径长度: {len(path[::-1])}")
                        return path[::-1], cost
            
            if iteration % 500 == 0 and iteration > 0:
                print(f"    动态RRT 进度: {iteration}/{max_iterations} 迭代")
        
        print("  未找到路径")
        return [], float('inf')
    
    def visualize_dynamic_path(self, path: List[Tuple[float, float, float, int]], 
                              algorithm_name: str, output_path: str):
        """可视化动态路径"""
        if not path:
            print(f"无法可视化空路径: {algorithm_name}")
            return
        
        fig = plt.figure(figsize=(15, 12))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取路径坐标
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        z_coords = [p[2] for p in path]
        t_coords = [p[3] for p in path]
        
        # 1. 三维路径视图
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        # 绘制地形
        step = max(1, min(self.height, self.width) // 20)
        X, Y = np.meshgrid(range(0, self.width, step), range(0, self.height, step))
        Z_terrain = self.terrain[::step, ::step]
        ax1.plot_surface(X, Y, Z_terrain, alpha=0.3, cmap='terrain')
        
        # 绘制路径
        ax1.plot(x_coords, y_coords, z_coords, 'r-', linewidth=3, label='动态路径')
        ax1.scatter([self.start[0]], [self.start[1]], [self.start[2]], 
                   c='green', s=100, label='起点')
        ax1.scatter([self.goal[0]], [self.goal[1]], [self.goal[2]], 
                   c='red', s=100, label='终点')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z (高度)')
        ax1.set_title(f'{algorithm_name} - 三维路径')
        ax1.legend()
        
        # 2. 俯视图
        ax2 = fig.add_subplot(2, 2, 2)
        im2 = ax2.imshow(self.terrain, cmap='terrain', alpha=0.7, 
                        extent=[0, self.width, 0, self.height])
        ax2.plot(x_coords, y_coords, 'r-', linewidth=2, label='路径投影')
        ax2.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='起点')
        ax2.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='终点')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'{algorithm_name} - 俯视图')
        ax2.legend()
        plt.colorbar(im2, ax=ax2, label='高程')
        
        # 3. 高度随时间变化
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t_coords, z_coords, 'b-', linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('飞行高度')
        ax3.set_title('飞行高度随时间变化')
        ax3.grid(True, alpha=0.3)
        
        # 4. 路径长度统计
        ax4 = fig.add_subplot(2, 2, 4)
        
        # 计算累积距离
        cumulative_distance = [0]
        for i in range(1, len(path)):
            dist = self._euclidean_distance_3d(path[i-1][:3], path[i][:3])
            cumulative_distance.append(cumulative_distance[-1] + dist)
        
        ax4.plot(t_coords, cumulative_distance, 'g-', linewidth=2, marker='s', markersize=4)
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('累积距离')
        ax4.set_title('累积飞行距离')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{algorithm_name} 动态路径规划结果\n'
                    f'路径长度: {len(path)} 步, 总距离: {cumulative_distance[-1]:.1f}', 
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{algorithm_name} 路径可视化已保存: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("优化的动态水汽/雾气障碍物三维路径规划系统")
    print("=" * 60)
    
    # 查找TIF文件
    tif_files = []
    tif_dir = Path('tif')
    if tif_dir.exists():
        tif_files = list(tif_dir.glob('*.tif')) + list(tif_dir.glob('*.TIF'))
    
    if not tif_files:
        print("错误: 未在tif文件夹中找到TIF文件")
        return
    
    # 使用第一个找到的TIF文件
    tif_file = tif_files[0]
    print(f"使用TIF文件: {tif_file}")
    
    # 加载DEM数据
    try:
        with rasterio.open(str(tif_file)) as src:
            dem_data = src.read(1)
            if hasattr(src, 'nodata') and src.nodata is not None:
                dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)
    except Exception as e:
        print(f"加载TIF文件失败: {e}")
        return
    
    # 下采样
    if dem_data.shape[0] > 80 or dem_data.shape[1] > 80:
        print("DEM数据较大，进行下采样...")
        step = max(dem_data.shape[0] // 60, dem_data.shape[1] // 60, 1)
        dem_data = dem_data[::step, ::step]
        print(f"下采样后形状: {dem_data.shape}")
    
    # 创建优化的动态雾气环境
    print("\n创建优化的动态雾气环境...")
    fog_env = OptimizedDynamicFogEnvironment(dem_data, time_steps=15)
    
    # 创建优化的动态路径规划器
    print("\n创建优化的动态路径规划器...")
    planner = OptimizedDynamicPathPlanning(fog_env, max_altitude=40.0)
    
    # 保存结果
    output_dir = Path('output/optimized_dynamic_fog_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行多种动态路径规划算法
    print("\n运行优化的动态路径规划...")
    algorithms = {
        'Dynamic_A*': planner.dynamic_a_star,
        'Dynamic_RRT': planner.dynamic_rrt
    }
    
    results = {}
    start_times = [0, 3, 6]
    
    for start_time in start_times:
        for alg_name, alg_func in algorithms.items():
            test_name = f'{alg_name}_t{start_time}'
            print(f"\n--- {test_name} ---")
            
            start_exec_time = time.time()
            path, cost = alg_func(start_time)
            end_exec_time = time.time()
            
            execution_time = end_exec_time - start_exec_time
            success = len(path) > 0
            
            results[test_name] = {
                'execution_time': execution_time,
                'path_cost': cost,
                'path_length': len(path),
                'success': success,
                'start_time': start_time,
                'path': path
            }
            
            print(f"  执行时间: {execution_time:.4f}s")
            print(f"  路径代价: {cost:.2f}")
            print(f"  路径长度: {len(path)}")
            print(f"  成功: {success}")
            
            # 可视化成功的路径
            if success:
                # 处理文件名中的特殊字符
                safe_test_name = test_name.replace('*', 'Star').replace('/', '_').replace('\\', '_')
                viz_path = output_dir / f'{safe_test_name}_path.png'
                planner.visualize_dynamic_path(path, test_name, str(viz_path))
    
    # 保存结果到CSV
    csv_file = output_dir / 'optimized_dynamic_results.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Algorithm', 'Start_Time', 'Execution_Time_s', 'Path_Cost', 'Path_Length', 'Success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for alg_name, result in results.items():
            writer.writerow({
                'Algorithm': alg_name,
                'Start_Time': result['start_time'],
                'Execution_Time_s': result['execution_time'],
                'Path_Cost': result['path_cost'],
                'Path_Length': result['path_length'],
                'Success': result['success']
            })
    
    print(f"\n结果已保存到: {csv_file}")
    
    # 显示总结
    print(f"\n=" * 60)
    print("优化的动态雾气路径规划测试完成!")
    print(f"结果保存在: {output_dir}")
    
    successful_results = {name: result for name, result in results.items() if result['success']}
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]['execution_time'])
        print(f"最快算法: {fastest[0]} ({fastest[1]['execution_time']:.4f}s)")
        
        lowest_cost = min(successful_results.items(), key=lambda x: x[1]['path_cost'])
        print(f"最优路径: {lowest_cost[0]} (代价: {lowest_cost[1]['path_cost']:.2f})")
        
        print(f"成功率: {len(successful_results)}/{len(results)} = {len(successful_results)/len(results)*100:.1f}%")
    else:
        print("所有算法都未找到路径，建议进一步降低雾气密度")
    
    print(f"=" * 60)


if __name__ == "__main__":
    main()