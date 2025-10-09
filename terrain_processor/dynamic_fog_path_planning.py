#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态水汽/雾气障碍物三维路径规划系统
实现随时间变化的雾气障碍物，并进行动态路径规划
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

class DynamicFogEnvironment:
    """动态雾气环境类"""
    
    def __init__(self, terrain_data: np.ndarray, time_steps: int = 50):
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
        
        # 雾气参数
        self.fog_base_height = self.min_elevation + 0.3 * self.elevation_range  # 雾气基础高度
        self.fog_max_height = self.min_elevation + 0.7 * self.elevation_range   # 雾气最大高度
        self.fog_thickness = self.fog_max_height - self.fog_base_height
        
        # 生成动态雾气数据
        self.fog_data = self._generate_dynamic_fog()
        
        print(f"动态雾气环境初始化完成:")
        print(f"  地形尺寸: {self.height} x {self.width}")
        print(f"  时间步数: {self.time_steps}")
        print(f"  地形高程范围: {self.min_elevation:.2f} - {self.max_elevation:.2f}")
        print(f"  雾气高度范围: {self.fog_base_height:.2f} - {self.fog_max_height:.2f}")
    
    def _generate_dynamic_fog(self) -> np.ndarray:
        """
        生成动态雾气数据
        
        Returns:
            fog_data: 四维数组 (time, height_levels, y, x)
        """
        print("生成动态雾气数据...")
        
        # 雾气高度层数
        fog_levels = 20
        fog_heights = np.linspace(self.fog_base_height, self.fog_max_height, fog_levels)
        
        # 初始化雾气数据 (time, height_levels, y, x)
        fog_data = np.zeros((self.time_steps, fog_levels, self.height, self.width))
        
        # 生成基础雾气模式
        base_fog_pattern = self._create_base_fog_pattern()
        
        for t in range(self.time_steps):
            for level_idx, fog_height in enumerate(fog_heights):
                # 时间变化因子
                time_factor = np.sin(2 * np.pi * t / self.time_steps) * 0.3 + 0.7
                
                # 高度衰减因子（越高雾气越少）
                height_factor = np.exp(-(fog_height - self.fog_base_height) / (self.fog_thickness * 0.5))
                
                # 地形影响因子（山谷容易有雾）
                terrain_factor = self._calculate_terrain_fog_factor(fog_height)
                
                # 随机扰动（保持整体变化不大）
                random_factor = 1.0 + 0.1 * np.sin(2 * np.pi * t / 10 + level_idx) * np.random.random((self.height, self.width)) * 0.5
                
                # 组合所有因子
                fog_intensity = (base_fog_pattern * time_factor * height_factor * 
                               terrain_factor * random_factor)
                
                # 应用平滑滤波
                fog_intensity = ndimage.gaussian_filter(fog_intensity, sigma=1.0)
                
                # 限制雾气强度范围 [0, 1]
                fog_intensity = np.clip(fog_intensity, 0, 1)
                
                fog_data[t, level_idx, :, :] = fog_intensity
        
        return fog_data
    
    def _create_base_fog_pattern(self) -> np.ndarray:
        """创建基础雾气分布模式"""
        # 使用多个正弦波创建自然的雾气分布
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        
        # 多尺度噪声
        pattern1 = 0.5 * (np.sin(X * 2 * np.pi / self.width * 3) + 1)
        pattern2 = 0.3 * (np.sin(Y * 2 * np.pi / self.height * 2) + 1)
        pattern3 = 0.2 * (np.sin((X + Y) * 2 * np.pi / (self.width + self.height) * 4) + 1)
        
        base_pattern = (pattern1 + pattern2 + pattern3) / 3
        
        # 添加随机噪声
        noise = np.random.random((self.height, self.width)) * 0.2
        base_pattern += noise
        
        return np.clip(base_pattern, 0, 1)
    
    def _calculate_terrain_fog_factor(self, fog_height: float) -> np.ndarray:
        """计算地形对雾气分布的影响"""
        # 计算相对高度（雾气高度相对于地形的高度）
        relative_height = fog_height - self.terrain
        
        # 雾气更容易在低洼地区聚集
        terrain_factor = np.where(relative_height > 0, 
                                np.exp(-relative_height / (self.fog_thickness * 0.3)), 
                                0)
        
        return terrain_factor
    
    def get_fog_at_time(self, t: int) -> np.ndarray:
        """
        获取指定时间的雾气数据
        
        Args:
            t: 时间步
            
        Returns:
            fog_slice: 三维数组 (height_levels, y, x)
        """
        t = int(np.clip(t, 0, self.time_steps - 1))
        return self.fog_data[t]
    
    def is_position_blocked_by_fog(self, x: float, y: float, z: float, t: int, 
                                  fog_threshold: float = 0.5) -> bool:
        """
        检查指定位置在指定时间是否被雾气阻挡
        
        Args:
            x, y, z: 三维坐标
            t: 时间步
            fog_threshold: 雾气阻挡阈值
            
        Returns:
            bool: 是否被雾气阻挡
        """
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
    
    def save_fog_data(self, filename: str):
        """保存雾气数据到文件"""
        np.save(filename, self.fog_data)
        print(f"雾气数据已保存到: {filename}")
    
    def load_fog_data(self, filename: str):
        """从文件加载雾气数据"""
        self.fog_data = np.load(filename)
        print(f"雾气数据已从文件加载: {filename}")


class DynamicPathPlanning:
    """动态环境路径规划类"""
    
    def __init__(self, fog_env: DynamicFogEnvironment, max_altitude: float = 50.0):
        """
        初始化动态路径规划
        
        Args:
            fog_env: 动态雾气环境
            max_altitude: 最大飞行高度
        """
        self.fog_env = fog_env
        self.terrain = fog_env.terrain
        self.height, self.width = fog_env.terrain.shape
        self.max_altitude = max_altitude
        self.time_steps = fog_env.time_steps
        
        # 起点和终点
        self.start = (0, 0, self.terrain[0, 0] + 10)
        self.goal = (self.width - 1, self.height - 1, self.terrain[-1, -1] + 10)
        
        print(f"动态路径规划初始化:")
        print(f"  起点: {self.start}")
        print(f"  终点: {self.goal}")
        print(f"  最大飞行高度: {self.fog_env.max_elevation + max_altitude:.2f}")
    
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
        if z < terrain_height or z > terrain_height + self.max_altitude:
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
        altitude_factor = 1.0 + (avg_height - terrain_height) / self.max_altitude * 0.5
        
        return distance * altitude_factor
    
    def _euclidean_distance_3d(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """三维欧几里得距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
    
    def dynamic_a_star(self, start_time: int = 0) -> Tuple[List[Tuple[float, float, float, int]], float]:
        """
        动态A*算法（考虑时间维度）
        
        Args:
            start_time: 起始时间
            
        Returns:
            path: 路径列表 [(x, y, z, t), ...]
            cost: 路径总代价
        """
        print(f"运行动态A*算法，起始时间: {start_time}")
        
        # 状态: (x, y, z, t)
        start_state = (*self.start, start_time)
        
        open_set = [(0, start_state)]
        came_from = {}
        g_score = {start_state: 0}
        f_score = {start_state: self._euclidean_distance_3d(self.start, self.goal)}
        
        max_iterations = 10000
        iteration = 0
        
        while open_set and iteration < max_iterations:
            iteration += 1
            current_state = heapq.heappop(open_set)[1]
            current_pos = current_state[:3]
            current_time = current_state[3]
            
            # 检查是否到达目标
            if self._euclidean_distance_3d(current_pos, self.goal) < 3.0:
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
            
            # 扩展邻居节点
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    for dz in [-2, 0, 2]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        
                        # 时间推进（每步消耗1个时间单位）
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
    
    def visualize_dynamic_environment(self, t: int, output_path: str):
        """可视化指定时间的动态环境"""
        fig = plt.figure(figsize=(15, 12))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 三维视图
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        # 绘制地形
        step = max(1, min(self.height, self.width) // 30)
        X, Y = np.meshgrid(range(0, self.width, step), range(0, self.height, step))
        Z_terrain = self.terrain[::step, ::step]
        ax1.plot_surface(X, Y, Z_terrain, alpha=0.3, cmap='terrain')
        
        # 绘制雾气
        fog_slice = self.fog_env.get_fog_at_time(t)
        fog_levels = fog_slice.shape[0]
        fog_heights = np.linspace(self.fog_env.fog_base_height, self.fog_env.fog_max_height, fog_levels)
        
        for level_idx in range(0, fog_levels, 3):  # 每3层显示一次
            fog_layer = fog_slice[level_idx, ::step, ::step]
            Z_fog = np.full_like(Z_terrain, fog_heights[level_idx])
            
            # 只显示雾气强度大于0.3的区域
            mask = fog_layer > 0.3
            if np.any(mask):
                # 使用平均雾气强度作为alpha值
                avg_fog_intensity = np.mean(fog_layer[mask])
                ax1.plot_surface(X, Y, Z_fog, alpha=min(avg_fog_intensity * 0.8, 0.6),
                               facecolors=plt.cm.Blues(fog_layer),
                               linewidth=0, antialiased=False)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z (高度)')
        ax1.set_title(f'动态环境 - 时间步 {t}')
        
        # 2. 俯视图 - 地形
        ax2 = fig.add_subplot(2, 2, 2)
        im2 = ax2.imshow(self.terrain, cmap='terrain', alpha=0.7, 
                        extent=[0, self.width, 0, self.height])
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('地形俯视图')
        plt.colorbar(im2, ax=ax2, label='高程')
        
        # 3. 俯视图 - 雾气分布（平均）
        ax3 = fig.add_subplot(2, 2, 3)
        fog_avg = np.mean(fog_slice, axis=0)  # 对所有高度层求平均
        im3 = ax3.imshow(fog_avg, cmap='Blues', alpha=0.8, 
                        extent=[0, self.width, 0, self.height])
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title(f'雾气分布俯视图 - 时间步 {t}')
        plt.colorbar(im3, ax=ax3, label='雾气强度')
        
        # 4. 雾气随时间变化曲线
        ax4 = fig.add_subplot(2, 2, 4)
        
        # 计算每个时间步的平均雾气强度
        fog_intensity_over_time = []
        for time_step in range(self.time_steps):
            fog_t = self.fog_env.get_fog_at_time(time_step)
            avg_intensity = np.mean(fog_t)
            fog_intensity_over_time.append(avg_intensity)
        
        ax4.plot(range(self.time_steps), fog_intensity_over_time, 'b-', linewidth=2)
        ax4.axvline(x=t, color='r', linestyle='--', label=f'当前时间 {t}')
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('平均雾气强度')
        ax4.set_title('雾气强度随时间变化')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'动态雾气环境可视化 - 时间步 {t}', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"动态环境可视化已保存: {output_path}")


def load_tif_dem(tif_path: str) -> np.ndarray:
    """从TIF文件加载DEM数据"""
    print(f"加载TIF文件: {tif_path}")
    
    with rasterio.open(tif_path) as src:
        dem_data = src.read(1)  # 读取第一个波段
        
        # 处理无效值
        if hasattr(src, 'nodata') and src.nodata is not None:
            dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)
        
        print(f"DEM数据形状: {dem_data.shape}")
        print(f"高程范围: {np.nanmin(dem_data):.2f} - {np.nanmax(dem_data):.2f}")
        
        return dem_data


def main():
    """主函数"""
    print("=" * 60)
    print("动态水汽/雾气障碍物三维路径规划系统")
    print("=" * 60)
    
    # 查找TIF文件
    tif_files = []
    tif_dir = Path('tif')
    if tif_dir.exists():
        tif_files = list(tif_dir.glob('*.tif')) + list(tif_dir.glob('*.TIF'))
    
    if not tif_files:
        print("错误: 未在tif文件夹中找到TIF文件")
        print("请将DEM的TIF文件放入tif文件夹中")
        return
    
    # 使用第一个找到的TIF文件
    tif_file = tif_files[0]
    print(f"使用TIF文件: {tif_file}")
    
    # 加载DEM数据
    try:
        dem_data = load_tif_dem(str(tif_file))
    except Exception as e:
        print(f"加载TIF文件失败: {e}")
        return
    
    # 如果数据太大，进行下采样
    if dem_data.shape[0] > 100 or dem_data.shape[1] > 100:
        print("DEM数据较大，进行下采样...")
        step = max(dem_data.shape[0] // 80, dem_data.shape[1] // 80, 1)
        dem_data = dem_data[::step, ::step]
        print(f"下采样后形状: {dem_data.shape}")
    
    # 创建动态雾气环境
    print("\n创建动态雾气环境...")
    fog_env = DynamicFogEnvironment(dem_data, time_steps=30)
    
    # 保存雾气数据
    output_dir = Path('output/dynamic_fog_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fog_data_file = output_dir / 'fog_data.npy'
    fog_env.save_fog_data(str(fog_data_file))
    
    # 创建动态路径规划器
    print("\n创建动态路径规划器...")
    planner = DynamicPathPlanning(fog_env, max_altitude=30.0)
    
    # 可视化不同时间步的环境
    print("\n生成环境可视化...")
    for t in [0, 10, 20, 29]:
        viz_path = output_dir / f'dynamic_environment_t{t:02d}.png'
        planner.visualize_dynamic_environment(t, str(viz_path))
    
    # 运行动态路径规划
    print("\n运行动态路径规划...")
    start_times = [0, 5, 10, 15]
    results = {}
    
    for start_time in start_times:
        print(f"\n--- 起始时间 {start_time} ---")
        start_exec_time = time.time()
        path, cost = planner.dynamic_a_star(start_time)
        end_exec_time = time.time()
        
        execution_time = end_exec_time - start_exec_time
        success = len(path) > 0
        
        results[f'Dynamic_A*_t{start_time}'] = {
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
    
    # 保存结果到CSV
    csv_file = output_dir / 'dynamic_path_planning_results.csv'
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
    print("动态雾气路径规划测试完成!")
    print(f"结果保存在: {output_dir}")
    
    successful_results = {name: result for name, result in results.items() if result['success']}
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]['execution_time'])
        print(f"最快算法: {fastest[0]} ({fastest[1]['execution_time']:.4f}s)")
        
        lowest_cost = min(successful_results.items(), key=lambda x: x[1]['path_cost'])
        print(f"最优路径: {lowest_cost[0]} (代价: {lowest_cost[1]['path_cost']:.2f})")
    
    print(f"=" * 60)


if __name__ == "__main__":
    main()