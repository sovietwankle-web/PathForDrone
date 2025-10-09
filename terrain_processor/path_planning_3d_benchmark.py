#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三维地形路径规划算法性能测试脚本
实现多种三维路径规划算法并进行性能对比测试
支持空中路径，不限制贴地面行走
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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

class PathPlanning3DBenchmark:
    """三维路径规划算法性能测试类"""
    
    def __init__(self, terrain_data: np.ndarray, max_altitude: float = 100.0):
        """
        初始化三维路径规划测试
        
        Args:
            terrain_data: 地形高程数据
            max_altitude: 最大飞行高度（相对于地面）
        """
        self.terrain = terrain_data
        self.height, self.width = terrain_data.shape
        self.max_altitude = max_altitude
        
        # 三维空间的起点和终点（x, y, z）
        self.start = (0, 0, self.terrain[0, 0] + 10)  # 左下角，高度+10
        self.goal = (self.width - 1, self.height - 1, self.terrain[-1, -1] + 10)  # 右上角，高度+10
        
        # 处理NaN值，用平均值替代
        if np.any(np.isnan(terrain_data)):
            mean_elevation = np.nanmean(terrain_data)
            self.terrain = np.where(np.isnan(terrain_data), mean_elevation, terrain_data)
        
        # 地形高度范围
        self.min_elevation = np.min(self.terrain)
        self.max_elevation = np.max(self.terrain)
        
        print(f"地形尺寸: {self.height} x {self.width}")
        print(f"起点: {self.start}, 终点: {self.goal}")
        print(f"地形高程范围: {self.min_elevation:.2f} - {self.max_elevation:.2f}")
        print(f"最大飞行高度: {self.max_elevation + self.max_altitude:.2f}")
    
    def _get_terrain_height(self, x: float, y: float) -> float:
        """获取指定位置的地形高度"""
        x_int = int(np.clip(x, 0, self.width - 1))
        y_int = int(np.clip(y, 0, self.height - 1))
        return self.terrain[y_int, x_int]
    
    def _is_valid_position(self, pos: Tuple[float, float, float]) -> bool:
        """检查位置是否有效（在边界内且高于地面）"""
        x, y, z = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        terrain_height = self._get_terrain_height(x, y)
        if z < terrain_height or z > terrain_height + self.max_altitude:
            return False
        
        return True
    
    def _get_neighbors_3d(self, pos: Tuple[float, float, float], step_size: float = 1.0) -> List[Tuple[float, float, float]]:
        """获取三维26邻域"""
        x, y, z = pos
        neighbors = []
        
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                for dz in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    new_pos = (x + dx, y + dy, z + dz)
                    if self._is_valid_position(new_pos):
                        neighbors.append(new_pos)
        
        return neighbors
    
    def _get_cost_3d(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """计算三维空间中两点间的移动代价"""
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        
        # 欧几里得距离
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        # 高度代价（飞得越高代价越大）
        avg_height = (z1 + z2) / 2
        terrain_height = (self._get_terrain_height(x1, y1) + self._get_terrain_height(x2, y2)) / 2
        altitude_factor = 1.0 + (avg_height - terrain_height) / self.max_altitude * 0.5
        
        # 地形复杂度代价
        terrain_gradient = abs(self._get_terrain_height(x2, y2) - self._get_terrain_height(x1, y1))
        terrain_factor = 1.0 + terrain_gradient / (self.max_elevation - self.min_elevation) * 0.3
        
        return distance * altitude_factor * terrain_factor
    
    def _euclidean_distance_3d(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """三维欧几里得距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
    
    def dijkstra_3d(self) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维Dijkstra算法"""
        distances = {self.start: 0}
        previous = {}
        pq = [(0, self.start)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if self._euclidean_distance_3d(current, self.goal) < 2.0:
                # 重构路径
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(self.start)
                return path[::-1], distances[current]
            
            for neighbor in self._get_neighbors_3d(current):
                if neighbor in visited:
                    continue
                
                cost = self._get_cost_3d(current, neighbor)
                distance = current_dist + cost
                
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return [], float('inf')
    
    def a_star_3d(self) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维A*算法"""
        open_set = [(0, self.start)]
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self._euclidean_distance_3d(self.start, self.goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if self._euclidean_distance_3d(current, self.goal) < 2.0:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1], g_score[current]
            
            for neighbor in self._get_neighbors_3d(current):
                tentative_g = g_score[current] + self._get_cost_3d(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._euclidean_distance_3d(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return [], float('inf')
    
    def rrt_3d(self, max_iterations: int = 1000, step_size: float = 3.0) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维RRT算法"""
        tree = {self.start: None}
        
        for _ in range(max_iterations):
            # 随机采样
            if random.random() < 0.1:  # 10%概率采样目标点
                rand_point = self.goal
            else:
                x = random.uniform(0, self.width - 1)
                y = random.uniform(0, self.height - 1)
                terrain_height = self._get_terrain_height(x, y)
                z = random.uniform(terrain_height, terrain_height + self.max_altitude)
                rand_point = (x, y, z)
            
            # 找到最近的树节点
            nearest = min(tree.keys(), key=lambda p: self._euclidean_distance_3d(p, rand_point))
            
            # 向随机点扩展
            direction = np.array(rand_point) - np.array(nearest)
            length = np.linalg.norm(direction)
            
            if length > 0:
                step = min(step_size, length)
                new_point = tuple(np.array(nearest) + direction * step / length)
                
                # 检查新点是否有效
                if self._is_valid_position(new_point):
                    tree[new_point] = nearest
                    
                    # 检查是否到达目标
                    if self._euclidean_distance_3d(new_point, self.goal) < step_size:
                        tree[self.goal] = new_point
                        
                        # 重构路径
                        path = []
                        current = self.goal
                        cost = 0
                        while current is not None:
                            path.append(current)
                            if tree[current] is not None:
                                cost += self._get_cost_3d(tree[current], current)
                            current = tree[current]
                        
                        return path[::-1], cost
        
        return [], float('inf')
    
    def straight_line_3d(self) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维直线路径"""
        start = np.array(self.start)
        goal = np.array(self.goal)
        
        # 生成直线路径点
        num_points = int(np.linalg.norm(goal - start)) + 1
        path = []
        total_cost = 0
        
        for i in range(num_points + 1):
            t = i / num_points
            point = tuple(start + t * (goal - start))
            
            # 检查点是否有效
            if self._is_valid_position(point):
                path.append(point)
                if len(path) > 1:
                    total_cost += self._get_cost_3d(path[-2], path[-1])
            else:
                # 如果直线路径无效，返回空路径
                return [], float('inf')
        
        return path, total_cost
    
    def run_all_algorithms_3d(self) -> Dict[str, Dict[str, Any]]:
        """运行所有三维算法并记录性能"""
        algorithms = {
            'Dijkstra_3D': self.dijkstra_3d,
            'A*_3D': self.a_star_3d,
            'RRT_3D': self.rrt_3d,
            'Straight_Line_3D': self.straight_line_3d
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"运行 {name} 算法...")
            
            try:
                start_time = time.time()
                path, cost = algorithm()
                end_time = time.time()
                
                execution_time = end_time - start_time
                path_length = len(path)
                success = len(path) > 0
                
                results[name] = {
                    'execution_time': execution_time,
                    'path_cost': cost,
                    'path_length': path_length,
                    'success': success,
                    'path': path
                }
                
                print(f"  {name}: {execution_time:.4f}s, 代价: {cost:.2f}, 路径长度: {path_length}, 成功: {success}")
                
            except Exception as e:
                print(f"  {name} 算法执行失败: {e}")
                results[name] = {
                    'execution_time': float('inf'),
                    'path_cost': float('inf'),
                    'path_length': 0,
                    'success': False,
                    'path': [],
                    'error': str(e)
                }
        
        return results
    
    def save_results_to_csv(self, results: Dict[str, Dict[str, Any]], filename: str):
        """保存结果到CSV文件"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Algorithm', 'Execution_Time_s', 'Path_Cost', 'Path_Length', 'Success', 'Error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for algorithm, result in results.items():
                writer.writerow({
                    'Algorithm': algorithm,
                    'Execution_Time_s': result['execution_time'],
                    'Path_Cost': result['path_cost'],
                    'Path_Length': result['path_length'],
                    'Success': result['success'],
                    'Error': result.get('error', '')
                })
        
        print(f"结果已保存到: {filename}")
    
    def visualize_paths_3d(self, results: Dict[str, Dict[str, Any]], output_dir: str):
        """生成三维路径的三视图可视化"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 选择成功的算法进行可视化
        successful_algorithms = {name: result for name, result in results.items() 
                               if result['success'] and len(result['path']) > 0}
        
        if not successful_algorithms:
            print("没有成功的算法可以可视化")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 为每个成功的算法生成三视图
        for name, result in successful_algorithms.items():
            path = result['path']
            if not path:
                continue
            
            # 提取路径坐标
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            z_coords = [p[2] for p in path]
            
            # 创建三视图
            fig = plt.figure(figsize=(15, 12))
            
            # 1. 三维视图
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            
            # 绘制地形表面（下采样以提高性能）
            step = max(1, min(self.height, self.width) // 50)
            X, Y = np.meshgrid(range(0, self.width, step), range(0, self.height, step))
            Z = self.terrain[::step, ::step]
            ax1.plot_surface(X, Y, Z, alpha=0.3, cmap='terrain')
            
            # 绘制路径
            ax1.plot(x_coords, y_coords, z_coords, 'r-', linewidth=3, label='路径')
            ax1.scatter([self.start[0]], [self.start[1]], [self.start[2]], 
                       c='green', s=100, label='起点')
            ax1.scatter([self.goal[0]], [self.goal[1]], [self.goal[2]], 
                       c='red', s=100, label='终点')
            
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z (高度)')
            ax1.set_title(f'{name} - 三维视图')
            ax1.legend()
            
            # 2. 俯视图 (XY平面)
            ax2 = fig.add_subplot(2, 2, 2)
            im = ax2.imshow(self.terrain, cmap='terrain', alpha=0.7, extent=[0, self.width, 0, self.height])
            ax2.plot(x_coords, y_coords, 'r-', linewidth=2, label='路径投影')
            ax2.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='起点')
            ax2.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='终点')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title(f'{name} - 俯视图 (XY)')
            ax2.legend()
            plt.colorbar(im, ax=ax2, label='高程')
            
            # 3. 侧视图 (XZ平面)
            ax3 = fig.add_subplot(2, 2, 3)
            # 绘制地形轮廓
            terrain_profile_x = []
            terrain_height_x = []
            for i, x in enumerate(x_coords):
                y = y_coords[i]
                terrain_profile_x.append(x)
                terrain_height_x.append(self._get_terrain_height(x, y))
            
            ax3.fill_between(range(self.width), 
                           [self._get_terrain_height(x, self.height//2) for x in range(self.width)], 
                           alpha=0.3, color='brown', label='地形')
            ax3.plot(x_coords, z_coords, 'r-', linewidth=2, label='路径')
            ax3.scatter([self.start[0]], [self.start[2]], c='green', s=100, label='起点')
            ax3.scatter([self.goal[0]], [self.goal[2]], c='red', s=100, label='终点')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Z (高度)')
            ax3.set_title(f'{name} - 侧视图 (XZ)')
            ax3.legend()
            
            # 4. 侧视图 (YZ平面)
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.fill_between(range(self.height), 
                           [self._get_terrain_height(self.width//2, y) for y in range(self.height)], 
                           alpha=0.3, color='brown', label='地形')
            ax4.plot(y_coords, z_coords, 'r-', linewidth=2, label='路径')
            ax4.scatter([self.start[1]], [self.start[2]], c='green', s=100, label='起点')
            ax4.scatter([self.goal[1]], [self.goal[2]], c='red', s=100, label='终点')
            ax4.set_xlabel('Y')
            ax4.set_ylabel('Z (高度)')
            ax4.set_title(f'{name} - 侧视图 (YZ)')
            ax4.legend()
            
            plt.suptitle(f'{name} 三维路径规划结果\n'
                        f'执行时间: {result["execution_time"]:.3f}s, '
                        f'路径代价: {result["path_cost"]:.1f}, '
                        f'路径长度: {result["path_length"]}', 
                        fontsize=14)
            
            plt.tight_layout()
            
            # 保存图片（处理文件名中的特殊字符）
            safe_name = name.replace('*', 'Star').replace('/', '_').replace('\\', '_')
            output_path = output_dir / f'{safe_name}_3d_views.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{name} 三视图已保存: {output_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("三维地形路径规划算法性能测试")
    print("=" * 60)
    
    # 查找可用的地形数据文件
    data_files = [
        'tif/output/bilateral_smooth_data.npy',
        'tif/output/smoothed_terrain.npy',
        'tif/output/DEM_5m_data.npy',
        'tif/output/DEM_1m_data.npy'
    ]
    
    terrain_file = None
    for file in data_files:
        if os.path.exists(file):
            terrain_file = file
            break
    
    if not terrain_file:
        print("错误: 未找到地形数据文件")
        print("请先运行地形处理脚本生成平滑后的地形数据")
        return
    
    print(f"使用地形数据文件: {terrain_file}")
    
    # 加载地形数据
    terrain_data = np.load(terrain_file)
    print(f"地形数据形状: {terrain_data.shape}")
    
    # 如果数据太大，进行下采样
    if terrain_data.shape[0] > 100 or terrain_data.shape[1] > 100:
        print("地形数据较大，进行下采样...")
        step = max(terrain_data.shape[0] // 80, terrain_data.shape[1] // 80, 1)
        terrain_data = terrain_data[::step, ::step]
        print(f"下采样后形状: {terrain_data.shape}")
    
    # 创建测试实例
    benchmark = PathPlanning3DBenchmark(terrain_data, max_altitude=50.0)
    
    # 运行所有算法
    print(f"\n开始三维路径规划算法性能测试...")
    results = benchmark.run_all_algorithms_3d()
    
    # 保存结果到CSV
    output_dir = Path('output/path_planning_3d_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_filename = output_dir / 'path_planning_3d_benchmark.csv'
    benchmark.save_results_to_csv(results, csv_filename)
    
    # 生成三视图可视化
    benchmark.visualize_paths_3d(results, output_dir)
    
    # 打印总结
    print(f"\n=" * 60)
    print("三维路径规划性能测试完成!")
    print(f"结果保存在: {output_dir}")
    print(f"CSV文件: {csv_filename}")
    
    # 显示最快的算法
    successful_results = {name: result for name, result in results.items() 
                         if result['success']}
    
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]['execution_time'])
        print(f"最快算法: {fastest[0]} ({fastest[1]['execution_time']:.4f}s)")
        
        lowest_cost = min(successful_results.items(), key=lambda x: x[1]['path_cost'])
        print(f"最优路径: {lowest_cost[0]} (代价: {lowest_cost[1]['path_cost']:.2f})")
    
    print(f"=" * 60)

if __name__ == "__main__":
    main()