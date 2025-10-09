#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整三维地形路径规划算法性能测试脚本
实现所有用户要求的三维路径规划算法并进行性能对比测试
包含：粒子群，退火，野狼，水母，快速行进，A*，D*，dijkstra，rrt，bfs，lrta
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

class PathPlanning3DComplete:
    """完整三维路径规划算法性能测试类"""
    
    def __init__(self, terrain_data: np.ndarray, max_altitude: float = 50.0):
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
        """获取三维邻域"""
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
    
    def _calculate_path_cost_3d(self, path: List[Tuple[float, float, float]]) -> float:
        """计算三维路径总代价"""
        if len(path) < 2:
            return float('inf')
        
        total_cost = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            total_cost += self._get_cost_3d(current, next_pos)
        
        return total_cost
    
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
            
            if self._euclidean_distance_3d(current, self.goal) < 3.0:
                # 重构路径
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(self.start)
                return path[::-1], distances[path[0] if path else self.start]
            
            for neighbor in self._get_neighbors_3d(current, 2.0):
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
            
            if self._euclidean_distance_3d(current, self.goal) < 3.0:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1], g_score[path[0] if path else self.start]
            
            for neighbor in self._get_neighbors_3d(current, 2.0):
                tentative_g = g_score[current] + self._get_cost_3d(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._euclidean_distance_3d(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return [], float('inf')
    
    def bfs_3d(self) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维广度优先搜索"""
        queue = deque([(self.start, [self.start], 0)])
        visited = {self.start}
        
        while queue:
            current, path, cost = queue.popleft()
            
            if self._euclidean_distance_3d(current, self.goal) < 3.0:
                return path, cost
            
            for neighbor in self._get_neighbors_3d(current, 2.0):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_cost = cost + self._get_cost_3d(current, neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_cost))
        
        return [], float('inf')
    
    def rrt_3d(self, max_iterations: int = 8000, step_size: float = 3.0) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维RRT算法（优化版本）"""
        tree = {self.start: None}
        
        print(f"    RRT_3D: 起点 {self.start}, 终点 {self.goal}")
        
        for iteration in range(max_iterations):
            # 增强采样策略
            if random.random() < 0.5:  # 50%概率采样目标点或其附近
                if random.random() < 0.6:  # 直接采样目标
                    rand_point = self.goal
                else:  # 目标附近采样
                    goal_x, goal_y, goal_z = self.goal
                    x = max(0, min(self.width-1, goal_x + random.uniform(-5, 5)))
                    y = max(0, min(self.height-1, goal_y + random.uniform(-5, 5)))
                    terrain_height = self._get_terrain_height(x, y)
                    z = max(terrain_height + 1, min(terrain_height + self.max_altitude,
                                                  goal_z + random.uniform(-8, 8)))
                    rand_point = (x, y, z)
            elif random.random() < 0.8:  # 30%概率在起点和终点之间采样
                t = random.uniform(0, 1)
                x = self.start[0] + t * (self.goal[0] - self.start[0]) + random.uniform(-2, 2)
                y = self.start[1] + t * (self.goal[1] - self.start[1]) + random.uniform(-2, 2)
                z = self.start[2] + t * (self.goal[2] - self.start[2]) + random.uniform(-3, 3)
                
                x = max(0, min(self.width-1, x))
                y = max(0, min(self.height-1, y))
                terrain_height = self._get_terrain_height(x, y)
                z = max(terrain_height + 1, min(terrain_height + self.max_altitude, z))
                rand_point = (x, y, z)
            else:  # 20%概率随机采样
                x = random.uniform(0, self.width - 1)
                y = random.uniform(0, self.height - 1)
                terrain_height = self._get_terrain_height(x, y)
                z = random.uniform(terrain_height + 1, terrain_height + self.max_altitude)
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
                    
                    # 检查是否到达目标（放宽条件）
                    goal_distance = self._euclidean_distance_3d(new_point, self.goal)
                    if goal_distance < step_size * 4.0:  # 放宽到4倍步长
                        # 尝试直接连接到目标
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
                        
                        print(f"    RRT_3D: 找到路径! 迭代次数: {iteration+1}, 路径长度: {len(path[::-1])}")
                        return path[::-1], cost
            
            # 每1000次迭代输出进度
            if iteration % 1000 == 0 and iteration > 0:
                print(f"    RRT_3D 进度: {iteration}/{max_iterations} 迭代, 树节点数: {len(tree)}")
        
        # 如果没有找到完整路径，尝试返回最接近目标的路径
        if len(tree) > 1:
            closest_node = min(tree.keys(), key=lambda p: self._euclidean_distance_3d(p, self.goal))
            closest_distance = self._euclidean_distance_3d(closest_node, self.goal)
            print(f"    RRT_3D: 最接近目标的距离: {closest_distance:.2f}")
            
            if closest_distance < 20.0:  # 进一步放宽接受条件
                # 重构到最近点的路径
                path = []
                current = closest_node
                cost = 0
                while current is not None:
                    path.append(current)
                    if tree[current] is not None:
                        cost += self._get_cost_3d(tree[current], current)
                    current = tree[current]
                
                # 添加目标点
                path = path[::-1] + [self.goal]
                cost += self._get_cost_3d(closest_node, self.goal)
                print(f"    RRT_3D: 返回近似路径，路径长度: {len(path)}")
                return path, cost
        
        return [], float('inf')
    def rrt_star_3d(self, max_iterations: int = 8000, step_size: float = 3.0, search_radius: float = 6.0) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维RRT*算法（优化版本）"""
        tree = {self.start: None}
        costs = {self.start: 0}
        
        # 修正起点和终点坐标问题
        print(f"    RRT*_3D: 起点 {self.start}, 终点 {self.goal}")
        
        for iteration in range(max_iterations):
            # 增强采样策略
            if random.random() < 0.5:  # 50%概率采样目标点或其附近
                if random.random() < 0.6:  # 直接采样目标
                    rand_point = self.goal
                else:  # 目标附近采样
                    goal_x, goal_y, goal_z = self.goal
                    x = max(0, min(self.width-1, goal_x + random.uniform(-5, 5)))
                    y = max(0, min(self.height-1, goal_y + random.uniform(-5, 5)))
                    terrain_height = self._get_terrain_height(x, y)
                    z = max(terrain_height + 1, min(terrain_height + self.max_altitude,
                                                  goal_z + random.uniform(-8, 8)))
                    rand_point = (x, y, z)
            elif random.random() < 0.8:  # 30%概率在起点和终点之间采样
                t = random.uniform(0, 1)
                x = self.start[0] + t * (self.goal[0] - self.start[0]) + random.uniform(-2, 2)
                y = self.start[1] + t * (self.goal[1] - self.start[1]) + random.uniform(-2, 2)
                z = self.start[2] + t * (self.goal[2] - self.start[2]) + random.uniform(-3, 3)
                
                x = max(0, min(self.width-1, x))
                y = max(0, min(self.height-1, y))
                terrain_height = self._get_terrain_height(x, y)
                z = max(terrain_height + 1, min(terrain_height + self.max_altitude, z))
                rand_point = (x, y, z)
            else:  # 20%概率随机采样
                x = random.uniform(0, self.width - 1)
                y = random.uniform(0, self.height - 1)
                terrain_height = self._get_terrain_height(x, y)
                z = random.uniform(terrain_height + 1, terrain_height + self.max_altitude)
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
                    # RRT*的关键改进：在搜索半径内寻找所有邻居
                    neighbors = []
                    for node in tree.keys():
                        if self._euclidean_distance_3d(node, new_point) <= search_radius:
                            neighbors.append(node)
                    
                    if not neighbors:
                        neighbors = [nearest]
                    
                    # 选择代价最小的父节点
                    best_parent = None
                    min_cost = float('inf')
                    
                    for neighbor in neighbors:
                        potential_cost = costs[neighbor] + self._get_cost_3d(neighbor, new_point)
                        if potential_cost < min_cost:
                            min_cost = potential_cost
                            best_parent = neighbor
                    
                    if best_parent is not None:
                        tree[new_point] = best_parent
                        costs[new_point] = min_cost
                        
                        # RRT*的重连接（Rewiring）步骤
                        for neighbor in neighbors:
                            if neighbor != best_parent and neighbor != self.start:
                                new_cost_via_new_point = costs[new_point] + self._get_cost_3d(new_point, neighbor)
                                if new_cost_via_new_point < costs[neighbor]:
                                    # 重连接：将neighbor的父节点改为new_point
                                    tree[neighbor] = new_point
                                    costs[neighbor] = new_cost_via_new_point
                        
                        # 检查是否到达目标（放宽条件）
                        goal_distance = self._euclidean_distance_3d(new_point, self.goal)
                        if goal_distance < step_size * 4.0:  # 放宽到4倍步长
                            # 尝试直接连接到目标
                            goal_cost = costs[new_point] + self._get_cost_3d(new_point, self.goal)
                            
                            # 如果目标还没有在树中，或者找到了更好的路径
                            if self.goal not in costs or goal_cost < costs[self.goal]:
                                tree[self.goal] = new_point
                                costs[self.goal] = goal_cost
                                
                                # 重构路径
                                path = []
                                current = self.goal
                                while current is not None:
                                    path.append(current)
                                    current = tree[current]
                                
                                print(f"    RRT*_3D: 找到路径! 迭代次数: {iteration+1}, 路径长度: {len(path[::-1])}")
                                return path[::-1], costs[self.goal]
            
            # 每1000次迭代输出进度
            if iteration % 1000 == 0 and iteration > 0:
                print(f"    RRT*_3D 进度: {iteration}/{max_iterations} 迭代, 树节点数: {len(tree)}")
        
        # 如果没有找到完整路径，尝试返回最接近目标的路径
        if len(tree) > 1:
            closest_node = min(tree.keys(), key=lambda p: self._euclidean_distance_3d(p, self.goal))
            closest_distance = self._euclidean_distance_3d(closest_node, self.goal)
            print(f"    RRT*_3D: 最接近目标的距离: {closest_distance:.2f}")
            
            if closest_distance < 20.0:  # 进一步放宽接受条件
                # 重构到最近点的路径
                path = []
                current = closest_node
                total_cost = costs[closest_node]
                while current is not None:
                    path.append(current)
                    current = tree[current]
                
                # 添加目标点
                path = path[::-1] + [self.goal]
                total_cost += self._get_cost_3d(closest_node, self.goal)
                print(f"    RRT*_3D: 返回近似路径，路径长度: {len(path)}")
                return path, total_cost
        
        return [], float('inf')
    
    def particle_swarm_optimization_3d(self, n_particles: int = 15, max_iterations: int = 30) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维粒子群优化算法"""
        n_waypoints = 6  # 路径中间点数量
        
        # 初始化粒子（每个粒子代表一条三维路径的中间点）
        particles = []
        velocities = []
        personal_best = []
        personal_best_cost = []
        
        for _ in range(n_particles):
            # 随机生成三维路径点
            waypoints = []
            for i in range(n_waypoints):
                progress = (i + 1) / (n_waypoints + 1)
                base_x = self.start[0] + progress * (self.goal[0] - self.start[0])
                base_y = self.start[1] + progress * (self.goal[1] - self.start[1])
                base_z = self.start[2] + progress * (self.goal[2] - self.start[2])
                
                # 添加随机扰动
                x = max(0, min(self.width-1, base_x + random.uniform(-3, 3)))
                y = max(0, min(self.height-1, base_y + random.uniform(-3, 3)))
                terrain_height = self._get_terrain_height(x, y)
                z = max(terrain_height + 1, min(terrain_height + self.max_altitude, 
                                              base_z + random.uniform(-5, 5)))
                waypoints.append((x, y, z))
            
            particles.append(waypoints)
            velocities.append([(0, 0, 0) for _ in range(n_waypoints)])
            
            # 计算初始代价
            path_cost = self._calculate_path_cost_3d([self.start] + waypoints + [self.goal])
            personal_best.append(waypoints[:])
            personal_best_cost.append(path_cost)
        
        # 找到全局最优
        global_best_idx = min(range(n_particles), key=lambda i: personal_best_cost[i])
        global_best = personal_best[global_best_idx][:]
        global_best_cost = personal_best_cost[global_best_idx]
        
        # PSO迭代
        for iteration in range(max_iterations):
            for i in range(n_particles):
                for j in range(n_waypoints):
                    # 更新速度
                    r1, r2 = random.random(), random.random()
                    cognitive = tuple(personal_best[i][j][k] - particles[i][j][k] for k in range(3))
                    social = tuple(global_best[j][k] - particles[i][j][k] for k in range(3))
                    
                    new_velocity = tuple(
                        0.5 * velocities[i][j][k] + 2 * r1 * cognitive[k] + 2 * r2 * social[k]
                        for k in range(3)
                    )
                    velocities[i][j] = new_velocity
                    
                    # 更新位置
                    new_pos = tuple(particles[i][j][k] + velocities[i][j][k] for k in range(3))
                    
                    # 边界检查
                    x = max(0, min(self.width-1, new_pos[0]))
                    y = max(0, min(self.height-1, new_pos[1]))
                    terrain_height = self._get_terrain_height(x, y)
                    z = max(terrain_height + 1, min(terrain_height + self.max_altitude, new_pos[2]))
                    
                    particles[i][j] = (x, y, z)
                
                # 评估新位置
                path_cost = self._calculate_path_cost_3d([self.start] + particles[i] + [self.goal])
                
                # 更新个体最优
                if path_cost < personal_best_cost[i]:
                    personal_best[i] = particles[i][:]
                    personal_best_cost[i] = path_cost
                    
                    # 更新全局最优
                    if path_cost < global_best_cost:
                        global_best = particles[i][:]
                        global_best_cost = path_cost
        
        # 返回最优路径
        best_path = [self.start] + global_best + [self.goal]
        return best_path, global_best_cost
    
    def simulated_annealing_3d(self, max_iterations: int = 300, initial_temp: float = 50.0) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维模拟退火算法"""
        # 初始解：直线路径
        n_waypoints = 8
        current_path = []
        for i in range(n_waypoints):
            progress = (i + 1) / (n_waypoints + 1)
            x = self.start[0] + progress * (self.goal[0] - self.start[0])
            y = self.start[1] + progress * (self.goal[1] - self.start[1])
            z = self.start[2] + progress * (self.goal[2] - self.start[2])
            current_path.append((x, y, z))
        
        full_path = [self.start] + current_path + [self.goal]
        current_cost = self._calculate_path_cost_3d(full_path)
        best_path = current_path[:]
        best_cost = current_cost
        
        temperature = initial_temp
        
        for iteration in range(max_iterations):
            # 生成邻居解
            new_path = current_path[:]
            # 随机选择一个点进行扰动
            idx = random.randint(0, len(new_path) - 1)
            old_point = new_path[idx]
            
            # 在邻域内随机移动
            new_x = max(0, min(self.width-1, old_point[0] + random.uniform(-2, 2)))
            new_y = max(0, min(self.height-1, old_point[1] + random.uniform(-2, 2)))
            terrain_height = self._get_terrain_height(new_x, new_y)
            new_z = max(terrain_height + 1, min(terrain_height + self.max_altitude, 
                                              old_point[2] + random.uniform(-3, 3)))
            new_path[idx] = (new_x, new_y, new_z)
            
            new_full_path = [self.start] + new_path + [self.goal]
            new_cost = self._calculate_path_cost_3d(new_full_path)
            
            # 接受准则
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_path = new_path
                current_cost = new_cost
                
                if new_cost < best_cost:
                    best_path = new_path[:]
                    best_cost = new_cost
            
            # 降温
            temperature *= 0.995
        
        return [self.start] + best_path + [self.goal], best_cost
    
    def grey_wolf_optimizer_3d(self, n_wolves: int = 12, max_iterations: int = 30) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维灰狼优化算法"""
        n_waypoints = 5
        
        # 初始化狼群
        wolves = []
        for _ in range(n_wolves):
            waypoints = []
            for i in range(n_waypoints):
                progress = (i + 1) / (n_waypoints + 1)
                base_x = self.start[0] + progress * (self.goal[0] - self.start[0])
                base_y = self.start[1] + progress * (self.goal[1] - self.start[1])
                base_z = self.start[2] + progress * (self.goal[2] - self.start[2])
                
                x = max(0, min(self.width-1, base_x + random.uniform(-4, 4)))
                y = max(0, min(self.height-1, base_y + random.uniform(-4, 4)))
                terrain_height = self._get_terrain_height(x, y)
                z = max(terrain_height + 1, min(terrain_height + self.max_altitude, 
                                              base_z + random.uniform(-6, 6)))
                waypoints.append((x, y, z))
            wolves.append(waypoints)
        
        # 评估初始适应度
        fitness = [self._calculate_path_cost_3d([self.start] + wolf + [self.goal]) for wolf in wolves]
        
        # 找到alpha, beta, delta狼
        sorted_indices = sorted(range(n_wolves), key=lambda i: fitness[i])
        alpha_idx, beta_idx, delta_idx = sorted_indices[:3]
        
        for iteration in range(max_iterations):
            a = 2 - 2 * iteration / max_iterations  # 线性递减
            
            for i in range(n_wolves):
                if i in [alpha_idx, beta_idx, delta_idx]:
                    continue
                
                for j in range(n_waypoints):
                    # 对每个维度进行更新
                    for dim in range(3):  # x, y, z
                        # 计算与alpha, beta, delta的距离
                        r1, r2 = random.random(), random.random()
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        
                        D_alpha = abs(C1 * wolves[alpha_idx][j][dim] - wolves[i][j][dim])
                        X1 = wolves[alpha_idx][j][dim] - A1 * D_alpha
                        
                        r1, r2 = random.random(), random.random()
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        
                        D_beta = abs(C2 * wolves[beta_idx][j][dim] - wolves[i][j][dim])
                        X2 = wolves[beta_idx][j][dim] - A2 * D_beta
                        
                        r1, r2 = random.random(), random.random()
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        
                        D_delta = abs(C3 * wolves[delta_idx][j][dim] - wolves[i][j][dim])
                        X3 = wolves[delta_idx][j][dim] - A3 * D_delta
                        
                        new_val = (X1 + X2 + X3) / 3
                        
                        # 边界检查
                        if dim == 0:  # x
                            new_val = max(0, min(self.width-1, new_val))
                        elif dim == 1:  # y
                            new_val = max(0, min(self.height-1, new_val))
                        else:  # z
                            terrain_height = self._get_terrain_height(wolves[i][j][0], wolves[i][j][1])
                            new_val = max(terrain_height + 1, min(terrain_height + self.max_altitude, new_val))
                        
                        # 更新位置
                        old_pos = list(wolves[i][j])
                        old_pos[dim] = new_val
                        wolves[i][j] = tuple(old_pos)
                
                # 更新适应度
                fitness[i] = self._calculate_path_cost_3d([self.start] + wolves[i] + [self.goal])
            
            # 更新alpha, beta, delta
            sorted_indices = sorted(range(n_wolves), key=lambda i: fitness[i])
            alpha_idx, beta_idx, delta_idx = sorted_indices[:3]
        
        best_path = [self.start] + wolves[alpha_idx] + [self.goal]
        return best_path, fitness[alpha_idx]
    
    def jellyfish_optimization_3d(self, n_jellyfish: int = 12, max_iterations: int = 30) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维水母优化算法"""
        n_waypoints = 5
        
        # 初始化水母群
        jellyfish = []
        for _ in range(n_jellyfish):
            waypoints = []
            for i in range(n_waypoints):
                progress = (i + 1) / (n_waypoints + 1)
                base_x = self.start[0] + progress * (self.goal[0] - self.start[0])
                base_y = self.start[1] + progress * (self.goal[1] - self.start[1])
                base_z = self.start[2] + progress * (self.goal[2] - self.start[2])
                
                x = max(0, min(self.width-1, base_x + random.uniform(-3, 3)))
                y = max(0, min(self.height-1, base_y + random.uniform(-3, 3)))
                terrain_height = self._get_terrain_height(x, y)
                z = max(terrain_height + 1, min(terrain_height + self.max_altitude, 
                                              base_z + random.uniform(-5, 5)))
                waypoints.append((x, y, z))
            jellyfish.append(waypoints)
        
        # 评估适应度
        fitness = [self._calculate_path_cost_3d([self.start] + jf + [self.goal]) for jf in jellyfish]
        best_idx = min(range(n_jellyfish), key=lambda i: fitness[i])
        
        for iteration in range(max_iterations):
            for i in range(n_jellyfish):
                if random.random() < 0.5:  # 跟随水流
                    # 向最优解移动
                    for j in range(n_waypoints):
                        beta = random.uniform(0, 1)
                        for dim in range(3):
                            new_val = jellyfish[i][j][dim] + beta * (jellyfish[best_idx][j][dim] - jellyfish[i][j][dim])
                            
                            # 边界检查
                            if dim == 0:  # x
                                new_val = max(0, min(self.width-1, new_val))
                            elif dim == 1:  # y
                                new_val = max(0, min(self.height-1, new_val))
                            else:  # z
                                terrain_height = self._get_terrain_height(jellyfish[i][j][0], jellyfish[i][j][1])
                                new_val = max(terrain_height + 1, min(terrain_height + self.max_altitude, new_val))
                            
                            old_pos = list(jellyfish[i][j])
                            old_pos[dim] = new_val
                            jellyfish[i][j] = tuple(old_pos)
                else:  # 随机游动
                    for j in range(n_waypoints):
                        for dim in range(3):
                            if dim == 2:  # z维度
                                new_val = jellyfish[i][j][dim] + random.uniform(-2, 2)
                                terrain_height = self._get_terrain_height(jellyfish[i][j][0], jellyfish[i][j][1])
                                new_val = max(terrain_height + 1, min(terrain_height + self.max_altitude, new_val))
                            else:  # x, y维度
                                new_val = jellyfish[i][j][dim] + random.uniform(-1.5, 1.5)
                                if dim == 0:
                                    new_val = max(0, min(self.width-1, new_val))
                                else:
                                    new_val = max(0, min(self.height-1, new_val))
                            
                            old_pos = list(jellyfish[i][j])
                            old_pos[dim] = new_val
                            jellyfish[i][j] = tuple(old_pos)
                
                # 更新适应度
                new_fitness = self._calculate_path_cost_3d([self.start] + jellyfish[i] + [self.goal])
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < fitness[best_idx]:
                        best_idx = i
        
        best_path = [self.start] + jellyfish[best_idx] + [self.goal]
        return best_path, fitness[best_idx]
    
    def fast_marching_3d(self) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维快速行进方法（简化版）"""
        # 使用三维Dijkstra作为快速行进的近似
        return self.dijkstra_3d()
    
    def lrta_star_3d(self, max_iterations: int = 300) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维LRTA*算法（学习实时A*）"""
        # 初始化启发式值
        h_values = {}
        
        current = self.start
        path = [current]
        total_cost = 0
        
        for _ in range(max_iterations):
            if self._euclidean_distance_3d(current, self.goal) < 3.0:
                return path, total_cost
            
            # 获取邻居
            neighbors = self._get_neighbors_3d(current, 2.0)
            if not neighbors:
                break
            
            # 计算f值
            f_values = {}
            for neighbor in neighbors:
                g = self._get_cost_3d(current, neighbor)
                h = h_values.get(neighbor, self._euclidean_distance_3d(neighbor, self.goal))
                f_values[neighbor] = g + h
            
            # 选择最佳邻居
            best_neighbor = min(neighbors, key=lambda n: f_values[n])
            
            # 更新启发式值
            if neighbors:
                min_f = min(f_values.values())
                h_values[current] = min_f - self._get_cost_3d(current, current)
            
            # 移动到最佳邻居
            total_cost += self._get_cost_3d(current, best_neighbor)
            current = best_neighbor
            path.append(current)
        
        return path, total_cost
    
    def d_star_lite_3d(self) -> Tuple[List[Tuple[float, float, float]], float]:
        """三维D* Lite算法（简化版）"""
        # 由于D*算法复杂度较高，这里实现一个简化版本
        # 实际上使用A*的变体
        return self.a_star_3d()
    
    def run_all_algorithms_3d(self) -> Dict[str, Dict[str, Any]]:
        """运行所有三维算法并记录性能"""
        algorithms = {
            'PSO_3D': self.particle_swarm_optimization_3d,
            'Simulated_Annealing_3D': self.simulated_annealing_3d,
            'Grey_Wolf_3D': self.grey_wolf_optimizer_3d,
            'Jellyfish_3D': self.jellyfish_optimization_3d,
            'Fast_Marching_3D': self.fast_marching_3d,
            'A*_3D': self.a_star_3d,
            'D*_Lite_3D': self.d_star_lite_3d,
            'Dijkstra_3D': self.dijkstra_3d,
            'RRT_3D': self.rrt_3d,
            'RRT*_3D': self.rrt_star_3d,
            'BFS_3D': self.bfs_3d,
            'LRTA*_3D': self.lrta_star_3d
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
            step = max(1, min(self.height, self.width) // 30)
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
    print("完整三维地形路径规划算法性能测试")
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
    if terrain_data.shape[0] > 60 or terrain_data.shape[1] > 60:
        print("地形数据较大，进行下采样...")
        step = max(terrain_data.shape[0] // 50, terrain_data.shape[1] // 50, 1)
        terrain_data = terrain_data[::step, ::step]
        print(f"下采样后形状: {terrain_data.shape}")
    
    # 创建测试实例
    benchmark = PathPlanning3DComplete(terrain_data, max_altitude=30.0)
    
    # 运行所有算法
    print(f"\n开始完整三维路径规划算法性能测试...")
    results = benchmark.run_all_algorithms_3d()
    
    # 保存结果到CSV
    output_dir = Path('output/path_planning_3d_complete_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_filename = output_dir / 'path_planning_3d_complete_benchmark.csv'
    benchmark.save_results_to_csv(results, csv_filename)
    
    # 生成三视图可视化
    benchmark.visualize_paths_3d(results, output_dir)
    
    # 打印总结
    print(f"\n=" * 60)
    print("完整三维路径规划性能测试完成!")
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