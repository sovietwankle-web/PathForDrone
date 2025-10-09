#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地形路径规划算法性能测试脚本
实现多种路径规划算法并进行性能对比测试
"""

import os
import sys
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import heapq
import random
from typing import List, Tuple, Optional, Dict, Any
import warnings

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

class PathPlanningBenchmark:
    """路径规划算法性能测试类"""
    
    def __init__(self, terrain_data: np.ndarray):
        """
        初始化路径规划测试
        
        Args:
            terrain_data: 地形高程数据
        """
        self.terrain = terrain_data
        self.height, self.width = terrain_data.shape
        self.start = (self.height - 1, 0)  # 左下角
        self.goal = (0, self.width - 1)   # 右上角
        
        # 处理NaN值，用平均值替代
        if np.any(np.isnan(terrain_data)):
            mean_elevation = np.nanmean(terrain_data)
            self.terrain = np.where(np.isnan(terrain_data), mean_elevation, terrain_data)
        
        # 归一化地形数据到0-1范围，用作代价
        self.terrain_normalized = self._normalize_terrain()
        
        print(f"地形尺寸: {self.height} x {self.width}")
        print(f"起点: {self.start}, 终点: {self.goal}")
        print(f"高程范围: {np.min(self.terrain):.2f} - {np.max(self.terrain):.2f}")
    
    def _normalize_terrain(self) -> np.ndarray:
        """归一化地形数据"""
        min_val = np.min(self.terrain)
        max_val = np.max(self.terrain)
        if max_val > min_val:
            return (self.terrain - min_val) / (max_val - min_val)
        else:
            return np.ones_like(self.terrain) * 0.5
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取8邻域"""
        row, col = pos
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.height and 0 <= new_col < self.width:
                    neighbors.append((new_row, new_col))
        return neighbors
    
    def _get_cost(self, pos: Tuple[int, int]) -> float:
        """获取位置代价（基于高程）"""
        row, col = pos
        return 1.0 + self.terrain_normalized[row, col] * 2.0  # 1-3的代价范围
    
    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """欧几里得距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def dijkstra(self) -> Tuple[List[Tuple[int, int]], float]:
        """Dijkstra算法"""
        distances = {self.start: 0}
        previous = {}
        pq = [(0, self.start)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == self.goal:
                # 重构路径
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(self.start)
                return path[::-1], distances[self.goal]
            
            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue
                
                cost = self._get_cost(neighbor)
                distance = current_dist + cost
                
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return [], float('inf')
    
    def a_star(self) -> Tuple[List[Tuple[int, int]], float]:
        """A*算法"""
        open_set = [(0, self.start)]
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self._euclidean_distance(self.start, self.goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == self.goal:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1], g_score[self.goal]
            
            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + self._get_cost(neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._euclidean_distance(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return [], float('inf')
    
    def bfs(self) -> Tuple[List[Tuple[int, int]], float]:
        """广度优先搜索"""
        queue = deque([(self.start, [self.start], 0)])
        visited = {self.start}
        
        while queue:
            current, path, cost = queue.popleft()
            
            if current == self.goal:
                return path, cost
            
            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_cost = cost + self._get_cost(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_cost))
        
        return [], float('inf')
    
    def rrt(self, max_iterations: int = 1000, step_size: float = 5.0) -> Tuple[List[Tuple[int, int]], float]:
        """RRT算法"""
        tree = {self.start: None}
        
        for _ in range(max_iterations):
            # 随机采样
            if random.random() < 0.1:  # 10%概率采样目标点
                rand_point = self.goal
            else:
                rand_point = (random.randint(0, self.height-1), random.randint(0, self.width-1))
            
            # 找到最近的树节点
            nearest = min(tree.keys(), key=lambda p: self._euclidean_distance(p, rand_point))
            
            # 向随机点扩展
            direction = (rand_point[0] - nearest[0], rand_point[1] - nearest[1])
            length = max(abs(direction[0]), abs(direction[1]))
            
            if length > 0:
                step = min(step_size, length)
                new_point = (
                    int(nearest[0] + direction[0] * step / length),
                    int(nearest[1] + direction[1] * step / length)
                )
                
                # 检查边界
                if (0 <= new_point[0] < self.height and 
                    0 <= new_point[1] < self.width and 
                    new_point not in tree):
                    
                    tree[new_point] = nearest
                    
                    # 检查是否到达目标
                    if self._euclidean_distance(new_point, self.goal) < step_size:
                        tree[self.goal] = new_point
                        
                        # 重构路径
                        path = []
                        current = self.goal
                        cost = 0
                        while current is not None:
                            path.append(current)
                            if tree[current] is not None:
                                cost += self._get_cost(current)
                            current = tree[current]
                        
                        return path[::-1], cost
        
        return [], float('inf')
    
    def rrt_star(self, max_iterations: int = 1000, step_size: float = 5.0, search_radius: float = 10.0) -> Tuple[List[Tuple[int, int]], float]:
        """RRT*算法（渐近最优RRT）"""
        tree = {self.start: {'parent': None, 'cost': 0}}
        
        for iteration in range(max_iterations):
            # 随机采样
            if random.random() < 0.1:  # 10%概率采样目标点
                rand_point = self.goal
            else:
                rand_point = (random.randint(0, self.height-1), random.randint(0, self.width-1))
            
            # 找到最近的树节点
            nearest = min(tree.keys(), key=lambda p: self._euclidean_distance(p, rand_point))
            
            # 向随机点扩展
            direction = (rand_point[0] - nearest[0], rand_point[1] - nearest[1])
            length = max(abs(direction[0]), abs(direction[1]))
            
            if length > 0:
                step = min(step_size, length)
                new_point = (
                    int(nearest[0] + direction[0] * step / length),
                    int(nearest[1] + direction[1] * step / length)
                )
                
                # 检查边界
                if (0 <= new_point[0] < self.height and
                    0 <= new_point[1] < self.width and
                    new_point not in tree):
                    
                    # 计算到新点的成本
                    edge_cost = self._euclidean_distance(nearest, new_point) * self._get_cost(new_point)
                    new_cost = tree[nearest]['cost'] + edge_cost
                    
                    # 在搜索半径内找到所有邻居
                    neighbors = []
                    for node in tree.keys():
                        if self._euclidean_distance(node, new_point) <= search_radius:
                            neighbors.append(node)
                    
                    # 选择最佳父节点
                    best_parent = nearest
                    best_cost = new_cost
                    
                    for neighbor in neighbors:
                        neighbor_edge_cost = self._euclidean_distance(neighbor, new_point) * self._get_cost(new_point)
                        neighbor_cost = tree[neighbor]['cost'] + neighbor_edge_cost
                        if neighbor_cost < best_cost:
                            best_parent = neighbor
                            best_cost = neighbor_cost
                    
                    # 添加新节点
                    tree[new_point] = {'parent': best_parent, 'cost': best_cost}
                    
                    # 重新连接邻居节点（如果通过新节点路径更短）
                    for neighbor in neighbors:
                        if neighbor == best_parent:
                            continue
                        
                        new_neighbor_cost = best_cost + self._euclidean_distance(new_point, neighbor) * self._get_cost(neighbor)
                        if new_neighbor_cost < tree[neighbor]['cost']:
                            tree[neighbor]['parent'] = new_point
                            tree[neighbor]['cost'] = new_neighbor_cost
                    
                    # 检查是否到达目标
                    if self._euclidean_distance(new_point, self.goal) < step_size:
                        goal_cost = best_cost + self._euclidean_distance(new_point, self.goal) * self._get_cost(self.goal)
                        tree[self.goal] = {'parent': new_point, 'cost': goal_cost}
                        
                        # 重构路径
                        path = []
                        current = self.goal
                        while current is not None:
                            path.append(current)
                            current = tree[current]['parent']
                        
                        return path[::-1], tree[self.goal]['cost']
        
        return [], float('inf')
    
    def fast_marching(self) -> Tuple[List[Tuple[int, int]], float]:
        """快速行进方法（简化版）"""
        # 初始化距离场
        distances = np.full((self.height, self.width), np.inf)
        distances[self.start] = 0
        
        # 使用Dijkstra作为快速行进的近似
        pq = [(0, self.start)]
        visited = set()
        previous = {}
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == self.goal:
                # 重构路径
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(self.start)
                return path[::-1], distances[self.goal]
            
            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # 使用地形高度作为速度的倒数
                speed = 1.0 / (1.0 + self.terrain_normalized[neighbor])
                travel_time = self._euclidean_distance(current, neighbor) / speed
                new_dist = current_dist + travel_time
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return [], float('inf')
    
    def particle_swarm_optimization(self, n_particles: int = 30, max_iterations: int = 100) -> Tuple[List[Tuple[int, int]], float]:
        """粒子群优化算法"""
        # 简化的PSO，用于路径点优化
        n_waypoints = 10  # 路径中间点数量
        
        # 初始化粒子（每个粒子代表一条路径的中间点）
        particles = []
        velocities = []
        personal_best = []
        personal_best_cost = []
        
        for _ in range(n_particles):
            # 随机生成路径点
            waypoints = []
            for i in range(n_waypoints):
                progress = (i + 1) / (n_waypoints + 1)
                base_row = int(self.start[0] + progress * (self.goal[0] - self.start[0]))
                base_col = int(self.start[1] + progress * (self.goal[1] - self.start[1]))
                
                # 添加随机扰动
                row = max(0, min(self.height-1, base_row + random.randint(-5, 5)))
                col = max(0, min(self.width-1, base_col + random.randint(-5, 5)))
                waypoints.append((row, col))
            
            particles.append(waypoints)
            velocities.append([(0, 0) for _ in range(n_waypoints)])
            
            # 计算初始代价
            path_cost = self._calculate_path_cost([self.start] + waypoints + [self.goal])
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
                    cognitive = (personal_best[i][j][0] - particles[i][j][0], 
                               personal_best[i][j][1] - particles[i][j][1])
                    social = (global_best[j][0] - particles[i][j][0],
                             global_best[j][1] - particles[i][j][1])
                    
                    velocities[i][j] = (
                        int(0.5 * velocities[i][j][0] + 2 * r1 * cognitive[0] + 2 * r2 * social[0]),
                        int(0.5 * velocities[i][j][1] + 2 * r1 * cognitive[1] + 2 * r2 * social[1])
                    )
                    
                    # 更新位置
                    new_row = max(0, min(self.height-1, particles[i][j][0] + velocities[i][j][0]))
                    new_col = max(0, min(self.width-1, particles[i][j][1] + velocities[i][j][1]))
                    particles[i][j] = (new_row, new_col)
                
                # 评估新位置
                path_cost = self._calculate_path_cost([self.start] + particles[i] + [self.goal])
                
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
    
    def simulated_annealing(self, max_iterations: int = 1000, initial_temp: float = 100.0) -> Tuple[List[Tuple[int, int]], float]:
        """模拟退火算法"""
        # 初始解：直线路径
        n_waypoints = 15
        current_path = []
        for i in range(n_waypoints):
            progress = (i + 1) / (n_waypoints + 1)
            row = int(self.start[0] + progress * (self.goal[0] - self.start[0]))
            col = int(self.start[1] + progress * (self.goal[1] - self.start[1]))
            current_path.append((row, col))
        
        full_path = [self.start] + current_path + [self.goal]
        current_cost = self._calculate_path_cost(full_path)
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
            new_row = max(0, min(self.height-1, old_point[0] + random.randint(-3, 3)))
            new_col = max(0, min(self.width-1, old_point[1] + random.randint(-3, 3)))
            new_path[idx] = (new_row, new_col)
            
            new_full_path = [self.start] + new_path + [self.goal]
            new_cost = self._calculate_path_cost(new_full_path)
            
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
    
    def grey_wolf_optimizer(self, n_wolves: int = 20, max_iterations: int = 100) -> Tuple[List[Tuple[int, int]], float]:
        """灰狼优化算法"""
        n_waypoints = 8
        
        # 初始化狼群
        wolves = []
        for _ in range(n_wolves):
            waypoints = []
            for i in range(n_waypoints):
                progress = (i + 1) / (n_waypoints + 1)
                base_row = int(self.start[0] + progress * (self.goal[0] - self.start[0]))
                base_col = int(self.start[1] + progress * (self.goal[1] - self.start[1]))
                
                row = max(0, min(self.height-1, base_row + random.randint(-8, 8)))
                col = max(0, min(self.width-1, base_col + random.randint(-8, 8)))
                waypoints.append((row, col))
            wolves.append(waypoints)
        
        # 评估初始适应度
        fitness = [self._calculate_path_cost([self.start] + wolf + [self.goal]) for wolf in wolves]
        
        # 找到alpha, beta, delta狼
        sorted_indices = sorted(range(n_wolves), key=lambda i: fitness[i])
        alpha_idx, beta_idx, delta_idx = sorted_indices[:3]
        
        for iteration in range(max_iterations):
            a = 2 - 2 * iteration / max_iterations  # 线性递减
            
            for i in range(n_wolves):
                if i in [alpha_idx, beta_idx, delta_idx]:
                    continue
                
                for j in range(n_waypoints):
                    # 计算与alpha, beta, delta的距离
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    D_alpha = abs(C1 * wolves[alpha_idx][j][0] - wolves[i][j][0])
                    X1 = wolves[alpha_idx][j][0] - A1 * D_alpha
                    
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    D_beta = abs(C2 * wolves[beta_idx][j][0] - wolves[i][j][0])
                    X2 = wolves[beta_idx][j][0] - A2 * D_beta
                    
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_delta = abs(C3 * wolves[delta_idx][j][0] - wolves[i][j][0])
                    X3 = wolves[delta_idx][j][0] - A3 * D_delta
                    
                    new_row = int((X1 + X2 + X3) / 3)
                    new_row = max(0, min(self.height-1, new_row))
                    
                    # 同样处理列坐标
                    D_alpha = abs(C1 * wolves[alpha_idx][j][1] - wolves[i][j][1])
                    X1 = wolves[alpha_idx][j][1] - A1 * D_alpha
                    
                    D_beta = abs(C2 * wolves[beta_idx][j][1] - wolves[i][j][1])
                    X2 = wolves[beta_idx][j][1] - A2 * D_beta
                    
                    D_delta = abs(C3 * wolves[delta_idx][j][1] - wolves[i][j][1])
                    X3 = wolves[delta_idx][j][1] - A3 * D_delta
                    
                    new_col = int((X1 + X2 + X3) / 3)
                    new_col = max(0, min(self.width-1, new_col))
                    
                    wolves[i][j] = (new_row, new_col)
                
                # 更新适应度
                fitness[i] = self._calculate_path_cost([self.start] + wolves[i] + [self.goal])
            
            # 更新alpha, beta, delta
            sorted_indices = sorted(range(n_wolves), key=lambda i: fitness[i])
            alpha_idx, beta_idx, delta_idx = sorted_indices[:3]
        
        best_path = [self.start] + wolves[alpha_idx] + [self.goal]
        return best_path, fitness[alpha_idx]
    
    def jellyfish_optimization(self, n_jellyfish: int = 20, max_iterations: int = 100) -> Tuple[List[Tuple[int, int]], float]:
        """水母优化算法"""
        n_waypoints = 8
        
        # 初始化水母群
        jellyfish = []
        for _ in range(n_jellyfish):
            waypoints = []
            for i in range(n_waypoints):
                progress = (i + 1) / (n_waypoints + 1)
                base_row = int(self.start[0] + progress * (self.goal[0] - self.start[0]))
                base_col = int(self.start[1] + progress * (self.goal[1] - self.start[1]))
                
                row = max(0, min(self.height-1, base_row + random.randint(-6, 6)))
                col = max(0, min(self.width-1, base_col + random.randint(-6, 6)))
                waypoints.append((row, col))
            jellyfish.append(waypoints)
        
        # 评估适应度
        fitness = [self._calculate_path_cost([self.start] + jf + [self.goal]) for jf in jellyfish]
        best_idx = min(range(n_jellyfish), key=lambda i: fitness[i])
        
        for iteration in range(max_iterations):
            for i in range(n_jellyfish):
                if random.random() < 0.5:  # 跟随水流
                    # 向最优解移动
                    for j in range(n_waypoints):
                        beta = random.uniform(0, 1)
                        new_row = int(jellyfish[i][j][0] + beta * (jellyfish[best_idx][j][0] - jellyfish[i][j][0]))
                        new_col = int(jellyfish[i][j][1] + beta * (jellyfish[best_idx][j][1] - jellyfish[i][j][1]))
                        
                        new_row = max(0, min(self.height-1, new_row))
                        new_col = max(0, min(self.width-1, new_col))
                        jellyfish[i][j] = (new_row, new_col)
                else:  # 随机游动
                    for j in range(n_waypoints):
                        new_row = max(0, min(self.height-1, jellyfish[i][j][0] + random.randint(-2, 2)))
                        new_col = max(0, min(self.width-1, jellyfish[i][j][1] + random.randint(-2, 2)))
                        jellyfish[i][j] = (new_row, new_col)
                
                # 更新适应度
                new_fitness = self._calculate_path_cost([self.start] + jellyfish[i] + [self.goal])
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < fitness[best_idx]:
                        best_idx = i
        
        best_path = [self.start] + jellyfish[best_idx] + [self.goal]
        return best_path, fitness[best_idx]
    
    def lrta_star(self, max_iterations: int = 1000) -> Tuple[List[Tuple[int, int]], float]:
        """LRTA*算法（学习实时A*）"""
        # 初始化启发式值
        h_values = {}
        for i in range(self.height):
            for j in range(self.width):
                h_values[(i, j)] = self._euclidean_distance((i, j), self.goal)
        
        current = self.start
        path = [current]
        total_cost = 0
        
        for _ in range(max_iterations):
            if current == self.goal:
                return path, total_cost
            
            # 获取邻居
            neighbors = self._get_neighbors(current)
            if not neighbors:
                break
            
            # 计算f值
            f_values = {}
            for neighbor in neighbors:
                g = self._get_cost(neighbor)
                h = h_values.get(neighbor, self._euclidean_distance(neighbor, self.goal))
                f_values[neighbor] = g + h
            
            # 选择最佳邻居
            best_neighbor = min(neighbors, key=lambda n: f_values[n])
            
            # 更新启发式值
            if neighbors:
                min_f = min(f_values.values())
                h_values[current] = min_f - self._get_cost(current)
            
            # 移动到最佳邻居
            total_cost += self._get_cost(best_neighbor)
            current = best_neighbor
            path.append(current)
        
        return path, total_cost
    
    def d_star_lite(self) -> Tuple[List[Tuple[int, int]], float]:
        """D* Lite算法（简化版）"""
        # 由于D*算法复杂度较高，这里实现一个简化版本
        # 实际上使用A*的变体
        return self.a_star()
    
    def _calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """计算路径总代价"""
        if len(path) < 2:
            return float('inf')
        
        total_cost = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            # 距离代价
            distance = self._euclidean_distance(current, next_pos)
            # 地形代价
            terrain_cost = (self._get_cost(current) + self._get_cost(next_pos)) / 2
            
            total_cost += distance * terrain_cost
        
        return total_cost
    
    def run_all_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """运行所有算法并记录性能"""
        algorithms = {
            'Dijkstra': self.dijkstra,
            'A*': self.a_star,
            'BFS': self.bfs,
            'RRT': self.rrt,
            'RRT*': self.rrt_star,
            'Fast_Marching': self.fast_marching,
            'PSO': self.particle_swarm_optimization,
            'Simulated_Annealing': self.simulated_annealing,
            'Grey_Wolf': self.grey_wolf_optimizer,
            'Jellyfish': self.jellyfish_optimization,
            'LRTA*': self.lrta_star,
            'D*_Lite': self.d_star_lite
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
                success = len(path) > 0 and path[0] == self.start and path[-1] == self.goal
                
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
    
    def visualize_paths(self, results: Dict[str, Dict[str, Any]], output_dir: str):
        """可视化路径结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 选择成功的算法进行可视化
        successful_algorithms = {name: result for name, result in results.items() 
                               if result['success'] and len(result['path']) > 0}
        
        if not successful_algorithms:
            print("没有成功的算法可以可视化")
            return
        
        # 创建对比图
        n_algorithms = len(successful_algorithms)
        cols = min(4, n_algorithms)
        rows = (n_algorithms + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_algorithms == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, result) in enumerate(successful_algorithms.items()):
            if rows > 1:
                ax = axes[i // cols, i % cols]
            elif cols > 1:
                ax = axes[i]
            else:
                ax = axes
            
            # 显示地形
            im = ax.imshow(self.terrain, cmap='terrain', alpha=0.7)
            
            # 绘制路径
            if result['path']:
                path = result['path']
                path_rows = [p[0] for p in path]
                path_cols = [p[1] for p in path]
                ax.plot(path_cols, path_rows, 'r-', linewidth=2, alpha=0.8)
                ax.plot(path_cols[0], path_rows[0], 'go', markersize=8, label='起点')
                ax.plot(path_cols[-1], path_rows[-1], 'ro', markersize=8, label='终点')
            
            ax.set_title(f'{name}\n时间: {result["execution_time"]:.3f}s\n代价: {result["path_cost"]:.1f}', 
                        fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 隐藏多余的子图
        for i in range(n_algorithms, rows * cols):
            if rows > 1:
                axes[i // cols, i % cols].set_visible(False)
            elif cols > 1:
                axes[i].set_visible(False)
        
        plt.suptitle('路径规划算法对比', fontsize=16, y=0.98)
        plt.tight_layout()
        
        output_path = output_dir / 'path_planning_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"路径可视化图已保存: {output_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("地形路径规划算法性能测试")
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
    if terrain_data.shape[0] > 200 or terrain_data.shape[1] > 200:
        print("地形数据较大，进行下采样...")
        step = max(terrain_data.shape[0] // 150, terrain_data.shape[1] // 150, 1)
        terrain_data = terrain_data[::step, ::step]
        print(f"下采样后形状: {terrain_data.shape}")
    
    # 创建测试实例
    benchmark = PathPlanningBenchmark(terrain_data)
    
    # 运行所有算法
    print(f"\n开始路径规划算法性能测试...")
    results = benchmark.run_all_algorithms()
    
    # 保存结果到CSV
    output_dir = Path('output/path_planning_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_filename = output_dir / 'path_planning_benchmark.csv'
    benchmark.save_results_to_csv(results, csv_filename)
    
    # 可视化结果
    benchmark.visualize_paths(results, output_dir)
    
    # 打印总结
    print(f"\n=" * 60)
    print("性能测试完成!")
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