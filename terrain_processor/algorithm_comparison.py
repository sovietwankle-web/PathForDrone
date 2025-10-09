"""
路径规划算法性能对比系统

对比基于Eikonal方程的FMM算法与传统路径规划算法
包括静态和动态环境下的性能测试

作者: Kilo Code
日期: 2025-08-26
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import rasterio

# 导入现有的算法
from eikonal_path_planning import EikonalPathPlanner
from dynamic_fog_path_planning_optimized import OptimizedDynamicFogEnvironment, OptimizedDynamicPathPlanning


class AlgorithmComparison:
    """算法性能对比类"""
    
    def __init__(self, terrain_data: np.ndarray):
        """
        初始化对比系统
        
        Args:
            terrain_data: 地形数据 (height, width)
        """
        self.terrain_2d = terrain_data
        self.height, self.width = terrain_data.shape
        
        # 创建三维地形
        self.nz = 15
        self.terrain_3d = np.zeros((self.nz, self.height, self.width))
        for i in range(self.nz):
            self.terrain_3d[i] = terrain_data + i * 3  # 每层高度增加3米
        
        # 创建动态雾气环境
        self.fog_env = OptimizedDynamicFogEnvironment(terrain_data, time_steps=8)
        
        # 设置测试点
        self.start_3d = (2, 2, 2)  # (z, y, x)
        self.goal_3d = (self.nz-3, self.height-5, self.width-5)
        
        print(f"算法对比系统初始化:")
        print(f"  地形尺寸: {self.height} x {self.width}")
        print(f"  三维地形: {self.nz} 层")
        print(f"  起点: {self.start_3d}")
        print(f"  终点: {self.goal_3d}")
    
    def test_static_algorithms(self) -> Dict[str, Any]:
        """测试静态环境算法"""
        print("\n=== 静态环境算法测试 ===")
        
        results = {}
        
        # 1. 测试Eikonal FMM算法
        print("\n1. 测试Eikonal FMM算法...")
        try:
            start_time = time.time()
            
            # 创建Eikonal规划器
            eikonal_planner = EikonalPathPlanner(
                self.terrain_3d, 
                spacing=(1.0, 1.0, 1.0)
            )
            
            # 规划路径
            path, stats = eikonal_planner.plan_path(self.start_3d, self.goal_3d)
            
            end_time = time.time()
            
            results['Eikonal_FMM'] = {
                'success': stats['success'],
                'path_length': stats['path_length'],
                'path_cost': stats['path_cost'],
                'execution_time': end_time - start_time,
                'solve_time': stats['solve_time'],
                'path': path
            }
            
            print(f"  成功: {stats['success']}")
            print(f"  路径长度: {stats['path_length']}")
            print(f"  路径代价: {stats['path_cost']:.2f}")
            print(f"  执行时间: {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"  Eikonal FMM算法失败: {e}")
            results['Eikonal_FMM'] = {
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'path_length': 0,
                'path_cost': float('inf')
            }
        
        # 2. 测试传统动态A*算法
        print("\n2. 测试传统动态A*算法...")
        try:
            start_time = time.time()
            
            # 创建动态路径规划器
            dynamic_planner = OptimizedDynamicPathPlanning(self.fog_env)
            
            # 规划路径
            path, cost = dynamic_planner.dynamic_a_star(start_time=0)
            
            end_time = time.time()
            
            results['Dynamic_AStar'] = {
                'success': len(path) > 0,
                'path_length': len(path),
                'path_cost': cost,
                'execution_time': end_time - start_time,
                'path': path
            }
            
            print(f"  成功: {len(path) > 0}")
            print(f"  路径长度: {len(path)}")
            print(f"  路径代价: {cost:.2f}")
            print(f"  执行时间: {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"  动态A*算法失败: {e}")
            results['Dynamic_AStar'] = {
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'path_length': 0,
                'path_cost': float('inf')
            }
        
        # 3. 测试传统动态RRT算法
        print("\n3. 测试传统动态RRT算法...")
        try:
            start_time = time.time()
            
            # 规划路径
            path, cost = dynamic_planner.dynamic_rrt(start_time=0, max_iterations=2000)
            
            end_time = time.time()
            
            results['Dynamic_RRT'] = {
                'success': len(path) > 0,
                'path_length': len(path),
                'path_cost': cost,
                'execution_time': end_time - start_time,
                'path': path
            }
            
            print(f"  成功: {len(path) > 0}")
            print(f"  路径长度: {len(path)}")
            print(f"  路径代价: {cost:.2f}")
            print(f"  执行时间: {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"  动态RRT算法失败: {e}")
            results['Dynamic_RRT'] = {
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'path_length': 0,
                'path_cost': float('inf')
            }
        
        return results
    
    def visualize_comparison(self, results: Dict[str, Any], save_dir: str):
        """可视化算法对比结果"""
        print("\n=== 生成对比可视化 ===")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 性能对比图表
        self._plot_performance_comparison(results, save_dir)
        
        # 2. 路径对比图
        self._plot_path_comparison(results, save_dir)
        
        # 3. 算法特性雷达图
        self._plot_algorithm_radar(results, save_dir)
    
    def _plot_performance_comparison(self, results: Dict[str, Any], save_dir: str):
        """绘制性能对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取数据
        algorithms = []
        execution_times = []
        path_costs = []
        path_lengths = []
        success_rates = []
        
        for alg_name, result in results.items():
            if 'error' not in result:
                algorithms.append(alg_name)
                execution_times.append(result['execution_time'])
                path_costs.append(result['path_cost'] if result['path_cost'] != float('inf') else 0)
                path_lengths.append(result['path_length'])
                success_rates.append(1 if result['success'] else 0)
        
        # 执行时间对比
        axes[0, 0].bar(algorithms, execution_times, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 路径代价对比
        axes[0, 1].bar(algorithms, path_costs, color=['blue', 'green', 'red'])
        axes[0, 1].set_title('Path Cost Comparison')
        axes[0, 1].set_ylabel('Cost')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 路径长度对比
        axes[1, 0].bar(algorithms, path_lengths, color=['blue', 'green', 'red'])
        axes[1, 0].set_title('Path Length Comparison')
        axes[1, 0].set_ylabel('Number of Steps')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 成功率对比
        axes[1, 1].bar(algorithms, success_rates, color=['blue', 'green', 'red'])
        axes[1, 1].set_title('Success Rate Comparison')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"性能对比图已保存: {save_dir}/performance_comparison.png")
    
    def _plot_path_comparison(self, results: Dict[str, Any], save_dir: str):
        """绘制路径对比图"""
        fig = plt.figure(figsize=(20, 6))
        
        successful_results = {name: result for name, result in results.items() 
                            if result.get('success', False) and 'path' in result and result['path']}
        
        if not successful_results:
            print("没有成功的路径可供可视化")
            return
        
        num_plots = len(successful_results)
        
        for idx, (alg_name, result) in enumerate(successful_results.items()):
            ax = fig.add_subplot(1, num_plots, idx + 1, projection='3d')
            
            path = result['path']
            
            # 绘制地形（下采样）
            step = max(1, min(self.height, self.width) // 15)
            X, Y = np.meshgrid(range(0, self.width, step), range(0, self.height, step))
            Z_terrain = self.terrain_2d[::step, ::step]
            ax.plot_surface(X, Y, Z_terrain, alpha=0.3, cmap='terrain')
            
            # 绘制路径
            if alg_name == 'Eikonal_FMM':
                # FMM路径格式: [(z, y, x), ...]
                if len(path) > 0:
                    path_array = np.array(path)
                    ax.plot(path_array[:, 2], path_array[:, 1], path_array[:, 0], 
                           'r-', linewidth=3, label='FMM Path')
            else:
                # 动态算法路径格式: [(x, y, z, t), ...]
                if len(path) > 0:
                    path_array = np.array(path)
                    ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                           'b-', linewidth=3, label='Dynamic Path')
            
            # 标记起点和终点
            ax.scatter([self.start_3d[2]], [self.start_3d[1]], [self.start_3d[0]], 
                      c='green', s=100, label='Start')
            ax.scatter([self.goal_3d[2]], [self.goal_3d[1]], [self.goal_3d[0]], 
                      c='red', s=100, label='Goal')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{alg_name}\nTime: {result["execution_time"]:.2f}s')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/path_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"路径对比图已保存: {save_dir}/path_comparison.png")
    
    def _plot_algorithm_radar(self, results: Dict[str, Any], save_dir: str):
        """绘制算法特性雷达图"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 定义评估维度
        categories = ['Speed', 'Optimality', 'Success Rate', 'Memory Efficiency', 'Robustness']
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形
        
        # 为每个算法计算评分
        algorithm_scores = {}
        
        for alg_name, result in results.items():
            if 'error' in result:
                continue
                
            # 标准化评分 (0-1)
            speed_score = 1.0 / (1.0 + result['execution_time']) if result['execution_time'] > 0 else 0
            optimality_score = 1.0 / (1.0 + result['path_cost'] / 100) if result['path_cost'] != float('inf') else 0
            success_score = 1.0 if result['success'] else 0.0
            
            # 基于算法类型的经验评分
            if 'Eikonal' in alg_name:
                memory_score = 0.6  # FMM需要存储整个势场
                robustness_score = 0.8  # 理论上更稳定
            elif 'AStar' in alg_name:
                memory_score = 0.8  # A*内存效率较高
                robustness_score = 0.7  # 启发式搜索
            else:  # RRT
                memory_score = 0.9  # RRT内存效率最高
                robustness_score = 0.6  # 随机性较大
            
            algorithm_scores[alg_name] = [
                speed_score, optimality_score, success_score, 
                memory_score, robustness_score
            ]
        
        # 绘制雷达图
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for idx, (alg_name, scores) in enumerate(algorithm_scores.items()):
            scores += scores[:1]  # 闭合图形
            ax.plot(angles, scores, 'o-', linewidth=2, 
                   label=alg_name, color=colors[idx % len(colors)])
            ax.fill(angles, scores, alpha=0.25, color=colors[idx % len(colors)])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Algorithm Performance Radar Chart', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/algorithm_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"算法雷达图已保存: {save_dir}/algorithm_radar.png")
    
    def save_results_csv(self, results: Dict[str, Any], save_path: str):
        """保存结果到CSV文件"""
        with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Algorithm', 'Success', 'Execution_Time_s', 'Path_Cost', 
                         'Path_Length', 'Notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for alg_name, result in results.items():
                writer.writerow({
                    'Algorithm': alg_name,
                    'Success': result.get('success', False),
                    'Execution_Time_s': result.get('execution_time', 0),
                    'Path_Cost': result.get('path_cost', float('inf')),
                    'Path_Length': result.get('path_length', 0),
                    'Notes': result.get('error', 'Normal execution')
                })
        
        print(f"结果已保存到CSV: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("路径规划算法性能对比系统")
    print("=" * 60)
    
    # 读取TIF文件
    tif_folder = Path("tif")
    if tif_folder.exists():
        tif_files = list(tif_folder.glob("*.tif"))
        if tif_files:
            print(f"使用TIF文件: {tif_files[0]}")
            with rasterio.open(tif_files[0]) as src:
                terrain_full = src.read(1)
                print(f"原始地形尺寸: {terrain_full.shape}")
                
                # 下采样以减少计算量
                downsample_factor = 12
                terrain_2d = terrain_full[::downsample_factor, ::downsample_factor]
                print(f"下采样后地形尺寸: {terrain_2d.shape}")
        else:
            print("未找到TIF文件，使用合成地形")
            terrain_2d = np.random.rand(40, 50) * 100
    else:
        print("TIF文件夹不存在，使用合成地形")
        terrain_2d = np.random.rand(40, 50) * 100
    
    # 创建算法对比系统
    comparison = AlgorithmComparison(terrain_2d)
    
    # 运行算法测试
    results = comparison.test_static_algorithms()
    
    # 创建输出目录
    output_dir = "output/algorithm_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成可视化
    comparison.visualize_comparison(results, output_dir)
    
    # 保存结果
    comparison.save_results_csv(results, f"{output_dir}/comparison_results.csv")
    
    # 打印总结
    print(f"\n=" * 60)
    print("算法对比测试完成!")
    print(f"结果保存在: {output_dir}")
    
    successful_algorithms = [name for name, result in results.items() 
                           if result.get('success', False)]
    
    if successful_algorithms:
        print(f"\n成功的算法: {', '.join(successful_algorithms)}")
        
        # 找出最快的算法
        fastest = min(results.items(), 
                     key=lambda x: x[1].get('execution_time', float('inf')) 
                     if x[1].get('success', False) else float('inf'))
        
        if fastest[1].get('success', False):
            print(f"最快算法: {fastest[0]} ({fastest[1]['execution_time']:.2f}s)")
        
        # 找出最优路径
        best_cost = min(results.items(),
                       key=lambda x: x[1].get('path_cost', float('inf'))
                       if x[1].get('success', False) else float('inf'))
        
        if best_cost[1].get('success', False) and best_cost[1]['path_cost'] != float('inf'):
            print(f"最优路径: {best_cost[0]} (代价: {best_cost[1]['path_cost']:.2f})")
    else:
        print("\n所有算法都未成功找到路径")
    
    print(f"=" * 60)


if __name__ == "__main__":
    main()