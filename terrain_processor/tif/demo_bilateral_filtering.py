#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双边滤波地形平滑演示脚本
展示双边滤波在地形数据平滑中的优势和应用
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from terrain_processor import TerrainProcessor

def create_bilateral_comparison(original_data, bilateral_data, output_path):
    """
    创建双边滤波前后对比图
    
    Args:
        original_data: 原始数据
        bilateral_data: 双边滤波后数据
        output_path: 输出路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 计算统计信息
    orig_valid = original_data[~np.isnan(original_data)]
    bilateral_valid = bilateral_data[~np.isnan(bilateral_data)]
    
    if len(orig_valid) == 0 or len(bilateral_valid) == 0:
        print("警告: 数据中没有有效值")
        return
    
    vmin = min(np.min(orig_valid), np.min(bilateral_valid))
    vmax = max(np.max(orig_valid), np.max(bilateral_valid))
    
    # 1. 原始地形数据
    im1 = axes[0, 0].imshow(original_data, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 0].contour(original_data, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    axes[0, 0].set_title('原始地形数据', fontsize=14, pad=10)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # 2. 双边滤波后数据
    im2 = axes[0, 1].imshow(bilateral_data, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 1].contour(bilateral_data, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    axes[0, 1].set_title('双边滤波平滑结果', fontsize=14, pad=10)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # 3. 差异图
    diff_data = bilateral_data - original_data
    diff_valid = diff_data[~np.isnan(diff_data)]
    if len(diff_valid) > 0:
        diff_max = max(abs(np.min(diff_valid)), abs(np.max(diff_valid)))
        im3 = axes[0, 2].imshow(diff_data, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        axes[0, 2].set_title('平滑差异 (平滑后 - 原始)', fontsize=14, pad=10)
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # 4. 高程分布对比
    axes[1, 0].hist(orig_valid, bins=50, alpha=0.7, label='原始数据', density=True, color='blue')
    axes[1, 0].hist(bilateral_valid, bins=50, alpha=0.7, label='双边滤波', density=True, color='red')
    axes[1, 0].set_xlabel('高程 (m)')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('高程分布对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 散点图对比
    sample_size = min(5000, len(orig_valid))
    if len(orig_valid) > sample_size:
        indices = np.random.choice(len(orig_valid), sample_size, replace=False)
        orig_sample = orig_valid[indices]
        bilateral_sample = bilateral_valid[indices]
    else:
        orig_sample = orig_valid
        bilateral_sample = bilateral_valid
    
    axes[1, 1].scatter(orig_sample, bilateral_sample, alpha=0.5, s=1)
    axes[1, 1].plot([vmin, vmax], [vmin, vmax], 'r--', label='y=x')
    axes[1, 1].set_xlabel('原始高程 (m)')
    axes[1, 1].set_ylabel('平滑后高程 (m)')
    axes[1, 1].set_title('高程值对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 统计信息表格
    axes[1, 2].axis('off')
    
    # 计算统计信息
    orig_stats = {
        '最小值': f"{np.min(orig_valid):.2f}",
        '最大值': f"{np.max(orig_valid):.2f}",
        '平均值': f"{np.mean(orig_valid):.2f}",
        '标准差': f"{np.std(orig_valid):.2f}",
        '中位数': f"{np.median(orig_valid):.2f}"
    }
    
    bilateral_stats = {
        '最小值': f"{np.min(bilateral_valid):.2f}",
        '最大值': f"{np.max(bilateral_valid):.2f}",
        '平均值': f"{np.mean(bilateral_valid):.2f}",
        '标准差': f"{np.std(bilateral_valid):.2f}",
        '中位数': f"{np.median(bilateral_valid):.2f}"
    }
    
    # 创建表格
    table_data = []
    for key in orig_stats.keys():
        table_data.append([key, orig_stats[key], bilateral_stats[key]])
    
    table = axes[1, 2].table(cellText=table_data,
                            colLabels=['统计量', '原始数据', '双边滤波'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 2].set_title('统计信息对比', fontsize=14, pad=20)
    
    plt.suptitle('双边滤波地形平滑效果分析', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"双边滤波对比分析图已保存: {output_path}")

def demo_bilateral_parameters():
    """演示不同双边滤波参数的效果"""
    print("=" * 60)
    print("双边滤波参数效果演示")
    print("=" * 60)
    
    # 检查DEM文件
    dem_files = ['DEM_5m.tif', 'DEM_1m.tif']
    available_files = [f for f in dem_files if os.path.exists(f)]
    
    if not available_files:
        print("错误: 未找到DEM文件 (DEM_5m.tif 或 DEM_1m.tif)")
        return
    
    dem_file = available_files[0]
    print(f"使用DEM文件: {dem_file}")
    
    try:
        # 初始化处理器
        processor = TerrainProcessor(dem_file)
        original_data = processor.load_terrain_data()
        
        # 创建输出目录
        output_dir = Path('output/bilateral_demo')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n地形信息:")
        info = processor.get_terrain_info()
        print(f"  数据尺寸: {info['data_shape']}")
        print(f"  数据类型: {info['data_type']}")
        
        # 测试不同的双边滤波参数
        print(f"\n开始双边滤波参数测试...")
        
        # 参数组合
        param_sets = [
            {'sigma_color': 0.05, 'sigma_spatial': 0.5, 'name': '轻度平滑'},
            {'sigma_color': 0.1, 'sigma_spatial': 1.0, 'name': '标准平滑'},
            {'sigma_color': 0.2, 'sigma_spatial': 2.0, 'name': '强度平滑'},
        ]
        
        results = {'original': original_data}
        
        for params in param_sets:
            print(f"  测试 {params['name']} (sigma_color={params['sigma_color']}, sigma_spatial={params['sigma_spatial']})...")
            
            try:
                smoothed = processor.smooth_terrain_data(
                    'bilateral', 
                    sigma_color=params['sigma_color'],
                    sigma_spatial=params['sigma_spatial']
                )
                results[params['name']] = smoothed
                
                # 保存数组
                array_path = output_dir / f"bilateral_{params['name'].replace(' ', '_')}.npy"
                np.save(array_path, smoothed)
                print(f"    数组已保存: {array_path}")
                
                # 生成等高线图
                contour_path = output_dir / f"bilateral_{params['name'].replace(' ', '_')}_contour.png"
                processor.generate_smoothed_contour(
                    contour_path,
                    smooth_method='bilateral',
                    levels=25,
                    dpi=300,
                    sigma_color=params['sigma_color'],
                    sigma_spatial=params['sigma_spatial']
                )
                print(f"    等高线图已保存: {contour_path}")
                
            except Exception as e:
                print(f"    {params['name']} 处理失败: {e}")
        
        # 创建参数对比图
        if len(results) > 1:
            print(f"\n生成参数对比图...")
            comparison_path = output_dir / f"{Path(dem_file).stem}_bilateral_params_comparison.png"
            create_parameter_comparison(results, comparison_path)
        
        # 创建详细的双边滤波分析
        if '标准平滑' in results:
            print(f"\n生成详细分析图...")
            analysis_path = output_dir / f"{Path(dem_file).stem}_bilateral_analysis.png"
            create_bilateral_comparison(original_data, results['标准平滑'], analysis_path)
        
        print(f"\n=" * 60)
        print(f"双边滤波演示完成!")
        print(f"输出目录: {output_dir}")
        print(f"生成文件:")
        for file in sorted(output_dir.glob('*')):
            print(f"  - {file.name}")
        print(f"=" * 60)
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def create_parameter_comparison(results, output_path):
    """创建参数对比图"""
    methods = list(results.keys())
    n_methods = len(methods)
    
    # 计算子图布局
    cols = min(4, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_methods == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # 计算全局统计信息用于统一颜色范围
    all_data = []
    for method, data in results.items():
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            all_data.extend(valid_data)
    
    if all_data:
        vmin, vmax = np.min(all_data), np.max(all_data)
    else:
        vmin, vmax = 0, 1
    
    for i, (method, data) in enumerate(results.items()):
        if rows > 1:
            ax = axes[i // cols, i % cols]
        elif cols > 1:
            ax = axes[i]
        else:
            ax = axes
        
        # 创建等高线图
        im = ax.imshow(data, cmap='terrain', vmin=vmin, vmax=vmax, aspect='equal')
        ax.contour(data, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        ax.set_title(method, fontsize=12, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 隐藏多余的子图
    for i in range(n_methods, rows * cols):
        if rows > 1:
            axes[i // cols, i % cols].set_visible(False)
        elif cols > 1:
            axes[i].set_visible(False)
    
    plt.suptitle('双边滤波参数对比', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"参数对比图已保存: {output_path}")

if __name__ == "__main__":
    demo_bilateral_parameters()