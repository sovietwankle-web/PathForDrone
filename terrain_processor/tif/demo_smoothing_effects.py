#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地形数据平滑效果演示脚本
展示不同插值平滑方法的效果对比
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from terrain_processor import TerrainProcessor

def create_comparison_plot(results, output_path, title="地形平滑效果对比"):
    """
    创建平滑效果对比图
    
    Args:
        results: 不同方法的结果字典
        output_path: 输出路径
        title: 图片标题
    """
    methods = list(results.keys())
    n_methods = len(methods)
    
    # 计算子图布局
    cols = min(3, n_methods)
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
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # 创建等高线图
        im = ax.imshow(data, cmap='terrain', vmin=vmin, vmax=vmax, aspect='equal')
        ax.contour(data, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # 设置标题
        method_names = {
            'original': '原始数据',
            'gaussian': '高斯平滑',
            'bilateral': '双边滤波',
            'spline': '样条插值',
            'savgol': 'Savitzky-Golay',
            'anisotropic': '各向异性扩散'
        }
        ax.set_title(method_names.get(method, method), fontsize=12, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 隐藏多余的子图
    for i in range(n_methods, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"平滑效果对比图已保存: {output_path}")

def create_detailed_analysis_plot(original_data, smoothed_data, method_name, output_path):
    """
    创建详细的分析对比图
    
    Args:
        original_data: 原始数据
        smoothed_data: 平滑后数据
        method_name: 方法名称
        output_path: 输出路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 计算统计信息
    orig_valid = original_data[~np.isnan(original_data)]
    smooth_valid = smoothed_data[~np.isnan(smoothed_data)]
    
    if len(orig_valid) == 0 or len(smooth_valid) == 0:
        print(f"警告: {method_name} 方法没有有效数据")
        return
    
    vmin = min(np.min(orig_valid), np.min(smooth_valid))
    vmax = max(np.max(orig_valid), np.max(smooth_valid))
    
    # 1. 原始数据
    im1 = axes[0, 0].imshow(original_data, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 0].contour(original_data, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    axes[0, 0].set_title('原始地形数据', fontsize=14)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # 2. 平滑后数据
    im2 = axes[0, 1].imshow(smoothed_data, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 1].contour(smoothed_data, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    axes[0, 1].set_title(f'{method_name} 平滑结果', fontsize=14)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # 3. 差异图
    diff_data = smoothed_data - original_data
    diff_valid = diff_data[~np.isnan(diff_data)]
    if len(diff_valid) > 0:
        diff_max = max(abs(np.min(diff_valid)), abs(np.max(diff_valid)))
        im3 = axes[0, 2].imshow(diff_data, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        axes[0, 2].set_title('差异 (平滑后 - 原始)', fontsize=14)
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # 4. 高程分布直方图
    axes[1, 0].hist(orig_valid, bins=50, alpha=0.7, label='原始数据', density=True)
    axes[1, 0].hist(smooth_valid, bins=50, alpha=0.7, label='平滑数据', density=True)
    axes[1, 0].set_xlabel('高程 (m)')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('高程分布对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 散点图对比
    # 随机采样以避免过多数据点
    sample_size = min(5000, len(orig_valid))
    if len(orig_valid) > sample_size:
        indices = np.random.choice(len(orig_valid), sample_size, replace=False)
        orig_sample = orig_valid[indices]
        smooth_sample = smooth_valid[indices]
    else:
        orig_sample = orig_valid
        smooth_sample = smooth_valid
    
    axes[1, 1].scatter(orig_sample, smooth_sample, alpha=0.5, s=1)
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
    
    smooth_stats = {
        '最小值': f"{np.min(smooth_valid):.2f}",
        '最大值': f"{np.max(smooth_valid):.2f}",
        '平均值': f"{np.mean(smooth_valid):.2f}",
        '标准差': f"{np.std(smooth_valid):.2f}",
        '中位数': f"{np.median(smooth_valid):.2f}"
    }
    
    # 创建表格
    table_data = []
    for key in orig_stats.keys():
        table_data.append([key, orig_stats[key], smooth_stats[key]])
    
    table = axes[1, 2].table(cellText=table_data,
                            colLabels=['统计量', '原始数据', '平滑数据'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 2].set_title('统计信息对比', fontsize=14)
    
    plt.suptitle(f'{method_name} 平滑效果详细分析', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{method_name} 详细分析图已保存: {output_path}")

def demo_smoothing_effects():
    """演示平滑效果"""
    print("=" * 60)
    print("地形数据平滑效果演示")
    print("=" * 60)
    
    # 检查DEM文件
    dem_files = ['DEM_5m.tif', 'DEM_1m.tif']
    available_files = [f for f in dem_files if os.path.exists(f)]
    
    if not available_files:
        print("错误: 未找到DEM文件 (DEM_5m.tif 或 DEM_1m.tif)")
        print("请确保DEM文件在当前目录中")
        return
    
    # 使用第一个可用文件
    dem_file = available_files[0]
    print(f"使用DEM文件: {dem_file}")
    
    try:
        # 初始化处理器
        processor = TerrainProcessor(dem_file)
        
        # 获取地形信息
        info = processor.get_terrain_info()
        print(f"\n地形信息:")
        print(f"  数据尺寸: {info['data_shape']}")
        print(f"  数据类型: {info['data_type']}")
        print(f"  文件大小: {info['file_size_mb']:.2f} MB")
        
        # 创建输出目录
        output_dir = Path('output/smoothing_demo')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n开始平滑效果演示...")
        
        # 比较不同平滑方法
        methods = ['gaussian', 'bilateral', 'spline', 'savgol', 'anisotropic']
        print(f"测试平滑方法: {', '.join(methods)}")
        
        results = processor.compare_smoothing_effects(methods)
        
        # 创建总体对比图
        comparison_output = output_dir / f"{Path(dem_file).stem}_smoothing_comparison.png"
        create_comparison_plot(results, comparison_output, 
                             f"{Path(dem_file).stem} 地形平滑效果对比")
        
        # 为每种方法创建详细分析
        original_data = results['original']
        
        method_names = {
            'gaussian': '高斯平滑',
            'bilateral': '双边滤波',
            'spline': '样条插值',
            'savgol': 'Savitzky-Golay滤波',
            'anisotropic': '各向异性扩散'
        }
        
        for method in methods:
            if method in results:
                print(f"  生成 {method_names[method]} 详细分析...")
                detail_output = output_dir / f"{Path(dem_file).stem}_{method}_analysis.png"
                create_detailed_analysis_plot(original_data, results[method], 
                                            method_names[method], detail_output)
        
        # 生成平滑后的等高线图
        print(f"\n生成平滑后的等高线图...")
        for method in ['gaussian', 'bilateral', 'spline']:
            if method in results:
                contour_output = output_dir / f"{Path(dem_file).stem}_{method}_contour.png"
                try:
                    processor.generate_smoothed_contour(
                        contour_output, 
                        smooth_method=method,
                        levels=30,
                        dpi=300
                    )
                    print(f"  {method_names[method]} 等高线图: {contour_output}")
                except Exception as e:
                    print(f"  {method_names[method]} 等高线图生成失败: {e}")
        
        # 保存平滑后的数组数据
        print(f"\n保存平滑后的数组数据...")
        for method in ['gaussian', 'spline']:
            if method in results:
                array_output = output_dir / f"{Path(dem_file).stem}_{method}_smoothed.npy"
                np.save(array_output, results[method])
                print(f"  {method_names[method]} 数组: {array_output}")
        
        print(f"\n=" * 60)
        print(f"平滑效果演示完成!")
        print(f"输出目录: {output_dir}")
        print(f"生成文件:")
        for file in sorted(output_dir.glob('*')):
            print(f"  - {file.name}")
        print(f"=" * 60)
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_smoothing_effects()