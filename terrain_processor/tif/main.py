#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地形处理器主程序
提供命令行接口和批处理功能
"""

import argparse
import sys
from pathlib import Path
import json
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from terrain_processor import TerrainProcessor


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='地形数据处理工具 - 读取TIF文件，生成三维数组和等高线图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s input.tif --array output.npy --contour output.png
  %(prog)s input.tif --info
  %(prog)s input.tif --batch config.json
        """
    )
    
    # 必需参数
    parser.add_argument('input', help='输入TIF文件路径')
    
    # 输出选项
    parser.add_argument('--array', '-a', help='保存三维数组到文件 (支持.npy, .npz, .csv)')
    parser.add_argument('--contour', '-c', help='生成等高线图片文件 (支持.png, .jpg)')
    parser.add_argument('--output-dir', '-o', help='输出目录 (默认: ./output)')
    
    # 可视化选项
    parser.add_argument('--levels', type=int, help='等高线级数 (默认: 自动计算)')
    parser.add_argument('--colormap', default='terrain', help='颜色映射方案 (默认: terrain)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], help='图片尺寸 (默认: 12 8)')
    parser.add_argument('--dpi', type=int, default=150, help='图片分辨率 (默认: 150)')
    
    # 信息选项
    parser.add_argument('--info', '-i', action='store_true', help='显示文件信息')
    parser.add_argument('--stats', '-s', action='store_true', help='显示统计信息')
    parser.add_argument('--validate', '-v', action='store_true', help='验证数据质量')
    
    # 高级功能
    parser.add_argument('--hillshade', help='生成山体阴影图')
    parser.add_argument('--3d', help='生成3D地形图')
    parser.add_argument('--multi-view', help='生成多视图综合图')
    
    # 平滑功能
    parser.add_argument('--smooth', help='生成平滑后的等高线图')
    parser.add_argument('--smooth-method', default='gaussian',
                       choices=['gaussian', 'bilateral', 'spline', 'savgol', 'anisotropic'],
                       help='平滑方法 (默认: gaussian)')
    parser.add_argument('--smooth-sigma', type=float, default=1.0,
                       help='高斯平滑的sigma参数 (默认: 1.0)')
    parser.add_argument('--smooth-array', help='输出平滑后的数组文件')
    parser.add_argument('--compare-smooth', help='输出平滑方法对比图')
    
    # 批处理
    parser.add_argument('--batch', help='批处理配置文件 (JSON格式)')
    
    # 其他选项
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}", file=sys.stderr)
        return 1
    
    try:
        # 创建地形处理器
        if not args.quiet:
            print(f"正在加载地形文件: {input_path}")
        
        processor = TerrainProcessor(input_path)
        
        # 批处理模式
        if args.batch:
            return run_batch_processing(processor, args.batch, args.quiet)
        
        # 设置输出目录
        output_dir = Path(args.output_dir) if args.output_dir else Path('./output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 显示信息
        if args.info:
            show_file_info(processor, args.verbose)
        
        if args.stats:
            show_statistics(processor)
        
        if args.validate:
            show_validation_results(processor)
        
        # 保存数组
        if args.array:
            array_path = output_dir / args.array if not Path(args.array).is_absolute() else Path(args.array)
            if not args.quiet:
                print(f"正在保存三维数组到: {array_path}")
            
            # 根据文件扩展名确定格式
            suffix = array_path.suffix.lower()
            if suffix == '.npy':
                format_type = 'npy'
            elif suffix == '.npz':
                format_type = 'npz'
            elif suffix == '.csv':
                format_type = 'csv'
            else:
                format_type = 'npy'
                array_path = array_path.with_suffix('.npy')
            
            saved_path = processor.save_array(array_path, format_type)
            if not args.quiet:
                print(f"数组已保存到: {saved_path}")
        
        # 生成等高线图
        if args.contour:
            contour_path = output_dir / args.contour if not Path(args.contour).is_absolute() else Path(args.contour)
            if not args.quiet:
                print(f"正在生成等高线图: {contour_path}")
            
            saved_path = processor.generate_contour_image(
                contour_path,
                levels=args.levels,
                colormap=args.colormap,
                figsize=tuple(args.figsize),
                dpi=args.dpi
            )
            if not args.quiet:
                print(f"等高线图已保存到: {saved_path}")
        
        # 生成山体阴影图
        if args.hillshade:
            hillshade_path = output_dir / args.hillshade if not Path(args.hillshade).is_absolute() else Path(args.hillshade)
            if not args.quiet:
                print(f"正在生成山体阴影图: {hillshade_path}")
            
            saved_path = processor.visualizer.generate_hillshade_plot(
                processor.load_terrain_data(),
                hillshade_path,
                figsize=tuple(args.figsize),
                dpi=args.dpi
            )
            if not args.quiet:
                print(f"山体阴影图已保存到: {saved_path}")
        
        # 生成3D图
        if getattr(args, '3d'):
            td_path = output_dir / getattr(args, '3d') if not Path(getattr(args, '3d')).is_absolute() else Path(getattr(args, '3d'))
            if not args.quiet:
                print(f"正在生成3D地形图: {td_path}")
            
            saved_path = processor.visualizer.generate_3d_surface(
                processor.load_terrain_data(),
                td_path,
                colormap=args.colormap,
                figsize=tuple(args.figsize),
                dpi=args.dpi
            )
            if not args.quiet:
                print(f"3D地形图已保存到: {saved_path}")
        
        # 生成多视图图
        if args.multi_view:
            multi_path = output_dir / args.multi_view if not Path(args.multi_view).is_absolute() else Path(args.multi_view)
            if not args.quiet:
                print(f"正在生成多视图综合图: {multi_path}")
            
            stats = processor.get_terrain_statistics()
            saved_path = processor.visualizer.generate_multi_view_plot(
                processor.load_terrain_data(),
                multi_path,
                statistics=stats,
                figsize=(16, 12),
                dpi=args.dpi
            )
            if not args.quiet:
                print(f"多视图图已保存到: {saved_path}")
        
        # 生成平滑后的等高线图
        if args.smooth:
            smooth_path = output_dir / args.smooth if not Path(args.smooth).is_absolute() else Path(args.smooth)
            if not args.quiet:
                print(f"正在生成平滑后的等高线图: {smooth_path}")
            
            # 准备平滑参数
            smooth_kwargs = {}
            if args.smooth_method == 'gaussian':
                smooth_kwargs['sigma'] = args.smooth_sigma
            
            saved_path = processor.generate_smoothed_contour(
                smooth_path,
                smooth_method=args.smooth_method,
                levels=args.levels,
                colormap=args.colormap,
                figsize=tuple(args.figsize),
                dpi=args.dpi,
                **smooth_kwargs
            )
            if not args.quiet:
                print(f"平滑后的等高线图已保存到: {saved_path}")
        
        # 保存平滑后的数组
        if args.smooth_array:
            smooth_array_path = output_dir / args.smooth_array if not Path(args.smooth_array).is_absolute() else Path(args.smooth_array)
            if not args.quiet:
                print(f"正在保存平滑后的数组: {smooth_array_path}")
            
            # 准备平滑参数
            smooth_kwargs = {}
            if args.smooth_method == 'gaussian':
                smooth_kwargs['sigma'] = args.smooth_sigma
            
            # 获取平滑后的数据
            smoothed_data = processor.smooth_terrain_data(args.smooth_method, **smooth_kwargs)
            
            # 根据文件扩展名确定格式
            suffix = smooth_array_path.suffix.lower()
            if suffix == '.npy':
                format_type = 'npy'
            elif suffix == '.npz':
                format_type = 'npz'
            elif suffix == '.csv':
                format_type = 'csv'
            else:
                format_type = 'npy'
                smooth_array_path = smooth_array_path.with_suffix('.npy')
            
            saved_path = processor.processor.save_array(smoothed_data, smooth_array_path, format_type)
            if not args.quiet:
                print(f"平滑后的数组已保存到: {saved_path}")
        
        # 生成平滑方法对比图
        if args.compare_smooth:
            compare_path = output_dir / args.compare_smooth if not Path(args.compare_smooth).is_absolute() else Path(args.compare_smooth)
            if not args.quiet:
                print(f"正在生成平滑方法对比图: {compare_path}")
            
            # 比较不同平滑方法
            results = processor.compare_smoothing_effects()
            
            # 创建对比图
            create_smoothing_comparison_plot(results, compare_path)
            if not args.quiet:
                print(f"平滑方法对比图已保存到: {compare_path}")
        
        if not args.quiet:
            print("处理完成!")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def show_file_info(processor: TerrainProcessor, verbose: bool = False):
    """显示文件信息"""
    info = processor.get_terrain_info()
    
    print("\n=== 文件信息 ===")
    print(f"文件路径: {info['file_path']}")
    print(f"文件大小: {info['file_size_mb']:.2f} MB")
    print(f"数据尺寸: {info['data_shape'][1]} x {info['data_shape'][0]} (宽 x 高)")
    print(f"数据类型: {info['data_type']}")
    print(f"坐标系统: {info['coordinate_system']}")
    
    if verbose and info['bounds']:
        bounds = info['bounds']
        print(f"地理边界:")
        print(f"  左: {bounds.left:.6f}")
        print(f"  右: {bounds.right:.6f}")
        print(f"  下: {bounds.bottom:.6f}")
        print(f"  上: {bounds.top:.6f}")
    
    if verbose and info['transform']:
        print(f"变换矩阵: {info['transform']}")


def show_statistics(processor: TerrainProcessor):
    """显示统计信息"""
    stats = processor.get_terrain_statistics()
    
    print("\n=== 统计信息 ===")
    print(f"总像素数: {stats['total_pixels']:,}")
    print(f"有效像素数: {stats['valid_pixels']:,}")
    print(f"有效数据比例: {stats['valid_percentage']:.1f}%")
    
    if stats['valid_pixels'] > 0:
        print(f"最低高程: {stats['min_elevation']:.2f} m")
        print(f"最高高程: {stats['max_elevation']:.2f} m")
        print(f"平均高程: {stats['mean_elevation']:.2f} m")
        print(f"中位数高程: {stats['median_elevation']:.2f} m")
        print(f"标准差: {stats['std_elevation']:.2f} m")
        print(f"高程范围: {stats['elevation_range']:.2f} m")


def show_validation_results(processor: TerrainProcessor):
    """显示验证结果"""
    validation = processor.reader.validate_data_integrity()
    
    print("\n=== 数据质量验证 ===")
    print(f"文件可读性: {'是' if validation['file_readable'] else '否'}")
    
    if validation['file_readable']:
        print(f"有效数据比例: {validation['valid_data_percentage']:.1f}%")
        
        if validation['data_range']['min'] is not None:
            print(f"数据范围: {validation['data_range']['min']:.2f} - {validation['data_range']['max']:.2f}")
        
        print(f"坐标系统: {'有' if validation['has_coordinate_system'] else '无'}")
        
        if validation['warnings']:
            print("警告:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
    else:
        print(f"错误: {validation.get('error', '未知错误')}")


def create_smoothing_comparison_plot(results, output_path):
    """
    创建平滑效果对比图
    
    Args:
        results: 不同方法的结果字典
        output_path: 输出路径
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
    
    plt.suptitle('地形平滑效果对比', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_batch_processing(processor: TerrainProcessor, config_file: str, quiet: bool = False) -> int:
    """运行批处理"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if not quiet:
            print(f"正在执行批处理配置: {config_file}")
        
        output_dir = Path(config.get('output_dir', './output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行配置中的任务
        tasks = config.get('tasks', [])
        
        for i, task in enumerate(tasks, 1):
            task_type = task.get('type')
            
            if not quiet:
                print(f"执行任务 {i}/{len(tasks)}: {task_type}")
            
            if task_type == 'array':
                output_path = output_dir / task['output']
                processor.save_array(output_path, task.get('format', 'npy'))
                
            elif task_type == 'contour':
                output_path = output_dir / task['output']
                processor.generate_contour_image(
                    output_path,
                    levels=task.get('levels'),
                    colormap=task.get('colormap', 'terrain'),
                    figsize=task.get('figsize', [12, 8]),
                    dpi=task.get('dpi', 150)
                )
                
            elif task_type == 'hillshade':
                output_path = output_dir / task['output']
                processor.visualizer.generate_hillshade_plot(
                    processor.load_terrain_data(),
                    output_path,
                    azimuth=task.get('azimuth', 315),
                    altitude=task.get('altitude', 45)
                )
                
            elif task_type == '3d':
                output_path = output_dir / task['output']
                processor.visualizer.generate_3d_surface(
                    processor.load_terrain_data(),
                    output_path,
                    colormap=task.get('colormap', 'terrain')
                )
                
            elif task_type == 'multi_view':
                output_path = output_dir / task['output']
                stats = processor.get_terrain_statistics()
                processor.visualizer.generate_multi_view_plot(
                    processor.load_terrain_data(),
                    output_path,
                    statistics=stats
                )
        
        if not quiet:
            print("批处理完成!")
        
        return 0
        
    except Exception as e:
        print(f"批处理错误: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())