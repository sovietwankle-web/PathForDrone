#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础使用示例
演示如何使用terrain_processor库的基本功能
"""

import sys
from pathlib import Path

# 添加父目录到路径以便导入terrain_processor
sys.path.insert(0, str(Path(__file__).parent.parent))

from terrain_processor import TerrainProcessor


def basic_example():
    """基础使用示例"""
    print("=== 地形处理器基础使用示例 ===\n")
    
    # 1. 创建处理器实例
    print("1. 创建地形处理器...")
    tif_file = "../test_terrain.tif"  # 使用我们创建的测试数据
    
    try:
        processor = TerrainProcessor(tif_file)
        print(f"✓ 成功加载地形文件: {tif_file}")
    except FileNotFoundError:
        print(f"✗ 文件不存在: {tif_file}")
        print("请先运行 create_test_data.py 创建测试数据")
        return
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return
    
    # 2. 获取文件信息
    print("\n2. 获取文件信息...")
    info = processor.get_terrain_info()
    print(f"   文件大小: {info['file_size_mb']:.2f} MB")
    print(f"   数据尺寸: {info['data_shape']}")
    print(f"   数据类型: {info['data_type']}")
    print(f"   坐标系统: {info['coordinate_system']}")
    
    # 3. 加载地形数据
    print("\n3. 加载地形数据...")
    terrain_data = processor.load_terrain_data()
    print(f"   数据形状: {terrain_data.shape}")
    print(f"   数据类型: {terrain_data.dtype}")
    
    # 4. 获取统计信息
    print("\n4. 获取统计信息...")
    stats = processor.get_terrain_statistics()
    print(f"   有效像素: {stats['valid_pixels']:,}")
    print(f"   最低高程: {stats['min_elevation']:.2f} m")
    print(f"   最高高程: {stats['max_elevation']:.2f} m")
    print(f"   平均高程: {stats['mean_elevation']:.2f} m")
    print(f"   高程范围: {stats['elevation_range']:.2f} m")
    
    # 5. 保存三维数组
    print("\n5. 保存三维数组...")
    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)
    
    # 保存为不同格式
    npy_path = processor.save_array(output_dir / "terrain_data.npy", "npy")
    print(f"   ✓ 保存为NPY格式: {npy_path}")
    
    npz_path = processor.save_array(output_dir / "terrain_data.npz", "npz")
    print(f"   ✓ 保存为NPZ格式: {npz_path}")
    
    # 6. 生成等高线图
    print("\n6. 生成等高线图...")
    contour_path = processor.generate_contour_image(
        output_dir / "contour_map.png",
        levels=20,
        colormap='terrain',
        figsize=(12, 8),
        dpi=150
    )
    print(f"   ✓ 等高线图已保存: {contour_path}")
    
    # 7. 获取特定点的高程
    print("\n7. 获取特定点的高程...")
    # 获取数据中心点的高程
    center_x = terrain_data.shape[1] // 2
    center_y = terrain_data.shape[0] // 2
    center_elevation = processor.get_elevation_at_point(center_x, center_y)
    print(f"   中心点 ({center_x}, {center_y}) 高程: {center_elevation:.2f} m")
    
    # 8. 获取高程剖面
    print("\n8. 获取高程剖面...")
    start_point = (0, terrain_data.shape[0] // 2)
    end_point = (terrain_data.shape[1] - 1, terrain_data.shape[0] // 2)
    distances, elevations = processor.get_elevation_profile(start_point, end_point, 50)
    print(f"   剖面线长度: {distances[-1]:.2f} 像素")
    print(f"   剖面高程范围: {min(elevations):.2f} - {max(elevations):.2f} m")
    
    print("\n✓ 基础示例完成!")


def advanced_example():
    """高级功能示例"""
    print("\n=== 高级功能示例 ===\n")
    
    tif_file = "../test_terrain.tif"
    
    try:
        processor = TerrainProcessor(tif_file)
    except:
        print("请先运行基础示例创建测试数据")
        return
    
    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 生成山体阴影图
    print("1. 生成山体阴影图...")
    hillshade_path = processor.visualizer.generate_hillshade_plot(
        processor.load_terrain_data(),
        output_dir / "hillshade.png",
        azimuth=315,
        altitude=45
    )
    print(f"   ✓ 山体阴影图: {hillshade_path}")
    
    # 2. 生成3D地形图
    print("\n2. 生成3D地形图...")
    try:
        td_path = processor.visualizer.generate_3d_surface(
            processor.load_terrain_data(),
            output_dir / "terrain_3d.png",
            colormap='terrain'
        )
        print(f"   ✓ 3D地形图: {td_path}")
    except Exception as e:
        print(f"   ✗ 3D图生成失败: {e}")
    
    # 3. 生成多视图综合图
    print("\n3. 生成多视图综合图...")
    stats = processor.get_terrain_statistics()
    multi_path = processor.visualizer.generate_multi_view_plot(
        processor.load_terrain_data(),
        output_dir / "multi_view.png",
        statistics=stats
    )
    print(f"   ✓ 多视图图: {multi_path}")
    
    # 4. 数据重采样
    print("\n4. 数据重采样...")
    original_data = processor.load_terrain_data()
    resampled_data = processor.resample_data(150, 150, method='bilinear')
    print(f"   原始尺寸: {original_data.shape}")
    print(f"   重采样后: {resampled_data.shape}")
    
    # 5. 应用滤波器
    print("\n5. 应用滤波器...")
    filtered_data = processor.apply_filter('gaussian', sigma=1.0)
    print(f"   ✓ 高斯滤波完成")
    
    print("\n✓ 高级示例完成!")


def main():
    """主函数"""
    print("地形处理器使用示例")
    print("=" * 50)
    
    # 运行基础示例
    basic_example()
    
    # 运行高级示例
    advanced_example()
    
    print("\n" + "=" * 50)
    print("所有示例完成! 请查看 output 目录中的生成文件。")


if __name__ == "__main__":
    main()