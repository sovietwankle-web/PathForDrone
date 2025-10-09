#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高分辨率DEM处理演示
展示处理DEM_1m.tif（1米分辨率）数据的能力
"""

import sys
from pathlib import Path
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def demo_high_resolution_dem():
    """演示高分辨率DEM数据处理"""
    print("高分辨率DEM数据处理演示")
    print("=" * 60)
    
    try:
        # 导入我们的地形处理库
        from terrain_processor import TerrainProcessor
        
        # 1. 加载高分辨率DEM数据
        print("1. 加载DEM_1m.tif文件（1米分辨率）...")
        dem_file = "../DEM_1m.tif"
        processor = TerrainProcessor(dem_file)
        
        # 2. 获取文件信息
        print("\n2. 文件信息分析...")
        info = processor.get_terrain_info()
        print(f"   文件大小: {info['file_size_mb']:.2f} MB")
        print(f"   数据尺寸: {info['data_shape'][1]} x {info['data_shape'][0]} 像素")
        print(f"   总像素数: {info['data_shape'][0] * info['data_shape'][1]:,} 个")
        print(f"   数据类型: {info['data_type']}")
        print(f"   坐标系统: {info['coordinate_system']}")
        
        # 3. 加载地形数据
        print("\n3. 读取高分辨率地形数据...")
        terrain_data = processor.load_terrain_data()
        print(f"   数据形状: {terrain_data.shape}")
        print(f"   内存占用: {terrain_data.nbytes / 1024 / 1024:.2f} MB")
        
        # 4. 详细统计分析
        print("\n4. 详细统计分析...")
        stats = processor.get_terrain_statistics()
        print(f"   高程范围: {stats['min_elevation']:.1f} - {stats['max_elevation']:.1f} m")
        print(f"   平均高程: {stats['mean_elevation']:.1f} m")
        print(f"   中位数高程: {stats['median_elevation']:.1f} m")
        print(f"   标准差: {stats['std_elevation']:.1f} m")
        print(f"   高程变化: {stats['elevation_range']:.1f} m")
        print(f"   有效像素: {stats['valid_pixels']:,} 个")
        print(f"   数据完整性: {stats['valid_percentage']:.1f}%")
        
        # 5. 高分辨率地形分析
        print("\n5. 高分辨率地形特征分析...")
        
        # 计算坡度和坡向
        slope_data, aspect_data = processor.utils.calculate_slope_aspect(terrain_data, pixel_size=1.0)
        
        print(f"   坡度统计:")
        print(f"     平均坡度: {np.nanmean(slope_data):.2f}°")
        print(f"     最大坡度: {np.nanmax(slope_data):.2f}°")
        print(f"     坡度标准差: {np.nanstd(slope_data):.2f}°")
        
        # 坡度分级统计
        flat_area = np.sum(slope_data < 5) / slope_data.size * 100
        gentle_slope = np.sum((slope_data >= 5) & (slope_data < 15)) / slope_data.size * 100
        moderate_slope = np.sum((slope_data >= 15) & (slope_data < 25)) / slope_data.size * 100
        steep_slope = np.sum(slope_data >= 25) / slope_data.size * 100
        
        print(f"   坡度分级:")
        print(f"     平坦区域 (<5°): {flat_area:.1f}%")
        print(f"     缓坡区域 (5-15°): {gentle_slope:.1f}%")
        print(f"     中坡区域 (15-25°): {moderate_slope:.1f}%")
        print(f"     陡坡区域 (>25°): {steep_slope:.1f}%")
        
        # 6. 地形复杂度分析
        print("\n6. 地形复杂度分析...")
        
        # 计算地形粗糙度
        roughness = processor.utils.calculate_roughness(terrain_data, window_size=3)
        mean_roughness = np.nanmean(roughness)
        
        print(f"   地形粗糙度: {mean_roughness:.2f}")
        
        # 计算曲率
        curvature = processor.utils.calculate_curvature(terrain_data)
        mean_curvature = np.nanmean(curvature['mean_curvature'])
        
        print(f"   平均曲率: {mean_curvature:.6f}")
        
        # 7. 高精度特征点检测
        print("\n7. 高精度地形特征点检测...")
        
        # 使用更严格的突出度阈值进行特征点检测
        peaks_valleys = processor.utils.detect_peaks_valleys(terrain_data, min_prominence=5.0)
        
        print(f"   检测到显著山峰: {len(peaks_valleys['peaks'])} 个")
        print(f"   检测到显著山谷: {len(peaks_valleys['valleys'])} 个")
        
        # 找到最高点和最低点的精确位置
        max_idx = np.unravel_index(np.nanargmax(terrain_data), terrain_data.shape)
        min_idx = np.unravel_index(np.nanargmin(terrain_data), terrain_data.shape)
        
        max_elevation = terrain_data[max_idx]
        min_elevation = terrain_data[min_idx]
        
        print(f"   最高点: 像素({max_idx[1]}, {max_idx[0]}), 高程{max_elevation:.1f}m")
        print(f"   最低点: 像素({min_idx[1]}, {min_idx[0]}), 高程{min_elevation:.1f}m")
        
        # 8. 高质量输出生成
        print("\n8. 生成高质量输出...")
        output_dir = Path("output/high_res_demo")
        output_dir.mkdir(exist_ok=True)
        
        # 保存完整分辨率数组
        array_path = processor.save_array(output_dir / "DEM_1m_full_res.npy", "npy")
        print(f"   完整分辨率数组: {array_path}")
        
        # 生成高质量等高线图
        contour_path = processor.generate_contour_image(
            output_dir / "DEM_1m_high_quality_contour.png",
            levels=50,  # 更多等高线级数
            colormap='terrain',
            figsize=(16, 12),  # 更大尺寸
            dpi=300,  # 高分辨率
            title="DEM_1m 高精度等高线图 (1米分辨率)",
            show_stats=True,
            show_grid=False
        )
        print(f"   高质量等高线图: {contour_path}")
        
        # 生成高质量山体阴影图
        hillshade_path = processor.visualizer.generate_hillshade_plot(
            terrain_data,
            output_dir / "DEM_1m_high_quality_hillshade.png",
            azimuth=315,
            altitude=45,
            figsize=(16, 12),
            dpi=300,
            title="DEM_1m 高精度山体阴影图"
        )
        print(f"   高质量山体阴影图: {hillshade_path}")
        
        # 9. 数据重采样演示
        print("\n9. 数据重采样演示...")
        
        # 重采样到不同分辨率
        resampled_5m = processor.resample_data(
            terrain_data.shape[1] // 5, terrain_data.shape[0] // 5, method='bilinear'
        )
        resampled_10m = processor.resample_data(
            terrain_data.shape[1] // 10, terrain_data.shape[0] // 10, method='bilinear'
        )
        
        print(f"   原始1m分辨率: {terrain_data.shape}")
        print(f"   重采样到5m分辨率: {resampled_5m.shape}")
        print(f"   重采样到10m分辨率: {resampled_10m.shape}")
        
        # 保存重采样数据
        processor.processor.save_array(resampled_5m, output_dir / "DEM_resampled_5m.npy", "npy")
        processor.processor.save_array(resampled_10m, output_dir / "DEM_resampled_10m.npy", "npy")
        
        # 10. 性能和质量评估
        print("\n10. 性能和质量评估...")
        
        quality_report = processor.utils.validate_data_quality(terrain_data)
        print(f"   数据质量评分: {quality_report['quality_score']}/100")
        print(f"   质量等级: {quality_report['quality_level']}")
        
        # 计算数据密度
        pixel_area = 1.0 * 1.0  # 1m x 1m per pixel
        total_area = terrain_data.size * pixel_area / 10000  # 转换为公顷
        
        print(f"   覆盖面积: {total_area:.2f} 公顷")
        print(f"   数据密度: {terrain_data.size / total_area:.0f} 点/公顷")
        
        print("\n" + "=" * 60)
        print("高分辨率DEM数据处理完成!")
        print(f"处理文件: {dem_file}")
        print(f"分辨率: 1米")
        print(f"数据点数: {terrain_data.size:,} 个")
        print(f"覆盖面积: {total_area:.2f} 公顷")
        print(f"输出目录: {output_dir}")
        
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {dem_file}")
        return False
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("地形处理器 - 高分辨率DEM数据处理演示")
    print("展示1米分辨率DEM数据的处理能力")
    print()
    
    success = demo_high_resolution_dem()
    
    if success:
        print("\n高分辨率DEM处理演示成功完成!")
        print("生成的高质量输出文件:")
        print("- output/high_res_demo/DEM_1m_full_res.npy (完整分辨率数组)")
        print("- output/high_res_demo/DEM_1m_high_quality_contour.png (高质量等高线图)")
        print("- output/high_res_demo/DEM_1m_high_quality_hillshade.png (高质量山体阴影图)")
        print("- output/high_res_demo/DEM_resampled_5m.npy (重采样到5m)")
        print("- output/high_res_demo/DEM_resampled_10m.npy (重采样到10m)")
    else:
        print("\n演示失败，请检查错误信息")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)