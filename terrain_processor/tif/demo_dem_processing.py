#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM_5m.tif处理演示
展示如何在其他Python程序中调用terrain_processor库
"""

import sys
from pathlib import Path
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def demo_dem_processing():
    """演示DEM数据处理"""
    print("DEM地形数据处理演示")
    print("=" * 50)
    
    try:
        # 导入我们的地形处理库
        from terrain_processor import TerrainProcessor
        
        # 1. 创建处理器实例
        print("1. 加载DEM_5m.tif文件...")
        dem_file = "../DEM_5m.tif"
        processor = TerrainProcessor(dem_file)
        
        # 2. 获取地形数据
        print("\n2. 读取地形数据...")
        terrain_data = processor.load_terrain_data()
        print(f"   数据尺寸: {terrain_data.shape}")
        print(f"   数据类型: {terrain_data.dtype}")
        
        # 3. 获取统计信息
        print("\n3. 地形统计分析...")
        stats = processor.get_terrain_statistics()
        print(f"   高程范围: {stats['min_elevation']:.1f} - {stats['max_elevation']:.1f} m")
        print(f"   平均高程: {stats['mean_elevation']:.1f} m")
        print(f"   高程变化: {stats['elevation_range']:.1f} m")
        print(f"   有效像素: {stats['valid_pixels']:,} 个")
        
        # 4. 地形特征分析
        print("\n4. 地形特征分析...")
        
        # 判断地形类型
        elevation_range = stats['elevation_range']
        if elevation_range < 50:
            terrain_type = "平原地形"
        elif elevation_range < 200:
            terrain_type = "丘陵地形"
        else:
            terrain_type = "山地地形"
        
        print(f"   地形类型: {terrain_type}")
        
        # 计算坡度统计
        slope_data, aspect_data = processor.utils.calculate_slope_aspect(terrain_data)
        mean_slope = np.nanmean(slope_data)
        max_slope = np.nanmax(slope_data)
        
        print(f"   平均坡度: {mean_slope:.1f}°")
        print(f"   最大坡度: {max_slope:.1f}°")
        
        # 5. 寻找地形特征点
        print("\n5. 地形特征点检测...")
        peaks_valleys = processor.utils.detect_peaks_valleys(terrain_data)
        print(f"   检测到山峰: {len(peaks_valleys['peaks'])} 个")
        print(f"   检测到山谷: {len(peaks_valleys['valleys'])} 个")
        
        # 找到最高点和最低点
        max_idx = np.unravel_index(np.nanargmax(terrain_data), terrain_data.shape)
        min_idx = np.unravel_index(np.nanargmin(terrain_data), terrain_data.shape)
        
        max_elevation = terrain_data[max_idx]
        min_elevation = terrain_data[min_idx]
        
        print(f"   最高点位置: ({max_idx[1]}, {max_idx[0]}), 高程: {max_elevation:.1f} m")
        print(f"   最低点位置: ({min_idx[1]}, {min_idx[0]}), 高程: {min_elevation:.1f} m")
        
        # 6. 保存处理结果
        print("\n6. 保存处理结果...")
        output_dir = Path("output/dem_demo")
        output_dir.mkdir(exist_ok=True)
        
        # 保存三维数组
        array_path = processor.save_array(output_dir / "dem_5m_array.npy", "npy")
        print(f"   三维数组已保存: {array_path}")
        
        # 生成等高线图
        contour_path = processor.generate_contour_image(
            output_dir / "dem_5m_contour_demo.png",
            levels=25,
            colormap='terrain',
            figsize=(12, 10),
            dpi=200,
            title="DEM_5m 等高线图",
            show_stats=True
        )
        print(f"   等高线图已保存: {contour_path}")
        
        # 7. 高程剖面分析
        print("\n7. 高程剖面分析...")
        
        # 创建一条从西到东的剖面线
        start_point = (0, terrain_data.shape[0] // 2)
        end_point = (terrain_data.shape[1] - 1, terrain_data.shape[0] // 2)
        
        distances, elevations = processor.get_elevation_profile(start_point, end_point, 100)
        
        print(f"   剖面线长度: {distances[-1]:.1f} 像素")
        print(f"   剖面高程变化: {np.min(elevations):.1f} - {np.max(elevations):.1f} m")
        print(f"   剖面高程差: {np.max(elevations) - np.min(elevations):.1f} m")
        
        # 8. 数据质量评估
        print("\n8. 数据质量评估...")
        quality_report = processor.utils.validate_data_quality(terrain_data)
        print(f"   数据质量评分: {quality_report['quality_score']}/100")
        print(f"   质量等级: {quality_report['quality_level']}")
        
        if quality_report['issues']:
            print("   发现的问题:")
            for issue in quality_report['issues']:
                print(f"     - {issue}")
        
        if quality_report['warnings']:
            print("   警告信息:")
            for warning in quality_report['warnings']:
                print(f"     - {warning}")
        
        print("\n" + "=" * 50)
        print("DEM地形数据处理完成!")
        print(f"处理的文件: {dem_file}")
        print(f"数据尺寸: {terrain_data.shape[1]} x {terrain_data.shape[0]} 像素")
        print(f"高程范围: {stats['min_elevation']:.1f} - {stats['max_elevation']:.1f} m")
        print(f"地形类型: {terrain_type}")
        print(f"输出目录: {output_dir}")
        
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {dem_file}")
        print("请确保DEM_5m.tif文件在正确的位置")
        return False
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("地形处理器 - DEM数据处理演示")
    print("这个脚本展示了如何在其他Python程序中使用terrain_processor库")
    print()
    
    success = demo_dem_processing()
    
    if success:
        print("\n演示成功完成!")
        print("您可以查看生成的文件:")
        print("- output/dem_demo/dem_5m_array.npy (三维数组)")
        print("- output/dem_demo/dem_5m_contour_demo.png (等高线图)")
    else:
        print("\n演示失败，请检查错误信息")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)