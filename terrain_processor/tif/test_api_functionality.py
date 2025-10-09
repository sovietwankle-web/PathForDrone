#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API功能测试脚本
验证terrain_processor作为Python库的API调用功能
"""

import sys
from pathlib import Path
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_api_integration():
    """测试API集成功能"""
    print("测试API集成功能...")
    print("=" * 40)
    
    try:
        # 导入库
        from terrain_processor import TerrainProcessor
        
        # 1. 简单的API调用示例
        print("1. 简单API调用测试...")
        
        def get_terrain_elevation_range(tif_file):
            """获取地形高程范围的简单函数"""
            processor = TerrainProcessor(tif_file)
            stats = processor.get_terrain_statistics()
            return stats['min_elevation'], stats['max_elevation']
        
        min_elev, max_elev = get_terrain_elevation_range("test_terrain.tif")
        print(f"   高程范围: {min_elev:.2f} - {max_elev:.2f} m")
        
        # 2. 数据处理API测试
        print("\n2. 数据处理API测试...")
        
        processor = TerrainProcessor("test_terrain.tif")
        
        # 获取原始数据
        original_data = processor.load_terrain_data()
        print(f"   原始数据形状: {original_data.shape}")
        
        # 重采样数据
        resampled_data = processor.resample_data(150, 150, method='bilinear')
        print(f"   重采样后形状: {resampled_data.shape}")
        
        # 应用滤波
        filtered_data = processor.apply_filter('gaussian', sigma=1.0)
        print(f"   滤波后形状: {filtered_data.shape}")
        
        # 3. 可视化API测试
        print("\n3. 可视化API测试...")
        
        output_dir = Path("output/api_test")
        output_dir.mkdir(exist_ok=True)
        
        # 生成等高线图
        contour_path = processor.generate_contour_image(
            output_dir / "api_contour.png",
            levels=20,
            colormap='terrain',
            figsize=(10, 8)
        )
        print(f"   等高线图: {contour_path}")
        
        # 生成山体阴影图
        hillshade_path = processor.visualizer.generate_hillshade_plot(
            original_data,
            output_dir / "api_hillshade.png"
        )
        print(f"   山体阴影图: {hillshade_path}")
        
        # 4. 地形分析API测试
        print("\n4. 地形分析API测试...")
        
        # 计算坡度和坡向
        slope, aspect = processor.utils.calculate_slope_aspect(original_data)
        print(f"   坡度范围: {np.nanmin(slope):.2f} - {np.nanmax(slope):.2f}°")
        print(f"   坡向范围: {np.nanmin(aspect):.2f} - {np.nanmax(aspect):.2f}°")
        
        # 检测山峰和山谷
        peaks_valleys = processor.utils.detect_peaks_valleys(original_data)
        print(f"   检测到山峰: {len(peaks_valleys['peaks'])} 个")
        print(f"   检测到山谷: {len(peaks_valleys['valleys'])} 个")
        
        # 5. 数据质量验证API测试
        print("\n5. 数据质量验证API测试...")
        
        quality_report = processor.utils.validate_data_quality(original_data)
        print(f"   数据质量评分: {quality_report['quality_score']}/100")
        print(f"   质量等级: {quality_report['quality_level']}")
        
        # 6. 高程剖面API测试
        print("\n6. 高程剖面API测试...")
        
        start_point = (0, original_data.shape[0] // 2)
        end_point = (original_data.shape[1] - 1, original_data.shape[0] // 2)
        
        distances, elevations = processor.get_elevation_profile(
            start_point, end_point, num_points=100
        )
        
        print(f"   剖面线长度: {distances[-1]:.2f} 像素")
        print(f"   剖面高程变化: {np.min(elevations):.2f} - {np.max(elevations):.2f} m")
        
        print("\n" + "=" * 40)
        print("API集成测试完成 - 所有功能正常!")
        return True
        
    except Exception as e:
        print(f"\nAPI测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing_api():
    """测试批量处理API"""
    print("\n测试批量处理API...")
    print("-" * 30)
    
    try:
        from terrain_processor import TerrainProcessor
        
        # 模拟批量处理多个文件
        files = ["test_terrain.tif"]  # 实际使用时可以是多个文件
        
        results = []
        for file_path in files:
            processor = TerrainProcessor(file_path)
            
            # 获取基本信息
            info = processor.get_terrain_info()
            stats = processor.get_terrain_statistics()
            
            result = {
                'file': file_path,
                'shape': info['data_shape'],
                'elevation_range': stats['elevation_range'],
                'mean_elevation': stats['mean_elevation']
            }
            
            results.append(result)
        
        print("批量处理结果:")
        for result in results:
            print(f"  文件: {result['file']}")
            print(f"    尺寸: {result['shape']}")
            print(f"    高程范围: {result['elevation_range']:.2f} m")
            print(f"    平均高程: {result['mean_elevation']:.2f} m")
        
        print("批量处理API测试成功!")
        return True
        
    except Exception as e:
        print(f"批量处理API测试失败: {e}")
        return False

def test_custom_analysis():
    """测试自定义分析功能"""
    print("\n测试自定义分析功能...")
    print("-" * 30)
    
    try:
        from terrain_processor import TerrainProcessor
        
        processor = TerrainProcessor("test_terrain.tif")
        data = processor.load_terrain_data()
        
        # 自定义分析：地形复杂度评估
        def calculate_terrain_complexity(terrain_data):
            """计算地形复杂度"""
            # 计算高程标准差
            elevation_std = np.nanstd(terrain_data)
            
            # 计算坡度变化
            gy, gx = np.gradient(terrain_data)
            slope = np.sqrt(gx**2 + gy**2)
            slope_std = np.nanstd(slope)
            
            # 综合复杂度指数
            complexity = (elevation_std + slope_std * 10) / 2
            
            return {
                'elevation_std': elevation_std,
                'slope_std': slope_std,
                'complexity_index': complexity
            }
        
        complexity = calculate_terrain_complexity(data)
        print(f"地形复杂度分析:")
        print(f"  高程标准差: {complexity['elevation_std']:.2f}")
        print(f"  坡度标准差: {complexity['slope_std']:.2f}")
        print(f"  复杂度指数: {complexity['complexity_index']:.2f}")
        
        # 自定义可视化
        output_path = Path("output/api_test/custom_analysis.png")
        
        # 使用多视图功能
        stats = processor.get_terrain_statistics()
        multi_view_path = processor.visualizer.generate_multi_view_plot(
            data, output_path, statistics=stats
        )
        
        print(f"自定义分析图表: {multi_view_path}")
        print("自定义分析测试成功!")
        return True
        
    except Exception as e:
        print(f"自定义分析测试失败: {e}")
        return False

def main():
    """主函数"""
    print("地形处理器API功能验证")
    print("=" * 60)
    
    # 检查测试数据
    if not Path("test_terrain.tif").exists():
        print("错误: 测试数据文件不存在")
        return False
    
    # 运行各项测试
    api_success = test_api_integration()
    batch_success = test_batch_processing_api()
    custom_success = test_custom_analysis()
    
    print("\n" + "=" * 60)
    print("API功能验证结果:")
    print(f"API集成测试: {'通过' if api_success else '失败'}")
    print(f"批量处理测试: {'通过' if batch_success else '失败'}")
    print(f"自定义分析测试: {'通过' if custom_success else '失败'}")
    
    all_success = api_success and batch_success and custom_success
    
    if all_success:
        print("\n所有API测试通过! 地形处理器可以完美作为Python库使用。")
        print("\n主要API功能:")
        print("✓ 地形数据读取和处理")
        print("✓ 三维数组操作和保存")
        print("✓ 等高线图生成")
        print("✓ 地形分析和统计")
        print("✓ 可视化和图表生成")
        print("✓ 批量处理支持")
        print("✓ 自定义分析扩展")
    else:
        print("\n部分API测试失败，请检查错误信息。")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)