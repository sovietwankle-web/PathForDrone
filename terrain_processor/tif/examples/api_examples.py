#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API调用示例
演示如何在其他Python程序中调用terrain_processor库
"""

import sys
from pathlib import Path
import numpy as np

# 添加父目录到路径以便导入terrain_processor
sys.path.insert(0, str(Path(__file__).parent.parent))

from terrain_processor import TerrainProcessor


def example_1_simple_integration():
    """示例1: 简单集成 - 在其他程序中快速获取地形数据"""
    print("=== 示例1: 简单集成 ===")
    
    def get_terrain_data(tif_file_path):
        """获取地形数据的简单函数"""
        try:
            processor = TerrainProcessor(tif_file_path)
            return processor.load_terrain_data()
        except Exception as e:
            print(f"加载地形数据失败: {e}")
            return None
    
    # 使用示例
    tif_file = "../test_terrain.tif"
    terrain_array = get_terrain_data(tif_file)
    
    if terrain_array is not None:
        print(f"✓ 成功获取地形数据，形状: {terrain_array.shape}")
        print(f"  高程范围: {np.nanmin(terrain_array):.2f} - {np.nanmax(terrain_array):.2f} m")
    else:
        print("✗ 获取地形数据失败")


def example_2_data_analysis():
    """示例2: 数据分析 - 分析地形特征"""
    print("\n=== 示例2: 地形数据分析 ===")
    
    class TerrainAnalyzer:
        """地形分析器类"""
        
        def __init__(self, tif_file_path):
            self.processor = TerrainProcessor(tif_file_path)
            self.data = self.processor.load_terrain_data()
            self.stats = self.processor.get_terrain_statistics()
        
        def analyze_terrain_features(self):
            """分析地形特征"""
            features = {}
            
            # 基本统计
            features['elevation_stats'] = {
                'min': self.stats['min_elevation'],
                'max': self.stats['max_elevation'],
                'mean': self.stats['mean_elevation'],
                'range': self.stats['elevation_range']
            }
            
            # 地形分类
            elevation_range = self.stats['elevation_range']
            if elevation_range < 50:
                terrain_type = "平原"
            elif elevation_range < 200:
                terrain_type = "丘陵"
            elif elevation_range < 500:
                terrain_type = "低山"
            else:
                terrain_type = "高山"
            
            features['terrain_type'] = terrain_type
            
            # 坡度分析
            slope_data = self.processor.utils.calculate_slope_aspect(self.data)[0]
            features['slope_stats'] = {
                'mean_slope': float(np.nanmean(slope_data)),
                'max_slope': float(np.nanmax(slope_data))
            }
            
            return features
        
        def find_peaks_and_valleys(self):
            """寻找山峰和山谷"""
            peaks_valleys = self.processor.utils.detect_peaks_valleys(self.data)
            return {
                'peaks_count': len(peaks_valleys['peaks']),
                'valleys_count': len(peaks_valleys['valleys']),
                'peaks_locations': peaks_valleys['peaks'].tolist(),
                'valleys_locations': peaks_valleys['valleys'].tolist()
            }
    
    # 使用示例
    try:
        analyzer = TerrainAnalyzer("../test_terrain.tif")
        
        # 分析地形特征
        features = analyzer.analyze_terrain_features()
        print(f"地形类型: {features['terrain_type']}")
        print(f"高程范围: {features['elevation_stats']['range']:.2f} m")
        print(f"平均坡度: {features['slope_stats']['mean_slope']:.2f}°")
        
        # 寻找山峰和山谷
        peaks_valleys = analyzer.find_peaks_and_valleys()
        print(f"山峰数量: {peaks_valleys['peaks_count']}")
        print(f"山谷数量: {peaks_valleys['valleys_count']}")
        
    except Exception as e:
        print(f"分析失败: {e}")


def example_3_batch_processing():
    """示例3: 批量处理 - 处理多个地形文件"""
    print("\n=== 示例3: 批量处理 ===")
    
    class BatchTerrainProcessor:
        """批量地形处理器"""
        
        def __init__(self):
            self.results = []
        
        def process_file(self, tif_file_path, output_dir):
            """处理单个文件"""
            try:
                processor = TerrainProcessor(tif_file_path)
                
                # 获取基本信息
                info = processor.get_terrain_info()
                stats = processor.get_terrain_statistics()
                
                # 生成输出
                file_stem = Path(tif_file_path).stem
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                
                # 保存数组
                array_path = processor.save_array(
                    output_path / f"{file_stem}_data.npy", "npy"
                )
                
                # 生成等高线图
                contour_path = processor.generate_contour_image(
                    output_path / f"{file_stem}_contour.png"
                )
                
                result = {
                    'file': tif_file_path,
                    'status': 'success',
                    'info': info,
                    'stats': stats,
                    'outputs': {
                        'array': array_path,
                        'contour': contour_path
                    }
                }
                
                return result
                
            except Exception as e:
                return {
                    'file': tif_file_path,
                    'status': 'error',
                    'error': str(e)
                }
        
        def process_multiple_files(self, file_list, output_dir):
            """批量处理多个文件"""
            results = []
            
            for file_path in file_list:
                print(f"处理文件: {file_path}")
                result = self.process_file(file_path, output_dir)
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"  ✓ 成功")
                else:
                    print(f"  ✗ 失败: {result['error']}")
            
            return results
    
    # 使用示例
    batch_processor = BatchTerrainProcessor()
    
    # 模拟文件列表（实际使用时替换为真实文件路径）
    file_list = ["../test_terrain.tif"]  # 可以添加更多文件
    
    results = batch_processor.process_multiple_files(file_list, "../output/batch")
    
    print(f"\n批量处理完成，共处理 {len(results)} 个文件")
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"成功: {success_count}, 失败: {len(results) - success_count}")


def example_4_custom_visualization():
    """示例4: 自定义可视化"""
    print("\n=== 示例4: 自定义可视化 ===")
    
    def create_custom_terrain_map(tif_file_path, output_path):
        """创建自定义地形图"""
        processor = TerrainProcessor(tif_file_path)
        data = processor.load_terrain_data()
        stats = processor.get_terrain_statistics()
        
        # 使用自定义参数生成等高线图
        contour_path = processor.generate_contour_image(
            output_path,
            levels=25,  # 25条等高线
            colormap='gist_earth',  # 地球色彩方案
            figsize=(15, 10),  # 大尺寸
            dpi=200,  # 高分辨率
            title="自定义地形等高线图",
            show_stats=True,
            show_grid=True,
            add_labels=True
        )
        
        return contour_path
    
    # 使用示例
    try:
        custom_map_path = create_custom_terrain_map(
            "../test_terrain.tif",
            "../output/custom_terrain_map.png"
        )
        print(f"✓ 自定义地形图已生成: {custom_map_path}")
    except Exception as e:
        print(f"✗ 生成失败: {e}")


def example_5_integration_with_other_libraries():
    """示例5: 与其他库集成"""
    print("\n=== 示例5: 与其他库集成 ===")
    
    def terrain_data_to_pandas():
        """将地形数据转换为pandas DataFrame"""
        try:
            import pandas as pd
            
            processor = TerrainProcessor("../test_terrain.tif")
            data = processor.load_terrain_data()
            
            # 创建坐标网格
            height, width = data.shape
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            # 展平数据
            df = pd.DataFrame({
                'x': x_coords.flatten(),
                'y': y_coords.flatten(),
                'elevation': data.flatten()
            })
            
            # 移除NaN值
            df = df.dropna()
            
            print(f"✓ 转换为DataFrame，形状: {df.shape}")
            print(f"  前5行数据:")
            print(df.head())
            
            return df
            
        except ImportError:
            print("✗ pandas未安装，跳过此示例")
            return None
        except Exception as e:
            print(f"✗ 转换失败: {e}")
            return None
    
    def save_as_geojson():
        """保存为GeoJSON格式（需要geopandas）"""
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            
            processor = TerrainProcessor("../test_terrain.tif")
            data = processor.load_terrain_data()
            info = processor.get_terrain_info()
            
            # 采样数据点（避免文件过大）
            step = max(1, min(data.shape) // 20)
            sampled_data = data[::step, ::step]
            
            # 创建点几何
            points = []
            elevations = []
            
            for i in range(sampled_data.shape[0]):
                for j in range(sampled_data.shape[1]):
                    if not np.isnan(sampled_data[i, j]):
                        # 转换为地理坐标（简化版本）
                        lon = info['bounds'].left + (j * step) * (info['bounds'].right - info['bounds'].left) / data.shape[1]
                        lat = info['bounds'].top - (i * step) * (info['bounds'].top - info['bounds'].bottom) / data.shape[0]
                        
                        points.append(Point(lon, lat))
                        elevations.append(sampled_data[i, j])
            
            # 创建GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'elevation': elevations,
                'geometry': points
            })
            
            # 保存为GeoJSON
            output_path = "../output/terrain_points.geojson"
            gdf.to_file(output_path, driver='GeoJSON')
            
            print(f"✓ 保存为GeoJSON: {output_path}")
            print(f"  点数量: {len(gdf)}")
            
        except ImportError:
            print("✗ geopandas未安装，跳过此示例")
        except Exception as e:
            print(f"✗ 保存失败: {e}")
    
    # 运行示例
    terrain_data_to_pandas()
    save_as_geojson()


def main():
    """主函数"""
    print("地形处理器API调用示例")
    print("=" * 60)
    
    # 检查测试数据
    test_file = Path("../test_terrain.tif")
    if not test_file.exists():
        print("⚠️  测试数据不存在，请先运行 create_test_data.py")
        return
    
    # 运行所有示例
    example_1_simple_integration()
    example_2_data_analysis()
    example_3_batch_processing()
    example_4_custom_visualization()
    example_5_integration_with_other_libraries()
    
    print("\n" + "=" * 60)
    print("✓ 所有API示例完成!")
    print("这些示例展示了如何在您的Python程序中集成terrain_processor库。")


if __name__ == "__main__":
    main()