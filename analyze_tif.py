#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIF地形文件分析脚本
用于分析TIF文件的格式、内容和元数据
"""

import os
import sys

def analyze_tif_basic():
    """基础文件信息分析"""
    tif_file = "Trend_未命名路径_GPX1(1).tif"
    
    if not os.path.exists(tif_file):
        print(f"错误：文件 {tif_file} 不存在")
        return False
    
    # 文件基本信息
    file_size = os.path.getsize(tif_file)
    print(f"文件名: {tif_file}")
    print(f"文件大小: {file_size:,} 字节 ({file_size/1024/1024:.2f} MB)")
    
    return True

def analyze_with_pillow():
    """使用Pillow库分析图像基本信息"""
    try:
        from PIL import Image
        
        tif_file = "Trend_未命名路径_GPX1(1).tif"
        
        with Image.open(tif_file) as img:
            print("\n=== Pillow 图像信息 ===")
            print(f"图像模式: {img.mode}")
            print(f"图像尺寸: {img.size} (宽x高)")
            print(f"图像格式: {img.format}")
            
            # 获取图像信息
            if hasattr(img, 'info'):
                print(f"图像信息: {img.info}")
            
            # 尝试获取像素值范围
            if img.mode in ['L', 'I', 'F']:
                extrema = img.getextrema()
                print(f"像素值范围: {extrema}")
        
        return True
        
    except ImportError:
        print("Pillow库未安装，跳过Pillow分析")
        return False
    except Exception as e:
        print(f"Pillow分析出错: {e}")
        return False

def analyze_with_rasterio():
    """使用rasterio库分析地理信息"""
    try:
        import rasterio
        import numpy as np
        
        tif_file = "Trend_未命名路径_GPX1(1).tif"
        
        with rasterio.open(tif_file) as src:
            print("\n=== Rasterio 地理信息 ===")
            print(f"驱动程序: {src.driver}")
            print(f"数据类型: {src.dtypes}")
            print(f"波段数量: {src.count}")
            print(f"图像尺寸: {src.width} x {src.height}")
            print(f"坐标参考系统: {src.crs}")
            print(f"仿射变换: {src.transform}")
            print(f"边界框: {src.bounds}")
            
            # 读取第一个波段的数据
            if src.count > 0:
                band1 = src.read(1)
                print(f"\n=== 数据统计 ===")
                print(f"数组形状: {band1.shape}")
                print(f"数据类型: {band1.dtype}")
                print(f"最小值: {np.nanmin(band1)}")
                print(f"最大值: {np.nanmax(band1)}")
                print(f"平均值: {np.nanmean(band1):.2f}")
                print(f"标准差: {np.nanstd(band1):.2f}")
                print(f"无效值数量: {np.sum(np.isnan(band1))}")
                
                # 检查是否有NoData值
                if src.nodata is not None:
                    print(f"NoData值: {src.nodata}")
                    nodata_count = np.sum(band1 == src.nodata)
                    print(f"NoData像素数量: {nodata_count}")
        
        return True
        
    except ImportError:
        print("rasterio库未安装，跳过rasterio分析")
        return False
    except Exception as e:
        print(f"rasterio分析出错: {e}")
        return False

def analyze_with_gdal():
    """使用GDAL库分析地理信息"""
    try:
        from osgeo import gdal
        import numpy as np
        
        tif_file = "Trend_未命名路径_GPX1(1).tif"
        
        # 打开数据集
        dataset = gdal.Open(tif_file)
        if dataset is None:
            print("GDAL无法打开文件")
            return False
        
        print("\n=== GDAL 地理信息 ===")
        print(f"驱动程序: {dataset.GetDriver().ShortName}")
        print(f"图像尺寸: {dataset.RasterXSize} x {dataset.RasterYSize}")
        print(f"波段数量: {dataset.RasterCount}")
        
        # 地理变换信息
        geotransform = dataset.GetGeoTransform()
        if geotransform:
            print(f"地理变换: {geotransform}")
            print(f"左上角坐标: ({geotransform[0]}, {geotransform[3]})")
            print(f"像素分辨率: ({geotransform[1]}, {geotransform[5]})")
        
        # 投影信息
        projection = dataset.GetProjection()
        if projection:
            print(f"投影信息: {projection}")
        
        # 波段信息
        for i in range(dataset.RasterCount):
            band = dataset.GetRasterBand(i + 1)
            print(f"\n波段 {i + 1}:")
            print(f"  数据类型: {gdal.GetDataTypeName(band.DataType)}")
            print(f"  NoData值: {band.GetNoDataValue()}")
            
            # 统计信息
            stats = band.GetStatistics(True, True)
            if stats:
                print(f"  最小值: {stats[0]}")
                print(f"  最大值: {stats[1]}")
                print(f"  平均值: {stats[2]:.2f}")
                print(f"  标准差: {stats[3]:.2f}")
        
        dataset = None  # 关闭数据集
        return True
        
    except ImportError:
        print("GDAL库未安装，跳过GDAL分析")
        return False
    except Exception as e:
        print(f"GDAL分析出错: {e}")
        return False

def main():
    """主函数"""
    print("TIF地形文件分析")
    print("=" * 50)
    
    # 基础文件信息
    if not analyze_tif_basic():
        return
    
    # 尝试不同的库进行分析
    success_count = 0
    
    if analyze_with_pillow():
        success_count += 1
    
    if analyze_with_rasterio():
        success_count += 1
    
    if analyze_with_gdal():
        success_count += 1
    
    if success_count == 0:
        print("\n警告：没有可用的地理数据处理库")
        print("建议安装: pip install rasterio pillow")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()