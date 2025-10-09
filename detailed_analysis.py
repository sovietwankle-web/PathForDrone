#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的TIF文件分析脚本
深入分析数据内容和可能的问题
"""

import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt

def detailed_analysis():
    """详细分析TIF文件"""
    tif_file = "Trend_未命名路径_GPX1(1).tif"
    
    print("=== 详细数据分析 ===")
    
    # 使用rasterio读取
    with rasterio.open(tif_file) as src:
        print(f"文件驱动: {src.driver}")
        print(f"数据类型: {src.dtypes[0]}")
        print(f"NoData值: {src.nodata}")
        print(f"坐标系: {src.crs}")
        print(f"变换矩阵: {src.transform}")
        
        # 读取所有数据
        data = src.read(1)
        print(f"\n数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        
        # 检查唯一值
        unique_values = np.unique(data)
        print(f"唯一值数量: {len(unique_values)}")
        print(f"前10个唯一值: {unique_values[:10]}")
        
        # 检查是否所有值都是0或NoData
        if src.nodata is not None:
            non_nodata_mask = data != src.nodata
            non_nodata_values = data[non_nodata_mask]
            print(f"非NoData值数量: {len(non_nodata_values)}")
            if len(non_nodata_values) > 0:
                print(f"非NoData值范围: {np.min(non_nodata_values)} 到 {np.max(non_nodata_values)}")
        
        # 检查是否有非零值
        non_zero_mask = data != 0
        non_zero_count = np.sum(non_zero_mask)
        print(f"非零值数量: {non_zero_count}")
        
        if non_zero_count > 0:
            non_zero_values = data[non_zero_mask]
            print(f"非零值范围: {np.min(non_zero_values)} 到 {np.max(non_zero_values)}")
        
        # 创建数据可视化
        plt.figure(figsize=(12, 8))
        
        # 子图1: 原始数据
        plt.subplot(2, 2, 1)
        plt.imshow(data, cmap='terrain')
        plt.title('原始数据')
        plt.colorbar()
        
        # 子图2: 数据直方图
        plt.subplot(2, 2, 2)
        plt.hist(data.flatten(), bins=50, alpha=0.7)
        plt.title('数据分布直方图')
        plt.xlabel('值')
        plt.ylabel('频次')
        
        # 子图3: 非零值掩码
        plt.subplot(2, 2, 3)
        plt.imshow(non_zero_mask, cmap='binary')
        plt.title('非零值分布')
        plt.colorbar()
        
        # 子图4: 如果有非零值，显示其分布
        plt.subplot(2, 2, 4)
        if non_zero_count > 0:
            # 只显示非零区域
            masked_data = np.where(data != 0, data, np.nan)
            plt.imshow(masked_data, cmap='terrain')
            plt.title('非零值区域')
            plt.colorbar()
        else:
            plt.text(0.5, 0.5, '没有非零值', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('无有效数据')
        
        plt.tight_layout()
        plt.savefig('tif_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n可视化图片已保存为: tif_analysis.png")
        
        return data, src.transform, src.crs

def check_file_integrity():
    """检查文件完整性"""
    tif_file = "Trend_未命名路径_GPX1(1).tif"
    
    print("\n=== 文件完整性检查 ===")
    
    try:
        # 尝试用不同方式读取
        with Image.open(tif_file) as img:
            print(f"PIL可以正常打开文件")
            print(f"图像模式: {img.mode}")
            print(f"是否有透明度: {img.mode in ['RGBA', 'LA']}")
            
            # 转换为数组查看
            img_array = np.array(img)
            print(f"PIL数组形状: {img_array.shape}")
            print(f"PIL数组数据类型: {img_array.dtype}")
            print(f"PIL数组值范围: {np.min(img_array)} 到 {np.max(img_array)}")
            
    except Exception as e:
        print(f"PIL读取出错: {e}")
    
    try:
        with rasterio.open(tif_file) as src:
            print(f"Rasterio可以正常打开文件")
            
            # 检查元数据
            print(f"元数据: {src.meta}")
            
            # 尝试读取不同区域
            window = rasterio.windows.Window(0, 0, 50, 50)  # 读取左上角50x50区域
            sample_data = src.read(1, window=window)
            print(f"样本区域数据: {sample_data}")
            
    except Exception as e:
        print(f"Rasterio读取出错: {e}")

def suggest_solutions():
    """根据分析结果提供解决方案建议"""
    print("\n=== 问题分析和建议 ===")
    print("根据分析结果，该TIF文件存在以下情况：")
    print("1. 所有像素值都是0.0")
    print("2. 文件大小较小(0.38MB)，但尺寸为251x262像素")
    print("3. 坐标系为EPSG:4326 (WGS84地理坐标系)")
    print("4. 位置在北京附近 (116.24°E, 40.04°N)")
    
    print("\n可能的原因：")
    print("- 文件可能是空的或损坏的")
    print("- 数据可能使用了特殊的编码或压缩")
    print("- 可能需要特定的NoData值处理")
    print("- 文件可能是轨迹文件而非地形高程文件")
    
    print("\n建议的处理方案：")
    print("1. 创建模拟地形数据用于测试")
    print("2. 设计程序时考虑处理空数据的情况")
    print("3. 添加数据验证和错误处理机制")
    print("4. 支持多种数据格式和异常情况")

def main():
    """主函数"""
    print("TIF文件详细分析")
    print("=" * 50)
    
    # 详细分析
    data, transform, crs = detailed_analysis()
    
    # 文件完整性检查
    check_file_integrity()
    
    # 提供建议
    suggest_solutions()
    
    print("\n分析完成！")
    return data, transform, crs

if __name__ == "__main__":
    main()