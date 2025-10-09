#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试地形数据
生成一个包含真实地形特征的TIF文件用于测试
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import matplotlib.pyplot as plt

def create_synthetic_terrain(width=300, height=300):
    """创建合成地形数据"""
    print("创建合成地形数据...")
    
    # 创建坐标网格
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)
    
    # 创建多层地形特征
    # 基础地形 - 缓慢变化的山丘
    base_terrain = 100 + 50 * np.sin(X * 0.5) * np.cos(Y * 0.3)
    
    # 添加山峰
    peak1 = 80 * np.exp(-((X - 3)**2 + (Y - 7)**2) / 2)
    peak2 = 60 * np.exp(-((X - 7)**2 + (Y - 3)**2) / 1.5)
    peak3 = 40 * np.exp(-((X - 8)**2 + (Y - 8)**2) / 1)
    
    # 添加山谷
    valley1 = -30 * np.exp(-((X - 2)**2 + (Y - 2)**2) / 3)
    valley2 = -25 * np.exp(-((X - 6)**2 + (Y - 6)**2) / 2)
    
    # 添加噪声模拟真实地形的复杂性
    noise = np.random.normal(0, 5, (height, width))
    
    # 组合所有特征
    terrain = base_terrain + peak1 + peak2 + peak3 + valley1 + valley2 + noise
    
    # 确保高程值为正数
    terrain = np.maximum(terrain, 10)
    
    print(f"地形数据统计:")
    print(f"  形状: {terrain.shape}")
    print(f"  最小高程: {np.min(terrain):.2f}m")
    print(f"  最大高程: {np.max(terrain):.2f}m")
    print(f"  平均高程: {np.mean(terrain):.2f}m")
    print(f"  高程范围: {np.max(terrain) - np.min(terrain):.2f}m")
    
    return terrain

def save_as_geotiff(terrain, filename="test_terrain.tif"):
    """将地形数据保存为GeoTIFF文件"""
    print(f"保存地形数据为: {filename}")
    
    height, width = terrain.shape
    
    # 定义地理边界 (使用北京附近的坐标)
    left, bottom, right, top = 116.2, 40.0, 116.3, 40.1
    
    # 创建仿射变换
    transform = from_bounds(left, bottom, right, top, width, height)
    
    # 定义坐标参考系统 (WGS84)
    crs = CRS.from_epsg(4326)
    
    # 保存为GeoTIFF
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=terrain.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'  # 使用LZW压缩
    ) as dst:
        dst.write(terrain, 1)
    
    print(f"GeoTIFF文件已保存: {filename}")
    return filename

def visualize_terrain(terrain, filename="terrain_preview.png"):
    """可视化地形数据"""
    print("生成地形可视化...")
    
    plt.figure(figsize=(15, 10))
    
    # 子图1: 地形高程图
    plt.subplot(2, 3, 1)
    im1 = plt.imshow(terrain, cmap='terrain', origin='lower')
    plt.title('地形高程图')
    plt.colorbar(im1, label='高程 (m)')
    
    # 子图2: 等高线图
    plt.subplot(2, 3, 2)
    contour_levels = np.linspace(np.min(terrain), np.max(terrain), 20)
    cs = plt.contour(terrain, levels=contour_levels, colors='black', linewidths=0.5)
    plt.contourf(terrain, levels=contour_levels, cmap='terrain', alpha=0.7)
    plt.title('等高线图')
    plt.colorbar(label='高程 (m)')
    
    # 子图3: 3D视图
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    x = np.arange(terrain.shape[1])
    y = np.arange(terrain.shape[0])
    X, Y = np.meshgrid(x, y)
    surf = ax3.plot_surface(X, Y, terrain, cmap='terrain', alpha=0.8)
    ax3.set_title('3D地形视图')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('高程 (m)')
    
    # 子图4: 高程分布直方图
    plt.subplot(2, 3, 4)
    plt.hist(terrain.flatten(), bins=50, alpha=0.7, color='green')
    plt.title('高程分布')
    plt.xlabel('高程 (m)')
    plt.ylabel('频次')
    
    # 子图5: 坡度分析
    plt.subplot(2, 3, 5)
    # 计算坡度
    gy, gx = np.gradient(terrain)
    slope = np.sqrt(gx**2 + gy**2)
    plt.imshow(slope, cmap='Reds', origin='lower')
    plt.title('坡度分析')
    plt.colorbar(label='坡度')
    
    # 子图6: 山体阴影效果
    plt.subplot(2, 3, 6)
    # 简单的山体阴影效果
    light_source = np.array([1, 1, 1])  # 光源方向
    light_source = light_source / np.linalg.norm(light_source)
    
    # 计算法向量
    gx, gy = np.gradient(terrain)
    normal = np.dstack([-gx, -gy, np.ones_like(terrain)])
    normal = normal / np.linalg.norm(normal, axis=2, keepdims=True)
    
    # 计算光照
    shading = np.sum(normal * light_source, axis=2)
    shading = np.clip(shading, 0, 1)
    
    plt.imshow(shading, cmap='gray', origin='lower')
    plt.title('山体阴影')
    plt.colorbar(label='光照强度')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"地形可视化已保存: {filename}")
    
    return filename

def verify_geotiff(filename):
    """验证生成的GeoTIFF文件"""
    print(f"验证GeoTIFF文件: {filename}")
    
    with rasterio.open(filename) as src:
        print(f"  驱动程序: {src.driver}")
        print(f"  数据类型: {src.dtypes[0]}")
        print(f"  图像尺寸: {src.width} x {src.height}")
        print(f"  坐标系: {src.crs}")
        print(f"  边界: {src.bounds}")
        
        # 读取数据
        data = src.read(1)
        print(f"  数据统计:")
        print(f"    最小值: {np.min(data):.2f}")
        print(f"    最大值: {np.max(data):.2f}")
        print(f"    平均值: {np.mean(data):.2f}")
        print(f"    非零值数量: {np.sum(data != 0)}")
    
    print("GeoTIFF文件验证完成!")

def main():
    """主函数"""
    print("创建测试地形数据")
    print("=" * 50)
    
    # 创建合成地形
    terrain = create_synthetic_terrain(300, 300)
    
    # 保存为GeoTIFF
    tif_filename = save_as_geotiff(terrain, "test_terrain.tif")
    
    # 生成可视化
    viz_filename = visualize_terrain(terrain, "terrain_preview.png")
    
    # 验证文件
    verify_geotiff(tif_filename)
    
    print("\n测试数据创建完成!")
    print(f"生成的文件:")
    print(f"  - {tif_filename} (GeoTIFF地形文件)")
    print(f"  - {viz_filename} (地形可视化)")
    
    return tif_filename

if __name__ == "__main__":
    main()