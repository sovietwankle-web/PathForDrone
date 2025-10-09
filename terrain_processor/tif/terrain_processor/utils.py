#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
提供各种辅助功能和计算函数
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import rasterio.transform
import warnings


class TerrainUtils:
    """地形处理工具类"""
    
    def __init__(self):
        """初始化工具类"""
        pass
    
    def calculate_statistics(self, data: np.ndarray, 
                           nodata_value: Optional[float] = None) -> Dict[str, float]:
        """
        计算地形统计信息
        
        Args:
            data: 地形数据
            nodata_value: NoData值
            
        Returns:
            dict: 统计信息字典
        """
        # 处理NoData值
        if nodata_value is not None:
            mask = (data != nodata_value) & (~np.isnan(data))
        else:
            mask = ~np.isnan(data)
        
        valid_data = data[mask]
        total_pixels = data.size
        valid_pixels = len(valid_data)
        
        if valid_pixels == 0:
            return {
                'total_pixels': total_pixels,
                'valid_pixels': 0,
                'invalid_pixels': total_pixels,
                'valid_percentage': 0.0,
                'min_elevation': np.nan,
                'max_elevation': np.nan,
                'mean_elevation': np.nan,
                'median_elevation': np.nan,
                'std_elevation': np.nan,
                'elevation_range': np.nan,
                'percentile_25': np.nan,
                'percentile_75': np.nan
            }
        
        # 计算统计值
        stats = {
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixels,
            'invalid_pixels': total_pixels - valid_pixels,
            'valid_percentage': (valid_pixels / total_pixels) * 100,
            'min_elevation': float(np.min(valid_data)),
            'max_elevation': float(np.max(valid_data)),
            'mean_elevation': float(np.mean(valid_data)),
            'median_elevation': float(np.median(valid_data)),
            'std_elevation': float(np.std(valid_data)),
            'elevation_range': float(np.max(valid_data) - np.min(valid_data)),
            'percentile_25': float(np.percentile(valid_data, 25)),
            'percentile_75': float(np.percentile(valid_data, 75))
        }
        
        return stats
    
    def geographic_to_pixel(self, lon: float, lat: float, 
                          transform: rasterio.transform.Affine) -> Tuple[float, float]:
        """
        将地理坐标转换为像素坐标
        
        Args:
            lon: 经度
            lat: 纬度
            transform: 仿射变换矩阵
            
        Returns:
            tuple: (x_pixel, y_pixel)
        """
        try:
            # 使用rasterio的逆变换
            col, row = ~transform * (lon, lat)
            return col, row
        except Exception as e:
            warnings.warn(f"坐标转换失败: {e}")
            return 0.0, 0.0
    
    def pixel_to_geographic(self, x: float, y: float,
                          transform: rasterio.transform.Affine) -> Tuple[float, float]:
        """
        将像素坐标转换为地理坐标
        
        Args:
            x: 像素X坐标
            y: 像素Y坐标
            transform: 仿射变换矩阵
            
        Returns:
            tuple: (longitude, latitude)
        """
        try:
            # 使用rasterio的变换
            lon, lat = transform * (x, y)
            return lon, lat
        except Exception as e:
            warnings.warn(f"坐标转换失败: {e}")
            return 0.0, 0.0
    
    def get_elevation_at_pixel(self, data: np.ndarray, x: int, y: int) -> float:
        """
        获取指定像素位置的高程值
        
        Args:
            data: 地形数据
            x: X坐标（列）
            y: Y坐标（行）
            
        Returns:
            float: 高程值
        """
        if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
            return float(data[y, x])
        else:
            return np.nan
    
    def get_elevation_profile(self, data: np.ndarray,
                            start_point: Tuple[float, float],
                            end_point: Tuple[float, float],
                            num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取两点间的高程剖面
        
        Args:
            data: 地形数据
            start_point: 起始点 (x, y)
            end_point: 结束点 (x, y)
            num_points: 采样点数量
            
        Returns:
            tuple: (距离数组, 高程数组)
        """
        x1, y1 = start_point
        x2, y2 = end_point
        
        # 生成剖面线上的点
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)
        
        # 计算距离
        total_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances = np.linspace(0, total_distance, num_points)
        
        # 获取高程值（使用双线性插值）
        elevations = []
        for x, y in zip(x_coords, y_coords):
            elevation = self._bilinear_interpolation(data, x, y)
            elevations.append(elevation)
        
        return distances, np.array(elevations)
    
    def _bilinear_interpolation(self, data: np.ndarray, x: float, y: float) -> float:
        """
        双线性插值获取指定位置的值
        
        Args:
            data: 数据数组
            x: X坐标
            y: Y坐标
            
        Returns:
            float: 插值结果
        """
        # 边界检查
        if x < 0 or x >= data.shape[1] - 1 or y < 0 or y >= data.shape[0] - 1:
            return np.nan
        
        # 获取四个邻近点的坐标
        x1, y1 = int(np.floor(x)), int(np.floor(y))
        x2, y2 = x1 + 1, y1 + 1
        
        # 获取四个邻近点的值
        q11 = data[y1, x1]
        q12 = data[y2, x1]
        q21 = data[y1, x2]
        q22 = data[y2, x2]
        
        # 检查是否有NaN值
        if np.isnan([q11, q12, q21, q22]).any():
            return np.nan
        
        # 双线性插值
        dx = x - x1
        dy = y - y1
        
        result = (q11 * (1 - dx) * (1 - dy) +
                 q21 * dx * (1 - dy) +
                 q12 * (1 - dx) * dy +
                 q22 * dx * dy)
        
        return result
    
    def create_hillshade(self, data: np.ndarray, 
                        azimuth: float = 315.0, 
                        altitude: float = 45.0,
                        z_factor: float = 1.0) -> np.ndarray:
        """
        创建山体阴影效果
        
        Args:
            data: 高程数据
            azimuth: 光源方位角（度）
            altitude: 光源高度角（度）
            z_factor: 高程缩放因子
            
        Returns:
            numpy.ndarray: 山体阴影数组
        """
        # 转换角度为弧度
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)
        
        # 计算梯度
        gy, gx = np.gradient(data * z_factor)
        
        # 计算坡度和坡向
        slope = np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(-gx, gy)
        
        # 计算山体阴影
        hillshade = (np.cos(altitude_rad) * np.cos(slope) +
                    np.sin(altitude_rad) * np.sin(slope) *
                    np.cos(azimuth_rad - aspect))
        
        # 归一化到0-255范围
        hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)
        
        return hillshade
    
    def calculate_slope_aspect(self, data: np.ndarray, 
                             pixel_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算坡度和坡向
        
        Args:
            data: 高程数据
            pixel_size: 像素大小（米）
            
        Returns:
            tuple: (坡度数组（度）, 坡向数组（度）)
        """
        # 计算梯度
        gy, gx = np.gradient(data, pixel_size)
        
        # 计算坡度（弧度转度）
        slope = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_degrees = np.degrees(slope)
        
        # 计算坡向（弧度转度，调整到0-360范围）
        aspect = np.arctan2(-gx, gy)
        aspect_degrees = np.degrees(aspect)
        aspect_degrees = (aspect_degrees + 360) % 360
        
        return slope_degrees, aspect_degrees
    
    def calculate_curvature(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算地形曲率
        
        Args:
            data: 高程数据
            
        Returns:
            dict: 包含不同类型曲率的字典
        """
        # 计算一阶和二阶偏导数
        gy, gx = np.gradient(data)
        gxy, gxx = np.gradient(gx)
        gyy, gyx = np.gradient(gy)
        
        # 计算平面曲率（plan curvature）
        denominator = (gx**2 + gy**2)**(3/2)
        plan_curvature = np.where(denominator != 0,
                                (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / denominator,
                                0)
        
        # 计算剖面曲率（profile curvature）
        profile_curvature = np.where(denominator != 0,
                                   (gxx * gx**2 + 2 * gxy * gx * gy + gyy * gy**2) / denominator,
                                   0)
        
        # 计算总曲率（mean curvature）
        mean_curvature = (plan_curvature + profile_curvature) / 2
        
        return {
            'plan_curvature': plan_curvature,
            'profile_curvature': profile_curvature,
            'mean_curvature': mean_curvature
        }
    
    def calculate_roughness(self, data: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        计算地形粗糙度
        
        Args:
            data: 高程数据
            window_size: 窗口大小
            
        Returns:
            numpy.ndarray: 粗糙度数组
        """
        from scipy import ndimage
        
        # 计算局部标准差作为粗糙度指标
        kernel = np.ones((window_size, window_size))
        
        # 计算局部均值
        local_mean = ndimage.uniform_filter(data, size=window_size)
        
        # 计算局部方差
        local_variance = ndimage.uniform_filter(data**2, size=window_size) - local_mean**2
        
        # 粗糙度为标准差
        roughness = np.sqrt(np.maximum(local_variance, 0))
        
        return roughness
    
    def detect_peaks_valleys(self, data: np.ndarray, 
                           min_prominence: float = None) -> Dict[str, np.ndarray]:
        """
        检测山峰和山谷
        
        Args:
            data: 高程数据
            min_prominence: 最小突出度
            
        Returns:
            dict: 包含山峰和山谷位置的字典
        """
        if min_prominence is None:
            data_range = np.nanmax(data) - np.nanmin(data)
            min_prominence = data_range * 0.1  # 默认为数据范围的10%
        
        # 使用局部最大值和最小值检测
        from scipy.ndimage import maximum_filter, minimum_filter
        
        # 检测局部最大值（山峰）
        local_maxima = maximum_filter(data, size=3) == data
        
        # 检测局部最小值（山谷）
        local_minima = minimum_filter(data, size=3) == data
        
        # 过滤掉边界和不显著的点
        h, w = data.shape
        local_maxima[0, :] = local_maxima[-1, :] = False
        local_maxima[:, 0] = local_maxima[:, -1] = False
        local_minima[0, :] = local_minima[-1, :] = False
        local_minima[:, 0] = local_minima[:, -1] = False
        
        # 应用突出度阈值
        peak_coords = np.where(local_maxima)
        valley_coords = np.where(local_minima)
        
        # 过滤突出度不足的点
        filtered_peaks = []
        for i, j in zip(peak_coords[0], peak_coords[1]):
            # 检查周围区域的高程差
            neighborhood = data[max(0, i-2):min(h, i+3), max(0, j-2):min(w, j+3)]
            if data[i, j] - np.nanmin(neighborhood) >= min_prominence:
                filtered_peaks.append((i, j))
        
        filtered_valleys = []
        for i, j in zip(valley_coords[0], valley_coords[1]):
            # 检查周围区域的高程差
            neighborhood = data[max(0, i-2):min(h, i+3), max(0, j-2):min(w, j+3)]
            if np.nanmax(neighborhood) - data[i, j] >= min_prominence:
                filtered_valleys.append((i, j))
        
        return {
            'peaks': np.array(filtered_peaks) if filtered_peaks else np.array([]).reshape(0, 2),
            'valleys': np.array(filtered_valleys) if filtered_valleys else np.array([]).reshape(0, 2)
        }
    
    def calculate_drainage_network(self, data: np.ndarray) -> np.ndarray:
        """
        计算简单的排水网络
        
        Args:
            data: 高程数据
            
        Returns:
            numpy.ndarray: 流向数组
        """
        # 计算梯度
        gy, gx = np.gradient(data)
        
        # 计算流向（8个方向）
        flow_direction = np.zeros_like(data, dtype=np.uint8)
        
        # 定义8个方向的偏移
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # 上左、上、上右
            (0, -1),           (0, 1),   # 左、右
            (1, -1),  (1, 0),  (1, 1)    # 下左、下、下右
        ]
        
        h, w = data.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if np.isnan(data[i, j]):
                    continue
                
                current_elevation = data[i, j]
                max_slope = 0
                flow_dir = 0
                
                # 检查8个方向
                for idx, (di, dj) in enumerate(directions):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and not np.isnan(data[ni, nj]):
                        # 计算坡度
                        elevation_diff = current_elevation - data[ni, nj]
                        distance = np.sqrt(di**2 + dj**2)
                        slope = elevation_diff / distance
                        
                        if slope > max_slope:
                            max_slope = slope
                            flow_dir = idx + 1  # 1-8编码
                
                flow_direction[i, j] = flow_dir
        
        return flow_direction
    
    def validate_data_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            data: 地形数据
            
        Returns:
            dict: 数据质量报告
        """
        report = {
            'data_shape': data.shape,
            'data_type': str(data.dtype),
            'total_pixels': data.size,
            'issues': [],
            'warnings': [],
            'quality_score': 100  # 满分100
        }
        
        # 检查有效数据比例
        valid_mask = ~np.isnan(data)
        valid_count = np.sum(valid_mask)
        valid_percentage = (valid_count / data.size) * 100
        
        report['valid_pixels'] = valid_count
        report['valid_percentage'] = valid_percentage
        
        if valid_percentage < 50:
            report['issues'].append(f"有效数据比例过低: {valid_percentage:.1f}%")
            report['quality_score'] -= 30
        elif valid_percentage < 80:
            report['warnings'].append(f"有效数据比例较低: {valid_percentage:.1f}%")
            report['quality_score'] -= 10
        
        # 检查数据范围
        if valid_count > 0:
            valid_data = data[valid_mask]
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            data_range = data_max - data_min
            
            report['elevation_range'] = {
                'min': float(data_min),
                'max': float(data_max),
                'range': float(data_range)
            }
            
            # 检查异常值
            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = np.sum((valid_data < lower_bound) | (valid_data > upper_bound))
            outlier_percentage = (outliers / valid_count) * 100
            
            report['outliers'] = {
                'count': int(outliers),
                'percentage': float(outlier_percentage)
            }
            
            if outlier_percentage > 5:
                report['warnings'].append(f"异常值比例较高: {outlier_percentage:.1f}%")
                report['quality_score'] -= 5
        
        # 检查数据连续性
        if valid_count > 0:
            # 检查是否有大片的NoData区域
            from scipy import ndimage
            
            # 标记连通的NoData区域
            invalid_mask = ~valid_mask
            labeled_regions, num_regions = ndimage.label(invalid_mask)
            
            if num_regions > 0:
                region_sizes = [np.sum(labeled_regions == i) for i in range(1, num_regions + 1)]
                max_invalid_region = max(region_sizes)
                max_invalid_percentage = (max_invalid_region / data.size) * 100
                
                report['largest_gap'] = {
                    'size': int(max_invalid_region),
                    'percentage': float(max_invalid_percentage)
                }
                
                if max_invalid_percentage > 10:
                    report['issues'].append(f"存在大片无效区域: {max_invalid_percentage:.1f}%")
                    report['quality_score'] -= 15
        
        # 总体质量评估
        if report['quality_score'] >= 90:
            report['quality_level'] = 'Excellent'
        elif report['quality_score'] >= 70:
            report['quality_level'] = 'Good'
        elif report['quality_score'] >= 50:
            report['quality_level'] = 'Fair'
        else:
            report['quality_level'] = 'Poor'
        
        return report