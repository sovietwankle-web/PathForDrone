#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心TerrainProcessor类
提供主要的API接口供其他Python程序调用
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path

from .reader import TIFReader
from .processor import DataProcessor
from .visualizer import ContourGenerator
from .utils import TerrainUtils


class TerrainProcessor:
    """
    地形数据处理器主类
    
    提供完整的地形数据处理功能，包括：
    - TIF文件读取
    - 数据处理和转换
    - 等高线生成
    - 统计分析
    """
    
    def __init__(self, tif_file_path: Union[str, Path]):
        """
        初始化地形处理器
        
        Args:
            tif_file_path: TIF文件路径
        """
        self.tif_file_path = Path(tif_file_path)
        if not self.tif_file_path.exists():
            raise FileNotFoundError(f"TIF文件不存在: {tif_file_path}")
        
        # 初始化组件
        self.reader = TIFReader(self.tif_file_path)
        self.processor = DataProcessor()
        self.visualizer = ContourGenerator()
        self.utils = TerrainUtils()
        
        # 缓存数据
        self._terrain_data = None
        self._metadata = None
        self._statistics = None
    
    def load_terrain_data(self, force_reload: bool = False) -> np.ndarray:
        """
        加载地形数据为三维数组
        
        Args:
            force_reload: 是否强制重新加载数据
            
        Returns:
            numpy.ndarray: 地形高程数据数组
        """
        if self._terrain_data is None or force_reload:
            self._terrain_data, self._metadata = self.reader.read_tif()
            
        return self._terrain_data
    
    def get_terrain_info(self) -> Dict[str, Any]:
        """
        获取地形文件的基本信息
        
        Returns:
            dict: 包含地形信息的字典
        """
        if self._metadata is None:
            self.load_terrain_data()
        
        data = self.load_terrain_data()
        
        info = {
            'file_path': str(self.tif_file_path),
            'file_size_mb': self.tif_file_path.stat().st_size / (1024 * 1024),
            'data_shape': data.shape,
            'data_type': str(data.dtype),
            'coordinate_system': self._metadata.get('crs'),
            'bounds': self._metadata.get('bounds'),
            'transform': self._metadata.get('transform'),
            'nodata_value': self._metadata.get('nodata'),
            'driver': self._metadata.get('driver')
        }
        
        return info
    
    def get_terrain_statistics(self) -> Dict[str, float]:
        """
        获取地形统计信息
        
        Returns:
            dict: 地形统计数据
        """
        if self._statistics is None:
            data = self.load_terrain_data()
            self._statistics = self.utils.calculate_statistics(data, self._metadata.get('nodata'))
        
        return self._statistics
    
    def save_array(self, output_path: Union[str, Path], format: str = 'npy') -> str:
        """
        保存地形数组到文件
        
        Args:
            output_path: 输出文件路径
            format: 保存格式 ('npy', 'npz', 'csv')
            
        Returns:
            str: 实际保存的文件路径
        """
        data = self.load_terrain_data()
        return self.processor.save_array(data, output_path, format)
    
    def generate_contour_image(self, 
                             output_path: Union[str, Path],
                             levels: Optional[int] = None,
                             colormap: str = 'terrain',
                             figsize: Tuple[int, int] = (12, 8),
                             dpi: int = 150,
                             **kwargs) -> str:
        """
        生成等高线图片
        
        Args:
            output_path: 输出图片路径
            levels: 等高线级数，None为自动计算
            colormap: 颜色映射方案
            figsize: 图片尺寸
            dpi: 图片分辨率
            **kwargs: 其他可视化参数
            
        Returns:
            str: 生成的图片文件路径
        """
        data = self.load_terrain_data()
        stats = self.get_terrain_statistics()
        
        return self.visualizer.generate_contour_plot(
            data, output_path, levels, colormap, figsize, dpi, stats, **kwargs
        )
    
    def get_elevation_at_point(self, x: float, y: float, 
                              coordinate_system: str = 'pixel') -> float:
        """
        获取指定点的高程值
        
        Args:
            x: X坐标
            y: Y坐标
            coordinate_system: 坐标系统 ('pixel' 或 'geographic')
            
        Returns:
            float: 高程值
        """
        data = self.load_terrain_data()
        
        if coordinate_system == 'geographic':
            # 将地理坐标转换为像素坐标
            x, y = self.utils.geographic_to_pixel(
                x, y, self._metadata.get('transform')
            )
        
        return self.utils.get_elevation_at_pixel(data, int(x), int(y))
    
    def get_elevation_profile(self, 
                            start_point: Tuple[float, float],
                            end_point: Tuple[float, float],
                            num_points: int = 100,
                            coordinate_system: str = 'pixel') -> Tuple[np.ndarray, np.ndarray]:
        """
        获取两点间的高程剖面
        
        Args:
            start_point: 起始点坐标 (x, y)
            end_point: 结束点坐标 (x, y)
            num_points: 采样点数量
            coordinate_system: 坐标系统
            
        Returns:
            tuple: (距离数组, 高程数组)
        """
        data = self.load_terrain_data()
        
        if coordinate_system == 'geographic':
            start_point = self.utils.geographic_to_pixel(
                start_point[0], start_point[1], self._metadata.get('transform')
            )
            end_point = self.utils.geographic_to_pixel(
                end_point[0], end_point[1], self._metadata.get('transform')
            )
        
        return self.utils.get_elevation_profile(data, start_point, end_point, num_points)
    
    def create_hillshade(self, 
                        azimuth: float = 315.0,
                        altitude: float = 45.0) -> np.ndarray:
        """
        创建山体阴影效果
        
        Args:
            azimuth: 光源方位角（度）
            altitude: 光源高度角（度）
            
        Returns:
            numpy.ndarray: 山体阴影数组
        """
        data = self.load_terrain_data()
        return self.utils.create_hillshade(data, azimuth, altitude)
    
    def resample_data(self, 
                     new_width: int, 
                     new_height: int,
                     method: str = 'bilinear') -> np.ndarray:
        """
        重采样地形数据
        
        Args:
            new_width: 新的宽度
            new_height: 新的高度
            method: 插值方法 ('nearest', 'bilinear', 'cubic')
            
        Returns:
            numpy.ndarray: 重采样后的数据
        """
        data = self.load_terrain_data()
        return self.processor.resample_data(data, new_width, new_height, method)
    
    def apply_filter(self, filter_type: str = 'gaussian', **kwargs) -> np.ndarray:
        """
        对地形数据应用滤波器
        
        Args:
            filter_type: 滤波器类型 ('gaussian', 'median', 'bilateral')
            **kwargs: 滤波器参数
            
        Returns:
            numpy.ndarray: 滤波后的数据
        """
        data = self.load_terrain_data()
        return self.processor.apply_filter(data, filter_type, **kwargs)
    
    def smooth_terrain_data(self, method: str = 'gaussian', **kwargs) -> np.ndarray:
        """
        平滑地形数据
        
        Args:
            method: 平滑方法 ('gaussian', 'bilateral', 'spline', 'savgol', 'anisotropic')
            **kwargs: 方法特定参数
            
        Returns:
            numpy.ndarray: 平滑后的地形数据
        """
        data = self.load_terrain_data()
        return self.processor.smooth_terrain(data, method, **kwargs)
    
    def generate_smoothed_contour(self, output_path: Union[str, Path],
                                 smooth_method: str = 'gaussian',
                                 levels: Optional[int] = None,
                                 colormap: str = 'terrain',
                                 figsize: Tuple[int, int] = (12, 8),
                                 dpi: int = 150,
                                 **smooth_kwargs) -> str:
        """
        生成平滑后的等高线图片
        
        Args:
            output_path: 输出图片路径
            smooth_method: 平滑方法
            levels: 等高线级数
            colormap: 颜色映射方案
            figsize: 图片尺寸
            dpi: 图片分辨率
            **smooth_kwargs: 平滑方法参数
            
        Returns:
            str: 生成的图片文件路径
        """
        # 获取平滑后的数据
        smoothed_data = self.smooth_terrain_data(smooth_method, **smooth_kwargs)
        
        # 重新计算统计信息
        stats = self.utils.calculate_statistics(smoothed_data, self._metadata.get('nodata'))
        
        return self.visualizer.generate_contour_plot(
            smoothed_data, output_path, levels, colormap, figsize, dpi, stats
        )
    
    def compare_smoothing_effects(self, methods: list = None) -> dict:
        """
        比较不同平滑方法的效果
        
        Args:
            methods: 要比较的方法列表
            
        Returns:
            dict: 包含不同方法结果的字典
        """
        data = self.load_terrain_data()
        return self.processor.compare_smoothing_methods(data, methods)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"TerrainProcessor('{self.tif_file_path}')"
    
    def __str__(self) -> str:
        """用户友好的字符串表示"""
        info = self.get_terrain_info()
        return (f"地形处理器\n"
                f"文件: {info['file_path']}\n"
                f"尺寸: {info['data_shape']}\n"
                f"数据类型: {info['data_type']}\n"
                f"坐标系: {info['coordinate_system']}")