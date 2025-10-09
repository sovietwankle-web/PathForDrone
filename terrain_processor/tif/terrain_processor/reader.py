#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIF文件读取模块
负责读取和解析TIF地形文件
"""

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from pathlib import Path
from typing import Tuple, Dict, Any, Union
import warnings


class TIFReader:
    """TIF文件读取器"""
    
    def __init__(self, file_path: Union[str, Path]):
        """
        初始化TIF读取器
        
        Args:
            file_path: TIF文件路径
        """
        self.file_path = Path(file_path)
        self._validate_file()
    
    def _validate_file(self):
        """验证文件是否存在且可读"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"TIF文件不存在: {self.file_path}")
        
        if not self.file_path.is_file():
            raise ValueError(f"路径不是文件: {self.file_path}")
        
        # 检查文件扩展名
        valid_extensions = {'.tif', '.tiff', '.TIF', '.TIFF'}
        if self.file_path.suffix not in valid_extensions:
            warnings.warn(f"文件扩展名可能不正确: {self.file_path.suffix}")
    
    def read_tif(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取TIF文件
        
        Returns:
            tuple: (地形数据数组, 元数据字典)
        """
        try:
            with rasterio.open(self.file_path) as src:
                # 读取元数据
                metadata = self._extract_metadata(src)
                
                # 读取数据
                data = self._read_data(src)
                
                # 数据预处理
                data = self._preprocess_data(data, metadata)
                
                return data, metadata
                
        except RasterioIOError as e:
            raise IOError(f"无法读取TIF文件: {e}")
        except Exception as e:
            raise RuntimeError(f"读取TIF文件时发生错误: {e}")
    
    def _extract_metadata(self, src) -> Dict[str, Any]:
        """
        提取文件元数据
        
        Args:
            src: rasterio数据源
            
        Returns:
            dict: 元数据字典
        """
        metadata = {
            'driver': src.driver,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'res': src.res,
            'units': src.crs.linear_units if src.crs else None,
            'file_size': self.file_path.stat().st_size
        }
        
        # 添加地理范围信息
        if src.crs:
            metadata['geographic_bounds'] = {
                'left': src.bounds.left,
                'bottom': src.bounds.bottom,
                'right': src.bounds.right,
                'top': src.bounds.top
            }
        
        return metadata
    
    def _read_data(self, src) -> np.ndarray:
        """
        读取栅格数据
        
        Args:
            src: rasterio数据源
            
        Returns:
            numpy.ndarray: 地形数据
        """
        if src.count == 1:
            # 单波段数据
            data = src.read(1)
        else:
            # 多波段数据，取第一个波段
            data = src.read(1)
            if src.count > 1:
                warnings.warn(f"文件包含{src.count}个波段，只使用第一个波段")
        
        return data
    
    def _preprocess_data(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        数据预处理
        
        Args:
            data: 原始数据
            metadata: 元数据
            
        Returns:
            numpy.ndarray: 预处理后的数据
        """
        # 处理NoData值
        nodata = metadata.get('nodata')
        if nodata is not None:
            # 将NoData值替换为NaN
            data = data.astype(np.float64)
            data[data == nodata] = np.nan
        
        # 检查数据有效性
        valid_data_count = np.sum(~np.isnan(data))
        total_pixels = data.size
        
        if valid_data_count == 0:
            warnings.warn("警告: 文件中没有有效的地形数据")
        elif valid_data_count < total_pixels * 0.1:
            warnings.warn(f"警告: 有效数据比例很低 ({valid_data_count/total_pixels*100:.1f}%)")
        
        return data
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        获取文件基本信息（不读取数据）
        
        Returns:
            dict: 文件信息
        """
        try:
            with rasterio.open(self.file_path) as src:
                info = {
                    'file_path': str(self.file_path),
                    'file_size_mb': self.file_path.stat().st_size / (1024 * 1024),
                    'driver': src.driver,
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'data_type': src.dtypes[0],
                    'coordinate_system': str(src.crs) if src.crs else 'Unknown',
                    'bounds': src.bounds,
                    'resolution': src.res
                }
                return info
        except Exception as e:
            raise RuntimeError(f"无法获取文件信息: {e}")
    
    def read_window(self, window: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取指定窗口的数据
        
        Args:
            window: 窗口范围 (col_off, row_off, width, height)
            
        Returns:
            tuple: (窗口数据, 元数据)
        """
        try:
            with rasterio.open(self.file_path) as src:
                # 创建窗口对象
                from rasterio.windows import Window
                win = Window(window[0], window[1], window[2], window[3])
                
                # 读取窗口数据
                data = src.read(1, window=win)
                
                # 获取窗口的元数据
                metadata = self._extract_metadata(src)
                
                # 更新变换矩阵以反映窗口位置
                window_transform = rasterio.windows.transform(win, src.transform)
                metadata['transform'] = window_transform
                metadata['width'] = window[2]
                metadata['height'] = window[3]
                
                # 预处理数据
                data = self._preprocess_data(data, metadata)
                
                return data, metadata
                
        except Exception as e:
            raise RuntimeError(f"读取窗口数据时发生错误: {e}")
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        验证数据完整性
        
        Returns:
            dict: 验证结果
        """
        try:
            data, metadata = self.read_tif()
            
            # 统计信息
            total_pixels = data.size
            valid_pixels = np.sum(~np.isnan(data))
            invalid_pixels = total_pixels - valid_pixels
            
            # 数据范围
            if valid_pixels > 0:
                valid_data = data[~np.isnan(data)]
                data_min = np.min(valid_data)
                data_max = np.max(valid_data)
                data_mean = np.mean(valid_data)
                data_std = np.std(valid_data)
            else:
                data_min = data_max = data_mean = data_std = np.nan
            
            validation_result = {
                'file_readable': True,
                'total_pixels': total_pixels,
                'valid_pixels': valid_pixels,
                'invalid_pixels': invalid_pixels,
                'valid_data_percentage': (valid_pixels / total_pixels) * 100,
                'data_range': {
                    'min': float(data_min) if not np.isnan(data_min) else None,
                    'max': float(data_max) if not np.isnan(data_max) else None,
                    'mean': float(data_mean) if not np.isnan(data_mean) else None,
                    'std': float(data_std) if not np.isnan(data_std) else None
                },
                'has_coordinate_system': metadata['crs'] is not None,
                'coordinate_system': str(metadata['crs']) if metadata['crs'] else None,
                'data_type': str(metadata['dtype']),
                'warnings': []
            }
            
            # 添加警告
            if valid_pixels == 0:
                validation_result['warnings'].append("文件中没有有效数据")
            elif valid_pixels < total_pixels * 0.5:
                validation_result['warnings'].append("有效数据比例较低")
            
            if metadata['crs'] is None:
                validation_result['warnings'].append("文件缺少坐标参考系统")
            
            return validation_result
            
        except Exception as e:
            return {
                'file_readable': False,
                'error': str(e),
                'warnings': [f"无法读取文件: {e}"]
            }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"TIFReader('{self.file_path}')"