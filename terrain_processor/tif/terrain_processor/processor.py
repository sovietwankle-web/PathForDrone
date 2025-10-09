#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块
负责三维数组处理、保存和各种数据操作
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import csv
from scipy import ndimage
from scipy.interpolate import griddata, RectBivariateSpline, interp2d
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
import warnings


class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        """初始化数据处理器"""
        pass
    
    def save_array(self, data: np.ndarray, output_path: Union[str, Path], 
                   format: str = 'npy') -> str:
        """
        保存数组到文件
        
        Args:
            data: 要保存的数组
            output_path: 输出路径
            format: 保存格式 ('npy', 'npz', 'csv', 'txt')
            
        Returns:
            str: 实际保存的文件路径
        """
        output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'npy':
            # 保存为numpy二进制格式
            if not output_path.suffix:
                output_path = output_path.with_suffix('.npy')
            np.save(output_path, data)
            
        elif format.lower() == 'npz':
            # 保存为numpy压缩格式
            if not output_path.suffix:
                output_path = output_path.with_suffix('.npz')
            np.savez_compressed(output_path, terrain_data=data)
            
        elif format.lower() == 'csv':
            # 保存为CSV格式
            if not output_path.suffix:
                output_path = output_path.with_suffix('.csv')
            np.savetxt(output_path, data, delimiter=',', fmt='%.6f')
            
        elif format.lower() == 'txt':
            # 保存为文本格式
            if not output_path.suffix:
                output_path = output_path.with_suffix('.txt')
            np.savetxt(output_path, data, fmt='%.6f')
            
        else:
            raise ValueError(f"不支持的保存格式: {format}")
        
        return str(output_path)
    
    def load_array(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        从文件加载数组
        
        Args:
            file_path: 文件路径
            
        Returns:
            numpy.ndarray: 加载的数组
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.npy':
            return np.load(file_path)
        elif suffix == '.npz':
            with np.load(file_path) as data:
                if 'terrain_data' in data:
                    return data['terrain_data']
                else:
                    # 取第一个数组
                    return data[list(data.keys())[0]]
        elif suffix in ['.csv', '.txt']:
            return np.loadtxt(file_path, delimiter=',' if suffix == '.csv' else None)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def resample_data(self, data: np.ndarray, new_width: int, new_height: int,
                     method: str = 'bilinear') -> np.ndarray:
        """
        重采样数据到新的尺寸
        
        Args:
            data: 输入数据
            new_width: 新宽度
            new_height: 新高度
            method: 插值方法 ('nearest', 'bilinear', 'cubic')
            
        Returns:
            numpy.ndarray: 重采样后的数据
        """
        old_height, old_width = data.shape
        
        # 创建坐标网格
        old_x = np.linspace(0, old_width - 1, old_width)
        old_y = np.linspace(0, old_height - 1, old_height)
        old_xx, old_yy = np.meshgrid(old_x, old_y)
        
        new_x = np.linspace(0, old_width - 1, new_width)
        new_y = np.linspace(0, old_height - 1, new_height)
        new_xx, new_yy = np.meshgrid(new_x, new_y)
        
        # 处理NaN值
        mask = ~np.isnan(data)
        if not np.any(mask):
            # 如果所有值都是NaN，返回相同形状的NaN数组
            return np.full((new_height, new_width), np.nan)
        
        # 提取有效数据点
        points = np.column_stack((old_xx[mask], old_yy[mask]))
        values = data[mask]
        
        # 插值方法映射
        method_map = {
            'nearest': 'nearest',
            'bilinear': 'linear',
            'cubic': 'cubic'
        }
        
        if method not in method_map:
            raise ValueError(f"不支持的插值方法: {method}")
        
        # 执行插值
        try:
            resampled = griddata(
                points, values, 
                (new_xx, new_yy), 
                method=method_map[method],
                fill_value=np.nan
            )
            return resampled
        except Exception as e:
            warnings.warn(f"插值失败，使用最近邻方法: {e}")
            # 回退到最近邻插值
            resampled = griddata(
                points, values, 
                (new_xx, new_yy), 
                method='nearest',
                fill_value=np.nan
            )
            return resampled
    
    def apply_filter(self, data: np.ndarray, filter_type: str = 'gaussian',
                    **kwargs) -> np.ndarray:
        """
        对数据应用滤波器
        
        Args:
            data: 输入数据
            filter_type: 滤波器类型 ('gaussian', 'median', 'uniform')
            **kwargs: 滤波器参数
            
        Returns:
            numpy.ndarray: 滤波后的数据
        """
        # 处理NaN值
        mask = ~np.isnan(data)
        if not np.any(mask):
            return data.copy()
        
        # 创建工作数据副本
        work_data = data.copy()
        
        if filter_type.lower() == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            # 对有效数据应用高斯滤波
            filtered = ndimage.gaussian_filter(work_data, sigma=sigma)
            # 保持NaN位置
            filtered[~mask] = np.nan
            
        elif filter_type.lower() == 'median':
            size = kwargs.get('size', 3)
            # 中值滤波
            filtered = ndimage.median_filter(work_data, size=size)
            filtered[~mask] = np.nan
            
        elif filter_type.lower() == 'uniform':
            size = kwargs.get('size', 3)
            # 均值滤波
            filtered = ndimage.uniform_filter(work_data, size=size)
            filtered[~mask] = np.nan
            
        else:
            raise ValueError(f"不支持的滤波器类型: {filter_type}")
        
        return filtered
    
    def calculate_gradient(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算数据的梯度
        
        Args:
            data: 输入数据
            
        Returns:
            tuple: (x方向梯度, y方向梯度)
        """
        # 使用numpy的梯度函数
        gy, gx = np.gradient(data)
        return gx, gy
    
    def calculate_slope(self, data: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
        """
        计算坡度
        
        Args:
            data: 高程数据
            pixel_size: 像素大小（米）
            
        Returns:
            numpy.ndarray: 坡度数组（度）
        """
        gx, gy = self.calculate_gradient(data)
        
        # 计算坡度（弧度）
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2) / pixel_size)
        
        # 转换为度
        slope_deg = np.degrees(slope_rad)
        
        return slope_deg
    
    def calculate_aspect(self, data: np.ndarray) -> np.ndarray:
        """
        计算坡向
        
        Args:
            data: 高程数据
            
        Returns:
            numpy.ndarray: 坡向数组（度，0-360）
        """
        gx, gy = self.calculate_gradient(data)
        
        # 计算坡向（弧度）
        aspect_rad = np.arctan2(-gy, -gx)
        
        # 转换为度并调整范围到0-360
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = (aspect_deg + 360) % 360
        
        return aspect_deg
    
    def fill_nodata(self, data: np.ndarray, method: str = 'nearest') -> np.ndarray:
        """
        填充NoData值
        
        Args:
            data: 输入数据
            method: 填充方法 ('nearest', 'mean', 'median', 'interpolate')
            
        Returns:
            numpy.ndarray: 填充后的数据
        """
        if not np.any(np.isnan(data)):
            return data.copy()
        
        filled_data = data.copy()
        nan_mask = np.isnan(data)
        
        if method == 'nearest':
            # 使用最近邻填充
            from scipy.ndimage import distance_transform_edt
            
            # 找到最近的有效像素
            valid_mask = ~nan_mask
            if np.any(valid_mask):
                indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
                filled_data[nan_mask] = data[tuple(indices[:, nan_mask])]
            
        elif method == 'mean':
            # 使用均值填充
            mean_value = np.nanmean(data)
            filled_data[nan_mask] = mean_value
            
        elif method == 'median':
            # 使用中位数填充
            median_value = np.nanmedian(data)
            filled_data[nan_mask] = median_value
            
        elif method == 'interpolate':
            # 使用插值填充
            valid_mask = ~nan_mask
            if np.any(valid_mask):
                y_coords, x_coords = np.mgrid[0:data.shape[0], 0:data.shape[1]]
                valid_points = np.column_stack((x_coords[valid_mask], y_coords[valid_mask]))
                valid_values = data[valid_mask]
                
                nan_points = np.column_stack((x_coords[nan_mask], y_coords[nan_mask]))
                
                try:
                    interpolated = griddata(valid_points, valid_values, nan_points, method='linear')
                    filled_data[nan_mask] = interpolated
                    
                    # 如果还有NaN，用最近邻填充
                    still_nan = np.isnan(filled_data)
                    if np.any(still_nan):
                        nearest_interp = griddata(valid_points, valid_values, 
                                                nan_points[still_nan[nan_mask]], method='nearest')
                        filled_data[still_nan] = nearest_interp
                except:
                    # 如果插值失败，回退到均值
                    filled_data[nan_mask] = np.nanmean(data)
        
        else:
            raise ValueError(f"不支持的填充方法: {method}")
        
        return filled_data
    
    def crop_data(self, data: np.ndarray, 
                  bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """
        裁剪数据
        
        Args:
            data: 输入数据
            bounds: 裁剪边界 (min_row, max_row, min_col, max_col)
            
        Returns:
            numpy.ndarray: 裁剪后的数据
        """
        min_row, max_row, min_col, max_col = bounds
        
        # 验证边界
        min_row = max(0, min_row)
        max_row = min(data.shape[0], max_row)
        min_col = max(0, min_col)
        max_col = min(data.shape[1], max_col)
        
        if min_row >= max_row or min_col >= max_col:
            raise ValueError("无效的裁剪边界")
        
        return data[min_row:max_row, min_col:max_col]
    
    def normalize_data(self, data: np.ndarray, 
                      method: str = 'minmax',
                      target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        数据归一化
        
        Args:
            data: 输入数据
            method: 归一化方法 ('minmax', 'zscore', 'robust')
            target_range: 目标范围（仅用于minmax）
            
        Returns:
            numpy.ndarray: 归一化后的数据
        """
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data.copy()
        
        normalized = data.copy()
        valid_data = data[valid_mask]
        
        if method == 'minmax':
            # 最小-最大归一化
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            
            if data_max > data_min:
                normalized[valid_mask] = (valid_data - data_min) / (data_max - data_min)
                normalized[valid_mask] = (normalized[valid_mask] * 
                                        (target_range[1] - target_range[0]) + target_range[0])
            
        elif method == 'zscore':
            # Z-score标准化
            data_mean = np.mean(valid_data)
            data_std = np.std(valid_data)
            
            if data_std > 0:
                normalized[valid_mask] = (valid_data - data_mean) / data_std
            
        elif method == 'robust':
            # 鲁棒标准化（使用中位数和MAD）
            data_median = np.median(valid_data)
            mad = np.median(np.abs(valid_data - data_median))
            
            if mad > 0:
                normalized[valid_mask] = (valid_data - data_median) / (1.4826 * mad)
        
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        return normalized
    
    def smooth_terrain(self, data: np.ndarray, method: str = 'gaussian',
                      **kwargs) -> np.ndarray:
        """
        使用插值方法平滑地形数据
        
        Args:
            data: 输入地形数据
            method: 平滑方法 ('gaussian', 'bilateral', 'spline', 'savgol', 'anisotropic')
            **kwargs: 方法特定参数
            
        Returns:
            numpy.ndarray: 平滑后的地形数据
        """
        if method == 'gaussian':
            return self._smooth_gaussian(data, **kwargs)
        elif method == 'bilateral':
            return self._smooth_bilateral(data, **kwargs)
        elif method == 'spline':
            return self._smooth_spline(data, **kwargs)
        elif method == 'savgol':
            return self._smooth_savgol(data, **kwargs)
        elif method == 'anisotropic':
            return self._smooth_anisotropic(data, **kwargs)
        else:
            raise ValueError(f"不支持的平滑方法: {method}")
    
    def _smooth_gaussian(self, data: np.ndarray, sigma: float = 1.0,
                        preserve_edges: bool = True) -> np.ndarray:
        """
        高斯平滑
        
        Args:
            data: 输入数据
            sigma: 高斯核标准差
            preserve_edges: 是否保持边缘
            
        Returns:
            numpy.ndarray: 平滑后的数据
        """
        # 处理NaN值
        mask = ~np.isnan(data)
        if not np.any(mask):
            return data.copy()
        
        smoothed = data.copy()
        
        if preserve_edges:
            # 使用边缘保持的高斯滤波
            # 先填充NaN值
            filled_data = self.fill_nodata(data, method='nearest')
            smoothed = gaussian_filter(filled_data, sigma=sigma)
            # 恢复原始NaN位置
            smoothed[~mask] = np.nan
        else:
            # 标准高斯滤波
            smoothed = gaussian_filter(data, sigma=sigma)
        
        return smoothed
    
    def _smooth_bilateral(self, data: np.ndarray, sigma_color: float = 0.1,
                         sigma_spatial: float = 1.0) -> np.ndarray:
        """
        双边滤波平滑（保持边缘）
        
        Args:
            data: 输入数据
            sigma_color: 颜色空间标准差
            sigma_spatial: 空间标准差
            
        Returns:
            numpy.ndarray: 平滑后的数据
        """
        try:
            from skimage import restoration
            
            # 处理NaN值
            mask = ~np.isnan(data)
            if not np.any(mask):
                return data.copy()
            
            # 填充NaN值进行处理
            filled_data = self.fill_nodata(data, method='nearest')
            
            # 归一化到0-1范围进行双边滤波
            data_min, data_max = np.nanmin(filled_data), np.nanmax(filled_data)
            if data_max > data_min:
                normalized = (filled_data - data_min) / (data_max - data_min)
                
                # 应用双边滤波
                smoothed_norm = restoration.denoise_bilateral(
                    normalized,
                    sigma_color=sigma_color,
                    sigma_spatial=sigma_spatial,
                    channel_axis=None
                )
                
                # 恢复原始范围
                smoothed = smoothed_norm * (data_max - data_min) + data_min
            else:
                smoothed = filled_data.copy()
            
            # 恢复NaN位置
            smoothed[~mask] = np.nan
            
            return smoothed
            
        except ImportError:
            warnings.warn("scikit-image未安装，使用高斯滤波替代双边滤波")
            return self._smooth_gaussian(data, sigma=sigma_spatial)
    
    def _smooth_spline(self, data: np.ndarray, smoothing: float = None,
                      spline_order: int = 3) -> np.ndarray:
        """
        样条插值平滑
        
        Args:
            data: 输入数据
            smoothing: 平滑参数，None为自动计算
            spline_order: 样条阶数
            
        Returns:
            numpy.ndarray: 平滑后的数据
        """
        # 处理NaN值
        mask = ~np.isnan(data)
        if not np.any(mask):
            return data.copy()
        
        height, width = data.shape
        
        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # 提取有效数据点
        valid_points_x = x_coords[mask]
        valid_points_y = y_coords[mask]
        valid_values = data[mask]
        
        # 如果数据点太少，回退到高斯平滑
        if len(valid_values) < 100:
            warnings.warn("有效数据点太少，使用高斯平滑替代样条插值")
            return self._smooth_gaussian(data, sigma=1.0)
        
        try:
            # 自动计算平滑参数
            if smoothing is None:
                smoothing = len(valid_values) * 0.1
            
            # 使用RectBivariateSpline进行插值
            # 需要规则网格，所以先进行网格化
            x_unique = np.unique(valid_points_x)
            y_unique = np.unique(valid_points_y)
            
            if len(x_unique) < 4 or len(y_unique) < 4:
                # 如果网格点太少，使用griddata插值
                smoothed = self._smooth_griddata_spline(data, valid_points_x,
                                                      valid_points_y, valid_values)
            else:
                # 创建规则网格进行样条插值
                smoothed = self._smooth_rect_spline(data, smoothing, spline_order)
            
            return smoothed
            
        except Exception as e:
            warnings.warn(f"样条插值失败: {e}，使用高斯平滑替代")
            return self._smooth_gaussian(data, sigma=1.0)
    
    def _smooth_griddata_spline(self, data: np.ndarray, valid_x: np.ndarray,
                               valid_y: np.ndarray, valid_values: np.ndarray) -> np.ndarray:
        """使用griddata进行三次样条插值"""
        height, width = data.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # 使用三次插值
        smoothed = griddata(
            np.column_stack((valid_x, valid_y)),
            valid_values,
            (x_coords, y_coords),
            method='cubic',
            fill_value=np.nan
        )
        
        # 如果有NaN，用线性插值填充
        nan_mask = np.isnan(smoothed)
        if np.any(nan_mask):
            linear_interp = griddata(
                np.column_stack((valid_x, valid_y)),
                valid_values,
                (x_coords[nan_mask], y_coords[nan_mask]),
                method='linear',
                fill_value=np.nan
            )
            smoothed[nan_mask] = linear_interp
        
        # 恢复原始NaN位置
        original_mask = ~np.isnan(data)
        smoothed[~original_mask] = np.nan
        
        return smoothed
    
    def _smooth_rect_spline(self, data: np.ndarray, smoothing: float,
                           spline_order: int) -> np.ndarray:
        """使用RectBivariateSpline进行插值"""
        # 填充NaN值
        filled_data = self.fill_nodata(data, method='nearest')
        height, width = filled_data.shape
        
        # 创建坐标
        x = np.arange(width)
        y = np.arange(height)
        
        try:
            # 创建样条插值器
            spline = RectBivariateSpline(y, x, filled_data,
                                       kx=min(spline_order, len(y)-1),
                                       ky=min(spline_order, len(x)-1),
                                       s=smoothing)
            
            # 生成平滑后的数据
            smoothed = spline(y, x)
            
            # 恢复原始NaN位置
            original_mask = ~np.isnan(data)
            smoothed[~original_mask] = np.nan
            
            return smoothed
            
        except Exception as e:
            warnings.warn(f"RectBivariateSpline失败: {e}")
            return self._smooth_gaussian(data, sigma=1.0)
    
    def _smooth_savgol(self, data: np.ndarray, window_length: int = 5,
                      polyorder: int = 2) -> np.ndarray:
        """
        Savitzky-Golay滤波平滑
        
        Args:
            data: 输入数据
            window_length: 窗口长度（必须为奇数）
            polyorder: 多项式阶数
            
        Returns:
            numpy.ndarray: 平滑后的数据
        """
        try:
            from scipy.signal import savgol_filter
            
            # 确保窗口长度为奇数
            if window_length % 2 == 0:
                window_length += 1
            
            # 处理NaN值
            mask = ~np.isnan(data)
            if not np.any(mask):
                return data.copy()
            
            # 填充NaN值
            filled_data = self.fill_nodata(data, method='nearest')
            
            # 对每一行应用Savitzky-Golay滤波
            smoothed = np.zeros_like(filled_data)
            for i in range(filled_data.shape[0]):
                if filled_data.shape[1] >= window_length:
                    smoothed[i, :] = savgol_filter(filled_data[i, :],
                                                 window_length, polyorder)
                else:
                    smoothed[i, :] = filled_data[i, :]
            
            # 对每一列应用Savitzky-Golay滤波
            for j in range(smoothed.shape[1]):
                if smoothed.shape[0] >= window_length:
                    smoothed[:, j] = savgol_filter(smoothed[:, j],
                                                 window_length, polyorder)
            
            # 恢复原始NaN位置
            smoothed[~mask] = np.nan
            
            return smoothed
            
        except ImportError:
            warnings.warn("scipy.signal未找到savgol_filter，使用高斯滤波替代")
            return self._smooth_gaussian(data, sigma=1.0)
    
    def _smooth_anisotropic(self, data: np.ndarray, iterations: int = 10,
                           kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        """
        各向异性扩散平滑（保持边缘特征）
        
        Args:
            data: 输入数据
            iterations: 迭代次数
            kappa: 扩散阈值
            gamma: 时间步长
            
        Returns:
            numpy.ndarray: 平滑后的数据
        """
        # 处理NaN值
        mask = ~np.isnan(data)
        if not np.any(mask):
            return data.copy()
        
        # 填充NaN值
        filled_data = self.fill_nodata(data, method='nearest').astype(np.float64)
        
        # 各向异性扩散
        smoothed = filled_data.copy()
        
        for _ in range(iterations):
            # 计算梯度
            gy, gx = np.gradient(smoothed)
            
            # 计算梯度幅度
            grad_mag = np.sqrt(gx**2 + gy**2)
            
            # 计算扩散系数
            c = np.exp(-(grad_mag / kappa)**2)
            
            # 计算扩散项
            cN = c
            cS = np.roll(c, 1, axis=0)
            cE = c
            cW = np.roll(c, 1, axis=1)
            
            # 计算邻域差分
            nN = np.roll(smoothed, -1, axis=0) - smoothed
            nS = np.roll(smoothed, 1, axis=0) - smoothed
            nE = np.roll(smoothed, -1, axis=1) - smoothed
            nW = np.roll(smoothed, 1, axis=1) - smoothed
            
            # 更新
            smoothed += gamma * (cN * nN + cS * nS + cE * nE + cW * nW)
        
        # 恢复原始NaN位置
        smoothed[~mask] = np.nan
        
        return smoothed
    
    def compare_smoothing_methods(self, data: np.ndarray,
                                 methods: list = None) -> dict:
        """
        比较不同平滑方法的效果
        
        Args:
            data: 输入数据
            methods: 要比较的方法列表
            
        Returns:
            dict: 包含不同方法结果的字典
        """
        if methods is None:
            methods = ['gaussian', 'bilateral', 'spline', 'savgol', 'anisotropic']
        
        results = {'original': data.copy()}
        
        for method in methods:
            try:
                if method == 'gaussian':
                    results[method] = self.smooth_terrain(data, method, sigma=1.0)
                elif method == 'bilateral':
                    results[method] = self.smooth_terrain(data, method,
                                                        sigma_color=0.1, sigma_spatial=1.0)
                elif method == 'spline':
                    results[method] = self.smooth_terrain(data, method)
                elif method == 'savgol':
                    results[method] = self.smooth_terrain(data, method,
                                                        window_length=5, polyorder=2)
                elif method == 'anisotropic':
                    results[method] = self.smooth_terrain(data, method,
                                                        iterations=5, kappa=50)
                else:
                    warnings.warn(f"未知的平滑方法: {method}")
                    
            except Exception as e:
                warnings.warn(f"方法 {method} 执行失败: {e}")
                results[method] = data.copy()
        
        return results