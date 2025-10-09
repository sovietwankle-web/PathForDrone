#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块
负责生成等高线图和其他地形可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.contour import ContourSet
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List
import warnings

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ContourGenerator:
    """等高线生成器"""
    
    def __init__(self):
        """初始化等高线生成器"""
        self.default_colormaps = {
            'terrain': 'terrain',
            'elevation': 'gist_earth',
            'topographic': 'terrain',
            'rainbow': 'rainbow',
            'viridis': 'viridis',
            'plasma': 'plasma'
        }
    
    def generate_contour_plot(self, 
                            data: np.ndarray,
                            output_path: Union[str, Path],
                            levels: Optional[Union[int, List[float]]] = None,
                            colormap: str = 'terrain',
                            figsize: Tuple[int, int] = (12, 8),
                            dpi: int = 150,
                            statistics: Optional[Dict[str, float]] = None,
                            **kwargs) -> str:
        """
        生成等高线图
        
        Args:
            data: 地形数据
            output_path: 输出路径
            levels: 等高线级数或具体高程值列表
            colormap: 颜色映射
            figsize: 图片尺寸
            dpi: 分辨率
            statistics: 地形统计信息
            **kwargs: 其他参数
            
        Returns:
            str: 生成的图片路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理数据
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            raise ValueError("数据中没有有效值")
        
        # 计算等高线级数
        if levels is None:
            levels = self._calculate_optimal_levels(data, statistics)
        elif isinstance(levels, int):
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            levels = np.linspace(data_min, data_max, levels)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 生成等高线
        try:
            # 填充等高线
            contourf = ax.contourf(data, levels=levels, cmap=colormap, alpha=0.8)
            
            # 等高线线条
            contour_lines = ax.contour(data, levels=levels, colors='black', 
                                     linewidths=0.5, alpha=0.6)
            
            # 添加等高线标签
            if kwargs.get('add_labels', True):
                ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')
            
            # 添加颜色条
            cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
            cbar.set_label('高程 (m)', rotation=270, labelpad=20)
            
            # 设置标题和标签
            title = kwargs.get('title', '地形等高线图')
            ax.set_title(title, fontsize=14, pad=20)
            ax.set_xlabel('X 坐标 (像素)', fontsize=12)
            ax.set_ylabel('Y 坐标 (像素)', fontsize=12)
            
            # 添加统计信息文本
            if statistics and kwargs.get('show_stats', True):
                self._add_statistics_text(ax, statistics)
            
            # 设置网格
            if kwargs.get('show_grid', True):
                ax.grid(True, alpha=0.3)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise RuntimeError(f"生成等高线图时发生错误: {e}")
    
    def generate_3d_surface(self,
                          data: np.ndarray,
                          output_path: Union[str, Path],
                          colormap: str = 'terrain',
                          figsize: Tuple[int, int] = (12, 9),
                          dpi: int = 150,
                          **kwargs) -> str:
        """
        生成3D地形表面图
        
        Args:
            data: 地形数据
            output_path: 输出路径
            colormap: 颜色映射
            figsize: 图片尺寸
            dpi: 分辨率
            **kwargs: 其他参数
            
        Returns:
            str: 生成的图片路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建3D图形
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建坐标网格
        y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        
        # 处理采样以提高性能
        step = kwargs.get('sampling_step', max(1, min(data.shape) // 100))
        x_sampled = x[::step, ::step]
        y_sampled = y[::step, ::step]
        data_sampled = data[::step, ::step]
        
        # 生成3D表面
        try:
            surf = ax.plot_surface(x_sampled, y_sampled, data_sampled,
                                 cmap=colormap, alpha=0.8,
                                 linewidth=0, antialiased=True)
            
            # 添加颜色条
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
            cbar.set_label('高程 (m)', rotation=270, labelpad=20)
            
            # 设置标签和标题
            ax.set_xlabel('X 坐标')
            ax.set_ylabel('Y 坐标')
            ax.set_zlabel('高程 (m)')
            ax.set_title(kwargs.get('title', '3D地形表面图'))
            
            # 设置视角
            ax.view_init(elev=kwargs.get('elevation', 30), 
                        azim=kwargs.get('azimuth', 45))
            
            # 保存图片
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise RuntimeError(f"生成3D表面图时发生错误: {e}")
    
    def generate_hillshade_plot(self,
                              data: np.ndarray,
                              output_path: Union[str, Path],
                              azimuth: float = 315.0,
                              altitude: float = 45.0,
                              figsize: Tuple[int, int] = (12, 8),
                              dpi: int = 150,
                              **kwargs) -> str:
        """
        生成山体阴影图
        
        Args:
            data: 地形数据
            output_path: 输出路径
            azimuth: 光源方位角
            altitude: 光源高度角
            figsize: 图片尺寸
            dpi: 分辨率
            **kwargs: 其他参数
            
        Returns:
            str: 生成的图片路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 计算山体阴影
        hillshade = self._calculate_hillshade(data, azimuth, altitude)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 显示山体阴影
        im = ax.imshow(hillshade, cmap='gray', origin='lower')
        
        # 可选：叠加等高线
        if kwargs.get('overlay_contours', False):
            levels = self._calculate_optimal_levels(data)
            contour = ax.contour(data, levels=levels, colors='black', 
                               linewidths=0.5, alpha=0.3)
        
        # 设置标题和标签
        ax.set_title(kwargs.get('title', '山体阴影图'))
        ax.set_xlabel('X 坐标 (像素)')
        ax.set_ylabel('Y 坐标 (像素)')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('光照强度')
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_multi_view_plot(self,
                               data: np.ndarray,
                               output_path: Union[str, Path],
                               statistics: Optional[Dict[str, float]] = None,
                               figsize: Tuple[int, int] = (16, 12),
                               dpi: int = 150) -> str:
        """
        生成多视图综合地形图
        
        Args:
            data: 地形数据
            output_path: 输出路径
            statistics: 统计信息
            figsize: 图片尺寸
            dpi: 分辨率
            
        Returns:
            str: 生成的图片路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('地形数据综合分析', fontsize=16)
        
        # 1. 原始高程图
        im1 = axes[0, 0].imshow(data, cmap='terrain', origin='lower')
        axes[0, 0].set_title('高程图')
        plt.colorbar(im1, ax=axes[0, 0], label='高程 (m)')
        
        # 2. 等高线图
        levels = self._calculate_optimal_levels(data, statistics)
        contourf = axes[0, 1].contourf(data, levels=levels, cmap='terrain')
        contour = axes[0, 1].contour(data, levels=levels, colors='black', linewidths=0.5)
        axes[0, 1].set_title('等高线图')
        plt.colorbar(contourf, ax=axes[0, 1], label='高程 (m)')
        
        # 3. 山体阴影
        hillshade = self._calculate_hillshade(data)
        axes[0, 2].imshow(hillshade, cmap='gray', origin='lower')
        axes[0, 2].set_title('山体阴影')
        
        # 4. 坡度图
        slope = self._calculate_slope(data)
        im4 = axes[1, 0].imshow(slope, cmap='Reds', origin='lower')
        axes[1, 0].set_title('坡度图')
        plt.colorbar(im4, ax=axes[1, 0], label='坡度 (度)')
        
        # 5. 高程分布直方图
        valid_data = data[~np.isnan(data)]
        axes[1, 1].hist(valid_data, bins=50, alpha=0.7, color='green')
        axes[1, 1].set_title('高程分布')
        axes[1, 1].set_xlabel('高程 (m)')
        axes[1, 1].set_ylabel('频次')
        
        # 6. 统计信息
        axes[1, 2].axis('off')
        if statistics:
            stats_text = self._format_statistics_text(statistics)
            axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 2].set_title('统计信息')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _calculate_optimal_levels(self, data: np.ndarray, 
                                statistics: Optional[Dict[str, float]] = None) -> np.ndarray:
        """计算最优等高线级数"""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return np.array([0])
        
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        data_range = data_max - data_min
        
        if data_range == 0:
            return np.array([data_min])
        
        # 根据数据范围自动确定等高线间距
        if data_range < 10:
            interval = 1
        elif data_range < 50:
            interval = 5
        elif data_range < 100:
            interval = 10
        elif data_range < 500:
            interval = 25
        elif data_range < 1000:
            interval = 50
        else:
            interval = 100
        
        # 生成等高线级数
        start = np.floor(data_min / interval) * interval
        end = np.ceil(data_max / interval) * interval
        levels = np.arange(start, end + interval, interval)
        
        return levels
    
    def _calculate_hillshade(self, data: np.ndarray, 
                           azimuth: float = 315.0, 
                           altitude: float = 45.0) -> np.ndarray:
        """计算山体阴影"""
        # 计算梯度
        gy, gx = np.gradient(data)
        
        # 计算坡度和坡向
        slope = np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(-gx, gy)
        
        # 转换角度为弧度
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)
        
        # 计算山体阴影
        hillshade = (np.cos(altitude_rad) * np.cos(slope) +
                    np.sin(altitude_rad) * np.sin(slope) *
                    np.cos(azimuth_rad - aspect))
        
        # 归一化到0-1范围
        hillshade = np.clip(hillshade, 0, 1)
        
        return hillshade
    
    def _calculate_slope(self, data: np.ndarray) -> np.ndarray:
        """计算坡度"""
        gy, gx = np.gradient(data)
        slope = np.arctan(np.sqrt(gx**2 + gy**2))
        return np.degrees(slope)
    
    def _add_statistics_text(self, ax, statistics: Dict[str, float]):
        """在图上添加统计信息文本"""
        stats_text = self._format_statistics_text(statistics)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _format_statistics_text(self, statistics: Dict[str, float]) -> str:
        """格式化统计信息文本"""
        text_lines = []
        
        if 'min_elevation' in statistics:
            text_lines.append(f"最低高程: {statistics['min_elevation']:.1f}m")
        if 'max_elevation' in statistics:
            text_lines.append(f"最高高程: {statistics['max_elevation']:.1f}m")
        if 'mean_elevation' in statistics:
            text_lines.append(f"平均高程: {statistics['mean_elevation']:.1f}m")
        if 'elevation_range' in statistics:
            text_lines.append(f"高程范围: {statistics['elevation_range']:.1f}m")
        if 'std_elevation' in statistics:
            text_lines.append(f"标准差: {statistics['std_elevation']:.1f}m")
        if 'valid_pixels' in statistics:
            text_lines.append(f"有效像素: {statistics['valid_pixels']:,}")
        
        return '\n'.join(text_lines)
    
    def create_custom_colormap(self, colors: List[str], name: str = 'custom') -> str:
        """
        创建自定义颜色映射
        
        Args:
            colors: 颜色列表
            name: 颜色映射名称
            
        Returns:
            str: 颜色映射名称
        """
        try:
            cmap = mcolors.LinearSegmentedColormap.from_list(name, colors)
            plt.cm.register_cmap(name=name, cmap=cmap)
            return name
        except Exception as e:
            warnings.warn(f"创建自定义颜色映射失败: {e}")
            return 'terrain'