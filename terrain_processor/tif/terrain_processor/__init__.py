#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terrain Processor - 地形数据处理库

一个用于读取TIF地形文件、处理三维数组数据和生成等高线图的Python库。

主要功能:
- 读取和解析TIF地形文件
- 转换为三维numpy数组
- 生成彩色等高线图
- 提供简洁的API供其他程序调用

作者: Kilo Code
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Kilo Code"
__email__ = "kilo@example.com"
__description__ = "A Python library for processing terrain data from TIF files"

# 导入主要类和函数
from .core import TerrainProcessor
from .reader import TIFReader
from .processor import DataProcessor
from .visualizer import ContourGenerator
from .utils import TerrainUtils

# 定义公共API
__all__ = [
    'TerrainProcessor',
    'TIFReader', 
    'DataProcessor',
    'ContourGenerator',
    'TerrainUtils'
]

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """获取版本信息"""
    return __version__

def get_info():
    """获取库信息"""
    return {
        'name': 'terrain_processor',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'python_requires': '>=3.7'
    }