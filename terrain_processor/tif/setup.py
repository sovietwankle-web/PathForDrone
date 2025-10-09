#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地形处理器安装配置
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# 读取requirements.txt
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)

setup(
    name="terrain-processor",
    version="1.0.0",
    author="Kilo Code",
    author_email="kilo@example.com",
    description="一个用于处理TIF地形文件的Python库",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kilocode/terrain-processor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
        "full": [
            "GDAL>=3.0.0",
            "plotly>=5.0.0",
            "folium>=0.12.0",
            "geopandas>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "terrain-processor=terrain_processor.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "terrain_processor": ["*.txt", "*.md"],
    },
    keywords="terrain, GIS, TIF, TIFF, elevation, contour, topography, raster",
    project_urls={
        "Bug Reports": "https://github.com/kilocode/terrain-processor/issues",
        "Source": "https://github.com/kilocode/terrain-processor",
        "Documentation": "https://github.com/kilocode/terrain-processor/wiki",
    },
)