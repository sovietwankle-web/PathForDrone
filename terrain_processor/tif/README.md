# Terrain Processor - 地形数据处理库

一个功能强大的Python库，用于读取TIF地形文件、处理三维数组数据和生成等高线图。

## 🌟 主要特性

- **TIF文件读取**: 支持读取各种格式的TIF/TIFF地形文件
- **三维数组处理**: 将地形数据转换为numpy数组，支持多种保存格式
- **等高线生成**: 自动生成彩色等高线图，支持自定义样式
- **API友好**: 提供简洁的API接口，易于集成到其他Python项目
- **多种可视化**: 支持2D等高线、3D地形、山体阴影等多种可视化方式
- **数据分析**: 提供地形统计、坡度分析、山峰检测等分析功能
- **批量处理**: 支持命令行工具和批量处理功能

## 📦 安装

### 使用pip安装（推荐）

```bash
pip install terrain-processor
```

### 从源码安装

```bash
git clone https://github.com/kilocode/terrain-processor.git
cd terrain-processor
pip install -e .
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 基础用法

```python
from terrain_processor import TerrainProcessor

# 创建处理器实例
processor = TerrainProcessor('your_terrain.tif')

# 加载地形数据
terrain_data = processor.load_terrain_data()
print(f"地形数据形状: {terrain_data.shape}")

# 获取统计信息
stats = processor.get_terrain_statistics()
print(f"高程范围: {stats['min_elevation']:.2f} - {stats['max_elevation']:.2f} m")

# 保存三维数组
processor.save_array('terrain_data.npy', format='npy')

# 生成等高线图
processor.generate_contour_image('contour_map.png', levels=20, colormap='terrain')
```

### 命令行使用

```bash
# 基本用法
terrain-processor input.tif --array output.npy --contour output.png

# 显示文件信息
terrain-processor input.tif --info --stats

# 生成多种输出
terrain-processor input.tif --contour map.png --hillshade shade.png --3d terrain3d.png

# 自定义参数
terrain-processor input.tif --contour map.png --levels 25 --colormap gist_earth --dpi 200
```

## 📖 详细文档

### 核心类和方法

#### TerrainProcessor

主要的处理器类，提供所有核心功能。

```python
class TerrainProcessor:
    def __init__(self, tif_file_path)
    def load_terrain_data() -> np.ndarray
    def get_terrain_info() -> dict
    def get_terrain_statistics() -> dict
    def save_array(output_path, format='npy') -> str
    def generate_contour_image(output_path, **kwargs) -> str
    def get_elevation_at_point(x, y) -> float
    def get_elevation_profile(start_point, end_point) -> tuple
```

### 支持的文件格式

#### 输入格式
- TIF/TIFF (GeoTIFF)
- 支持各种坐标参考系统
- 支持单波段和多波段数据

#### 输出格式
- **数组格式**: NPY, NPZ, CSV, TXT
- **图像格式**: PNG, JPG, SVG
- **地理格式**: GeoJSON (通过扩展)

### 可视化选项

#### 等高线图
```python
processor.generate_contour_image(
    'contour.png',
    levels=20,           # 等高线数量
    colormap='terrain',  # 颜色方案
    figsize=(12, 8),     # 图片尺寸
    dpi=150,             # 分辨率
    add_labels=True,     # 添加标签
    show_stats=True      # 显示统计信息
)
```

#### 3D地形图
```python
processor.visualizer.generate_3d_surface(
    data, 'terrain_3d.png',
    colormap='terrain',
    elevation=30,        # 视角高度
    azimuth=45          # 视角方位
)
```

#### 山体阴影图
```python
processor.visualizer.generate_hillshade_plot(
    data, 'hillshade.png',
    azimuth=315,        # 光源方位角
    altitude=45         # 光源高度角
)
```

## 🔧 高级功能

### 数据处理

```python
# 数据重采样
resampled = processor.resample_data(new_width=500, new_height=500, method='bilinear')

# 应用滤波器
filtered = processor.apply_filter('gaussian', sigma=1.0)

# 填充缺失值
filled = processor.processor.fill_nodata(data, method='nearest')
```

### 地形分析

```python
# 计算坡度和坡向
slope, aspect = processor.utils.calculate_slope_aspect(data)

# 检测山峰和山谷
peaks_valleys = processor.utils.detect_peaks_valleys(data)

# 创建山体阴影
hillshade = processor.create_hillshade(data, azimuth=315, altitude=45)
```

### 批量处理

创建批处理配置文件 `batch_config.json`:

```json
{
  "output_dir": "./output",
  "tasks": [
    {
      "type": "array",
      "output": "terrain_data.npy",
      "format": "npy"
    },
    {
      "type": "contour",
      "output": "contour_map.png",
      "levels": 25,
      "colormap": "terrain",
      "dpi": 200
    },
    {
      "type": "hillshade",
      "output": "hillshade.png",
      "azimuth": 315,
      "altitude": 45
    }
  ]
}
```

运行批处理:
```bash
terrain-processor input.tif --batch batch_config.json
```

## 📊 使用示例

### 示例1: 地形数据分析

```python
from terrain_processor import TerrainProcessor
import numpy as np

# 加载数据
processor = TerrainProcessor('terrain.tif')
data = processor.load_terrain_data()
stats = processor.get_terrain_statistics()

# 分析地形特征
print(f"地形类型判断:")
elevation_range = stats['elevation_range']
if elevation_range < 50:
    terrain_type = "平原"
elif elevation_range < 200:
    terrain_type = "丘陵"
else:
    terrain_type = "山地"

print(f"地形类型: {terrain_type}")
print(f"高程变化: {elevation_range:.2f} m")

# 寻找最高点
max_idx = np.unravel_index(np.nanargmax(data), data.shape)
max_elevation = data[max_idx]
print(f"最高点位置: {max_idx}, 高程: {max_elevation:.2f} m")
```

### 示例2: 自定义可视化

```python
# 创建多视图综合图
stats = processor.get_terrain_statistics()
processor.visualizer.generate_multi_view_plot(
    data, 'comprehensive_view.png',
    statistics=stats,
    figsize=(16, 12)
)

# 自定义颜色方案
custom_colors = ['#0066CC', '#00CC66', '#CCCC00', '#CC6600', '#CC0000']
colormap_name = processor.visualizer.create_custom_colormap(custom_colors, 'custom_terrain')

processor.generate_contour_image(
    'custom_contour.png',
    colormap=colormap_name
)
```

### 示例3: 与其他库集成

```python
import pandas as pd
import matplotlib.pyplot as plt

# 转换为pandas DataFrame
height, width = data.shape
y_coords, x_coords = np.mgrid[0:height, 0:width]

df = pd.DataFrame({
    'x': x_coords.flatten(),
    'y': y_coords.flatten(),
    'elevation': data.flatten()
}).dropna()

# 使用pandas进行分析
elevation_stats = df['elevation'].describe()
print(elevation_stats)

# 创建自定义图表
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['elevation'], alpha=0.5, s=1)
plt.xlabel('X坐标')
plt.ylabel('高程 (m)')
plt.title('高程分布散点图')
plt.show()
```

## 🛠️ 开发

### 项目结构

```
terrain_processor/
├── terrain_processor/          # 主包
│   ├── __init__.py            # 包初始化
│   ├── core.py                # 核心TerrainProcessor类
│   ├── reader.py              # TIF文件读取
│   ├── processor.py           # 数据处理
│   ├── visualizer.py          # 可视化功能
│   └── utils.py               # 工具函数
├── examples/                  # 使用示例
├── tests/                     # 测试文件
├── docs/                      # 文档
├── main.py                    # 命令行工具
├── setup.py                   # 安装配置
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```

### 运行测试

```bash
# 安装开发依赖
pip install -e .[dev]

# 运行测试
pytest tests/

# 运行示例
python examples/basic_usage.py
python examples/api_examples.py
```

### 代码风格

项目使用以下工具保证代码质量:
- **Black**: 代码格式化
- **Flake8**: 代码检查
- **pytest**: 单元测试

## 📋 系统要求

- Python 3.7+
- NumPy >= 1.19.0
- Rasterio >= 1.2.0
- Matplotlib >= 3.3.0
- SciPy >= 1.6.0
- Pillow >= 8.0.0

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Rasterio](https://rasterio.readthedocs.io/) - 地理栅格数据处理
- [Matplotlib](https://matplotlib.org/) - 数据可视化
- [NumPy](https://numpy.org/) - 数值计算
- [SciPy](https://scipy.org/) - 科学计算

## 📞 联系方式

- 作者: Kilo Code
- 邮箱: kilo@example.com
- 项目链接: [https://github.com/kilocode/terrain-processor](https://github.com/kilocode/terrain-processor)

## 🔄 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持TIF文件读取和三维数组输出
- 实现等高线图生成功能
- 提供命令行工具
- 添加API接口和使用示例

---

**如果这个项目对您有帮助，请给个⭐️支持一下！**