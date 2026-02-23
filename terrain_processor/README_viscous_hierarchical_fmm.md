# 粘滞分层快速行进法 (Viscous Hierarchical FMM)

优化的三维路径规划算法，结合粘滞系数场、多分辨率金字塔和内存优化的窄带FMM。

## 特性

| 特性 | 描述 |
|------|------|
| 粘滞系数场 | 综合坡度、粗糙度、雾气等环境因素计算通行代价 |
| 多分辨率金字塔 | 粗到细求解，远处低分辨率，近处高分辨率 |
| 内存优化 | 稀疏存储 + 势值剪枝 + Corridor限制 |
| 性能提升 | ~3.8x 加速，~86% 内存减少 |

## 算法原理

### 1. 粘滞系数场 (Viscosity Field)

```
速度场: f(x) = base_speed × viscosity_factor(x)

viscosity_factor = 1 / (1 + α×slope + β×roughness + γ×fog)

- slope: 坡度因子（梯度幅值）
- roughness: 地形粗糙度（局部高程方差）
- fog: 雾气/环境密度
```

### 2. 多分辨率金字塔 (Hierarchical Pyramid)

```
Level 2: 4x下采样 ──→ 粗略路径
    ↓
Level 1: 2x下采样 ──→ 中等精度路径
    ↓
Level 0: 原始分辨率 ──→ 最终路径

策略: 在粗分辨率求解后，提取路径Corridor，仅在Corridor内进行细化求解
```

### 3. 内存优化策略

| 优化 | 方法 |
|------|------|
| 稀疏存储 | 使用dict存储phi/status，而非全尺寸数组 |
| 势值剪枝 | 到达目标后，phi > threshold × 1.5 的点停止扩展 |
| Corridor限制 | 细化层级仅在粗路径周围的带状区域内求解 |
| 内存释放 | 可选：释放远离当前前沿的已处理点 |

## 类结构

```
viscous_hierarchical_fmm.py
├── ViscosityParams          # 参数配置数据类
├── ViscosityField           # 粘滞系数场计算
├── TerrainPyramid           # 多分辨率地形金字塔
├── NarrowBandFMM            # 内存优化的窄带FMM求解器
└── HierarchicalPathPlanner  # 分层路径规划主类
```

## 快速开始

### 基本用法

```python
from viscous_hierarchical_fmm import HierarchicalPathPlanner, ViscosityParams
import numpy as np

# 准备数据
terrain = np.load('terrain.npy')  # 3D地形数据 (nz, ny, nx)
fog_data = np.load('fog.npy')     # 可选：雾气数据

# 配置参数
params = ViscosityParams(
    slope_weight=0.3,      # 坡度影响权重
    roughness_weight=0.2,  # 粗糙度影响权重
    fog_weight=0.5         # 雾气影响权重
)

# 创建规划器
planner = HierarchicalPathPlanner(
    terrain=terrain,
    fog_data=fog_data,
    n_levels=3,
    viscosity_params=params,
    spacing=(1.0, 1.0, 1.0)
)

# 规划路径
start = (2, 5, 5)
goal = (25, 45, 45)
path, stats = planner.plan(start, goal)

# 结果
print(f"路径点数: {len(path)}")
print(f"总耗时: {stats['total_time']:.3f}s")
print(f"路径代价: {stats['path_cost']:.2f}")
```

### 仅使用粘滞系数场

```python
from viscous_hierarchical_fmm import ViscosityField, ViscosityParams

vf = ViscosityField(ViscosityParams(slope_weight=0.5))

# 计算各分量
slope = vf.compute_slope(terrain, spacing=(1.0, 1.0, 1.0))
roughness = vf.compute_roughness(terrain)
viscosity = vf.compute_viscosity(terrain, fog_data)

# 获取FMM速度场
speed_field = vf.get_speed_field(terrain, fog_data, base_speed=1.0)
```

### 仅使用窄带FMM

```python
from viscous_hierarchical_fmm import NarrowBandFMM

fmm = NarrowBandFMM(shape=(30, 50, 50), spacing=(1.0, 1.0, 1.0))
fmm.set_speed_field(speed_field)
fmm.set_phi_threshold(1000.0)  # 可选：势值上限

success = fmm.solve(source=goal, goal=start)

# 获取结果
phi_value = fmm.get_phi_value(start)
phi_array = fmm.get_phi_array()  # 转为密集数组

# 内存统计
stats = fmm.get_memory_stats()
print(f"存储点数: {stats['phi_points']}")
print(f"剪枝点数: {stats['pruned_count']}")
```

## 参数说明

### ViscosityParams

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `slope_weight` | 0.3 | 坡度对粘滞系数的影响权重 |
| `roughness_weight` | 0.2 | 粗糙度影响权重 |
| `fog_weight` | 0.5 | 雾气/环境因子影响权重 |
| `roughness_window` | 3 | 计算粗糙度的滑动窗口大小 |
| `min_viscosity` | 0.01 | 最小粘滞系数（防止除零） |

### HierarchicalPathPlanner

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_levels` | 3 | 金字塔层级数 |
| `corridor_width` | 10 | 路径Corridor宽度（像素） |
| `phi_threshold_factor` | 1.5 | 势值剪枝阈值因子 |

## 性能对比

测试环境：30×50×50 三维网格

| 指标 | 单分辨率FMM | 分层FMM | 提升 |
|------|-------------|---------|------|
| 求解时间 | 1.48s | 0.39s | **3.8x** |
| 存储点数 | 75,000 | 10,525 | **86%↓** |
| 路径质量 | 基准 | 相当 | - |

## 输出统计

`planner.plan()` 返回的 `stats` 字典包含：

```python
{
    'success': True,           # 是否成功找到路径
    'total_time': 0.391,       # 总耗时（秒）
    'path_length': 64,         # 路径点数
    'path_cost': 207.27,       # 路径总代价
    'n_levels': 3,             # 金字塔层级数
    'level_stats': [           # 各层级统计
        {
            'level': 2,
            'shape': (8, 12, 12),
            'solve_time': 0.012,
            'accepted_count': 1152,
            'pruned_count': 0
        },
        ...
    ]
}
```

## 可视化

运行主模块会生成可视化结果：

```bash
python viscous_hierarchical_fmm.py
```

输出 `hierarchical_fmm_result.png`，包含：
- 3D路径可视化
- 粘滞系数场切片
- 地形切片
- 统计信息

## 文件结构

```
terrain_processor/
├── viscous_hierarchical_fmm.py      # 本模块
├── hierarchical_fmm_result.png      # 可视化结果
├── eikonal_path_planning.py         # 原始FMM实现
├── dynamic_eikonal_path_planning.py # 动态FMM
└── ...
```

## 依赖

```
numpy
scipy (ndimage, zoom)
matplotlib (可选，用于可视化)
```

## 扩展建议

1. **动态障碍物**: 结合 `dynamic_eikonal_path_planning.py` 支持时变环境
2. **GPU加速**: 使用CuPy替换NumPy进行并行计算
3. **自适应分辨率**: 根据地形复杂度动态调整局部分辨率
4. **路径平滑**: 添加B样条或贝塞尔曲线后处理

## 许可证

MIT License
