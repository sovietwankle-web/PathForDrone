# terrain_processor

无人机 (UAV) 三维路径规划工具集：粘滞分层快速行进法 (Viscous Hierarchical FMM)、
多路径规划、Q 值多路径、多机器人路径点分配、UAV 运动学约束、实时天气重规划、
神经路径点分配，以及对比/消融实验框架。

---

## 模块总览

```
terrain_processor/
├── viscous_hierarchical_fmm.py   ← 核心：FMM/多路径/Q值/多机器人
├── uav_constraints.py            ← UAV 运动学约束 + FMM 速度场修饰
├── online_replanner.py           ← 实时天气下的在线重规划闭环
├── neural_allocator/             ← 神经路径点分配 (imitation + RL)
└── experiments/                  ← 数据集 / 基线 / 对比 / 消融
```

| 文件 | 模块 | 类/函数入口 |
|------|------|------------|
| `viscous_hierarchical_fmm.py` | 粘滞速度场、金字塔、窄带 FMM | `ViscosityField`, `TerrainPyramid`, `NarrowBandFMM`, `HierarchicalPathPlanner` |
| `viscous_hierarchical_fmm.py` | 多路径 / Q值多路径 / 多机器人 | `MultiPathPlanner`, `QValueMultiPathPlanner`, `MultiRobotWaypointAllocator` |
| `uav_constraints.py` | 运动学约束 | `UAVKinematics`, `apply_uav_speed_modifier`, `attach_uav_constraints`, `enforce_turn_radius`, `check_path_feasibility` |
| `online_replanner.py` | 闭环重规划 | `WeatherSource`, `SyntheticWeatherSource`, `ReplanTrigger`, `OnlineReplanner` |
| `experiments/` | 实验工具 | `datasets.generate_dataset`, `baselines.*Baseline`, `comparison.main`, `ablation.main` |

---

## 1. 核心算法：粘滞分层 FMM

### 速度场公式

```
viscosity(x) = 1 / (1 + α·slope + β·roughness + γ·fog)
speed(x)     = base_speed × viscosity(x)
arrival_time φ(x) 满足 ‖∇φ‖ = 1/speed(x)
```

参数权重通过 `ViscosityParams(slope_weight=α, roughness_weight=β, fog_weight=γ)` 配置。

### 分层金字塔

| 层级 | 分辨率 | 用途 |
|------|--------|------|
| Level N-1 | 4× / 8× 下采样 | 粗求解，得到走廊 |
| Level k   | 2× 下采样     | 在上层走廊内细化 |
| Level 0   | 原始分辨率     | 最终路径 |

每一层只在上一层路径周围的 corridor 内运行窄带 FMM，节省时间和内存。

### 内存优化

* 稀疏字典存储 phi/status，而非全尺寸数组
* 到达目标后用 `phi_threshold_factor × phi_goal` 剪枝
* 可选 `release_far_points()` 释放远端已处理点

### 性能（30×50×50）

| 指标 | 单分辨率 | 分层 | 提升 |
|------|----------|------|------|
| 求解时间 | 1.48s | 0.39s | 3.8× |
| 存储点数 | 75 000 | 10 525 | -86 % |

### 最小用例

```python
from viscous_hierarchical_fmm import HierarchicalPathPlanner, ViscosityParams

planner = HierarchicalPathPlanner(
    terrain=terrain, fog_data=fog,
    n_levels=3,
    viscosity_params=ViscosityParams(slope_weight=0.3, fog_weight=0.5),
    spacing=(1.0, 1.0, 1.0),
)
path, stats = planner.plan(start=(2, 5, 5), goal=(25, 45, 45))
```

---

## 2. 多路径与多机器人扩展

### `MultiPathPlanner` — 迭代惩罚得到 K 条多样化路径

```python
from viscous_hierarchical_fmm import MultiPathPlanner, MultiPathParams

mp = MultiPathPlanner(terrain=terrain, fog_data=fog,
    multi_path_params=MultiPathParams(n_paths=3, penalty_weight=0.5,
                                       min_path_separation=8.0))
result = mp.plan_multi_path(start, goal)
# result.paths, result.costs, result.diversity_matrix
```

### `QValueMultiPathPlanner` — 在目标区域内寻找综合得分最高的终点

```
score = q_weight × Q(goal) − cost_weight × normalized_cost
```

### `MultiRobotWaypointAllocator` — 不互溶气体扩散 + 惠更斯原理

多机器人共享优先队列，按 phi 值竞争领土，自然形成路径点的就近指派与到达顺序。

---

## 3. UAV 运动学约束 (`uav_constraints.py`)

### 约束维度

| 维度 | 字段 | 作用入口 |
|------|------|---------|
| 水平 / 爬升 / 下降速度 | `vmax_horizontal` / `vmax_climb` / `vmax_descent` | 速度场修饰 + 后处理 |
| 最小转弯半径 | `min_turn_radius` | 路径平滑 (`enforce_turn_radius`) |
| 角度-速度 (各向异性) | `angle_decay_exponent`, `angle_speed_breakpoint` | `apply_uav_speed_modifier` |
| 海拔-速度 | `altitude_decay_per_m`, `altitude_ceiling` | `apply_uav_speed_modifier` |
| 最大坡度可飞 | `max_slope_angle` | `apply_uav_speed_modifier` |
| 最大俯仰角 | `max_pitch_angle` | `check_path_feasibility` |

### 角度-速度衰减曲线

```
f(θ) = cos(θ/2) ^ k          (θ < θ_break)
f(θ) ≈ 0.05                  (θ ≥ θ_break)
```

### 接入 FMM 的最小调用

```python
from uav_constraints import UAVKinematics, attach_uav_constraints, \
                            enforce_turn_radius, check_path_feasibility

kin = UAVKinematics(min_turn_radius=4.0, vmax_horizontal=15.0)
attach_uav_constraints(planner, kin)        # 一行接入：修饰每层 speed_field
path, _ = planner.plan(start, goal)
path    = enforce_turn_radius(path, kin, spacing)
metrics = check_path_feasibility(path, kin, spacing)
```

`metrics` 输出：`max_speed_mps`、`max_climb_rate_mps`、`max_descent_rate_mps`、
`min_curvature_radius_m`、`max_pitch_rad`、`n_violations` 等。

---

## 4. 实时天气重规划闭环 (`online_replanner.py`)

```
┌────────────────┐  fog(t) ┌─────────────────┐
│ WeatherSource  ├─────────► OnlineReplanner │
└────────────────┘         └────┬────────────┘
                                │ 阈值触发
                                ▼
                ┌──────────────────────────┐
                │ ReplanTrigger 判据        │
                │  fog_delta / speed_drop  │
                │  periodic / interval     │
                └──────┬───────────────────┘
                       │
                       ▼ 复用基速度场缓存
                ┌──────────────────────────┐
                │ HierarchicalPathPlanner  │
                │ + UAV speed modifier     │
                └──────────────────────────┘
```

### 触发器（`ReplanTrigger`）

* `fog_delta_threshold`：`‖fog_t − fog_last‖ / ‖fog_last‖` 超阈值
* `speed_drop_threshold`：当前路径走廊雾气均值上升比例
* `periodic_interval`：兜底周期触发
* `min_replan_interval`：节流，避免抖动

### 优化与安全

* `_base_speed_levels`：UAV 修饰后的"无雾基速度场"在 `__init__` 一次性构造，
  每次重规划只把新雾下采样并叠加，**不重建金字塔**
* `max_iterations` / `max_replans=12` / stuck 检测，避免极端环境死循环

### 典型输出

```
t= 0.00  reason=initial            new_path_len=29  plan=0.149s
t= 3.04  reason=fog_delta=0.642    new_path_len=18  plan=0.029s   ← 天气触发
t= 6.04  reason=fog_delta=0.433    new_path_len= 2  plan=0.007s   ← 天气触发
t= 9.04  reason=fog_delta=0.351    new_path_len= 2  plan=0.007s   ← 天气触发
```

12 次重规划总墙钟时间 0.29 秒。

### 用法

```python
from online_replanner import OnlineReplanner, SyntheticWeatherSource, ReplanTrigger
from uav_constraints   import UAVKinematics

weather = SyntheticWeatherSource(shape=terrain.shape, n_clouds=4, seed=0,
                                  drift_speed=1.0, amplitude=1.0)
trigger = ReplanTrigger(fog_delta_threshold=0.10, speed_drop_threshold=0.15,
                        min_replan_interval=2.0, periodic_interval=20.0)
rep = OnlineReplanner(terrain=terrain, weather_source=weather,
                      kinematics=UAVKinematics(), trigger=trigger,
                      n_levels=3, dt=1.0)
trace = rep.run(start=(2,5,5), goal=(25,45,45), t_max=120.0)
print(trace.summary())
```

---

## 5. 神经路径点分配 (`neural_allocator/`)

把多机器人路径点分配 (assignment) 学成一个 cross-attention 网络。
当前提供 imitation learning + PPO 强化学习两条流水线，已在 `checkpoints/` 落地若干模型。

```
neural_allocator/
├── model.py                  WaypointAllocationNet (RobotEnc + WPEnc + CrossAttn + ScoreHead)
├── allocator.py              NeuralWaypointAllocator (推理封装)
├── features.py               FeatureExtractor
├── hierarchical_features.py  HierarchicalFeatureExtractor (大场景)
├── geotiff_loader.py         GeoTIFF 数据接入
├── decoder.py                匈牙利 / 贪心 / 自回归解码
├── training/
│   ├── data_generator.py     合成场景
│   ├── augmentation.py       旋转/翻转/抖动
│   ├── dataset.py            torch Dataset + collate
│   ├── optimal_solver.py     ground-truth (匈牙利 / 暴力 TSP)
│   ├── reward.py             makespan reward shaping
│   ├── imitation_trainer.py  监督训练
│   └── rl_trainer.py         PPO
└── scripts/
    ├── train_imitation.py
    ├── train_rl.py
    ├── train_curriculum.py   课程式训练
    ├── evaluate.py
    └── benchmark_large.py
```

---

## 6. 实验框架 (`experiments/`)

### 数据集

```bash
python -m experiments.datasets --n 20 --shape 12,18,18 --out experiments_data/dataset.pkl
```

`generate_dataset()` 输出 `List[ScenarioRecord]`，每条含
`(terrain, fog, start, goal, spacing, seed, meta)`，可直接喂入下游训练或基线评估。

### 对比实验（4 基线）

```bash
python -m experiments.comparison --n 6 --shape 10,15,15 --turn-radius 4 --vmax 15
```

| 基线 | 描述 |
|------|------|
| `AStar26` | 26 邻域 A* on speed_field（无 UAV 约束） |
| `FMM_single` | 单分辨率粘滞 FMM |
| `FMM_hier` | 分层粘滞 FMM |
| `FMM_hier_UAV` | 分层粘滞 FMM + UAV 约束 + 转弯半径平滑 |

### 消融实验（9 配置）

```bash
python -m experiments.ablation --n 6 --shape 10,15,15
```

| ID | 关闭项 | 期望影响 |
|----|--------|---------|
| A0 | 全开（基线） | — |
| A1 | `slope_weight=0` | 路径穿越陡坡 |
| A2 | `roughness_weight=0` | 不绕行崎岖 |
| A3 | `fog_weight=0` | 忽略天气 |
| A4 | UAV 修饰 | 转弯/爬升违例剧增 |
| A5 | 转弯半径平滑 | 最小半径下降 |
| A6 | 方向各向异性保守化 | 速度更激进 |
| A7 | 海拔速度衰减 | 高空速度不变 |
| A8 | 用 `OnlineReplanner` 替换静态规划 | 闭环开销 + 重规划日志 |

### 评测指标

`check_path_feasibility` 给每条路径输出：

```
total_distance_m, total_time_s,
max_speed_mps, max_climb_rate_mps, max_descent_rate_mps,
min_curvature_radius_m, max_pitch_rad,
n_violations { speed, climb_rate, descent_rate, pitch_angle, turn_radius }
```

汇总表（CSV + 文本）：`success_rate`, `cost_mean/std`, `plan_time_mean/p95`,
`path_points_mean`, `viol_mean`, `min_R_mean_m`。

### 典型 smoke 数据 (n=3, 8×12×12)

```
对比：
  AStar26       cost=23.6  plan=0.090s  viol=2.0  R_min=2.60m
  FMM_single    cost=31.3  plan=0.020s  viol=2.0  R_min=0.71m
  FMM_hier      cost=31.3  plan=0.049s  viol=2.0  R_min=0.71m
  FMM_hier_UAV  cost=10.3  plan=0.061s  viol=1.0  R_min=4.09m  ✓

消融：
  A0_full        cost=9.7   viol=0.67  R_min=4.66m
  A4_no_uav      cost=31.5  viol=2.00  R_min=0.71m   (UAV 关闭后违例翻倍)
  A5_no_smooth   cost=9.7   viol=1.67  R_min=0.71m   (半径平滑失效)
```

---

## 7. 训练数据 → 模型回路

```
ScenarioRecord (datasets.py)
       │
       ├─► UAVHierarchicalFMMBaseline.run()  ── ground-truth path
       │                              │
       │                              └─► neural_allocator/training/imitation_trainer.py
       │
       └─► OnlineReplanner.run()              ── 在线重规划轨迹（用于 RL rollout）
                                       │
                                       └─► neural_allocator/training/rl_trainer.py
```

任意一条 `ScenarioRecord` 都可以同时用作：
* 静态规划基准评测
* 在线重规划基准评测
* 模仿学习 / 强化学习训练数据

---

## 8. 安装

```
numpy, scipy            (必需)
matplotlib              (可选，可视化)
torch                   (神经分配训练，仅 neural_allocator/)
rasterio / GDAL         (可选，GeoTIFF 加载)
```

---

## 9. 文件清单

```
viscous_hierarchical_fmm.py        3094 行  核心 FMM + 多路径 + 多机器人
uav_constraints.py                  494 行  UAV 运动学约束
online_replanner.py                 442 行  实时天气在线重规划
experiments/datasets.py             166 行  场景生成
experiments/baselines.py            227 行  4 基线
experiments/comparison.py           135 行  对比跑批
experiments/ablation.py             212 行  9 组消融跑批
neural_allocator/                  ~2900 行  神经分配 + 训练
```

---

## 10. License

MIT.
