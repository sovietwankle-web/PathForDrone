# Experiments: UAV Path Planning

这是新增的对比与消融实验框架，配合：
* `terrain_processor/uav_constraints.py` — UAV 运动学约束
* `terrain_processor/online_replanner.py` — 在线重规划闭环

## 模块

| 文件 | 作用 |
|-----|-----|
| `datasets.py` | 程序化生成 (terrain, fog, start, goal) 场景 |
| `baselines.py` | 4 种基线：A*26 / 单分辨率 FMM / 分层 FMM / 分层 FMM + UAV |
| `comparison.py` | 跑全部基线并输出 CSV + 文本表 |
| `ablation.py` | 9 组消融 (含在线重规划) 并输出 CSV + 文本表 |

## 用法

```bash
cd terrain_processor

# 生成数据集
python -m experiments.datasets --n 10 --shape 12,18,18 --out experiments_data/dataset.pkl

# 对比实验
python -m experiments.comparison --n 6 --shape 10,15,15 --turn-radius 4 --vmax 15

# 消融实验
python -m experiments.ablation --n 6 --shape 10,15,15
```

## 评测指标

每条路径都用 `check_path_feasibility` 给出：
* `total_distance_m`、`total_time_s`
* `max_speed_mps` / `max_climb_rate_mps` / `max_descent_rate_mps`
* `min_curvature_radius_m`、`max_pitch_rad`
* `n_violations`：以 UAV 约束为基准的违例累计

汇总表给出：
* `success_rate` — 成功率
* `cost_mean` — 平均路径代价（单位：FMM 解出的到达时间，UAV 修饰下隐式带速度量纲）
* `plan_time_mean / p95` — 规划耗时
* `viol_mean` — 平均违例数
* `min_R_mean_m` — 平均最小曲率半径

## 消融组

| ID | 配置 | 期望影响 |
|----|------|---------|
| A0 | 全开 | 基线 |
| A1 | `slope_weight=0` | 路径会穿越陡坡 |
| A2 | `roughness_weight=0` | 路径在崎岖区不再绕行 |
| A3 | `fog_weight=0` | 不考虑天气 |
| A4 | 关闭 UAV 修饰 | 转弯/爬升率违例剧增 |
| A5 | 关闭转弯半径平滑 | 最小半径下降 |
| A6 | 关闭方向各向异性保守化 | 速度可能更激进 |
| A7 | 关闭海拔速度衰减 | 高空速度不变 |
| A8 | 用在线重规划 (静态雾) | 引入闭环开销 |

## 训练数据

`generate_dataset()` 已为后续监督学习/强化学习准备好可复现场景。
建议结合 `neural_allocator/training/` 现有 imitation/RL 流水线：
* 把 `ScenarioRecord.terrain/fog/start/goal` 输入；
* 用 `UAVHierarchicalFMMBaseline` 求解 ground-truth 路径；
* 学一个轻量代理网络在线预测速度场修正或下一步方向。
