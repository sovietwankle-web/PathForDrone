# UAV 三维路径规划：思路总览

本仓库实现一套面向无人机的三维路径规划工具链，核心放在 `terrain_processor/`。
本文档不重复 API 细节（`terrain_processor/README.md` 已覆盖），而是把
**整体思路**说清楚，重点四块：

1. **雾气（fog）的综合评定过程**：天气数据 → 粘滞系数 → FMM 速度场
2. **天气变化下的闭环重规划过程**：触发 → 增量更新 → 局部重规划
3. **对比 / 消融实验结果**：4 基线 + 9 配置，量化每个模块的贡献
4. **神经网络逻辑**：Cross-Attention 多机路径点分配器

---

## 总体框架

```
       ┌──────────┐   坡度 / 粗糙度 / 雾                ┌──────────────┐
DEM ──►│ 粘滞场   │──────────────────────────────────►│ 速度场 f(x)  │
       └──────────┘                                    └──────┬───────┘
                                                              │ UAV 修饰
                                                              ▼
                                                      ┌──────────────┐
                                                      │ 分层窄带 FMM │ ──► 路径
                                                      └──────┬───────┘
                                                             │ 转弯半径平滑
                                                             ▼
                                                      ┌──────────────┐
              天气变化   ──── 触发器 ────────────────►│ 闭环重规划   │ ──► 闭环轨迹
                                                      └──────────────┘
```

底层算法是**粘滞分层快速行进法 (Viscous Hierarchical FMM)**：把环境扰动写
进一个速度场 `f(x)`，求 Eikonal 方程 `‖∇φ‖ = 1/f(x)` 得到到达时间 φ，再沿
∇φ 反向回溯即得最优路径。所有"地形+天气+UAV"的影响最终统一在 `f(x)` 上。

---

## 1. 雾气的综合评定过程

雾气从来不是一个"二值禁飞区"，而是一个**连续的减速因子**，与坡度、粗糙度
一起进入同一个粘滞模型。

### 1.1 粘滞公式

```
viscosity(x) = 1 / (1 + α·slope_n(x) + β·roughness_n(x) + γ·fog_n(x))
speed(x)     = base_speed × viscosity(x)
```

* `slope_n`：地形高程梯度幅值，全场归一到 [0, 1]
* `roughness_n`：局部窗口内高程的标准差（3×3×3 默认），归一到 [0, 1]
* `fog_n`：雾密度场，`fog_data / max(fog_data)`，归一到 [0, 1]
* 默认权重 `α=0.3, β=0.2, γ=0.5`（在 `ViscosityParams` 中可调）

雾权重 γ 是三者中最大的——这是有意的：地形是相对稳定的物理量，雾是
**主动驱使 UAV 改道**的扰动，必须对速度场有显著拉低能力。

参考实现：`terrain_processor/viscous_hierarchical_fmm.py` 的
`ViscosityField.compute_viscosity` (terrain_processor/viscous_hierarchical_fmm.py:219).

### 1.2 综合评定的几个关键点

**(a) 归一而非裁剪**。雾密度直接除以全场最大值。这意味着同样的物理雾团，
在"晴天偶尔几片云"的场景里影响有限，在"全场大雾"的场景里影响相对被压平。
这是有意的——绝对的雾量没有意义，**对该 batch 内的最差路段的相对惩罚**才
有意义。

**(b) 加性而非乘性叠加**。slope、roughness、fog 三个 cost 在分母里相加，
不是相乘。带来的好处：单一极端值不会把 `speed` 直接打到 0，避免规划早期
出现 "完全不可达" 的死路；同时三者的边际效应**可独立调参**——消融时把
某个权重置 0 即可干净地剥离它的贡献。

**(c) 下界保护**。`viscosity ≥ min_viscosity = 0.01`，让最糟糕的格点仍
有可达性——FMM 永远能在有限时间内求出 φ，重规划永远有解（哪怕是"硬着头
皮穿过去"）。

**(d) UAV 后修饰**。雾决定的是"环境想让你慢多少"，UAV 修饰
（`apply_uav_speed_modifier`）再叠加"飞机能飞多快"：方向各向异性
（爬升 < 平飞）、海拔衰减、超坡度禁飞。两者乘在一起得到最终 `speed`。

### 1.3 雾在分层金字塔里的传播

`HierarchicalPathPlanner` 的多分辨率金字塔（默认 3 层，每层下采样 2×）
对雾做**同步降采样**：

```
雾原图 fog (Nz, Ny, Nx)
    │ scipy.ndimage.zoom (order=1, 线性)
    ▼
fog_L1 (Nz/2, Ny/2, Nx/2)   ← 粗层求走廊
    │
    ▼
fog_L0  (Nz, Ny, Nx)         ← 细层在走廊里精修
```

雾下采样到粗层时**不做阈值化**——保留连续梯度，让粗层 FMM 也能"轻轻绕开"
雾的边缘，而不是只能绕开雾的中心。

---

## 2. 天气变化下的闭环重规划过程

`online_replanner.py` 把雾从一个"输入"变成**一个时变信号源**，让规划
器从开环变成闭环。整个回路设计上有三个关键决策。

### 2.1 闭环结构

```
t=0 :  fog(0) ─► initial plan ─► path₀
       ┌─────────────────────────────────────────────────────────┐
while not at goal:
       ├ step along path (vmax × dt)
       ├ fog_now = source.get_fog(t)
       ├ fog_delta_L2 = ‖fog_now − fog_last‖ / ‖fog_last‖
       ├ speed_drop  = 当前路径前半段的雾增长比
       ├ if trigger.should_replan(...):
       │     replan from current position, fog_now
       │     fog_last ← fog_now
       └────────────────────────────────────────────────────────┘
```

参考实现：`OnlineReplanner.run` 主循环
(terrain_processor/online_replanner.py:306).

### 2.2 触发器：三条线 + 一条防抖

`ReplanTrigger` (terrain_processor/online_replanner.py:134) 用三条独立判
据，**任一满足即触发**，再用一条防抖线压制振荡：

| 判据 | 默认阈 | 物理含义 |
|------|--------|---------|
| `fog_delta_threshold` | 0.15 | 全局雾场 L2 相对变化（"天气整体变了"） |
| `speed_drop_threshold` | 0.30 | 当前路径走廊上雾均值的上升比例（"我前方变糟了"） |
| `periodic_interval` | 30 s | 周期兜底（"以防有什么我没检测到的"） |
| `min_replan_interval` | 2 s | 任何触发都至少间隔 2 秒（防抖） |

**为什么需要 speed_drop 而不只用 fog_delta**：全局 L2 大变化未必影响我的
路径——雾团可能在场景的另一头。`speed_drop` 只看路径前半段，能避免"远处
打雷我就改道"的过度反应。

### 2.3 复用基速度场缓存（关键性能优化）

如果每次重规划都从头构造金字塔、做坡度/粗糙度/UAV 修饰，**每次都是 0.15 s
量级**——12 次重规划就是 1.8 s，闭环延迟太大。

解决办法（`OnlineReplanner.__init__` 缓存层 + `_inject_fog`）：

```
init: 把"无雾基速度场" base = UAV_modifier(viscosity_no_fog(terrain)) 缓存为
      每层金字塔的常量。

replan: speed_level = base_level × 1/(1 + γ·fog_level)
         ↑ 只乘一个雾惩罚因子，不重建任何静态字段
```

效果：首次规划 ~0.15 s，后续每次重规划 ~0.01 s（典型 10× 以上加速）。
量化：12 次重规划总墙钟 ~0.29 s（实测见 `online_replanner` 模块顶部样例
日志）。

### 2.4 安全网

闭环算法最怕"卡死"。三层兜底：

* **max_replans = 12**：硬上限，超过即认输
* **stuck 检测**：最近 `stuck_window=10` 步累计位移 < `stuck_distance=0.05`
  视为卡住，触发 `reason="stuck"` 事件并退出
* **`path_idx >= len(path)-1` 兜底重规划**：路径走完了但没到目标 → 重规划

### 2.5 典型闭环输出

```
t= 0.00  reason=initial            new_path_len=29  plan=0.149s
t= 3.04  reason=fog_delta=0.642    new_path_len=18  plan=0.029s
t= 6.04  reason=fog_delta=0.433    new_path_len= 2  plan=0.007s
t= 9.04  reason=fog_delta=0.351    new_path_len= 2  plan=0.007s
...
```

每个事件都带了**触发原因**——这对调试触发器超参很关键。

---

## 3. 对比 / 消融实验结果

实验框架在 `terrain_processor/experiments/`，用同一份数据集
（`datasets.generate_dataset`）喂给所有方法，保证可比性。

### 3.1 对比实验（4 个基线）

```bash
python -m experiments.comparison --n 6 --shape 10,15,15
```

| 基线 | 描述 |
|------|------|
| `AStar26` | 26 邻域离散 A* on 速度场，**不带** UAV 约束 |
| `FMM_single` | 单分辨率粘滞 FMM |
| `FMM_hier` | 分层粘滞 FMM（金字塔，**无** UAV 修饰） |
| `FMM_hier_UAV` | 分层粘滞 FMM **+** UAV 速度修饰 **+** 转弯半径平滑 |

实测结果（n=3, shape=10×15×15，见 `experiments_data/cmp.csv`）：

| Method | cost | plan_t (s) | viol | min_R (m) | success |
|---|---|---|---|---|---|
| AStar26 | **23.63** | 0.090 | 2.0 | 2.60 | 1.00 |
| FMM_single | 31.26 | 0.020 | 2.0 | 0.71 | 1.00 |
| FMM_hier | 31.26 | 0.049 | 2.0 | 0.71 | 1.00 |
| **FMM_hier_UAV** | **10.28** | 0.061 | **1.0** | **4.09** | 1.00 |

**怎么读这张表**：

* **cost** 不是"走了多远"，是路径总成本（slowness × distance）；
  `FMM_hier_UAV` cost 最低，因为 UAV 修饰把陡峭/逆风段权重打高，
  规划器主动绕开了高成本区。
* **viol** = 路径上违反 UAV 约束（爬升率、俯仰角、转弯半径等）的次数；
  AStar/FMM 系列因为不知道 UAV 约束，每条路径有 2 处违例；UAV 版只剩 1 处。
* **min_R** = 路径最小曲率半径。FMM 系列默认输出"折线",曲率半径只有
  √2/2 ≈ 0.71 m（一个体素的对角）；转弯半径平滑后提升到 4.09 m，超过
  `min_turn_radius = 4 m` 的设定。
* AStar26 单步代价小（cost 看上去合理），但 plan_time 是 FMM 的 4–5 倍，
  在大场景下不可扩展。

**结论**：完整管线（分层 FMM + UAV 修饰 + 半径平滑）相比裸 FMM，**cost
↓ 67 %**、违例数 ↓ 50 %、最小曲率半径 ↑ 5.8 倍，规划时间只增加 30 %。

### 3.2 消融实验（9 配置）

```bash
python -m experiments.ablation --n 6 --shape 10,15,15
```

基线 = `A0_full`（全功能），其余 8 项各关一个模块：

| ID | 关闭项 |
|----|--------|
| A0 | 全开（基线） |
| A1 | `slope_weight = 0` |
| A2 | `roughness_weight = 0` |
| A3 | `fog_weight = 0` |
| A4 | UAV 速度修饰 |
| A5 | 转弯半径平滑 |
| A6 | UAV 方向各向异性保守化 |
| A7 | 海拔速度衰减 |
| A8 | 静态 → `OnlineReplanner`（同一份雾，看闭环开销） |

实测结果（`experiments_data/abl.csv`）：

| ID | cost | viol | min_R (m) | success |
|---|---|---|---|---|
| A0_full | 9.70 | 0.67 | **4.66** | 1.00 |
| A1_no_slope | 8.50 | 1.00 | 4.16 | 1.00 |
| A2_no_roughness | 9.00 | 1.00 | 4.39 | 1.00 |
| A3_no_fog | 8.37 | 0.67 | 4.27 | 1.00 |
| **A4_no_uav** | **31.48** | **2.00** | **0.71** | 1.00 |
| **A5_no_smooth** | 9.70 | 1.67 | **0.71** | 1.00 |
| A6_no_anisotropy | 7.27 | 0.67 | 4.66 | 1.00 |
| A7_no_alt_decay | 9.64 | 0.67 | 4.64 | 1.00 |
| A8_online | 13.40 | 1.00 | 4.88 | **0.67** |

**关键发现**：

1. **UAV 修饰 (A4) 是最大贡献者**——关掉它 cost 升 3.2 倍、违例翻倍、
   曲率半径降到 0.71 m（原始体素水平）。说明在速度场里就把 UAV 能力建
   模进去，比事后修正有效得多。
2. **半径平滑 (A5) 单独抓"几何可行性"**——关掉它 cost 不变（路径形状
   依旧合理），但 min_R 立刻塌回 0.71 m，违例 1.67。半径平滑是
   **几何后处理**，不影响 cost 但影响可飞性。
3. **slope / roughness / fog 各自影响在 5–10 %**（A1/A2/A3 vs A0）——
   说明三个环境因子彼此**有一定冗余**（都倾向于绕开困难地形），但每个
   都贡献一部分违例下降。
4. **conservative_anisotropy (A6) 反而让 cost 略降**——保守的各向异性
   在该数据集上略偏紧，关掉后规划器更激进（cost 更低）；但实际飞行
   时保守值更安全，这是一个**离线指标 vs 飞行安全**的 trade-off。
5. **A8 闭环 vs 静态**：success 从 1.00 掉到 0.67——闭环里 UAV 跟着
   `dt=1.0` 离散积分，且雾被动态触发重规划，**消耗更多时间预算**，部分
   场景在 `t_max=60s` 内没走到 goal。这反映了"静态规划做出来的路径是
   纸面最优，闭环里要再付一次时间开销"。

### 3.3 评测指标的统一口径

`check_path_feasibility` (terrain_processor/uav_constraints.py) 对每条路径
计算同一组指标：

```
total_distance_m, total_time_s,
max_speed_mps, max_climb_rate_mps, max_descent_rate_mps,
min_curvature_radius_m, max_pitch_rad,
n_violations { speed, climb_rate, descent_rate, pitch_angle, turn_radius }
```

所有基线 / 消融用同一函数，**不会出现"自己评自己"**的偏差。

---

## 4. 神经网络逻辑

`terrain_processor/neural_allocator/` 把**多机路径点分配**（multi-robot
waypoint assignment）从规则式（不互溶气体扩散 + 惠更斯原理）升级为可学习
的神经网络。规划本身仍由 FMM 完成；网络的任务是**说哪台 UAV 应该去哪些
路径点**。

### 4.1 任务定义

输入：
* `n_r` 台 UAV：位置 + 速度 + 当前已分配负载
* `n_w` 个路径点：3D 坐标 + 局部地形 cost + 局部 Q 值
* 两两特征：FMM-cost(robot_i, wp_j) + 欧氏距离

输出：
* 分配矩阵 `S ∈ R^{n_r × n_w}`：每个 UAV 对每个 wp 的"想要程度"得分
* 经匈牙利 / 贪心 / 自回归解码 → 最终 `robot_id → [wp_1, wp_2, ...]`

目标：**最小化 makespan**（最慢那台 UAV 的完成时间，比 sum-of-costs 更
符合多机协作的现实）。

### 4.2 网络架构（SADCHER 风格的 cross-attention）

```
robot_feats (B, n_r, 7)
   │  RobotEncoder (MLP + LayerNorm)
   ▼
robot_emb (B, n_r, 128)
                                        ┌─►  CrossAttn × 3 layers  ◄─┐
waypoint_feats (B, n_w, 5)              │                              │
   │  WaypointEncoder                   │   robot ⇄ waypoint           │
   ▼                                    │   双向注意力（含偏置）        │
waypoint_emb (B, n_w, 128) ─────────────┘                              │
                                                                       │
pairwise_feats (B, n_r, n_w, 2)                                        │
   │  PairwiseEncoder                                                  │
   ▼                                                                   │
pairwise_bias (B, n_heads, n_r, n_w) ──────────────────────────────────┘

                ┌──── ScoreHead (bilinear)  ─► (B, n_r, n_w)  分数矩阵
                │
                └──── ValueHead (mean-pool + MLP) ─► (B, 1)  状态价值 V(s) [PPO]
```

参考实现：`neural_allocator/model.py`。

### 4.3 三个设计点

**(a) Cross-attention 双向、带成对偏置**。
传统做法是直接拼接 robot/waypoint 特征丢给 MLP——丢失了"哪台 UAV 配哪个
wp 便宜"这种结构。这里：
* `PairwiseEncoder` 把 (FMM-cost, 欧氏距离) 映射成**每头注意力的偏置标
  量**（`(B, n_heads, n_r, n_w)`），直接加到 softmax 之前的 score。
* `CrossAttentionBlock` 做**两轮**注意力：robot → waypoint，再 waypoint
  → robot（转置 bias）。让两边各自"看见"对方的整体配置，而不是只单向。
* 这是 SADCHER 论文的核心思想：assignment 是一个**对称的二部图问题**，
  bias 编码图的边权。

**(b) FMM-cost 作为成对特征**。
`FeatureExtractor.compute_cost_matrix` 用两种方法估算 robot↔wp 的代价：
* `fast=True`：沿直线采 30 个点，做 `mean(1/speed) × distance` 的线积分。
  向量化实现，~1000× 比 FMM 快，用于训练。
* `fast=False`：跑窄带 FMM，得到精确 φ。慢但精确，用于推理。

训练时用 fast 是因为 batch size 大、追求吞吐；推理时用 exact 因为只跑
一次、要求精度。这是**训-推不一致**的有意设计——网络学的是"在近似代价
下也能给出好的分配"，鲁棒性更好。

**(c) 解码三选一**。`decoder.py` 提供：
* **匈牙利**：把 `S` 当代价矩阵，全局最优分配。`O(n³)`。
* **贪心**：每次取分数最大的 (i, j) 对，标记 j 已用，迭代。`O(n²)`。
* **自回归**：用 RL 时按时间步逐个采样 wp，已分配的 mask 掉。

推理时根据 `n_robots × n_waypoints` 规模在三者间切换。

### 4.4 两阶段训练

```
Stage 1 - Imitation Learning (监督)
  ground-truth: optimal_solver.py（小规模匈牙利或暴力 TSP）
  loss:         cross-entropy(scores, optimal_assignment) + label smoothing
  目的:          快速冷启动，让网络学会"长得像最优的"分配

Stage 2 - PPO (强化学习)
  reward:       -makespan + makespan_improvement shaping (reward.py)
  loss:         clipped policy gradient + value loss + entropy bonus
  目的:          在 imitation 的最优解不存在 / 不唯一时，自己发现更好的
                启发式（尤其是大规模 n_r > 8）
```

课程学习（`scripts/train_curriculum.py`）按以下阶段升难度，避免 RL 直接
在 hard 场景上 collapse：

| Stage | 地形大小 | UAV 数 | 路径点数 | Epochs |
|-------|----------|--------|----------|--------|
| 1 | 32–64    | 2–4    | 4–10     | 30 |
| 2 | 48–128   | 3–8    | 8–25     | 30 |
| 3 | 96–256   | 4–12   | 15–50    | 30 |
| 4 | 128–512  | 6–20   | 30–100   | 30 |

### 4.5 与下游规划的接口

```
ScenarioRecord (terrain, fog, robot_starts, waypoints)
   │
   ▼
FeatureExtractor   ───► robot_feats / waypoint_feats / pairwise_feats
   │
   ▼
WaypointAllocationNet ─► score matrix S
   │
   ▼
Decoder (Hungarian/Greedy/AR)  ───► assignment {robot_id: [wp_idx, ...]}
   │
   ▼
HierarchicalPathPlanner.plan(start_i, wp) 顺序规划每段
   │
   ▼
最终多机轨迹
```

分配器**不**自己规��路径——它把 NP-hard 的组合分配问题交给网络，把
**连续路径规划**留给 FMM。两者职责分离，相互独立可替换。

---

## 5. 仓库结构

```
terrain_processor/
├── viscous_hierarchical_fmm.py     核心 FMM + 粘滞场 + 多路径
├── uav_constraints.py              UAV 运动学约束 + 速度场修饰
├── online_replanner.py             实时天气闭环重规划
├── experiments/                    数据集 / 4 基线 / 9 消融
├── neural_allocator/               Cross-Attention 分配网络
└── README.md                       API 级文档（与本文件互补）
```

完整 API 入口、参数表、最小用例：见 `terrain_processor/README.md`。

---

## 6. License

MIT.
