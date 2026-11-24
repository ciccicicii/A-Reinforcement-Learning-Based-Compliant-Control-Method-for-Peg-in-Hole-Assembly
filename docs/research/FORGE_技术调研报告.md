# FORGE PegInsert 技术调研报告（面向《基于残差强化学习与柔顺控制的机器人精密插接方法研究》）

> 调研对象：Isaac Lab `Isaac-Forge-PegInsert-Direct-v0`（v2.3.0，仓库 `source/isaaclab_tasks/isaaclab_tasks/direct/forge/`）
> 调研日期：2026-11-22
> 证据等级标注：**✅已核验**（论文原文/官方源码/官方文档逐行核对）｜**⚠️推断**（由源码逻辑推导，未运行验证）｜**❓未确认**（缺公开资料或本机无法实测）
> 本机限制：调研在沙箱内完成，`torch.cuda.is_available()=False`、`nvidia-smi` 报 NVML 错误，**无法实跑训练/测显存**；第 4 节中所有显存/时长数字均为估算，已明确标注。

---

## 0. 一页速览（先看这张表）

| 项目 | 结论 | 证据 |
|---|---|---|
| 论文全称 | FORGE: **FOR**ce-**G**uided **E**xploration for Robust Contact-Rich Manipulation under Uncertainty（名字取自 "FORce-Guided Exploration"） | ✅ arXiv/项目页 |
| arXiv | **2408.04587**（v1 2024-08-08，v2 2025-01-02） | ✅ |
| 正式发表 | **IEEE RA-L 2025**, vol.10, pp.4436–4443, DOI `10.1109/LRA.2025.3551637` | ✅ Crossref |
| 单位 | MIT（Noseworthy, Roy）、USC（B. Tang）、NVIDIA（Wen, Handa, Fox, Ramos, Narang, Akinola）、DEVCOM ARL（Kessens） | ✅ |
| 官方开源 | **没有独立仓库**。项目页 "Code (Coming Soon)"，截至调研日仍未放出 | ✅ 项目页 |
| 事实上的参考实现 | Isaac Lab v2.2.0 起内置（PR #2968，作者 @noseworm 本人提交），v2.3.0 源码可用 | ✅ |
| 许可证 | 代码 BSD-3-Clause；**资产（USD）走 NVIDIA Isaac 资产许可，不是 BSD** | ✅ |
| 训练框架 | 论文用 **IsaacGym + Factory**；Isaac Lab 版是移植（rl_games PPO） | ✅ 论文 IV-B |
| 动作空间 | **7 维**：3 位置 + 3 姿态（roll/pitch 被强制置零，实际只有 yaw）+ 1 成功预测 | ✅ 源码 |
| 观测 | actor **24 维**（含 3 维力 + 1 维力阈值），critic **61 维**（特权信息） | ⚠️ 由源码推导 |
| 控制频率 | 仿真 120 Hz，`decimation=8` → **策略 15 Hz** | ✅ |
| 回合长度 | PegInsert **10 s / 150 步** | ✅ |
| 名义间隙 | 销 ⌀7.986 mm，孔 ⌀8.1 mm → 直径间隙 **0.114 mm**（半径 57 µm） | ✅ 配置 |
| 实际间隙 | USD 网格实测孔内径≈9.0 mm → 直径间隙≈1.0 mm，**与配置差 9 倍** | ⚠️ issue #8850（第三方测量，建议自测） |
| 官方并行环境数 | `scene.num_envs = 128` | ✅ |
| 官方训练量 | 200 个 PPO iteration ≈ **3.28 M 样本**（128×128×200） | ⚠️ 由配置推算 |
| 是否有预训练权重 | **无**（`--use_pretrained_checkpoint` 对 FORGE 无效） | ✅ 源码检索 |
| 8 GB 单卡能否跑 | **没实测**；建议 32 env 起，128 基本不现实（PhysX 缓冲+渲染占主导） | ❓ |

---

## 1. 论文本身

### 1.1 元信息

- **标题**：FORGE: Force-Guided Exploration for Robust Contact-Rich Manipulation under Uncertainty
- **arXiv**：[arxiv.org/abs/2408.04587](https://arxiv.org/abs/2408.04587)（HTML 全文：[v2](https://arxiv.org/html/2408.04587v2)，PDF：[v2](https://arxiv.org/pdf/2408.04587v2)）
- **发表**：IEEE Robotics and Automation Letters (RA-L)，2025 年第 10 卷，pp. 4436–4443，DOI [10.1109/LRA.2025.3551637](https://doi.org/10.1109/LRA.2025.3551637)（Crossref 核验；arXiv v2 即接收版）
- **作者与单位**：Michael Noseworthy¹、Bingjie Tang²、Bowen Wen³、Ankur Handa³、Chad Kessens⁴、Nicholas Roy¹、Dieter Fox³、Fabio Ramos³、Yashraj Narang³、Iretiayo Akinola³
  ¹ MIT，² USC，³ NVIDIA Research，⁴ DEVCOM Army Research Laboratory
- **项目页**：[noseworm.github.io/forge](https://noseworm.github.io/forge/)（含视频 [youtu.be/5WveyMyHX48](https://youtu.be/5WveyMyHX48)、supplementary PDF）
- **血统**：环境与奖励直接继承 **Factory**（Narang et al., RSS 2022, [arXiv:2205.03532](https://arxiv.org/abs/2205.03532)）；主要对比基线是 **IndustReal**（Tang et al., [arXiv:2305.17110](https://arxiv.org/abs/2305.17110)）。这两篇是读懂 FORGE 的前置阅读。

### 1.2 问题设定（POMDP）

- **任务**：销孔插入（8 mm 圆销）、齿轮啮合（medium gear）、M16 螺母拧入（只需 1/4 圈），以及补充的 snap-fit 卡扣（需 ~15 N 力）。
- **状态** $s$：EE 位姿/速度、固定件位姿、夹持件位姿、接触力、以及动力学参数（质量、摩擦、控制器增益等）。
- **观测** $o$（策略真正看到的）：含噪 EE 位姿与速度、**估计接触力**、**含噪的固定件位姿估计**、上一动作。**不观测夹持件位姿**（夹持后会滑动），**不观测动力学参数**（用上一动作隐式推断）。
- **动作** $a$：任务空间阻抗控制器的**目标位姿增量**，因假设零件都竖直，只需 $(x,y,z,\text{yaw})$ 四个自由度；再加 1 维成功预测，共 5 维"语义动作"。
- **成功判据**：横向居中 + 高度到位（销插入：销底距孔底 ≤1 mm，等价于低于孔口 24 mm）。

### 1.3 核心方法：三个组件（**这里必须纠正一个常见误读**）

**(1) Force threshold（力阈值条件化）**
策略额外观测一个标量 $F^{th}$，训练时每回合从 **[5, 10] N** 均匀采样；奖励里加**超额力惩罚**：

$$r^{FP}_t = -\beta \cdot \max(0,\ \|F_t\| - F^{th})$$

PegInsert 的 $\beta=0.2$（齿轮/螺母 0.05）。部署时可以**不改策略、只改这个标量**来调"允许的最大接触力"。论文的关键卖点：即使部署时控制器增益变了，策略也会自动输出更小的动作来压住力（Fig. 5 的增益实验）。

**(2) Dynamics randomization（动力学随机化）**
随机化对象：控制器 P/D 增益（$k_p\in[400,800]$，临界阻尼 $k_d=2\sqrt{k_p}$）、动作尺度 $\lambda\in[1.6,2.5]$ cm、零件摩擦（销 [0.5,1.0]）、零件质量、以及**执行器死区**（dead-zone，$[0,5]$ N，模拟关节摩擦导致"小指令力不生效"）。
**控制器参数不进入观测**——这是设计要点：策略不能"查表"知道自己增益多大，只能靠力反馈闭环。

**(3) Success prediction（成功预测 = 第 5/7 维动作）**
把动作空间扩一维 $a^{ET}\in[0,1]$，与策略共享网络权重；奖励为 $-\lvert \mathbb{1}_{success} - a^{ET}\rvert$，训练前期（成功率 <25%）**延迟开启**该惩罚，否则全是负样本。用途：① 部署时按置信阈值 $p_{term}$ 提前终止，缩短节拍；② 自动调力阈值（失败就把 $F^{th}$ 调大重试）。

**⚠️ 关于"探索噪声如何被力反馈调制"——答案是：没有显式调制。**
FORGE 的"force-guided exploration"指的是：**探索被力惩罚项塑形 + 策略被力阈值条件化 + 动力学随机化迫使策略依赖力反馈**，而不是"用力度去缩放高斯噪声"。论文用的是标准 recurrent PPO，高斯策略的 $\sigma$ 由 `fixed_sigma: False` 学习得到，与力无显式耦合。写毕设时如果写成"用力反馈调制探索噪声"就是错的；正确表述是"以超额接触力惩罚作为探索的安全约束，使探索轨迹落在低接触力流形附近"。

**(4) 是否 privileged / teacher-student？**
**没有 teacher-student 蒸馏**。用的是**非对称 actor-critic（asymmetric actor-critic）**：critic 拿全状态（含关节角、夹持件真实位姿、真实增益/阈值），actor 只拿 24 维可部署观测。特权信息只用于降低 value 估计方差，不参与策略蒸馏。

**(5) sim-to-real 怎么做？**
无系统辨识、无在线自适应。靠五件事叠加：
1. 观测与动作都**以固定件（孔）为原点**表达（工作空间泛化）；
2. 位姿估计误差建模（仿真里给固定件位姿加高斯噪声，$\sigma=2.5$ mm；实测 0–5 mm 分档评估）；
3. 动力学随机化（上）；
4. 动作低通（PLAI/EMA）与 5 cm 操作域限幅（避免末端跑飞）；
5. 力阈值 + 超额力惩罚（安全）。

真实系统：Franka Panda + **FrankaPy** 做 1 kHz 阻抗控制，策略 **15 Hz** 下发目标；Panda 的**关节力矩传感投影成 EE 力**（也可换 F/T 传感器）；固定件位姿先人工标定，再**人为注入噪声**做受控实验；最后的行星齿轮箱装配用 IndustReal 的感知系统。真机共 **>500 次试验**。

### 1.4 关键数值（论文 Table II，已从 PDF 还原）

| 类别 | 参数 | 8 mm Peg | Medium Gear | M16 Nut |
|---|---|---|---|---|
| 初始随机化 | 固定件 $x,y,z$ | $[0.55,0.65]$ m, $[-0.05,0.05]$ m, $[0,0.1]$ m | 同左 | 同左 |
| | 手 $x,y$（相对孔） | $\pm2$ cm | $\pm2$ cm | $\pm2$ cm |
| | 手 $z$（相对孔） | $[3.7,5.7]$ cm | $[2.5,4.5]$ cm | $[0.5,2.5]$ cm |
| | 手 yaw | $\pm45^\circ$ | $\pm45^\circ$ | $[-120,-90]^\circ$ |
| | 夹持件 $x,y$（相对夹爪） | $[-3,3]$ mm, 0 | 同左 | 同左 |
| | 夹持件 $z$ | $[14,20]$ mm | $[12,15]$ mm | $[10,16]$ mm |
| 观测噪声 | 位姿估计（固定件） | 2.5 mm | 2.5 mm | 2.5 mm |
| | 力 | 1 N | 1 N | 1 N |
| | EE 位置 | 0.25 mm | 0.25 mm | 0.25 mm |
| 动力学随机化 | 零件摩擦 | $[0.5,1.0]$ | $[0.38,0.75]$ | $[0.1,0.38]$ |
| | 控制器增益 $k_p$ | $[400,800]$ | 同左 | 同左 |
| | 动作尺度 $\lambda$ | $[1.6,2.5]$ cm | 同左 | 同左 |
| | 死区 | $[0,5]$ N | 同左 | 同左 |
| | **力阈值** | $[5,10]$ N | 同左 | 同左 |
| 奖励 | coarse $(a^c,b^c)$ | (50, 2) | (50, 2) | (100, 2) |
| | fine $(a^f,b^f)$ | (100, 0) | (100, 0) | (500, 0) |
| | 超额力惩罚 $\beta$ | **0.2** | 0.05 | 0.05 |
| | 成功距离 | 24 mm | 19 mm | 2.5 mm |
| | 放置距离（engage） | 2.5 mm | 2 mm | 2.5 mm |
| | 回合长度 | **150 步（10 s）** | 300 步（20 s） | 450 步（30 s） |

> 论文明确写了：**位姿估计噪声超过 $\sigma=2.5$ mm 训练会不稳定**（§V-B）。这句话很重要——做"精密插接"时不要盲目把噪声拉大。

### 1.5 真机结果（节选，均值(标准差)；每个指标 45 次试验 = 5 个工位 × 3 个误差档）

| 任务 | 方法 | 成功率↑ | 时长(s)↓ | $F_{mean}$(N)↓ | $F_{max}$(N)↓ |
|---|---|---|---|---|---|
| 8 mm Peg | **FORGE** | **0.84** | 2.82 | **5.51** | **12.84** |
| | 无力度量 | 0.82 | 3.18 | 7.09 | 14.16 |
| | 无动力学随机化 | 0.91 | 2.03 | 6.28 | 13.08 |
| | Baseline（无力/无 DR/无惩罚） | 0.64 | 2.35 | 11.81 | 17.93 |
| | IndustReal | 0.82 | 3.41 | 9.45 | 21.15 |
| Medium Gear | **FORGE** | **0.98** | 3.14 | 7.95 | 15.10 |
| | Baseline | 0.69 | 2.90 | 11.67 | 18.29 |
| M16 Nut | **FORGE** | **0.69** | 11.38 | 7.82 | 14.52 |
| | Baseline | 0.40 | 14.09 | 8.34 | 15.04 |

解读（对毕设有用的三条）：
1. **力观测本身不提升成功率，但显著降低接触力**（7.09→5.51 N，$F_{max}$ 14.16→12.84 N）；成功率主要来自动力学随机化。
2. 无惩罚时想靠"调低增益"压力会掉成功率（No FP 400kp: 0.64；600kp: 0.71，但 $F_{max}$ 16.58 N）。
3. 螺母任务最难（0.69），且"无力观测"直接崩到 0.40——力反馈在**判断是否对中/入牙**上不可替代。

**参考链接**
- 论文摘要页 <https://arxiv.org/abs/2408.04587>
- 论文 HTML（v2）<https://arxiv.org/html/2408.04587v2>
- DOI <https://doi.org/10.1109/LRA.2025.3551637>
- 项目页 <https://noseworm.github.io/forge/>
- Factory <https://arxiv.org/abs/2205.03532>；IndustReal <https://arxiv.org/abs/2305.17110>

---

## 2. 官方代码

### 2.1 开源状态（重要）

- 项目页明确写 **"Code (Coming Soon)"**，论文里也只承诺 "We plan to release the code with the paper"。截至 2026-11-22，**没有独立的 FORGE 仓库**。
- **事实上的官方实现**是 Isaac Lab 的 `direct/forge/`：由第一作者 @noseworm 提交（PR [#2968](https://github.com/isaac-sim/IsaacLab/pull/2968)，2025-07-31 合入），并在 **v2.2.0 发行说明**里作为新环境公布，v2.3.0 中代码结构未变。
- 差异：论文训练用 **IsaacGym + Factory**；Isaac Lab 版是 **Isaac Sim 5.x + DirectRLEnv** 的移植，配置数值有若干改动（见 §3.7），**不建议把 Isaac Lab 版结果直接当作论文数值复现**。

### 2.2 文件清单与职责

| 文件（v2.3.0） | 行数 | 职责 |
|---|---|---|
| `direct/forge/__init__.py` | 44 | 注册 3 个 Gym 环境，绑定 `env_cfg_entry_point` 与 `rl_games_cfg_entry_point` |
| `direct/forge/forge_env.py` | 383 | 核心：力传感器读取/平滑/坐标变换、7 维动作解释（目标位姿相对孔尖）、额外奖励项（超额力、成功预测）、重置期随机化、成功预测指标日志 |
| `direct/forge/forge_env_cfg.py` | 148 | 继承 Factory 配置，改 `obs_order/state_order`（加 `ft_force`、`force_threshold`）、控制器/噪声/事件配置、三个任务子类 |
| `direct/forge/forge_tasks_cfg.py` | 33 | 在 Factory 任务上覆写 FORGE 专属奖励系数与力阈值范围 |
| `direct/forge/forge_events.py` | 15 | `randomize_dead_zone`：每 2 s 重采样死区 |
| `direct/forge/forge_utils.py` | 35 | 控制器增益随机化（乘/除对称扰动）、`change_FT_frame`（力/力矩换系） |
| `direct/forge/agents/rl_games_ppo_cfg.yaml` | 123 | PPO + LSTM 超参（Peg/Gear 用） |
| `direct/forge/agents/rl_games_ppo_cfg_nut_thread.yaml` | 123 | 仅 `horizon_length: 256` 不同 |
| **依赖的父类** `direct/factory/*` | 1840 | 场景/资产/OSC 控制/关键点奖励/成功判据/重置同步逻辑——**读 FORGE 有一半时间在读 factory** |

### 2.3 任务变体与注册名

| 注册 ID | 任务 | `episode_length_s` | RL 配置 |
|---|---|---|---|
| `Isaac-Forge-PegInsert-Direct-v0` | 销孔插入（Hole8mm + Peg8mm） | 10.0 | `rl_games_ppo_cfg.yaml` |
| `Isaac-Forge-GearMesh-Direct-v0` | 齿轮啮合（GearBase + MediumGear，含两个侧齿轮） | 20.0 | `rl_games_ppo_cfg.yaml` |
| `Isaac-Forge-NutThread-Direct-v0` | M16 螺母拧入（BoltM16 + NutM16） | 30.0 | `rl_games_ppo_cfg_nut_thread.yaml` |

三者**共用同一个 `ForgeEnv` 类**，只换 cfg；`task_name` 分流成功判据与资产。

### 2.4 许可与依赖

- **代码许可**：BSD-3-Clause（Isaac Lab 全仓）。可自由修改、商用需保留版权声明。
- **资产许可**：`franka_mimic.usd`、`factory_*.usd`、`SeattleLabTable` 来自 NVIDIA Nucleus/S3，属 **Isaac Sim 资产许可**，**不是 BSD**，随仓库分发时要注意。
- **运行依赖**：Isaac Lab v2.3.0 + Isaac Sim 5.1 + Python 3.11 + PyTorch 2.7.0+cu128 + `rl_games`（v2.3.0 自带）。
- **是否需要真实 Franka + 力传感器？** 训练与仿真评估**完全不需要**。仿真机器人在腕部有一个名为 `force_sensor` 的刚体链接，用 `root_physx_view.get_link_incoming_joint_force()` 读取关节反作用力当作 6 维力/力矩（源码 `forge_env.py:36, 95`）。真机部署才需要 Franka（有 joint torque sensing 即可，不必额外 F/T 传感器）。
- **资产清单（S3，均已 HEAD 200 验证）**：
  - `.../Assets/Isaac/5.1/Isaac/IsaacLab/Factory/franka_mimic.usd`
  - `.../factory_peg_8mm.usd`、`.../factory_hole_8mm.usd`
  - `.../factory_gear_base.usd`、`.../factory_gear_medium.usd`、`.../factory_gear_small.usd`、`.../factory_gear_large.usd`
  - `.../factory_bolt_m16.usd`、`.../factory_nut_m16.usd`
  - 桌面：`.../Assets/Isaac/5.1/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd`
  根地址由 `carb` 设置 `/persistent/isaac/asset_root/cloud` 提供，在 `apps/isaaclab.python.kit` 中写死为 `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1`。**首次运行必须联网下载**（可预先缓存到 OV 缓存目录）。

**参考链接**
- FORGE 文档页（v2.3.0）<https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html#forge>
- 源码目录 <https://github.com/isaac-sim/IsaacLab/tree/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge>
- `forge_env.py` <https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py>
- `forge_env_cfg.py` <https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env_cfg.py>
- `forge_tasks_cfg.py` <https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_tasks_cfg.py>
- PR #2968 <https://github.com/isaac-sim/IsaacLab/pull/2968>
- v2.2.0 release notes <https://github.com/isaac-sim/IsaacLab/releases/tag/v2.2.0>

---

## 3. 任务细节（源码级，v2.3.0 逐行核对）

### 3.1 场景、几何与物理参数

| 项 | 值 | 出处 |
|---|---|---|
| 销（held） | `factory_peg_8mm.usd`，直径 **7.986 mm**，高 50 mm，质量 **0.019 kg** | `factory_tasks_cfg.py` |
| 孔（fixed） | `factory_hole_8mm.usd`，直径 **8.1 mm**，高 **25 mm**，质量 0.05 kg（继承默认值） | 同上 |
| 名义直径间隙 | 8.1 − 7.986 = **0.114 mm**（半径 57 µm） | 计算 |
| 论文口径 | "8 mm peg + **0.5 mm diametral clearance**" | 论文 §II-A |
| 网格实测 | 孔内径 ≈ **9.0 mm** → 直径间隙 ≈1.0 mm（issue 报告并给出 SHA256） | issue #8850 |
| 摩擦（配置） | 夹持件 0.75、机器人 0.75、固定件静摩擦 $U[0.25,1.25]$（128 桶）/动摩擦 0.25；PhysX 默认取平均 → **有效摩擦 [0.5,1.0]**，与论文一致 | `forge_env_cfg.py:51-85` |
| 质量随机化 | 夹持件 $+U[-0.005,0.005]$ kg（每回合 reset 时） | `forge_env_cfg.py:40-49` |
| 重力 | 机器人 `disable_gravity=True`、夹持件 `disable_gravity=True`；reset 期间全局关重力，结束再恢复 | `factory_tasks_cfg.py` / `factory_env.py:618, 820` |
| 求解器 | TGS（`solver_type=1`），**位置迭代 192**、速度迭代 1，`gpu_max_rigid_contact_count=2^23`、`gpu_max_rigid_patch_count=2^23`、`gpu_collision_stack_size=2^28`、`gpu_max_num_partitions=1` | `factory_env_cfg.py` |
| 接触参数 | `contact_offset=0.005`、`rest_offset=0.0`、`bounce_threshold_velocity=0.2`、`friction_offset_threshold=0.01`、`friction_correlation_distance=0.00625` | 同上 |
| 场景 | `num_envs=128`、`env_spacing=2.0`、`clone_in_fabric=True` | 同上 |

> **间隙是本毕设最该较真的数字**：配置写的 0.114 mm 属于"精密插接"级别（半径 57 µm，和 0.25 mm 的位姿噪声同量级）；但如果 USD 网格真是 9 mm 孔，实际是 1 mm 间隙的"粗插"。**建议自己用 SDF/网格量一遍**（issue #8850 给了方法），并在论文里写清用的是哪个数——否则"精密"二字站不住。

### 3.2 动作空间（7 维，语义要写清楚）

| 维度 | 索引 | 语义 | 缩放 / 映射 | 备注 |
|---|---|---|---|---|
| 位置 x,y,z | 0–2 | 相对**孔尖（固定件顶点）**的目标位置 | `a * pos_action_bounds` = `a * 0.05 m`，然后**单步位移被 clip 到 ±`pos_threshold`=0.02 m**（该阈值每回合再乘 $1\pm25\%$ 随机化） | 绝对目标（相对孔尖），不是相对当前 EE 的增量 |
| 姿态 roll,pitch | 3,4 | **被强制置零**（`rot_actions[:,0:2]=0`） | — | 观测里的 `prev_actions` 也把这两维清零 |
| 姿态 yaw | 5 | 世界系 yaw | 映射到 $[-180^\circ, +90^\circ]$：`-180 + 270*(a+1)/2` | 避开关节限位（限位在 +x/−y 象限） |
| 成功预测 $a^{ET}$ | 6 | 预测"任务已成功" | $[0,1]=(a+1)/2$ | 训练时有惩罚，**仿真里不用于终止** |

要点：
1. **有效自由度只有 4 个**（$x,y,z,\text{yaw}$）+ 1 个预测位；rl_games 仍输出 7 维。
2. 目标姿态 = 绕 x 轴 180°（末端朝下）× `bolt_frame_quat(roll=0,pitch=0,yaw)` → **始终竖直向下**，只调 yaw。
3. 动作先经 **EMA 低通**：`a_t = α·a_new + (1−α)·a_{t−1}`，`α ∈ [0.025, 0.1]`（每回合随机）→ 时间常数 10–40 个策略步（0.7–2.7 s），**非常强**的平滑，等价于 IndustReal 的 PLAI。
4. 工作域限制：目标点相对孔尖不超过 **5 cm**（`pos_action_bounds`），防止跑飞。

### 3.3 观测（actor 24 维 / critic 61 维）

**Actor（部署可用，`obs_order` + `prev_actions`）**

| # | 名称 | 维度 | 含义 / 噪声 |
|---|---|---|---|
| 1 | `fingertip_pos_rel_fixed` | 3 | 末端位置 − **含噪**孔尖位置；位置噪声 $\sigma=0.25$ mm，孔位姿估计噪声 $\sigma=1$ mm（**每回合采样一次**） |
| 2 | `fingertip_quat` | 4 | 末端姿态（世界系四元数，roll/pitch 各加 $\sigma=0.1^\circ$ 噪声，并随机整体取反号） |
| 3 | `ee_linvel` | 3 | 位置差分速度（**v2.3.0 有 bug，见 §5.1**） |
| 4 | `ee_angvel` | 3 | 姿态差分角速度，**只有 yaw 分量非零**（x,y 被清零） |
| 5 | `ft_force` | 3 | 平滑后的接触力 + $\sigma=1$ N 高斯噪声 |
| 6 | `force_threshold` | 1 | 本回合的力阈值（5–10 N），**这就是"力条件化"的入口** |
| 7 | `prev_actions` | 7 | 上一步（EMA 后）动作，roll/pitch 置零；用来隐式推断动力学 |

合计 **3+4+3+3+3+1+7 = 24**。

**Critic（特权，61 维）**：`fingertip_pos(3) + quat(4) + ee_linvel(3) + ee_angvel(3) + joint_pos(7) + held_pos(3) + held_pos_rel_fixed(3) + held_quat(4) + fixed_pos(3) + fixed_quat(4) + task_prop_gains(6) + ema_factor(1) + ft_force(3，无噪) + pos_threshold(3) + rot_threshold(3) + force_threshold(1) + prev_actions(7) = 61`（末尾 7 为动作）。

注意：
- **不含关节速度**（critic 也只有 `joint_pos`）；**不含显式 goal**——goal 被"孔尖为原点"的编码隐式表达了；**力矩不进观测**（只用力 3 维）。
- `cfg.observation_space/state_space` 在 `__init__` 里被按 `obs_order/state_order` 重算（`factory_env.py:28-33`），配置文件里写的 21/72 是占位值。

### 3.4 奖励函数（逐项 + 数值）

基类（Factory）的**多尺度关键点奖励**，对 4 个关键点的平均距离 $d$ 过一个 logistic 核 $\mathcal{K}_{a,b}(d)=1/(e^{-ad}+b+e^{ad})$：

| 项 | 公式 | 权重 | 数值（PegInsert） |
|---|---|---|---|
| `kp_baseline` | $\mathcal{K}_{5,4}(d)$ | +1 | 距离 0 时 0.2 |
| `kp_coarse` | $\mathcal{K}_{50,2}(d)$ | +1 | 距离 0 时 0.333 |
| `kp_fine` | $\mathcal{K}_{100,0}(d)$ | +1 | 距离 0 时 1.0 |
| `curr_engaged` | 高度进入孔深 90%（<22.5 mm） | +1 | 每步 |
| `curr_success` | 居中（xy<2.5 mm）且高度 <1 mm | +1 | 每步 |
| `action_penalty_ee` | $\|a\|_2$ | **0**（FORGE 置零） | — |
| `action_penalty_asset` | $\|\Delta p\|/0.02 + \lvert\Delta yaw\rvert/0.097$ | **−0.001** | 满量程约 −0.002/步 |
| `action_grad_penalty` | $\|a_t-a_{t-1}\|_2$ | **−0.1** | 因 EMA 平滑而很小 |
| **`contact_penalty`** | $\max(0,\|F\|-F^{th})$ | **−0.2**（Peg）／−0.05（Gear/Nut） | 超出阈值 5 N 即 −1/步 |
| **`success_pred_error`** | $\lvert\mathbb{1}_{succ}-(a_6+1)/2\rvert$ | −1.0，**全局成功率≥25% 后才启用** | 满量程 −1/步 |

单步奖励上限约 1.53（三项核）+1（engage）+1（success）≈ **3.5**；一旦稳定插到位，每步 +2 的"占位奖励"会主导 return——**所以训练出来的是"插进去并保持"的策略**。

### 3.5 终止、回合、控制

| 项 | 值 | 说明 |
|---|---|---|
| 仿真步长 | 1/120 s | `sim.dt` |
| `decimation` | 8 | 策略周期 8/120 = **1/15 s → 15 Hz** |
| 回合 | 10 s = **150 步** | `episode_length_s=10.0` |
| 终止 | **只有超时** | `_get_dones` 返回 `(time_out, time_out)`；成功不终止、超力也不终止 |
| 重置同步 | **所有 env 同时重置** | 源码注释 "We assume all envs will always be reset at the same time" |

**低层控制律（`factory_control.compute_dof_torque`，120 Hz）**：任务空间阻抗（OSC）

$$\tau = J^\top\big[k_p(p^*-p) + k_d(\dot p^*-\dot p)\big] + \text{nullspace}(k_{p,n},k_{d,n})$$

- 默认增益 `default_task_prop_gains = [565, 565, 565, 28, 28, 28]`（N/m, N·m/rad）；
- 每回合随机化：`get_random_prop_gains` 用 $1\pm U[0,0.41]$ **或取其倒数**（对称乘性扰动）→ 实际 $k_p\in[401,797]$，落在论文 [400,800] 内；
- **死区**：`default_dead_zone=[5,5,5,1,1,1]`，每回合 $\times U[0,1]$，且每 2 s 重采样一次（`forge_events.py`）——指令力小于死区直接归零，模拟关节摩擦；
- 夹爪用力控（`set_joint_position_target`，stiffness 7500），手臂用 `set_joint_effort_target`；
- 力矩限幅 ±100 N·m；nullspace 保持默认姿态。
- **力平滑**：`ft_smoothing_factor = 0.25` 的 EMA 在 120 Hz 上滤波（≈4 步时间常数，约 33 ms）。

### 3.6 随机化范围（代码 vs 论文）

| 项 | 代码（v2.3.0） | 论文 Table II | 一致性 |
|---|---|---|---|
| 固定件位置 | 每轴 $\pm5$ cm 均匀（+环境原点） | $x\in[0.55,0.65]$, $y\in[-0.05,0.05]$, $z\in[0,0.1]$ m | 等价 |
| 固定件 yaw | $U[0,360^\circ]$ | 同 | ✅ |
| 手初始位置 | 孔上方 +$z$(0.047 m) + $\pm(2,2,1)$ cm | 手 $z\in[3.7,5.7]$ cm → 编码为 4.7±1 cm | ✅ |
| 手初始 yaw 噪声 | $\pm0.785$ rad (±45°) | ±45° | ✅ |
| 夹持件在夹爪内位置 | $\pm(3,0,3)$ mm | $[-3,3]$ mm, 0 | ✅ |
| 孔位姿**估计**噪声 | $\sigma=1$ mm/轴（每回合一次） | **2.5 mm** | ❌ **差 2.5 倍** |
| EE 位置噪声 | 0.25 mm | 0.25 mm | ✅ |
| 力噪声 | 1 N | 1 N | ✅ |
| 力阈值 | $U[5,10]$ N | $[5,10]$ N | ✅ |
| 控制器增益扰动 | ±41%（乘/除） | $k_p\in[400,800]$ | ✅ 等效 |
| 单步位移阈值扰动 | ±25% | （论文的 $\lambda$ 随机化） | ✅ 等效 |
| 姿态阈值扰动 | ±29% | — | 代码新增 |
| 死区 | $[0,5]$ N / $[0,1]$ N·m，每 2 s 重采样 | $[0,5]$ N | ✅ |
| 固定件摩擦 | 静 $U[0.25,1.25]$、动 0.25（平均合成→0.5–1.0） | $[0.5,1.0]$ | ✅ 等效 |
| 夹持件质量 | ±5 g | 论文只说"随机化" | ⚠️ |

> **复现论文级鲁棒性时，第一件要改的就是 `ForgeObsRandCfg.fixed_asset_pos`：`[0.001]*3` → `[0.0025]*3`。** 当前 1 mm 的设置让"搜索"行为弱化很多，成功率会虚高。

### 3.7 Isaac Lab 版 vs 论文的 6 处实质差异（写论文时要说明用哪个）

1. **位姿估计噪声**：1 mm vs 2.5 mm（见上）。
2. **动作/观测的坐标系**：论文说"相对固定件的位姿变换"（含旋转）；**代码里只有原点相对于孔尖，坐标轴仍是世界系**——`ctrl_target = 孔尖位置 + a*0.05`（世界系平移），yaw 目标也是世界系绝对角。⚠️由源码推断，未运行验证。
3. **力的参考系**：论文说"投影到 EE 系"；代码调用 `change_FT_frame` 时把 `source_frame` 传成**单位变换**，即只做了平移（力臂）变换，**力矢量没有旋转到任何工具/孔坐标系**；而 PhysX 的 `get_link_incoming_joint_force()` 按 Isaac Lab 文档是"parent body frame"。→ 力观测的坐标系语义与论文不一致（这也是 issue #3513 争议的一部分）。⚠️推断。
4. **成功预测不参与终止**：论文用它做 early termination，Isaac Lab 版 `_get_dones` 里**没有接**，只训练不推理。
5. **无课程**：论文只在 IndustReal 基线里提到 SBC 课程；FORGE 本体无课程，靠 DR。
6. **仿真器**：IsaacGym → Isaac Sim/PhysX（接触求解器、摩擦模型都变了），绝对成功率不可直接对比。

### 3.8 参考链接
- `factory_env.py`（奖励/成功/重置）<https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py>
- `factory_env_cfg.py`（decimation/仿真/机器人）<https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py>
- `factory_tasks_cfg.py`（几何/质量/奖励系数）<https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py>
- `factory_control.py`（OSC 阻抗律）<https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py>
- 论文 Table II 原始页 <https://arxiv.org/html/2408.04587v2#A1>

---

## 4. 复现门槛（8 GB / RTX 4060 Laptop）

> **重要声明**：本节所有显存与时长数字 **❓未实测**（本沙箱无 CUDA）。下面给出的是"由配置推算的量级 + 可立即执行的压榨方案 + 测量方法"，请以你自己跑出来的数为准。

### 4.1 官方配置的算力画像

| 项 | 值 | 影响 |
|---|---|---|
| 并行环境 | **128**（`scene.num_envs`，`--num_envs` 缺省即 128） | PhysX 每步要解 128×2 个 articulations |
| 每 iteration 样本 | 128 env × `horizon_length` 128 = **16 384** | |
| PPO iterations | **200**（`max_epochs`，`--max_iterations` 可覆盖） | → **≈3.28 M 策略步** |
| 网络 | 策略 LSTM(1024)×2 层 + MLP [512,128,64] ELU；critic 同规模（独立 central value） | LSTM 1024 是显存/时间大头 |
| minibatch | 512，4 mini-epoch → 每 iteration 32 个 minibatch × 4 | |
| 混合精度 | 开（`mixed_precision: True`） | |
| 物理 | 120 Hz × 位置迭代 192，接触缓冲 2^23/2^23/2^28 | **每步物理成本极高** |
| 论文规模 | 未披露环境数与机时（❓未确认）；只说明"每个任务 3 个随机种子" | |

**时长估算（❓估算，非实测）**：3.28 M 步 ≈ 60.7 小时"单环境仿真时间"。若 4060L 上 32 env 能达到 ~600–1200 策略步/s，则 rollout ≈ 45–90 min；加 LSTM 反传（128 长序列 × 32 env）后，200 iteration 大概 **4–10 h**；4090 桌面级大约 **1–2 h**。注意：**减 env 会让每 iteration 样本变少**，要凑同样的 3.28 M 样本，32 env 需要 **800 iteration**。

### 4.2 8 GB 能压到多少 env？（结论 + 依据）

**结论（❓未实测，建议按此起测）**：从 **`--num_envs 32`** 开始；32 稳了再试 64；**128 在 8 GB 上大概率 OOM**。

依据：
- 显存主要不是被 env 数吃掉的，而是 **PhysX 的全局缓冲 + Isaac Sim 渲染/材质系统**。本仓库 `SimulationCfg` 把 `gpu_max_rigid_contact_count`、`gpu_max_rigid_patch_count` 都设成 `2^23`（约 840 万），`gpu_collision_stack_size=2^28`，这是为"低间隙、多接触"任务准备的激进值。
- 参考你已有记录：空场景 headless ≈ 2.2 GB。FORGE 在此之上叠加固定 PhysX 缓冲（量级 0.5–1.5 GB）+ 每 env 的 articulation/contact 缓冲，128 env 很可能落在 5–8 GB 区间并触发 OOM；32 env 大约 3–5 GB。
- 渲染侧：FORGE **没有相机传感器**，`--headless` 即可关掉视口渲染（**不要加 `--enable_cameras`，不要加 `--video`**）。

**必须做的减法（按收益排序）**

| 手段 | 做法 | 风险 |
|---|---|---|
| ① headless | `--headless`（不创建视口） | 无 |
| ② 降 env 数 | `--num_envs 32`（先 8 试通） | 样本变少，需加 iteration |
| ③ 降接触缓冲 | 覆写 `env_cfg.sim.physx`：`gpu_max_rigid_contact_count=2**20`、`gpu_max_rigid_patch_count=2**20`、`gpu_collision_stack_size=2**25` | 接触多时可能截断，影响穿透/成功率 |
| ④ 降求解迭代 | `max_position_iteration_count: 192 → 64`（源码注释说 192 是"避免互穿"的关键） | **会改变接触动力学**，成功率不可比；只用于"跑通" |
| ⑤ 关掉不用的传感器 | 不加相机；`activate_contact_sensors` 只在需要时留 | 无 |
| ⑥ `clone_in_fabric` | 保持 `True`（Isaac Sim 5.x 下更省内存/更快） | 无 |

### 4.3 命令（可直接抄）

```bash
# 0) 环境（按你现有记录）
source /home/mi/miniforge3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/opt/isaac-workspace/conda-pkgs XDG_CACHE_HOME=/opt/isaac-workspace/cache TMPDIR=/opt/isaac-workspace/tmp
conda activate /opt/isaac-workspace/envs/isaaclab
cd /home/mi/graduation_design/RL/IsaacLab

# 1) 冒烟：单环境加载（不训练）
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Forge-PegInsert-Direct-v0 --num_envs 8 --headless

# 2) 训练（headless，32 env，先跑 50 iteration 看显存与吞吐）
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Forge-PegInsert-Direct-v0 \
  --num_envs 32 --headless --max_iterations 50

# 3) 续训
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Forge-PegInsert-Direct-v0 --num_envs 32 --headless \
  --checkpoint logs/rl_games/Forge/test/nn/Forge.pth

# 4) 回放 / 评估（注意：FORGE 无预训练权重，--use_pretrained_checkpoint 无效）
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Forge-PegInsert-Direct-v0 --num_envs 16 \
  --checkpoint logs/rl_games/Forge/test/nn/Forge.pth
```

- 日志目录：`logs/rl_games/Forge/test/`（`name: Forge` + `full_experiment_name: test` 写死在 yaml 里，**每次训练都会覆盖同一个目录**——第二次训练前务必手动改名或改 yaml，否则权重被覆盖）。
- `--max_iterations` 会覆盖 yaml 里的 `max_epochs: 200`（`train.py:112`）。
- `num_actors` 由 `env.unwrapped.num_envs` 覆盖写入（`train.py:210`），所以 `--num_envs` 就是 PPO 的 batch 基数。

### 4.4 怎么统计成功率

环境在**每次重置时**计算 `extras["successes"] = 当前成功 env 数 / num_envs`（因为所有 env 同步重置，这就是"本批回合成功率"），同时记录 `success_times`。最小评估循环：

```python
obs, _ = env.reset()
for _ in range(env.unwrapped.max_episode_length):
    with torch.inference_mode():
        act = policy(obs)                 # 7 维
    obs, rew, term, trunc, extras = env.step(act)
assert "successes" in extras              # 最后一步（reset 时）才有
print("success rate:", float(extras["successes"]))
```

更稳的做法：直接调用 `env.unwrapped._get_curr_successes(success_threshold=0.04)` 拿布尔掩码自己累计；或读 `env.unwrapped.ep_succeeded`。想复刻论文的"分误差档"评估，就把 `ForgeObsRandCfg.fixed_asset_pos` 在 `[0.0, 0.001, 0.0025, 0.005]` 之间扫一遍。

### 4.5 参考链接
- 训练脚本 <https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/scripts/reinforcement_learning/rl_games/train.py>
- 回放脚本 <https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/scripts/reinforcement_learning/rl_games/play.py>
- RL 脚本用法文档 <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/reinforcement-learning/rl_existing_scripts.html>

---

## 5. 已知坑（按"会不会毁掉复现"排序）

### 5.1 【致命】速度观测在 v2.3.0 是"纯噪声"（issue #8874）

`forge_env.py:57-92` 调用 `super()._compute_intermediate_values(dt)`，父类已经用**真实位置**算完 `ee_linvel_fd` 并把 `prev_fingertip_pos` 更新为**当前真实位置**；子类随后又用"含噪位置"重算：

$$\text{obs\_linvel} = \frac{(p_t+\epsilon_t) - p_t}{dt} = \frac{\epsilon_t}{dt}$$

→ **线性速度观测里没有真实运动，只有噪声/dt**（0.25 mm / 8.33 ms ≈ 0.03 m/s 量级的白噪声）。角速度同理（且 x,y 被清零，只剩 yaw 噪声）。**24 维观测里有 6 维（25%）是废的**。issue 明确写了"also present in v2.3.0"，修复 PR #8887 截至调研日仍 open。
→ 影响：策略失去阻尼信息，接触阶段更容易振荡；这可能是 Isaac Lab 版难训/成功率与论文差距的主因之一。
→ 建议修复（自己打补丁）：为 FORGE 单独维护 `prev_noisy_pos/prev_noisy_quat`，用"上一帧含噪位置"做差分。这是一条**很好的毕设"工程贡献"**。

### 5.2 【严重】`change_FT_frame` 的换系实现有问题（issue #3513）

代码只对力矩做了平移，且用的是 $+p\times F$；按 Modern Robotics 3.95，力矩项符号/顺序不对；同时 `source_frame` 传单位变换，导致**力矢量没有被旋转**（见 §3.7）。力矩不进观测，所以对策略影响有限，但**力的方向语义**（世界系/腕部系）和论文不一致，调试时容易误判"力方向反了"。修复在 v2.3.0 之后才进主干。

### 5.3 【严重】`pos_action_bounds` / `pos_action_threshold` 语义与 Factory 相反（issue #5424）

Factory 里 `threshold` 是"单步增量尺度"、`bounds` 是"工作域硬限"；**FORGE 里正好反过来**：`bounds(0.05)` 当动作尺度、`threshold(0.02)` 当单步 clip 且被随机化。读代码/改代码时极易搞混。写论文时按"$\lambda$ = 单步最大位移 2 cm（±25%）、操作域半径 5 cm"来描述才与论文对得上。

### 5.4 孔的实际间隙可能是配置值的 9 倍（issue #8850）

配置 8.1 mm vs 网格 ≈9.0 mm。**先量再写**。若实际是 1 mm 间隙，所谓"精密插接"要重新定位（或者自己换更紧的资产）。

### 5.5 FORGE 没有预训练权重

`play.py --use_pretrained_checkpoint` 对 FORGE 会失败（注册表里没有条目）。必须自己训。

### 5.6 日志目录固定为 `test`，会覆盖

`rl_games_ppo_cfg.yaml` 里 `full_experiment_name: test`，两次训练互相覆盖。改 yaml 或用 `--seed` 区分不了，必须改名字。

### 5.7 同步重置假设（加 early termination 会炸）

`factory_env._get_dones` 返回全 true/全 false，`_reset_idx` 注释明说"假设所有 env 同时重置"。若你按论文实现"成功预测终止"，必须自己处理**部分 env 重置**（`randomize_initial_state` 里有全局 `set_gravity(0)` 和 IK 重试循环，部分重置容易出状态不一致）。这也是 §6 里"保留同步回合"的理由。

### 5.8 资产必须联网下载

首次 `zero_agent` 会从 S3 拉 `franka_mimic.usd`、`factory_*.usd`、桌面 USD。国内网络建议先配代理或预缓存；离线机器会卡在 `omni.client` 超时（表现为长时间无输出，不是崩溃）。

### 5.9 版本兼容性

- **v2.3.0 官方就是配 Isaac Sim 5.1**（release notes 明确"built on Isaac Sim 5.1"），你现在的 5.1.0 组合是官方支持的，不需要降级。
- 但 v2.3.0 **早于** #3513/#8779/#8874 这些修复，所以上面 5.1/5.2 的 bug 一定会遇到。
- Isaac Sim 6.x + Isaac Lab 3.x 把任务路径改成了 `contrib/forge/`、任务名也改了（#6163 "Rename all contrib tasks"），网上教程的路径/ID 会对不上，**别混用**。

### 5.10 其它小坑

- `episode_length_s` 改了要同步检查 `max_episode_length`（`DirectRLEnv` 用 `episode_length_s/(dt*decimation)` 取整）。
- `--num_envs` 太小时 `minibatch_size=512` 相对 batch 偏大（32×128=4096，尚可；8 env 就只有 1024，会退化成几乎 full-batch）。
- 渲染窗口 + 训练同开必炸显存；`--video` 会强制开相机。

---

## 6. 与"残差强化学习 + 柔顺控制"的衔接点

### 6.1 先把 FORGE baseline 的性质说准

FORGE 不是"纯位置控制的端到端"，它其实是**两层**：

```
策略(15 Hz, 24 维观测) --目标位姿(孔尖原点, 4 DoF)--> EMA 低通 --clip 2cm--> OSC 阻抗(120 Hz, kp≈565, kd 临界阻尼, 死区 0-5N) --> 关节力矩
```

也就是说：**内层已经是一个笛卡尔阻抗（柔顺）控制器**，FORGE 学的是"往哪儿放目标位姿"。所谓"端到端"是指**目标位姿由网络直接给出、没有解析搜索律**，而不是"没有低层控制器"。

这个认识直接决定你的论文定位：**你的"解析/柔顺控制器 + 残差策略"不是替换掉 FORGE 的控制器，而是替换掉它的"目标位姿生成器"**——把"网络从零生成 4 DoF 目标"换成"解析搜索/力控律生成名义目标 + 网络只学残差"。

### 6.2 保留 / 替换清单

| 模块 | 处置 | 判断依据 |
|---|---|---|
| 场景、资产、USD、桌面 | **保留** | 与算法无关，重做成本高；且是论文可比性的基础 |
| 15 Hz 策略 / 120 Hz 物理 / decimation=8 | **保留** | 与真机 15 Hz 一致；改了对齐不了论文 |
| OSC 阻抗 + 临界阻尼 + nullspace | **保留**（可加阻抗参数调度） | 这就是"柔顺控制"的落点；$\tau=J^\top(k_p e + k_d \dot e)$ 已是标准阻抗 |
| 关键点奖励（5,4)/(50,2)/(100,0) | **保留** | 多尺度、平滑、与几何无关，做 residual 时直接复用，省一大块调参 |
| 成功/engage 判据（2.5 mm 居中 + 1 mm 深度） | **保留** | 与论文/Factory 对齐，便于横向比较 |
| 观测里的力 + 力阈值 | **保留并扩展** | 这是"精密插接"的信息源；建议再加"力变化率/接触状态标志" |
| 动力学随机化（增益、死区、摩擦、质量、力阈值） | **保留全部** | 这是 sim2real 的主力，删了真机必崩（论文 No DR 成功率反而升、但泛化差） |
| 位姿估计噪声 | **保留并加大到 2.5 mm** | 当前 1 mm 太容易，体现不出"搜索" |
| 动作：网络直接输出 4 DoF 绝对目标 | **替换为残差** | 见 6.3 |
| EMA 0.025–0.1 的动作低通 | **保留但降到残差通道之后**，α 调大（0.2–0.5） | 原 α 太小（时间常数 0.7–2.7 s），会把残差的高频修正**吃掉**；残差需要更快响应 |
| 第 7 维成功预测 | **保留为辅助头**（不接终止） | 免费的多任务正则；若要接终止，先解决 5.7 的同步重置 |
| 超额力惩罚 | **上移为硬约束/投影** | 见 6.4，这是相对 FORGE 的"方法增量" |

### 6.3 残差接口的三种设计（推荐第 2 种）

记孔尖估计位姿为 $T_{h}$，解析/柔顺控制器给出的名义目标为 $T^{nom}_t$，网络输出 $r_t\in[-1,1]^k$：

| 方案 | 形式 | 动作维度 | 优点 | 风险 |
|---|---|---|---|---|
| A. 目标位姿残差 | $T^{cmd}=T^{nom}\oplus \Delta(r)$，$\Delta$ 限幅 1–3 mm / 2° | 2–6 | 与 FORGE 动作语义完全兼容（同一个 OSC 入口），改动最小 | 残差被"绝对目标"尺度淹没，需重设限幅 |
| **B. 横向残差 + 解析轴向（推荐）** | $x,y$ 由网络残差（±1–2 mm），$z$ 由"力控下压/螺旋搜索"解析给出，yaw 固定 | **2**（或 3 含 yaw） | 维度最低、样本效率最高、8 GB 友好；轴向由力闭环保证安全，正好对应"柔顺" | 需要写一个可靠的轴向力控律（用 $F_z$ 做 PI 或恒速下压 + 力限幅） |
| C. 力/导纳参数残差 | 网络调 $k_p$、$F^{th}$、下压速度 | 1–3 | 直接对应"柔顺参数学习"，讲故事最漂亮 | 参数→行为的映射非线性强，PPO 方差大，难训 |

推荐 **B**，理由：
1. FORGE 的动作本来就是"以孔尖为原点的绝对目标"，把**名义点**设成"孔尖正上方 + 解析下压"，残差只负责 $x,y$ 的**对中修正**，语义干净，动作维度 7→2，训练量可降一个数量级——这对 8 GB 卡是决定性的。
2. 力反馈在"何时开始下压/是否卡住"上比网络更可靠（论文 nut 任务无力观测直接 0.40 也印证了力是"对中判据"）；把这件事交给解析律，网络只需学"横向往哪边挪"。
3. 保留 FORGE 的 $F^{th}$ 条件化：解析层做**力限幅投影**（$F_z>F^{th}$ 就停止下压/回退），网络残差在剩余自由度里探索——这比 FORGE"只用奖励软惩罚"更安全，是真增量。

### 6.4 可以写成"贡献"的三个点（有据可依）

1. **力约束从"奖励软惩罚"升级为"动作空间投影"**：FORGE 的 $\max(0,\|F\|-F^{th})$ 只是惩罚，策略仍可能瞬间超力（论文 $F_{max}$ 12.84 N vs 阈值 5–10 N）；你在解析层做硬限幅 + 网络残差，可以把 $F_{max}$ 压到阈值附近，直接对标论文 Table 的 $F_{mean}/F_{max}$ 指标。
2. **修好速度观测（#8874）后再做 residual**：残差策略对阻尼信息更敏感，修完这个 bug 的对比实验本身就是可写的一节。
3. **残差幅度 vs 力阈值的权衡曲线**：扫"残差限幅 × 力阈值 × 位姿噪声"，画出成功率/接触力/时长三维 Pareto，正好复用论文的评估口径（0/1/2.5/5 mm 误差档、45 次/档）。

### 6.5 对比实验设计（建议 5 组，控制变量）

| 组 | 目标位姿来源 | 说明 |
|---|---|---|
| G0 | 零动作（不插） | 下界 / 环境健全性检查 |
| G1 | FORGE 原版（7 维端到端） | 你的 baseline，同 env 同奖励同随机化 |
| G2 | 解析：定速下压 + 螺旋搜索 | 无学习，验证"解析够不够" |
| G3 | 解析 + 2 维残差（你的方法） | 主方法 |
| G4 | G3 去掉力限幅投影 | 消融"硬约束"的贡献 |
| G5 | G3 去掉动力学随机化 | 消融 DR（对齐论文 No DR） |

评价指标照抄论文：成功率、回合时长、$F_{mean}$、$F_{max}$、成功预测 precision/recall/delay。**注意**：G1 在 8 GB 上可能训不动 128 env，必须记录 env 数与样本量，否则对比不公平。

### 6.6 风险与判据（提前想清楚）

- **EMA 吞残差**：`_apply_action` 前对 7 维动作统一 EMA，残差会被同样平滑。要么把 α 调大，要么把残差加在 EMA **之后**（改 `_apply_action`）。
- **世界系 vs 孔轴系**（§3.7）：残差若定义在孔坐标系，必须自己加旋转 $R_{h}$；否则孔 yaw 随机 0–360° 时，残差方向会和孔的实际横轴错开——**这是最容易埋的坑**。
- **clip 2 cm**：残差叠加后仍会被 `pos_threshold` 截断，设计限幅时要小于 2 cm。
- **同步重置**：想加 early termination 必须重写 `_reset_idx` 的批量逻辑，建议第一版不做。
- **可信度**：论文自己说训练噪声 >2.5 mm 不稳定；不要把噪声调大到 5 mm 训练。
- **"精密"的定义**：先把 §3.1 的间隙量清楚，再决定你的 success 判据是 1 mm（论文）还是更严（0.2 mm）。

### 6.7 参考链接
- Residual RL（Johannink et al., ICRA 2019）<https://arxiv.org/abs/1812.03201>
- Residual Policy Learning（Silver et al., 2018）<https://arxiv.org/abs/1812.06298>
- FORGE issue #8874（速度观测）<https://github.com/isaac-sim/IsaacLab/issues/8874>
- FORGE issue #3513（FT 换系）<https://github.com/isaac-sim/IsaacLab/issues/3513>
- FORGE issue #5424（bounds/threshold 语义）<https://github.com/isaac-sim/IsaacLab/issues/5424>
- FORGE issue #8850（孔间隙）<https://github.com/isaac-sim/IsaacLab/issues/8850>

---

## 7. 核验状态总表（哪些能直接引用，哪些要自己测）

| 结论 | 状态 | 核验方式 |
|---|---|---|
| arXiv 编号、作者、单位、RA-L 2025/DOI | ✅ | arXiv abs 页 + Crossref |
| 论文 Table II 全部数值 | ✅ | PDF → MathML alttext 还原（`docs/forge_research/forge_paper_math.txt`） |
| 方法三组件、无 teacher-student、无力调制噪声 | ✅ | 论文 §III、§IV-B 原文 |
| 无官方 FORGE 仓库、Code Coming Soon | ✅ | 项目页（2026-11-22） |
| Isaac Lab 文件清单/注册名/行数 | ✅ | v2.3.0 本地源码 |
| 动作/观测/奖励/随机化具体数值 | ✅ | 源码逐行 |
| actor 24 维 / critic 61 维 | ⚠️ | 源码推导（未运行时打印） |
| 世界系动作、力未旋转、yaw 绝对角 | ⚠️ | 源码推断（未运行验证） |
| 孔实际间隙 ≈1 mm | ⚠️ | 依赖 issue #8850 的第三方测量，**建议自测** |
| 128 env 在 8 GB 是否 OOM | ❓ | 本机 CUDA 不可用，**必须自测** |
| 训练时长/收敛所需 iteration | ❓ | 论文未披露，只能实测 |
| v2.3.0 上 FORGE 能否学到 0.8+ 成功率 | ❓ | 未训练，**这是复现的第一风险点** |

### 附：本次调研落盘的证据文件

- `docs/forge_research/forge_paper.html`（arXiv v2 HTML 原文）
- `docs/forge_research/forge_paper.pdf`（arXiv v2 PDF）
- `docs/forge_research/forge_paper.txt`（去标签纯文本）
- `docs/forge_research/forge_paper_math.txt`（含 MathML alttext 还原，Table II 数值来源）
- `docs/forge_research/gh_forge_issues.json`（GitHub issue 检索原始结果）
