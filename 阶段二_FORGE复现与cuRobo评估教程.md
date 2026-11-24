# 阶段二：复现 FORGE PegInsert 并评估 cuRobo 的插入价值

> **目标**：在本机（RTX 4060 Laptop 8 GB）上完成三件事：
> 1. 把 `Isaac-Forge-PegInsert-Direct-v0` 从"能加载"推进到"能训练、能评估、能复现论文趋势"；
> 2. 得到两个可对比的基线：**解析柔顺基线**（不学习）与 **FORGE 官方 PPO**（端到端学习）；
> 3. 用**分段实验**回答"cuRobo 对插接到底有没有价值"，而不是笼统地说"有用/没用"。
>
> **本教程不做**：真机迁移、VLA、RLinf（那是阶段三）、相机视觉、变阻抗联合训练（阶段三的消融项）。
>
> **前置**：阶段一已完成（环境可启动、`run_diff_ik.py` 可跑、FORGE 零动作可加载）。
> 环境版本与排错见 [依赖安装与排错记录.md](依赖安装与排错记录.md)；阶段一操作见 [ISAAC_LAB_FORGE_基础运控搭建教程.md](ISAAC_LAB_FORGE_基础运控搭建教程.md)。
>
> **调研依据**：论文与源码的对照、方案取舍见 [docs/调研_阶段二三技术选型.md](docs/调研_阶段二三技术选型.md)。

---

## 0. 本阶段代码清单

本阶段新增的脚本都在 `RL/demos/`，**不修改 IsaacLab 源码**：

| 文件 | 作用 | 是否已在本机实测 |
|---|---|---|
| `demos/02_forge_inspect.py` | 环境体检：观测/动作真实维度、控制频率、力读数、成功判据 | 未实测（需 GPU） |
| `demos/03_scripted_insert.py` | 解析柔顺基线（下降 + 恒压 + 螺旋搜索） | 未实测（需 GPU） |
| `demos/06_train_forge_rl_games.py` | FORGE 官方 PPO 复现（rl_games，带成功率日志、支持消融） | 未实测（需 GPU） |
| `forge_ext/logged_env.py` | 与官方任务动力学等价、只把指标写进 `extras["log"]` 的壳 | 未实测（需 GPU） |
| `forge_ext/base_policy.py` | 柔顺基线策略（阶段三作为残差 base） | 未实测（需 GPU） |

> ⚠️ **诚实声明**：本文档编写时，本会话所在的沙箱环境**看不到 GPU**（`nvidia-smi` 报 `Failed to initialize NVML`，torch 报 `Error 304`），因此以上脚本只做了语法检查（`py_compile`），**没有实际跑过仿真**。你在自己的终端里运行，若与文档不符，以终端报错为准，并把报错贴回给我修正。
> 沙箱外你自己的终端里 `nvidia-smi` 是正常的（内核模块 `580.126.09`、`/dev/nvidia*` 都在）。

---

## 1. 每次开终端：环境准备

```bash
source /home/mi/miniforge3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/opt/isaac-workspace/conda-pkgs
export XDG_CACHE_HOME=/opt/isaac-workspace/cache
export TMPDIR=/opt/isaac-workspace/tmp

conda activate /opt/isaac-workspace/envs/isaaclab
cd /home/mi/graduation_design/RL/IsaacLab
```

四步确认，缺一不可：

```bash
which python
# 必须是 /opt/isaac-workspace/envs/isaaclab/bin/python

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
# 必须能看到 RTX 4060 Laptop；memory.used 应该是 0 MiB 左右

ps -eo pid,etime,args | grep -E 'create_empty|run_diff_ik|zero_agent|train.py|06_train' | grep -v grep
# 必须为空：不要同时开两个 Isaac Sim

df -h /home | tail -1
# 训练日志和 checkpoint 会吃空间，留够 15 GB
```

**为什么强调"不要两个实例"**：阶段一已经踩过（依赖记录 4.7），`another kit process is locking it` 之后 GUI 会被 SIGKILL。训练时更容易犯：一边 `train.py` 一边 `play.py`。

---

## 2. 步骤一：环境体检（必须先做）

```bash
cd /home/mi/graduation_design/RL/IsaacLab
./isaaclab.sh -p ../demos/02_forge_inspect.py --num_envs 4 --steps 150 --headless
```

### 2.1 你要从输出里确认的 8 件事

| # | 打印项 | 期望值 | 不对说明什么 |
|---|---|---|---|
| 1 | 控制频率 | **15.00 Hz** | 有人改了 `decimation` 或 `dt` |
| 2 | 每回合步数 | **150** | `episode_length_s` 被改 |
| 3 | policy 观测维度 | **24**（`[4, 24]` 这种形状） | cfg 里写的 21 是过期常量，真实是 24 |
| 4 | critic 状态维度 | **61**（含 7 维 prev_actions） | 与 `state_order` 逐项相加对不上就是源码被改过 |
| 5 | 动作维度 | **7** | 第 7 维是成功预测 |
| 6 | 孔/销直径 | 8.100 / 7.986 mm，单边间隙 **0.057 mm** | 这是任务难度的手册数字，论文里要写 |
| 7 | 成功判据 | xy < 2.5 mm 且 z_disp < **1.00 mm** | `success_threshold=0.04 × 孔深 25 mm` |
| 8 | 力读数 | 零动作下 `|F|` 应该很小（< 2 N，主要是夹爪自重残差） | 若几百 N，说明力坐标系/索引取错 |

> 如果第 4 项与你手算不一致，用 `python -c` 打印 `ForgeEnvCfg().state_order` 逐项核对，不要凭记忆。

### 2.2 两个必须自己量一次的坑（源码/上游已确认）

**(1) 观测里 6 维速度是"噪声 / dt"（IsaacLab issue #8874，v2.3.0 存在）**

`FactoryEnv._compute_intermediate_values` 先用**真值**位置算速度并覆盖 `prev_fingertip_pos`，
`ForgeEnv` 紧接着又用**含噪**位置重算一遍。于是 `ee_linvel`(3) + `ee_angvel`(3) 实际等于
`位置噪声 / dt`（dt=1/120 → 1 mm 噪声变成 0.12 m/s 的假速度）。

- 对"复现官方结果"：**保持原样**（否则与官方曲线不可比）；
- 对"做残差策略"：`forge_ext/residual_env.py` 里已加 `fix_velocity_obs=True` 开关，
  用真值历史重新差分，把这 6 维修好。两版可以做对照，是论文里一个不错的消融点。

**(2) 间隙到底是多少（issue #8850）**

配置写的是销 ⌀7.986 / 孔 ⌀8.1（单边 57 µm），但上游 issue 实测 USD 里孔内径约 **9.0 mm**
（单边约 1 mm），差 9 倍。经验判别法（30 秒）：

```bash
# 把孔位观测噪声置零，跑解析基线：
#   成功率接近 100% → 间隙宽裕（实际约 1 mm 那一档）
#   成功率仍然很低且频繁卡住 → 间隙确实是几十微米那一档
./isaaclab.sh -p ../demos/03_scripted_insert.py --num_envs 32 --episodes 5 \
  --obs_noise_mm 0 --headless
```

（`--obs_noise_mm` 需要你自己在 `03` 脚本里加一行：`env_cfg.obs_rand.fixed_asset_pos = [0.001*m, ...] * 0.001`。）

**这一步不能省**：论文里"精密插接""亚毫米间隙"这类措辞，必须建立在你亲自量过的数字上。

### 2.3 三种动作下的现象（用来确认"任务不是碰巧成功"）

```bash
# 全零动作：机械臂基本不动，peg 停在初始高度，成功率应该一直是 0
./isaaclab.sh -p ../demos/02_forge_inspect.py --num_envs 4 --steps 150 --action zero --headless

# 随机动作：抖动，可能偶尔碰一下，几乎不可能插进去
./isaaclab.sh -p ../demos/02_forge_inspect.py --num_envs 4 --steps 150 --action random --headless

# 恒定下压：能建立接触，力会上升；如果 z 一路下压成功率为 0 说明"压不出插入"
./isaaclab.sh -p ../demos/02_forge_inspect.py --num_envs 4 --steps 150 --action down --headless
```

**判读标准**：`--action down` 下 `|F|` 应该随步数明显上升（几十牛），且 `z_disp` 不再下降（被孔口顶住）。这说明"接触确实发生了、力确实测到了"，是后续一切实验的前提。

产出文件：`RL/outputs/forge_inspect_<时间戳>.csv`，字段 `step, env_id, force_x/y/z, force_norm, xy_err_mm, z_disp_mm, success`。**这张表可以直接画"力-位移曲线"，是论文里很漂亮的一张图**。

---

## 3. 步骤二：解析柔顺基线（不学习能做到什么）

```bash
cd /home/mi/graduation_design/RL/IsaacLab
./isaaclab.sh -p ../demos/03_scripted_insert.py --num_envs 32 --episodes 10 --headless
```

### 3.1 它在做什么

```
阶段 1（0–25 步）  自由空间：z 从 47 mm 线性降到 +3 mm（相对孔口顶面）
阶段 2（接触后）  恒压：每步 z -0.5 mm；一旦 |F| > 30 N 就冻结 z（轴向导纳）
阶段 3（接触后）  螺旋搜索：半径从 0.2 mm 长到 2 mm（100 步），极角每步 +0.5 rad
```

三个阶段的代码在 `forge_ext/base_policy.py::ScriptedInsertPolicy.act()`，只有 60 行，**建议先把这 60 行读懂**——阶段三的残差策略就是在这个基础上加两个数。

### 3.2 预期结果与判读

| 现象 | 含义 | 下一步 |
|---|---|---|
| 成功率 0%，末态 `z ≈ +40 mm` | 根本没下去（动作被裁/符号错） | 检查 `z_cmd/pos_bound` 的符号；确认 `fixed_pos_obs_frame` 是孔口顶面 |
| 成功率 0%，末态 `z ≈ -1 mm` 且力很大 | 压住了但没进孔（卡在孔口） | 增大 `search_radius_max`、减小 `press_step`、降低 `hold_force` |
| 成功率 5%–40% | **正常**。螺旋搜索本身低效，论文也这么说 | 记录数字，作为残差 RL 的下界 |
| 成功率 > 60% | 参数偏乐观（可能搜索半径刚好覆盖） | 用更难的初始误差重测（见 3.4） |

### 3.3 调参顺序（不要乱调）

按这个顺序调，每次只改一个：

1. `hold_force`：先确保**不会压爆**（看 CSV 里力峰值 < 50 N）；
2. `press_step`：太大→力峰值高；太小→150 步内下不到底。建议 0.3–0.6 mm/step；
3. `contact_force`：触发搜索的阈值。太小（<2 N）会误触发；太大（>10 N）会"压死"才搜；
4. `search_radius_max`：孔位噪声 1σ = 1 mm，取 2 mm 覆盖 2σ；
5. `search_radial_steps`：决定螺旋的细密度。每步径向增量应显著小于倒角（约 0.1–0.2 mm），所以 100 步从 0.2→2 mm（18 µm/step）是合理的。

**先知道一个会限制上限的实现细节**：控制器死区 `default_dead_zone=[5,5,5,1,1,1]` 是在 wrench 域做减法，
折算成位置约 `5 N / 565 N·m⁻¹ ≈ 8.8 mm`、角度约 `1/28 ≈ 2.05°`。也就是说**小于 9 mm 的位置误差不产生力**。
这让"靠微小位移精确对中"在仿真里根本做不到，只能靠 peg 在夹爪里的被动自对齐。

→ 想验证这个判断，把死区改小再跑一次基线：

```python
env_cfg.ctrl.default_dead_zone = [0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
```

成功率如果明显上升，说明"死区是精密段的主要瓶颈"——这本身就是阶段三"柔顺参数整定"章节的一个好实验（阶段二三都用得上）。

> 想系统刷参数，不要手改代码：`InsertBaseCfg` 是 dataclass，写个 10 行脚本循环覆盖字段再 `ScriptedInsertPolicy` 重建即可。**每个配置至少 10 回合 × 32 环境**，否则成功率的标准误太大（32 环境 × 10 回合 = 320 次试验，标准误约 √(0.3×0.7/320) ≈ 2.6%）。

### 3.4 难度分级（论文里要用）

默认随机化里孔位是 ±50 mm（`fixed_asset_init_pos_noise=[0.05,0.05,0.05]`），但**观测噪声只有 1 mm**。真正决定难度的是"估计值 vs 真值"的偏差。做三档：

```python
# 在 03 脚本里加一个 --difficulty 参数（示例片段）
env_cfg.obs_rand.fixed_asset_pos = [0.001, 0.001, 0.001]   # 低：1 mm
env_cfg.obs_rand.fixed_asset_pos = [0.0025, 0.0025, 0.001] # 中：2.5 mm（论文的 medium 上界）
env_cfg.obs_rand.fixed_asset_pos = [0.005, 0.005, 0.001]   # 高：5 mm（论文 high）
```

论文 Fig.4 就是按 low(0–1 mm)/medium(1–2.5 mm)/high(2.5–5 mm) 分档的，**复现这张图是本阶段的高价值产出**：它直接说明"为什么需要学搜索"。

---

## 4. 步骤三：复现 FORGE 官方 PPO

### 4.1 为什么不用官方 `train.py`

两个硬原因（已在源码里核实）：

1. 官方 `scripts/reinforcement_learning/rl_games/train.py` 只 `import isaaclab_tasks`，**不会导入本仓库的 `forge_ext`**，所以它看不到我们注册的任务；
2. 官方 FORGE 把指标写在 `extras["successes"]` 这一层，而 rl_games 只记录 `extras["episode"]`（`source/isaaclab_rl/isaaclab_rl/rl_games/rl_games.py` 里有 `if "log" in extras: extras["episode"] = extras.pop("log")`）。**结果是：用官方命令训练 FORGE，TensorBoard 里没有成功率曲线**。

`demos/06_train_forge_rl_games.py` 解决这两点：用 `forge_ext` 注册的 `Isaac-Forge-PegInsert-Logged-v0`（动力学与官方**完全等价**，只多写一行日志），并原样加载官方 `rl_games_ppo_cfg.yaml`（LSTM 1024×2 + 非对称 AC + central value，`num_actors` 由 env 数决定）。

另外两个事实要提前知道：

- **没有预训练权重**。官方没有发布 checkpoint，项目页仍是 "Code (Coming Soon)"；Isaac Lab 里的 `direct/forge/` 就是事实上的官方实现（PR #2968）。所以阶段二必须自己训。
- **`full_experiment_name: test` 写死在 yaml 里**。用官方 `train.py` 每次都会覆盖同一个日志目录，历史曲线全丢。`demos/06` 已改成按时间戳建目录，并额外 `dump_yaml` 保存 env/agent 配置（复现性）。
- **FORGE 不是"无控制器的端到端"**。它已经是两层：策略 15 Hz 输出 4 自由度目标位姿（孔尖为原点）→ EMA → 2 cm clip → **120 Hz 任务空间阻抗控制器**（kp≈565 N/m、带死区）。
  → 阶段三要替换的不是控制器，而是"**目标位姿生成器**"。这句话是你论文第 4 章方法部分的立论基础，务必写清。

### 4.2 冒烟（先 20 epoch）

```bash
cd /home/mi/graduation_design/RL/IsaacLab
./isaaclab.sh -p ../demos/06_train_forge_rl_games.py \
  --num_envs 16 --max_epochs 20 --seed 0 --headless
```

成功标志（终端）：

```text
[INFO] agent 配置: .../isaaclab_tasks/direct/forge/agents/rl_games_ppo_cfg.yaml
[INFO] 日志目录: .../IsaacLab/logs/rl_games/Forge/<时间戳>
=> saving checkpoint ...
fps step: ...
```

同时另开一个终端看显存：

```bash
watch -n 2 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

**16 envs 时显存应在 3–5 GB**。如果一开始就 OOM，先降到 8 envs，再考虑 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

### 4.3 正式训练

```bash
./isaaclab.sh -p ../demos/06_train_forge_rl_games.py \
  --num_envs 64 --max_epochs 500 --seed 0 --headless
```

**规模参考（`horizon_length=128`）**

| num_envs | 每 epoch 步数 | 500 epoch 总步数 | 约合回合数 |
|---|---|---|---|
| 16 | 2048 | 1.0 M | 6.8 k |
| 32 | 4096 | 2.0 M | 13.6 k |
| 64 | 8192 | 4.1 M | 27 k |
| 96 | 12288 | 6.1 M | 41 k |

> **显存不够时怎么退**：减 `num_envs` 必须**按比例加 epoch**，否则样本量不足、结果不可比。
> 官方默认是 `128 env × 200 epoch ≈ 3.28 M 步`；你跑 32 env 就要 **800 epoch** 才等量。
> 论文没披露训练步数，200 epoch 只是 yaml 里的演示值。**判据不是"跑完"，而是 `episode/successes` 有没有从 0 抬起来。**

### 4.4 读曲线

```bash
cd /home/mi/graduation_design/RL/IsaacLab
tensorboard --logdir logs/rl_games --port 6006
```

| 曲线 | 含义 | 健康表现 |
|---|---|---|
| `episode/successes` | 回合结束时仍插在孔里的比例 | 从 0 单调上升到 > 0.5 |
| `episode/logs_rew_kp_fine` | 细粒度关键点奖励 | 上升 |
| `episode/logs_rew_contact_penalty` | 超阈值力惩罚 | 先上升（开始接触）后下降（学会控制力） |
| `episode/early_term_recall/0.5` | 成功预测的召回 | 后期 > 0.8 |
| `losses/a_loss`、`losses/entropy` | PPO 健康度 | 熵缓慢下降，不塌到 0 |
| `rewards/step` | 原始回报 | 上升 |

**常见不健康模式**：

- `successes` 一直是 0，`contact_penalty` 也不动 → 策略根本没接触（动作尺度/坐标系问题），先去跑步骤二的 `--action down` 验证接触链；
- `successes` 一直 0，但 `contact_penalty` 很大 → 一直在"撞"，力阈值太低或 `press_step` 太大；
- 熵迅速塌陷 → 学习率过高 / `entropy_coef=0`（官方就是 0，靠 adaptive lr 稳）。

### 4.5 断点续训

```bash
./isaaclab.sh -p ../demos/06_train_forge_rl_games.py \
  --num_envs 64 --max_epochs 1000 --seed 0 --headless \
  --checkpoint logs/rl_games/Forge/<时间戳>/nn/Forge.pth
```

### 4.6 录视频（答辩素材）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Forge-PegInsert-Direct-v0 \
  --num_envs 4 \
  --checkpoint logs/rl_games/Forge/<时间戳>/nn/Forge.pth \
  --video --video_length 300
```

> `play.py` 用的是官方任务 id，不影响 checkpoint 加载（网络结构由 yaml 决定）。

---

## 5. 步骤四：消融（复现论文趋势）

论文的三个消融是 **No Force / No DR / No FP**。本仓库能直接做前两个中的 No Force 和 No FP：

```bash
# 去掉过力惩罚（对应论文 FORGE (No FP)）
./isaaclab.sh -p ../demos/06_train_forge_rl_games.py --num_envs 64 --max_epochs 500 --ablate no_fp --headless

# 策略不看力（对应论文 FORGE (No Force)）
./isaaclab.sh -p ../demos/06_train_forge_rl_games.py --num_envs 64 --max_epochs 500 --ablate no_force --headless
```

**No DR** 需要手改配置（`06` 脚本没实现，因为要改 6 个字段）。最小改法：在 `06` 脚本里 `env_cfg = load_cfg_from_registry(...)` 之后插入：

```python
# 关掉控制器增益 / 阈值 / 死区随机化
env_cfg.ctrl.task_prop_gains_noise_level = [0.0] * 6
env_cfg.ctrl.pos_threshold_noise_level = [0.0] * 3
env_cfg.ctrl.rot_threshold_noise_level = [0.0] * 3
env_cfg.ctrl.ema_factor_range = [0.1, 0.1]
env_cfg.events.dead_zone_thresholds = None          # 不周期性随机化死区
# 关掉摩擦/质量随机化：把 EventCfg 里对应的 range 收成单点
env_cfg.events.fixed_physics_material.params["static_friction_range"] = (0.75, 0.75)
env_cfg.events.object_scale_mass.params["mass_distribution_params"] = (0.0, 0.0)
# 初始位姿噪声收敛
env_cfg.task.fixed_asset_init_pos_noise = [0.001, 0.001, 0.001]
env_cfg.obs_rand.fixed_asset_pos = [0.001, 0.001, 0.001]
```

> 论文的结论是：No DR 反而成功率更高（0.91 vs 0.84，因为任务变简单）、但力更大（6.28 vs 5.51 N）、早停精度下降。**你复现时只要趋势方向一致就写"复现趋势"，不要硬对数字。**

### 5.1 消融记录表（直接抄进论文）

| 配置 | 仿真成功率 | 平均插接时间(s) | 力峰值 mean (N) | 力峰值 p95 (N) | 早停 precision | 早停 recall | 训练步数 |
|---|---|---|---|---|---|---|---|
| FORGE (full) | | | | | | | |
| FORGE (No FP) | | | | | | | |
| FORGE (No Force) | | | | | | | |
| FORGE (No DR) | | | | | | | |
| 解析柔顺基线 | — | | | | — | — | 0 |

（早停指标在 TB 里是 `episode/early_term_precision/*`、`episode/early_term_recall/*`；插接时间取 `episode/success_times`。）

---

## 6. 步骤五：评估协议（三组对象用同一把尺）

阶段二末、阶段三全程都用**同一套评估**，否则数字不可比。

### 6.1 统一规则

1. **固定评估种子**：`env_cfg.seed = 1234`（`demos/05_eval_success.py` 已写死），训练种子不参与评估；
2. **固定回合数**：≥10 回合 × ≥32 环境 = ≥320 次试验；
3. **固定判据**：直接用环境自带 `_get_curr_successes`（xy<2.5 mm 且 z<1 mm），**不要自定义更宽松判据**；
4. **报区间**：成功率给 Wilson 95% 置信区间：`p̂ ± 1.96√(p̂(1-p̂)/n)`；
5. **报失败模式**：超时 / 卡在孔口 / 侧向滑出 / 力超限，各占多少。

### 6.2 三组对象

| 对象 | 命令 | 说明 |
|---|---|---|
| 解析柔顺基线 | `03_scripted_insert.py` | 不学习 |
| 零残差（等价于只有 base） | `05_eval_success.py`（不给 checkpoint） | 阶段三的对照 |
| FORGE 端到端 PPO | TB `episode/successes` 或 `play.py` 目视 | 学习基线 |

> 阶段三的残差策略用 `05_eval_success.py --checkpoint ...` 评估，与本表同协议。

### 6.3 记录模板

`outputs/stage2_summary.csv`：

```csv
method,difficulty,n_trials,success_rate,ci95_low,ci95_high,time_mean_s,time_p95_s,force_mean_N,force_p95_N,fail_timeout,fail_jam,fail_slip
scripted,low,320,,,,,,,,
scripted,medium,320,,,,,,,,
forge_ppo,low,320,,,,,,,,
```

---

## 7. 步骤六：cuRobo 安装与"插入价值"评估

> **时间盒：3 天。** 超时就降级为"仅调研 + 理论分析 + 用差分 IK 做对照"，不要为它拖垮主线。
> 详细调研见 `docs/research/curobo.md`（版本谱系 + E1–E7 实验矩阵 + 踩坑）与 `docs/research/cuRobo_IsaacLab_技术调研.md`（显存/依赖细节）。

### 7.1 先明确它管哪一段

```
[初始位姿] --A: 自由空间运动 --> [预插入位姿] --B: 接触逼近 --> [孔口接触] --C: 插入/柔顺 --> [到位]
              ↑ cuRobo 的战场                                   ↑ 力控/柔顺/残差RL 的战场
```

- **A 段**：cuRobo 能贡献无碰撞轨迹、批量 IK、时间参数化；
- **C 段**：cuRobo 不建模接触刚度、没有力闭环，**帮不上忙**；它的碰撞模型是球体近似（`surface_sphere_radius=5 mm`），精度也不支持亚毫米对位。
- 所以**必须分段评估**，不能把"插接成功率"当成 cuRobo 的指标。

### 7.2 安装（照 SkillGen 文档，锁 v1 commit）

```bash
# 注意：必须在"没有 source 过 Isaac Sim setup_conda_env.sh"的干净终端里装，
# 否则 PYTHONHOME/PYTHONPATH 会被 Kit 污染，conda 会崩（官方文档明确警告）

conda activate /opt/isaac-workspace/envs/isaaclab

conda install -c nvidia cuda-toolkit=12.8 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"

# ⚠️ 关键：cuda-toolkit 12.8 的 nvcc 要求 GCC < 14，而 conda 会顺手装 GCC 14
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++

# 4060 是 Ada（算力 8.9），不是 8.0
export TORCH_CUDA_ARCH_LIST="8.9+PTX"

cd /opt/isaac-workspace      # 不要放在 /home（空间紧）
pip install -e "git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8#egg=nvidia-curobo" \
  --no-build-isolation
```

预计：源码 clone ~1 分钟、编译 10–25 分钟（nvcc 单架构；官方原文 "20 minutes"）。

**这一步有两个高频失败点**（都有上游证据）：

1. **GCC 版本**：`conda install cuda-toolkit` 会带进 GCC 14，nvcc 12.8 直接报 "unsupported GNU version"。修法就是上面三行 `CC/CXX/CUDAHOSTCXX`。
2. **commit 归属**：`ebb7170` 是 2025-08-04 的提交，属于 v1 线（等价于 v0.7.8 往前 4 个提交）。Isaac Lab 2.3.0/2.3.1/2.3.2/main/3.0.0 **全部锁同一个 hash**——也就是说官方从没更新过；不要自己换到 `v0.7.8` 或 v2，除非你愿意自己修 API。

装完验证：

```bash
python -c "import curobo; print(curobo.__file__)"
```

**不要做的事**：`pip install curobo`（PyPI 上是第三方占位空包）；装 v2（`v0.8.0+`，入口从 `MotionGen` 换成 `MotionPlanner`，与 Isaac Lab 2.3.0 的封装不兼容）。

### 7.3 冒烟测试（用官方 test）

```bash
cd /home/mi/graduation_design/RL/IsaacLab
./isaaclab.sh -p -m pytest source/isaaclab_mimic/test/test_curobo_planner_franka.py -s
```

这个测试会：建一个带挡墙的 stack-cube 场景 → 用 `CuroboPlanner` 规划到 5 个目标位姿 → 检查是否成功。

- 通过：说明 cuRobo + Isaac Sim 共存 OK，可以做 E1；
- OOM：说明 8 GB 共用不了，直接跳到 7.6 的降级方案。

### 7.4 E1：规划器对比（A 段）

被测对象（A 段都用同一个起点、同一批目标位姿）：

| 对象 | 实现方式 | 关键参数 |
|---|---|---|
| 差分 IK（基线） | `DifferentialIKController` + `ik_method="dls"` | `lambda=0.1`，迭代 10 次 |
| cuRobo IK | `CuroboPlanner.plan_motion(target_pose)` + `enable_retiming=True` | `num_seeds` 小、关 collision 或开 collision 各一组 |
| cuRobo 轨迹规划 | 同上，但读回 `_current_plan` 的完整轨迹 | `step_size` 控制重采样 |

指标（**只在这一段测**）：

| 指标 | 怎么取 |
|---|---|
| 规划成功率 | 成功次数 / 总次数 |
| 单次耗时 p50 / p95 | `torch.cuda.synchronize()` 后 `time.perf_counter()`，warmup 至少 10 次 |
| 末端位姿误差 | 到达后 `fingertip_pos - target_pos` 的范数（mm）与姿态角误差（deg） |
| 碰撞率 | 只在开 world collision 的组测 |

**控制变量**：同一 USD 场景、同一 Franka、同一 TCP（`panda_fingertip_centered`）、同一批目标位姿（≥50 个，固定 seed）、同一物理步长。

### 7.5 E2 / E3：碰撞消融与端到端影响

- **E2**：cuRobo 开/关 `CollisionCheckerType`，看规划耗时与成功率变化 → 回答"碰撞规避值多少毫秒"；
- **E3（主实验）**：A 段固定用最优方案，C 段比 **纯位置 / 定阻抗+螺旋 / 残差 RL / FORGE 端到端** → 这是阶段三的表，阶段二只需先把"纯位置"和"定阻抗+螺旋"两列填上。

### 7.5.1 更严谨的做法：用 handoff 文件把两段解耦

直接在一个脚本里"A 段规划 → B 段执行"会让两段互相干扰（规划失败时执行段拿不到输入）。
更干净的实现是**落盘交接**（下面是设计，不是现成脚本——需要你自己写两个小脚本）：

```text
# 脚本 1（A 段，只需 cuRobo + 一个空场景，不加载 FORGE）：
#   curobo_plan_offline.py --num_poses 50 --out outputs/handoff.npz
#   写出的字段：
#     init_joint_pos[N,7]      起点关节角
#     target_pose[N,4,4]       预插入位姿（孔口上方 3 mm，与 InsertBaseCfg.z_press_start 一致）
#     plan_joint_traj[list]    规划出的关节轨迹（可选）
#     plan_ok[N], plan_time[N], e_pos[N], e_rot[N]

# 脚本 2（C 段）：在 demos/03 里加一个 --handoff 参数
#   读 npz → 按 init_joint_pos 设置机械臂 → 执行 handoff 位姿 → 交给解析基线继续
```

好处：A 段与 C 段可以**分开跑**（8 GB 上必须分开），指标正交，归因清楚。

**误差解耦（论文 Fig 的关键）**：把总成功率写成全概率形式

$$
P(\text{succ}) = \int P(\text{succ}\mid e)\,p_{\text{planner}}(e)\,de
$$

做法：注入孔位误差 δ ∈ {0, 2, 5} mm → 测各规划器的 `p_planner(e)`；再固定控制器，在 e 网格 {0.2, 1, 3} mm 上测 `P(succ|e)`。这样能回答"成功率下降是规划器给的初始误差造成的，还是控制器本身不行"——这是答辩时最难被问倒的一张图。

### 7.5.2 三个必须知道的工程约束

1. **CUDA graph 会把形状锁死**：cuRobo 一旦 capture，`num_seeds` / `trajopt_tsteps` / dof 就固定了，改参数必须**重启进程**。所以扫参脚本要"每个配置一个进程"，不要在同一进程里循环改配置。
2. **`CuroboPlannerCfg` 没暴露 `use_cuda_graph`**：`curobo_planner.py` 里没有把该参数传下去（实际用默认 True）。要关只能改源码或直接用 `MotionGenConfig`。
3. **Isaac Lab 2.3.x 的碰撞世界同步 bug**（上游 PR #4212，2026-09 才合入 develop）：机器人 base 不在世界原点时，障碍物位姿的坐标换算不一致（`root_pos_w - env_origin` vs 世界系）。
   **规避**：单环境 + `env_origin=(0,0,0)`，或等升级。表现是"没碰却报碰撞 / 撞了却不报"，排查起来极费时间。

### 7.6 降级方案（很可能用得上）

| 情况 | 降级做法 | 论文怎么写 |
|---|---|---|
| 编译失败 | 放弃 cuRobo，用 `DifferentialIKController` 做 A 段基线 | "cuRobo 在 Isaac Sim 5.1 + 8 GB 环境下编译/运行受限，本文以差分 IK 作为运动生成基线，并在第 6 章讨论 cuRobo 的适用边界" |
| 编译成功但 OOM | 只跑"纯 cuRobo 规划器 + 刚体小球场景"的微基准，不进 FORGE 场景 | 报"规划耗时/精度微基准"，不报插接成功率 |
| 都失败 | 只做文献分析 + 理论分工论证 | 第 2 章相关工作里写清楚"运动生成 vs 接触控制"的边界 |

**统一口径**：无论哪种结果，论文结论都应是"**cuRobo 解决 A 段，不解决 C 段；插接的瓶颈在接触阶段**"。这个结论从论文与本仓库实现都能推出来，不依赖你是否装上 cuRobo。

---

## 8. 阶段二验收清单

- [ ] `02_forge_inspect.py` 跑通，8 项体检数字与 §2.1 表一致，CSV 已保存
- [ ] 零动作 / 随机动作基线成功率 ≈ 0（有记录）
- [ ] `03_scripted_insert.py` 跑通，解析基线成功率与力峰值有记录
- [ ] 至少做过一次 `search_radius_max` 或 `press_step` 的对比（≥3 组参数）
- [ ] `06_train_forge_rl_games.py` 20 epoch 冒烟通过，TensorBoard 里能看到 `episode/successes`
- [ ] 正式训练 ≥500 epoch，`successes` 曲线已导出（PNG + CSV）
- [ ] 至少完成 1 个消融（No FP 或 No Force），趋势与论文一致
- [ ] 三组对象用同一评估协议出表（`outputs/stage2_summary.csv`）
- [ ] cuRobo：安装成功→E1 表；失败→降级方案已执行并记录原因
- [ ] 所有图/表/日志归档到 `RL/docs/`、`RL/outputs/`，并在论文里能一键追溯

---

## 9. 排错 FAQ

**Q1：`CUDA out of memory` 一开始就报**

```bash
# 1) 确认没有别的进程占卡
nvidia-smi
# 2) 降环境数（16 → 8）
# 3) 开可扩展显存分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

FORGE 的 PhysX 配置很重（`max_position_iteration_count=192`、`gpu_max_rigid_contact_count=2^23`），显存主要吃在 PhysX 而不是网络。**不要为了省显存去改这些参数**——它们是为"避免穿透"特意设的，改了任务就变了。

**Q2：训练能跑但 `episode/successes` 永远是 0**

按顺序查：
1. 跑 `02 --action down`，确认接触与力读数正常；
2. 跑 `03_scripted_insert.py`，确认"能插进去"这件事在环境里成立；
3. 看 TB 的 `logs_rew_kp_fine` 是否在上升（上升但没成功=接近了但精度不够，考虑延长训练）；
4. 看 `logs_rew_contact_penalty`，若一直很大=在撞，考虑降 `press_step`；
5. 确认 `num_actors=64` 时 `minibatch_size=512` 能整除（64×128/512=16，OK）。

**Q3：`episode/successes` 有值但抖动很大**

Factory 系列**所有环境同步重置**（`_get_dones` 返回全 True/全 False），所以一个点就是"这一整批 150 步结束时的成功率"，抖动大是正常的。看移动平均（TB 里开 smoothing 0.9）。

**Q4：`ImportError: cannot import name 'forge_ext'`**

脚本用 `sys.path.insert(0, RL根目录)` 引入，不受工作目录影响；若你把脚本复制到别处，需要手动加 `export PYTHONPATH=/home/mi/graduation_design/RL:$PYTHONPATH`。

**Q5：`play.py` 加载 checkpoint 报 shape mismatch**

说明你用了 `--ablate no_force` 训的模型（观测 21 维）却用默认配置（24 维）去加载。评估时要带同样的 `--ablate`。

**Q6：cuRobo 安装时 conda 崩、报 `omni.kit.pip_archive`**

你在一个 source 过 Isaac Sim 脚本的终端里装。新开一个干净终端（只 activate conda 环境，不 source 任何 `setup_conda_env.sh`）。

**Q7：`nvidia-smi` 报 `Failed to initialize NVML`**

先确认不是"两个 Isaac Sim 抢卡"（`ps` 查）。若进程干净仍报错，重启机器；若是容器/沙箱环境，可能是设备权限被限制（本会话的沙箱就是这种情况），需在宿主机终端跑。

**Q8：训练曲线里 `episode/successes` 是 0，但肉眼在 `play.py` 里看到插进去了**

两种可能：
1. `successes` 只在**所有环境同步重置的那一步**记录（Factory 的设计），你看的是平滑后的点，末端可能已经掉了；
2. 训练用的是官方任务 id（`Isaac-Forge-PegInsert-Direct-v0`）→ 根本没这条曲线，必须用 `Isaac-Forge-PegInsert-Logged-v0`。

**Q9：想加"成功即终止"（early termination）**

官方 `_get_dones` 对 Factory/FORGE **只返回超时**（`time_out, time_out`），所有环境同步重置。加提前终止必须重写批量 reset 逻辑，工作量大且会改变任务语义。
**建议不要改**：论文里的早停是**部署策略**（预测成功就停），训练阶段仍然跑满 150 步。你可以在评估脚本里实现"预测成功即停"，效果更接近论文。

**Q10：`change_FT_frame` 是不是把力转到了孔口系？**

**不是**。上游 issue #3513 指出它传入的是单位四元数，只有平移项生效，`ft_force` 实际在世界系。
对本任务无实质影响（圆孔轴对称、固定件只随机 yaw），但论文里**不要写"力已转到孔口坐标系"**；要写"力在世界系，孔轴与 z 轴重合"。

---

## 10. 参考

- FORGE 论文：<https://arxiv.org/abs/2408.04587> · 项目页 <https://noseworm.github.io/forge/>
- FORGE 源码（Isaac Lab v2.3.0）：`source/isaaclab_tasks/isaaclab_tasks/direct/forge/`
- Factory 任务与控制器：`source/isaaclab_tasks/isaaclab_tasks/direct/factory/`
- Isaac Lab 环境文档（Factory/FORGE 任务列表）：<https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html>
- SkillGen（cuRobo 官方安装与用法）：<https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/imitation-learning/skillgen.html>
- cuRobo 仓库：<https://github.com/NVlabs/curobo>
- 本仓库调研：`docs/research/curobo.md`、`docs/research/cuRobo_IsaacLab_技术调研.md`、`docs/research/FORGE_技术调研报告.md`、`docs/research/forge/`（论文原文）
