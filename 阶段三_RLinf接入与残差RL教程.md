# 阶段三：残差强化学习训练与 RLinf 接入

> **目标**：把"解析柔顺基线 + 二维横向残差策略"训练出来、评估出来、对比出来，并完成 RLinf 的框架层接入验证。
>
> **两条路线，先做 A 再做 B**：
> - **路线 A（主线，必须有结果）**：Isaac Lab 原生 `rsl_rl` PPO 训练残差策略 → 与本仓库基线、FORGE 端到端策略对比；
> - **路线 B（接入验证，时间盒 3 天）**：把同一个残差任务接进 RLinf，用 MLP+PPO 跑通冒烟。失败就降级，如实记录阻塞点。
>
> **为什么这样排**：RLinf 官方对 Isaac Lab 的支持是 **8 卡 + VLA 后训练**口径（详见 §6.1），
> 而你的机器是 8 GB / 14 GB 内存。把 RLinf 设成硬门槛会把整个毕设拖死；设成"接入验证 + 系统层对照"则是稳赢的写法。
>
> **前置**：阶段二已完成（FORGE 可训练、解析基线有数字）。方案依据见 [docs/调研_阶段二三技术选型.md](docs/调研_阶段二三技术选型.md)。

---

## 0. 代码清单

| 文件 | 作用 |
|---|---|
| `forge_ext/base_policy.py` | 解析柔顺基线（自由空间下降 + 力限幅 + 螺旋搜索），同时是残差 base |
| `forge_ext/residual_env.py` | 残差任务：2 维残差动作 + base 动作叠加 + 残差正则 + 速度观测修复（issue #8874） |
| `forge_ext/agents.py` | rsl_rl PPO 超参（MLP [256,128,64]，非对称 AC，γ=0.99） |
| `forge_ext/__init__.py` | 注册 `Isaac-Forge-PegInsert-Residual-v0` 与 `Isaac-Forge-PegInsert-Logged-v0` |
| `demos/04_train_residual.py` | 残差 PPO 训练入口（显式构图，不走 hydra） |
| `demos/05_eval_success.py` | 统一评估：脚本基线 / 零残差 / 任意 checkpoint，同一随机化、同一指标 |
| `rlinf_adapter/forge_peginsert_env.py` | RLinf 任务适配器（state-only） |
| `rlinf_adapter/config/forge_peginsert_ppo_mlp.yaml` | RLinf MLP-PPO 配置模板 |

> 所有脚本**只做语法检查，未在本机真跑**（本会话沙箱看不到 GPU）。跑不通就把报错贴回来。

---

## 1. 架构设计（先定死语义，再写代码）

### 1.1 分层结构

```
┌─────────────────────────────────────────────────────────────────────┐
│ 残差策略 πθ  (MLP 256-128-64)        15 Hz                          │
│   输入: 19 维观测（含 3 维力 + 1 维力阈值）                          │
│   输出: a_res ∈ [-1,1]²  ──►  Δx, Δy ∈ ±3 mm                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ 叠加
┌───────────────────────────────▼─────────────────────────────────────┐
│ 解析柔顺基线 (无参数)                15 Hz                          │
│   z: 47mm 线性下降到 +3mm，之后每步 -0.5mm；|F|>30N 冻结（导纳）    │
│   xy: 接触后螺旋搜索，半径 0.2→2mm，极角每步 +0.5rad                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ 7 维归一化动作（孔尖为原点的世界轴系）
┌───────────────────────────────▼─────────────────────────────────────┐
│ 任务空间阻抗控制器 (Factory/FORGE 原生)      120 Hz                 │
│   kp≈565 N/m (每回合随机 ±41%)、kd=2√kp、死区、动作 EMA 0.025-0.1   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ 关节力矩
                            Franka Panda
```

**关键点：你替换的不是控制器，而是"目标位姿生成器"。** 柔顺性由最下层阻抗环保证，残差只负责"往哪边找孔"。

### 1.2 动作语义（必须在论文里写死）

| 项 | 定义 | 实现位置 |
|---|---|---|
| 动作空间 | 2 维连续，[-1,1] | `ForgeResidualCfg.action_space=2` |
| 归一化动作 → 物理量 | `Δxy = a_res × 3 mm` | `residual_pos_scale=0.003` |
| 叠加位置 | base 的**绝对目标**（相对孔尖）上叠加，再 clip 到 ±5 cm | `_pre_physics_step` |
| EMA 位置 | 只对 base 做 EMA，残差**不过 EMA**（否则高频被吃掉，学不动） | 同上 |
| 频率 | 15 Hz（与 FORGE 一致） | `decimation=8, dt=1/120` |
| 限幅层级 | ① clip(a_res) ② 叠加后 clip 到 `pos_action_bounds` ③ 控制器内部再 clip `pos_action_threshold` | 三层，防"接管" |

### 1.3 观测（19 维）

```
fingertip_pos_rel_fixed(3)   # 带 1 mm 噪声
fingertip_quat(4)            # 带 0.1° 噪声
ee_linvel(3)                 # 已修 #8874，用真值差分
ee_angvel(3)                 # 同上，且仅 yaw 非零
ft_force(3)                  # 带 1 N 噪声的力
force_threshold(1)           # 每个环境随机的力阈值 5-10 N
residual_action(2)           # 上一步残差
```

critic 额外看到 37 维特权信息（关节角、工件真值位姿、控制器增益、阈值），总 56 维。**非对称 AC**：actor 只看可部署量，critic 看全状态。

### 1.4 奖励

$$
r = \underbrace{K_{5,4}+K_{50,2}+K_{100,0}}_{\text{FORGE 多尺度关键点}} + \underbrace{1_{\text{engage}} + 1_{\text{success}}}_{\text{阶段奖励}} - \underbrace{0.2\max(0,\|F\|-F^{th})}_{\text{过力惩罚}} - \underbrace{0.05\|\Delta\|^2 - 0.2\|\Delta_t-\Delta_{t-1}\|^2}_{\text{残差正则}}
$$

- 关键点核：`K_{a,b}(x) = 1/(e^{ax}+b+e^{-ax})`，`x` 为多尺度关键点距离；
- 过力惩罚沿用 FORGE（`contact_penalty_scale=0.2`），阈值每环境随机 5–10 N；
- **残差正则只惩罚残差本身**，不惩罚 base —— 这样"残差趋于 0"等价于"退化成解析基线"，是一个天然的安全退路。

> 想做 A/B：把 `residual_penalty_scale` 设 0，看残差会不会学成"大范围乱扫"（通常会，且力峰值上升）。这是论文里一个好消融。

### 1.5 为什么必须给 base 加随机化（否则残差学不到东西）

螺旋搜索是**时间确定性**的：第 k 步该朝哪个方向转是算好的。如果所有环境用同一条螺旋，
残差只需要记住"第 30 步往右推 2 mm"就能刷分，**根本没学会看力**。换个孔位就崩。

`forge_ext/base_policy.py` 因此提供三项基座随机化（`ForgeResidualCfg.base` 里默认已开）：

| 参数 | 默认（残差训练） | 含义 |
|---|---|---|
| `phase_noise` | π | 初相位 U[−π, π] → 每个环境从螺旋的不同位置开始 |
| `press_step_rand` | 0.3 | 下压速度 ±30% → 残差不能依赖"固定时刻接触" |
| `radius_max_rand` | 0.5 | 螺旋最大半径 ±50% → 残差不能依赖"固定搜索幅度" |

阶段二的解析基线用的是**全 0**（确定性），所以两边数字不可直接混用——这也是为什么
`demos/05_eval_success.py` 里"零残差"和 `demos/03` 的"脚本基线"必须分开跑、分别报告。

### 1.6 两个会吃掉"精密"的实现细节

**(1) 死区比容差大两个数量级。** `default_dead_zone=[5,5,5,1,1,1]` 在 wrench 域做减法，
折算成位置约 $5/565\approx8.8$ mm，而几何容差是 0.057 mm。也就是说：位置误差小于 9 mm 时
控制器**根本不产生力**，精密段完全靠 peg 在夹爪里的被动自对齐。

→ 这是本毕设最值得做的一个消融：把 `default_dead_zone` 调到 `[0.5,0.5,0.5,0.1,0.1,0.1]`，
扫"死区 vs 成功率 / 力峰值"。别人没做过（FORGE 论文无此消融），而且直接属于"柔顺控制参数整定"。

**(2) 力观测滞后约 1 个策略步。** EMA α=0.25 @120 Hz → 约 4.8 Hz 截止，群延迟 33 ms；
15 Hz 决策 67 ms/步。想更准就把观测里的 `ft_force` 扩成 6 维（当前 + 上一步），
或者接受它并在论文里说明"力反馈存在约 67 ms 延迟"。


---

## 2. 步骤零：安装 rsl_rl（一次）

本机当前**只装了 rl_games**，没有 rsl_rl：

```bash
/opt/isaac-workspace/envs/isaaclab/bin/python -c "import rsl_rl" 
# ModuleNotFoundError: No module named 'rsl_rl'   ← 当前状态
```

安装（官方推荐路径）：

```bash
cd /home/mi/graduation_design/RL/IsaacLab
./isaaclab.sh --install rsl_rl
```

它会装 `rsl-rl-lib==3.0.1` + `onnxscript`，并连带 `tensordict`、`onnx`、`GitPython`（已核对 PyPI 元数据：`torch>=2.6`、`tensordict>=0.7`、`onnx`、`GitPython`）。

**装完必须复查阶段一锁定的四个包**（`--install` 曾经把 `click/psutil/typing_extensions` 升级坏过，见依赖记录 4.1）：

```bash
/opt/isaac-workspace/envs/isaaclab/bin/python -m pip list 2>/dev/null | \
  grep -E "^(rsl-rl-lib|tensordict|click|psutil|typing_extensions|setuptools|torch) "

# 期望：rsl-rl-lib 3.0.1 / click 8.1.7 / psutil 5.9.8 / typing_extensions 4.12.2
#       setuptools 80.9.0 / torch 2.7.0
```

若被升级坏：

```bash
python -m pip install --no-cache-dir "click==8.1.7" "psutil==5.9.8" "typing_extensions==4.12.2" "setuptools==80.9.0"
```

最后验证：

```bash
/opt/isaac-workspace/envs/isaaclab/bin/python -c "import rsl_rl, tensordict; print('rsl_rl OK')"
```

> **降级方案**：如果这一步把环境搞坏且修不回来，就把残差训练改到 rl_games（`demos/06` 的写法 + 一份新的 yaml），评估改用 `play.py` 目视 + 训练日志里的 `episode/successes`。功能上不影响方法，只影响"逐 checkpoint 定量评估"的便利性。

---

## 3. 步骤一：冒烟（确认残差链路通）

```bash
cd /home/mi/graduation_design/RL/IsaacLab
./isaaclab.sh -p ../demos/04_train_residual.py \
  --num_envs 16 --max_iterations 20 --seed 0 --headless
```

### 3.1 成功标志

```text
[INFO] 日志目录: .../RL/outputs/rsl_rl/<时间戳>_residual
[INFO] num_envs=16  max_iterations=20  seed=0
Learning iteration 0/20 ...
...
```

**20 次迭代后不要求任何成功率**，只要求：环境能 step、loss 能出、checkpoint 能存。

### 3.2 必须看的三个量（在终端或 TensorBoard）

| 量 | 位置 | 健康值 |
|---|---|---|
| `Loss/Policy` | TB | 量级 1e-3 ~ 1e-1，不 NaN |
| `Loss/Value` | TB | 缓慢下降 |
| `Mean episode length` | TB | **恒等于 150**（Factory 只超时终止） |
| `Mean reward` | TB | 应该在 -1 ~ +5 之间；若恒为 0 说明奖励没接上 |

> rsl_rl 3.x **不会自动把环境 `extras` 里的指标写进 TensorBoard**，所以成功率不在 TB 里。
> 这是正常的：**成功率统一用 `demos/05_eval_success.py` 在固定种子上评估**（见 §5）。

### 3.3 冒烟阶段常见错误

| 报错 | 原因 | 处理 |
|---|---|---|
| `KeyError: 'residual_action'` | `obs_order` 与 `_get_observations` 的 key 不一致 | 确认用的是 `forge_ext` 的 cfg，不是官方 cfg |
| `shape mismatch: [16,2] vs [16,7]` | `_pre_physics_step` 把 2 维残差当 7 维用了 | 检查 `action_space=2` 是否生效 |
| `RuntimeError: The size of tensor a (19) must match ...` | `observation_space` 写错（gym 空间与实际张量不一致） | 改成 19 / 56 |
| `AttributeError: 'ForgeResidualEnv' object has no attribute '_true_prev_pos'` | 用了旧版本代码 | 重新拉 `forge_ext/residual_env.py` |

---

## 4. 步骤二：正式训练

```bash
cd /home/mi/graduation_design/RL/IsaacLab
./isaaclab.sh -p ../demos/04_train_residual.py \
  --num_envs 64 --max_iterations 1500 --seed 0 --run_name residual_seed0 --headless
```

### 4.1 规模与时间

`num_steps_per_env=32`：

| num_envs | 每迭代步数 | 1500 迭代总步数 | 约合回合 |
|---|---|---|---|
| 16 | 512 | 0.77 M | 5.1 k |
| 32 | 1024 | 1.54 M | 10.2 k |
| 64 | 2048 | 3.07 M | 20.5 k |

**为什么 32 步/迭代比 FORGE 的 128 步小**：残差策略不需要长 history（base 已经把状态带到接触附近），短 rollout + 多迭代收敛更快、显存更省。

**建议训练 3 个种子**（0/1/2），论文报均值 ± 标准差。8 GB 上一次 64 envs 预计 6–12 小时（未实测），可以晚上跑。

### 4.2 什么时候停

不是看迭代数，而是看**评估成功率是否平台化**：

```bash
# 每 200 迭代评估一次（另开终端，或训练停掉后跑）
./isaaclab.sh -p ../demos/05_eval_success.py \
  --checkpoint ../outputs/rsl_rl/<时间戳>_residual_seed0/model_200.pt \
  --num_envs 32 --episodes 5 --headless
```

评估曲线（成功率 vs 迭代）**才是论文里的学习曲线**，比 TB 的 reward 曲线更有说服力。

---

## 5. 步骤三：评估与对比（论文主表）

### 5.1 三组对象，同一把尺

```bash
# ① 纯解析基线（残差恒 0）
./isaaclab.sh -p ../demos/05_eval_success.py --num_envs 32 --episodes 10 --headless

# ② 脚本基线（同一套 base 参数，独立跑）
./isaaclab.sh -p ../demos/03_scripted_insert.py --num_envs 32 --episodes 10 --headless

# ③ 残差策略
./isaaclab.sh -p ../demos/05_eval_success.py \
  --checkpoint ../outputs/rsl_rl/<run>/model_1400.pt --num_envs 32 --episodes 10 --headless
```

> ① 和 ② 应该给出几乎相同的数字（同一 base，同一判据）。**如果差很多，说明脚本有 bug**，先修再往下走。这个交叉验证能省掉后面很多解释不了的数字。

### 5.2 主表模板

| 方法 | 成功率 | Wilson 95% CI | 插接时间 (s) | 力峰值 mean (N) | 力峰值 p95 (N) | 训练样本数 |
|---|---|---|---|---|---|---|
| 纯位置控制（无搜索） | | | | | | 0 |
| 解析柔顺（螺旋搜索） | | | | | | 0 |
| FORGE 端到端 PPO | | | | | | |
| **残差 PPO（本文）** | | | | | | |
| 残差 PPO（残差 3 维） | | | | | | |

Wilson 区间直接算：

```python
import math
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c-r)/d, (c+r)/d
```

### 5.2.1 建议的消融清单（按性价比排序）

| # | 消融 | 怎么改 | 想说明什么 |
|---|---|---|---|
| 1 | 残差维度 2 vs 3（加 yaw） | `residual_dim=3`，动作前 3 维用；`_pre_physics_step` 里把第 3 维加到 `actions[:,5]` | 横向修正够不够，yaw 有没有额外价值 |
| 2 | 残差尺度 1 / 3 / 6 mm | `residual_pos_scale` | 权限太小学不动、太大破坏安全 |
| 3 | 残差正则开 / 关 | `residual_penalty_scale=0` | 正则是否必要（通常力峰值会变差） |
| 4 | 观测含力 / 不含力 | `obs_order` 去掉 `ft_force`（注意同步改 `observation_space`） | 力反馈对残差的价值（对标 FORGE 的 No Force） |
| 5 | base 随机化开 / 关 | `base.phase_noise=0, press_step_rand=0, radius_max_rand=0` | 证明"残差学到的是力反馈修正，不是相位记忆"（**最有说服力的一个**） |
| 6 | 死区 5 N vs 0.5 N | 覆盖 `ctrl.default_dead_zone` | 柔顺参数对精密插接的影响 |
| 7 | 残差 + 变阻抗（可选） | §7 | 刚度自适应是否进一步降低力峰值 |

> 第 5 项一定要做：它是回应"你的残差是不是只是背下了螺旋轨迹"这个质疑的唯一硬证据。

### 5.3 必做的统计检验

- 同一批初始位姿集（固定 seed=1234）→ 配对样本 → **配对 Wilcoxon 符号秩检验**（成功率）或 bootstrap；
- 时间指标报 p50/p95，不要只报均值（接触任务的长尾很长）；
- 每个格子 ≥320 次试验（32 env × 10 回合）。

---

## 6. 路线 B：RLinf 接入

### 6.1 先把现实说清楚

| 事实 | 出处 | 对你的含义 |
|---|---|---|
| Isaac Lab 官方已集成 RLinf（`rlinf/envs/sim/isaaclab/`） | [RLinf 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/isaaclab.html) | 不用自己从零写接口 |
| 但只注册了 **1 个任务**（stack-cube），且 `_wrap_obs` 依赖两路相机 | `tasks/stack_cube.py` | 你的 FORGE 任务要自己写一个子类 |
| 官方硬件口径 **1 节点 8 GPU（8×80 GB）** | [requirements](https://rlinf.readthedocs.io/en/latest/rst_source/start/requirements.html) | 8 GB 上不可能跑官方示例 |
| **单卡 Isaac Lab 有已知静默挂死**（issue #1384：`env_fn()` 在 try 外 + `obs_queue.get()` 无 timeout） | [issue #1384](https://github.com/RLinf/RLinf/issues/1384) | 启动失败时你会看到"无任何日志地卡住"，必须加超时/心跳 |
| RLinf v0.4 默认 torch 2.11，Isaac Sim 5.1 侧是 2.7.x | `requirements/install.sh` | 必须**独立 venv**，靠子进程隔离 |
| RLinf **没有 residual 支持** | 全仓库检索 | 要么加 `model_type`，要么在 `action_utils` 里组合 |

### 6.2 三步验证法（不要一上来就接 FORGE）

**第 1 步：RLinf 自身能跑（1–2 天）**

```bash
# 官方推荐 docker，但 8 GB 卡跑不动 VLA；用源码装最小依赖
git clone https://github.com/RLinf/RLinf.git ~/RLinf && cd ~/RLinf
bash requirements/install.sh embodied --model none --env mujoco_warp
# 跑最轻的 MLP+PPO（MuJoCo Warp 或 EmbodiChain CartPole）
bash examples/embodiment/run_embodiment.sh mujoco_warp_ppo_mlp
```

**通过标准**：`env/success_once` 出现、loss 下降、TensorBoard 有图。这一步只验证"RLinf 能在这台机器上训练一个小 MLP 策略"。

> 注意：必须用 `run_embodiment.sh`（它设置 `EMBODIED_PATH`），直接 `python` 跑会 `InterpolationKeyError`。

**第 2 步：接上 FORGE 任务（1–2 天）**

```bash
cp /home/mi/graduation_design/RL/rlinf_adapter/forge_peginsert_env.py \
   ~/RLinf/rlinf/envs/sim/isaaclab/tasks/forge_peginsert.py
```

在 `~/RLinf/rlinf/envs/sim/isaaclab/__init__.py` 里注册：

```python
from .tasks.forge_peginsert import IsaaclabForgeResidualEnv

REGISTER_ISAACLAB_ENVS = {
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0": IsaaclabStackCubeEnv,
    "Isaac-Forge-PegInsert-Residual-v0": IsaaclabForgeResidualEnv,
}
```

配置：复制 `examples/embodiment/config/maniskill_ppo_mlp.yaml` → `forge_peginsert_ppo_mlp.yaml`，把 `env` 段换成 `rlinf_adapter/config/forge_peginsert_ppo_mlp.yaml` 里的内容，把 `model` 段改成 `obs_dim: 19, action_dim: 2`。

**关键前提**：RLinf 的子进程要能 `import forge_ext`，所以启动前：

```bash
export PYTHONPATH=/home/mi/graduation_design/RL:$PYTHONPATH
export EMBODIED_PATH=$HOME/RLinf/examples/embodiment
```

**第 3 步：加超时保护（针对 #1384）**

#1384 的根因是子进程启动失败时父进程永久阻塞。最简单的自保：**启动前手工验证 Isaac Sim 能起来**，并在配置文件里把 `total_num_envs` 压到 8–16：

```bash
# 先单独验证 Isaac Sim headless 能启动（这一步能过，#1384 概率大幅降低）
cd /home/mi/graduation_design/RL/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

如果 RLinf 启动后 **GPU 占用为 0 且 30 分钟无日志**，直接 `kill -9` 并认定为 #1384 触发，转降级方案。

### 6.3 显存预算（8 GB）

| 组件 | 估计占用 | 说明 |
|---|---|---|
| Isaac Sim（headless，FORGE 场景，16 envs） | 3.5–5 GB | PhysX 全局缓冲是大头 |
| RLinf 训练进程（MLP + PPO + Ray worker） | 1–2 GB | MLP 很小，Ray 才是内存杀手 |
| 共享显存余量 | 1–2 GB | 需要 PyTorch caching 留缓冲 |

→ **只跑 8–16 envs**，`actor.micro_batch_size` 压到 512，关 `video_cfg.save_video`、关 `torch_compile`。
强行跑 64 envs 大概率 OOM；而 16 envs 又基本训不出东西。**所以 B 路线的目标只能是"接入成功"，不是"训出好策略"**，这一点要在论文里写清楚。

### 6.4 如果坚持在 RLinf 里做残差

RLinf 没有 residual 概念，两条实现路径：

| 路径 | 做法 | 评价 |
|---|---|---|
| **推荐** | 在 `rlinf/models/embodiment/` 新增 `residual_mlp`：forward 里 `a = a_base + α·a_res`，base 从环境额外返回的 `base_action` 取 | 梯度语义干净，方法章节好写 |
| 不推荐 | 在 `rlinf/envs/action_utils.py` 的 `prepare_actions_for_*` 里把残差和 base 相加 | 改动最小，但"残差"变成环境内部实现，论文里说不清 |

相应的 env 适配器要多返回一个 `base_action` 字段（`_wrap_obs` 里从 `env.unwrapped` 读）。

### 6.5 时间盒与降级

| 用时 | 里程碑 | 未达成怎么办 |
|---|---|---|
| 第 1 天 | RLinf 装好 + 最小 MLP 示例跑通 | 装不上 → 直接降级 |
| 第 2 天 | FORGE 适配器能 reset/step（用脚本调，不进 RLinf） | 调不通 → 降级 |
| 第 3 天 | RLinf 里跑 5 次迭代 | 挂死/OOM → 降级 |

**降级方案**（论文写法）：

> "本文完成了 RLinf 侧的任务适配器实现（`rlinf/envs/sim/isaaclab/tasks/forge_peginsert.py`），并在单卡 8 GB 环境下尝试接入；
> 因 RLinf 官方 Isaac Lab 支持面向 8 卡 VLA 后训练场景，且上游存在单卡静默挂死问题（issue #1384）与本机显存/内存限制，
> 本文的训练以 Isaac Lab 原生 rsl_rl 完成，RLinf 作为系统层对照框架在第 6 章讨论。"

这段话是**合格的工程结论**，比"跑失败了"强得多，也比硬凑一个跑不动的实验诚实。

---

## 7. 进阶（可选）：变阻抗 + 残差

如果路线 A 顺利、还有时间，做一个"残差不仅改位置，还改刚度"的实验：

`OperationalSpaceControllerCfg` 支持 `impedance_mode="variable"` / `"variable_kp"`，刚度范围由 `motion_stiffness_limits_task=(0, 1000)` 限制。做法：

1. 把 Factory 的控制器换成 `OperationalSpaceController`（`target_types=("pose_abs",)`）；
2. 残差动作扩成 3 维：`[Δx, Δy, Δkp]`，`Δkp` 映射到 `[100, 900] N/m`；
3. 奖励里加"刚度变化率"惩罚，避免抖振。

**风险**：换控制器 = 不再是 FORGE 复现，需要重跑基线；变阻抗在接触瞬间容易不稳定。**建议只作为第 5 章最后一个消融，不作为主线。**

---

## 8. 与论文的对应 & sim-to-real 讨论

| 章节 | 内容 | 材料 |
|---|---|---|
| 4.1 | 分层控制架构（阻抗内环 + 导纳外环 + 残差） | §1.1 图 |
| 4.2 | 动作空间与频率设计（为什么 2 维、为什么 15 Hz） | §1.2 表 |
| 4.3 | 奖励设计（关键点 + 过力 + 残差正则） | §1.4 公式 |
| 4.4 | 与 FORGE 的关系（替换目标位姿生成器，保留其余） | §4.1 三条事实 |
| 5.x | 主表 + 学习曲线 + 消融 | §5、§5.2 |
| 6.x | RLinf 接入、cuRobo 分工、sim-to-real | §6、阶段二 §7 |

**sim-to-real 要提前想清楚的 5 件事**（答辩必问）：

1. **力从哪来**：仿真里是 `get_link_incoming_joint_force`；真机用 Franka 关节力矩投影到 EE（FORGE 就是这么做的），或加腕部 F/T 传感器。**零漂与重力补偿必须处理**，否则恒压环会漂。
2. **控制频率**：仿真 15 Hz 策略 + 120 Hz 阻抗；真机 FORGE 用 15 Hz 策略 + 1 kHz 阻抗。**保持 15 Hz 策略频率不变**，底层提高即可。
3. **阻抗参数**：仿真 kp=565 N/m 是"能插进去又不压坏"的折中；真机要按工件材料重标定，建议先在自由空间扫 kp。
4. **位置噪声**：仿真 1 mm 是训练用的"困难设定"；真机标定精度可能更好（<0.5 mm），此时残差策略会"过度搜索"，需要调小残差尺度。
5. **安全**：残差限幅 ±3 mm/步 是硬约束；再加上力阈值硬限幅（base 的导纳环），构成双保险。论文里这是卖点。

---

## 9. 验收清单

**路线 A（必须全绿）**

- [ ] `rsl_rl` 安装完成，四个锁定包未被升级（§2 复查命令通过）
- [ ] 20 次迭代冒烟通过（§3）
- [ ] 正式训练 ≥1000 迭代 × 3 个种子，checkpoint 齐全
- [ ] 评估脚本能加载 checkpoint 并输出成功率 / 时间 / 力峰值（§5.1）
- [ ] ① 零残差 与 ② 脚本基线数字一致（交叉验证通过）
- [ ] 主表填完（≥4 行），带 Wilson 区间
- [ ] 至少 2 个消融（残差维度 / 残差尺度 / 残差正则 / 观测含力）
- [ ] 学习曲线（成功率 vs 迭代）出图

**路线 B（尽力，时间盒 3 天）**

- [ ] RLinf 安装成功，最小 MLP 示例跑通（§6.2 第 1 步）
- [ ] 适配器文件已放入 RLinf 并注册（§6.2 第 2 步）
- [ ] 5 次迭代冒烟（成功 → 写"已接入"；失败 → 写阻塞点 + 降级说明）
- [ ] 阻塞点有日志/截图证据（issue #1384 的判定依据）

---

## 10. 排错 FAQ

**Q1：训练 loss 正常但成功率一直是 0**

先跑 `05_eval_success.py`（不带 checkpoint，即零残差）看基线是否非零。如果基线也是 0：
- base 的 `hold_force` / `press_step` 没调好（回阶段二 §3.3）；
- 或者 `z_min=-0.027` 不够深（孔深 25 mm，成功需要 z < -24 mm）。

如果基线非零但残差为 0：残差在"帮倒忙"。检查 `residual_pos_scale` 是否过大（3 mm 已经不小，试 1 mm），以及残差正则是否太弱。

**Q2：残差一直饱和在 ±1**

- 残差尺度太大 → 减小 `residual_pos_scale`；
- 奖励里的 `kp_fine` 权重不够 → 残差学不到"精细对中"的梯度；
- 观测没给力 → 打开 `ft_force`（默认已开）。

**Q3：训练出来的残差几乎恒 0（退化成基线）**

这**不一定是失败**：说明 base 已经足够好。但如果论文要"残差有效"，就需要让任务更难：
- 提高孔位噪声（`obs_rand.fixed_asset_pos` → 2.5 mm）；
- 缩小间隙（改 USD 或换孔件）；
- 降低 base 的搜索能力（`search_radius_max` → 0.5 mm，逼残差接手）。

**这是最容易被答辩问到的一点**："你的残差学到了什么？" 提前准备好"任务难度 vs 残差贡献"的曲线。

**Q4：评估脚本 `--checkpoint` 报 `KeyError: 'policy'`**

说明 checkpoint 不是 rsl_rl 的（可能是 rl_games 的 `.pth`）。rl_games checkpoint 只能用 `play.py` 或阶段二的日志看，不要混用。

**Q5：`runner.load()` 报 missing keys**

`obs_groups` 或网络结构变了（比如 `--ablate` 改过观测维度）。加载时必须用**同一份 `agents.py` 配置**。

**Q6：RLinf 启动后卡住不动**

按 §6.2 第 3 步处理：先手工验证 Isaac Sim headless 能起；再压 `total_num_envs` 到 8；仍卡就是 #1384，记录并降级。

**Q7：多跑几个种子，显存越跑越少**

PyTorch caching 碎片。每个种子单独起进程（不要在一个进程里循环），或设 `PYTORCH_CUDA_ALLOC_CONF=expandable_background`。

---

## 11. 参考

- 残差 RL 原始论文：<https://arxiv.org/abs/1812.03201>（Johannink et al., ICRA 2019）
- Residual Policy Learning：<https://arxiv.org/abs/1812.06298>
- FORGE（RA-L 2025）：<https://arxiv.org/abs/2408.04587>
- Isaac Lab rsl_rl 教程：<https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/reinforcement-learning/rl_existing_scripts.html>
- `OperationalSpaceController` 文档：<https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.controllers.html>
- RLinf 仓库与文档：<https://github.com/RLinf/RLinf>、<https://rlinf.readthedocs.io/en/latest/>
- RLinf 的 Isaac Lab 示例：<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/isaaclab.html>
- RLinf 接入新环境：<https://rlinf.readthedocs.io/en/latest/rst_source/extending/new_env.html>
- 上游 issue：#8874（速度观测）、#3513（力坐标系）、#5424（动作界语义）、#8850（间隙）、RLinf #1384（单卡挂死）
- 本仓库调研：`docs/research/RLinf_技术调研报告.md`、`docs/research/cuRobo_IsaacLab_技术调研.md`
