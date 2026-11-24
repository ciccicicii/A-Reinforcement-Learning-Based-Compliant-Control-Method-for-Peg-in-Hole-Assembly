# 基于残差强化学习与柔顺控制的机器人精密插接方法研究

> 本科毕业设计代码与文档仓库
> 环境：Ubuntu 24.04 / Isaac Sim 5.1.0 / Isaac Lab v2.3.0 / Python 3.11 / torch 2.7.0+cu128 / RTX 4060 Laptop 8 GB
> 更新：2026-11-22

---

## 三个阶段与当前状态

| 阶段 | 内容 | 状态 | 入口文档 |
|---|---|---|---|
| 一 | 4060 上跑通不含 cuRobo 的基础运控闭环 | ✅ 已完成 | [ISAAC_LAB_FORGE_基础运控搭建教程.md](ISAAC_LAB_FORGE_基础运控搭建教程.md) |
| 二 | 复现 FORGE PegInsert，评估 cuRobo 的插入价值 | 🚧 教程已就绪 | [阶段二_FORGE复现与cuRobo评估教程.md](阶段二_FORGE复现与cuRobo评估教程.md) |
| 三 | RLinf 接入与残差 RL 训练 | 🚧 教程已就绪 | [阶段三_RLinf接入与残差RL教程.md](阶段三_RLinf接入与残差RL教程.md) |

方案依据与文献综述：[docs/调研_阶段二三技术选型.md](docs/调研_阶段二三技术选型.md)
环境安装与排错：[依赖安装与排错记录.md](依赖安装与排错记录.md)
原始调研材料（论文、issue、版本核实）：[docs/research/](docs/research/README.md)

---

## 目录结构

```text
RL/
├── IsaacLab/                 # 固定 v2.3.0，尽量不改（官方参考实现）
├── forge_ext/                # 本仓库的任务扩展（不侵入 IsaacLab）
│   ├── base_policy.py        #   解析柔顺基线（下降+恒压+螺旋搜索）
│   ├── residual_env.py       #   残差任务：2 维残差 + base 叠加 + 正则
│   ├── logged_env.py         #   与官方等价、只补成功率日志的壳
│   ├── agents.py             #   rsl_rl PPO 超参
│   └── __init__.py           #   注册两个 gym 任务
├── demos/
│   ├── 01_franka_hold.py     #   阶段一：末端位姿保持
│   ├── 02_forge_inspect.py   #   阶段二：环境体检（维度/力/成功判据）
│   ├── 03_scripted_insert.py #   阶段二：解析柔顺基线
│   ├── 04_train_residual.py  #   阶段三：残差 PPO 训练
│   ├── 05_eval_success.py    #   阶段三：统一评估（脚本/零残差/checkpoint）
│   └── 06_train_forge_rl_games.py  # 阶段二：FORGE 官方 PPO + 消融
├── rlinf_adapter/            # 阶段三：RLinf 任务适配器与配置模板
├── docs/
│   ├── 调研_阶段二三技术选型.md
│   └── research/             # 原始调研（含论文 PDF 与 issue 证据）
├── configs/                  # 预留
└── outputs/                  # CSV / 训练日志 / 图表
```

---

## 快速开始

```bash
source /home/mi/miniforge3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/opt/isaac-workspace/conda-pkgs
export XDG_CACHE_HOME=/opt/isaac-workspace/cache
export TMPDIR=/opt/isaac-workspace/tmp
conda activate /opt/isaac-workspace/envs/isaaclab
cd /home/mi/graduation_design/RL/IsaacLab
```

**体检（阶段二第一步）**

```bash
./isaaclab.sh -p ../demos/02_forge_inspect.py --num_envs 4 --steps 150 --headless
# 确认：15 Hz、150 步/回合、obs 24 维、动作 7 维、间隙 0.057 mm、成功阈值 1 mm
```

**解析柔顺基线（不学习能到多少）**

```bash
./isaaclab.sh -p ../demos/03_scripted_insert.py --num_envs 32 --episodes 10 --headless
```

**训练（两条主线）**

```bash
# FORGE 端到端 PPO（阶段二）
./isaaclab.sh -p ../demos/06_train_forge_rl_games.py --num_envs 64 --max_epochs 500 --headless

# 残差 PPO（阶段三，需先装 rsl_rl：./isaaclab.sh --install rsl_rl）
./isaaclab.sh -p ../demos/04_train_residual.py --num_envs 64 --max_iterations 1500 --headless
```

**评估（三组对象同一把尺）**

```bash
./isaaclab.sh -p ../demos/05_eval_success.py --num_envs 32 --episodes 10 --headless
./isaaclab.sh -p ../demos/05_eval_success.py --checkpoint ../outputs/rsl_rl/<run>/model_1400.pt \
  --num_envs 32 --episodes 10 --headless
```

---

## 三条必须记住的事实

1. **控制频率 15 Hz**（`dt=1/120`，`decimation=8`），回合 150 步，**只有超时终止**，所有环境同步重置。
2. **FORGE 是两层结构**：策略出 4 自由度目标位姿 → 120 Hz 任务空间阻抗控制器。所以"残差"替换的是**目标位姿生成器**，不是控制器。
3. **成功率不在官方 TensorBoard 里**（`extras["successes"]` 不被 rl_games 记录）。用 `forge_ext/logged_env.py`（与官方等价）或 `demos/05_eval_success.py`。

## 上游已知问题（会影响你复现出的数字）

| 问题 | 影响 | 处理 |
|---|---|---|
| IsaacLab #8874 | v2.3.0 的 `ee_linvel/ee_angvel` 实为"噪声/dt"，24 维观测里 6 维是废的 | `forge_ext` 里 `fix_velocity_obs=True` 已修；复现官方时设 False |
| IsaacLab #3513 | `change_FT_frame` 未真正旋转力矢量，`ft_force` 在世界系 | 论文写"世界系，孔轴与 z 同轴"，别写"已转孔口系" |
| IsaacLab #8850 | 实测孔内径≈9.0 mm，实际间隙约 1 mm（配置写 0.057 mm） | **自己量一次**再写"精密" |
| IsaacLab #5424 | `pos_action_bounds` / `pos_action_threshold` 语义与 Factory 相反 | 改动作前先读 `forge_env.py` |
| RLinf #1384 | 单卡 IsaacLab 静默挂死（子进程启动失败 + 无 timeout 的 queue.get） | 先单独验证 headless 启动；30 分钟无日志就判定触发 |
| cuRobo 编译 | `cuda-toolkit=12.8` 带 GCC 14，nvcc 要求 <14 | `export CC/CXX/CUDAHOSTCXX` 指向系统 gcc/g++ |

## 主要参考

- FORGE: arXiv [2408.04587](https://arxiv.org/abs/2408.04587)，IEEE RA-L 2025, 10:4436–4443，DOI 10.1109/LRA.2025.3551637
- Factory: arXiv [2205.03532](https://arxiv.org/abs/2205.03532)
- Residual RL: arXiv [1812.03201](https://arxiv.org/abs/1812.03201)；Residual Policy Learning: arXiv [1812.06298](https://arxiv.org/abs/1812.06298)
- RLinf: arXiv [2509.15965](https://arxiv.org/abs/2509.15965)，OSDI 2026
- Isaac Lab v2.3.0 文档：<https://isaac-sim.github.io/IsaacLab/v2.3.0/>
# A-Reinforcement-Learning-Based-Compliant-Control-Method-for-Peg-in-Hole-Assembly
