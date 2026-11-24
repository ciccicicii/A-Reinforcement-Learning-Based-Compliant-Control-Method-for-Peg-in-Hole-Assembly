# Isaac Lab + FORGE：Franka 基础运控操作教程

> **目的**：在本机 RTX 4060 Laptop（8 GB 显存）上完成三件事——启动空场景、跑通 Franka 末端运控、加载 FORGE 插接场景。
>
> **本阶段不做**：PPO 训练、RLinf 适配、残差强化学习、相机传感器、cuRobo 集成、大规模并行环境。
>
> **验收标准**：
> 1. Isaac Sim 空场景可以启动；
> 2. 单台 Franka 能按多个末端目标位姿运动；
> 3. `Isaac-Forge-PegInsert-Direct-v0` 能创建、重置并推进仿真。

环境已安装完成（Isaac Lab v2.3.0 + Isaac Sim 5.1.0 + Python 3.11 + PyTorch 2.7.0+cu128）。
安装过程与依赖问题的记录见 [依赖安装与排错记录.md](依赖安装与排错记录.md)，**日常操作不需要看它**。

---

## 0. 每次开新终端，先激活环境

```bash
source /home/mi/miniforge3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/opt/isaac-workspace/conda-pkgs
export XDG_CACHE_HOME=/opt/isaac-workspace/cache
export TMPDIR=/opt/isaac-workspace/tmp

conda activate /opt/isaac-workspace/envs/isaaclab

cd /home/mi/graduation_design/RL/IsaacLab
```

确认环境正确：

```bash
which python
```

必须是：

```text
/opt/isaac-workspace/envs/isaaclab/bin/python
```

如果是 `/home/mi/miniforge3/bin/python`，说明没激活成功，后面所有命令都会报 `No module named 'isaaclab'`。

运行前顺手确认没有遗留的仿真进程：

```bash
nvidia-smi
ps -eo pid,etime,args | grep -E 'create_empty|run_diff_ik|zero_agent' | grep -v grep
```

有残留就 `kill -TERM <PID>`，无效再用 `kill -KILL <PID>`。**不要同时开两个 Isaac Sim 实例。**

---

## 1. 验收一：启动空场景（已通过）

```bash
cd /home/mi/graduation_design/RL/IsaacLab

./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

成功标志是终端出现：

```text
[INFO]: Setup complete...
```

**视口全黑是正常的**：这个脚本不创建地面、灯光和物体，渲染不出任何内容。它的作用是验证仿真器能启动、物理循环能推进。

（本次已确认该场景可启动，显存占用约 2.2 GB。）

无窗口快速验证：

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

---

## 2. 验收二：Franka 差分 IK 基础运控

```bash
cd /home/mi/graduation_design/RL/IsaacLab

./isaaclab.sh -p \
  scripts/tutorials/05_controllers/run_diff_ik.py \
  --robot franka_panda \
  --num_envs 1
```

### 预期现象

- 视口出现一张灰色桌面和一台 Franka Panda；
- 末端有表示当前位姿的坐标系标记，另有表示目标位姿的标记；
- 机械臂向目标位姿收敛；
- 约每 150 个仿真步重置一次，并切换到下一个目标位姿。

周期性重置是脚本设计，不是控制失败。

### 参数说明

`--robot` 只能是 `franka_panda` 或 `ur10`，写 `franka` 会报错。
`--num_envs 1` 必须显式指定——脚本默认会创建 128 个环境。

### 本步骤验证的内容

验证了「末端目标位姿 → 差分 IK → 关节目标 → Franka 执行」这条控制链，以及资产加载和物理推进。

**尚未验证**：插接、接触力反馈、柔顺/阻抗控制、强化学习。

### 记录材料

这一步是毕设"基础运控验证"的素材来源，建议保存：

- 视口截图或录屏；
- 终端输出（含 `Using device: cuda:0`）；
- `nvidia-smi` 显示的显存占用。

存放到 `RL/docs/` 或 `RL/outputs/`。

### 显存不足时

先关闭浏览器等占用 GPU 的程序，再试：

```bash
./isaaclab.sh -p \
  scripts/tutorials/05_controllers/run_diff_ik.py \
  --robot franka_panda \
  --num_envs 1 \
  --headless
```

---

## 3. 验收三：加载 FORGE Peg Insertion 场景

source /home/mi/miniforge3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/opt/isaac-workspace/conda-pkgs
export XDG_CACHE_HOME=/opt/isaac-workspace/cache
export TMPDIR=/opt/isaac-workspace/tmp

conda activate /opt/isaac-workspace/envs/isaaclab

### 3.1 确认任务已注册
cd /home/mi/graduation_design/RL/IsaacLab
```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

在输出中查找：

```text
Isaac-Forge-PegInsert-Direct-v0
```

该任务是 Franka 的插销插孔接触任务，也是后续基础控制、柔顺控制和残差强化学习的共同环境基础。

### 3.2 零动作加载

```bash
./isaaclab.sh -p \
  scripts/environments/zero_agent.py \
  --task Isaac-Forge-PegInsert-Direct-v0 \
  --num_envs 1
```

该脚本每步输入全零动作，**不会完成插接**。它验证的是：

- Franka、插销、插座与工作台资产能否加载；
- 环境能否初始化与重置；
- 观测空间、动作空间能否创建；
- 物理仿真能否持续推进。

零动作下没插进去是预期行为，不能据此判定环境失败。

无窗口版本：

```bash
./isaaclab.sh -p \
  scripts/environments/zero_agent.py \
  --task Isaac-Forge-PegInsert-Direct-v0 \
  --num_envs 1 \
  --headless
```

---

## 4. 后续：自己的 Demo 开发顺序

source /home/mi/miniforge3/etc/profile.d/conda.sh

export CONDA_PKGS_DIRS=/opt/isaac-workspace/conda-pkgs
export XDG_CACHE_HOME=/opt/isaac-workspace/cache
export TMPDIR=/opt/isaac-workspace/tmp

conda activate /opt/isaac-workspace/envs/isaaclab
cd /home/mi/graduation_design/RL/IsaacLab

不要直接改官方 `run_diff_ik.py`。官方仓库保持固定版本参考，自己的代码放在 `RL/demos/`。

```text
RL/
├── IsaacLab/                          # 固定 v2.3.0，尽量不改
├── demos/
│   ├── 01_franka_hold.py              # 保持指定末端位姿
│   ├── 02_franka_waypoints.py         # 多个末端航点间运动
│   └── 03_franka_preinsert.py         # 到达插孔上方的预插入位姿
├── configs/
├── outputs/
└── docs/
```

## 用键盘实时控制机械臂
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device keyboard


## 基线模型和柔顺控制
cd /home/mi/graduation_design/RL/IsaacLab
conda activate /opt/isaac-workspace/envs/isaaclab

# 1. 先复验纯位置基线（我改过策略：恒定目标而非增量下压）
./isaaclab.sh -p /home/mi/graduation_design/RL/demos/04_forge_baseline.py \
  --episodes 10 --headless

# 2. 再跑柔顺版，先观察，不急着调参
./isaaclab.sh -p /home/mi/graduation_design/RL/demos/05_forge_compliance.py \
  --episodes 10 --debug_every 30 --headless



实施顺序：

1. Franka 末端保持；
2. 空桌面上的多航点运动；
3. 到达插孔上方的预插入位姿；
4. FORGE 环境的状态、动作与接触信息检查；
5. 基础轴向推进控制；
6. 固定参数柔顺或阻抗控制；
7. 二维横向残差动作接口；
8. 官方训练闭环验证； 
9. RLinf 的低维观测与连续动作适配；
10. PPO 残差强化学习和对比实验。

> **阶段二、三的操作手册**：[阶段二_FORGE复现与cuRobo评估教程.md](阶段二_FORGE复现与cuRobo评估教程.md)、[阶段三_RLinf接入与残差RL教程.md](阶段三_RLinf接入与残差RL教程.md)。
> 方案与文献依据：[docs/调研_阶段二三技术选型.md](docs/调研_阶段二三技术选型.md)。

两个约束：

- **差分 IK 不等于柔顺控制。** 末端能到目标只说明运动通路成立，接触阶段的柔顺性需要另外实现。
- **残差动作的语义要先定死**：是"每步位移增量"、"目标位姿偏置"还是"速度修正"。这决定后续动作累积、限幅和控制频率怎么设计。

**cuRobo 暂不接入。** 后续可作为"从初始位姿规划到预插入位姿"的可选组件；它替代不了孔口接触阶段的柔顺控制、接触力约束与残差强化学习。

---

## 5. 执行清单

- [x] 空场景启动成功（`[INFO]: Setup complete...`，视口黑屏属正常）
- [x] 运行单环境 Franka `run_diff_ik.py`，确认机械臂追踪目标位姿
- [x] 任务列表中确认 `Isaac-Forge-PegInsert-Direct-v0`
- [x] 用 `zero_agent.py` 加载 FORGE 单环境
- [x] `demos/01_franka_hold.py` 固定位姿保持（末段误差 1e-6 m，曲线见 `outputs/franka_hold_*.csv`）
- [ ] 运行 `demos/02_franka_waypoints.py`，确认 5 个航点全部 CONVERGED
- [ ] 保存截图、终端输出、显存占用到 `docs/`
- [ ] 开始写 `demos/03_franka_preinsert.py`（预插入位姿）

---

## 6. 参考资料

- [Isaac Lab v2.3.0：Franka 差分 IK 示例源码](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/scripts/tutorials/05_controllers/run_diff_ik.py)
- [Isaac Lab v2.3.0：FORGE 任务注册源码](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge/__init__.py)
- [Isaac Lab v2.3.0：安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
