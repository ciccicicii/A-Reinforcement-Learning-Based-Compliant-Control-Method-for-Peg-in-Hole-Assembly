# cuRobo × Isaac Lab 技术调研报告

> 面向本科毕业设计《基于残差强化学习与柔顺控制的机器人精密插接方法研究》
> 调研时间：2026-11-22；本机事实取自 `/home/mi/graduation_design/RL/依赖安装与排错记录.md`（Isaac Sim 5.1.0 / Isaac Lab v2.3.0 / torch 2.7.0+cu128 / warp-lang 1.19.0 / RTX 4060 Laptop 8 GB）
> 所有结论附可核验链接；无法核实的写「未确认」。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 最新版本 | v0.9.0（2026-11-05，cuRoboV2-Fast）；v0.8.0（2026-04-18）是 v2 重写；**v1 最后 tag = v0.7.8** |
| 许可证 | Apache-2.0（代码）；自带机器人资产另有许可（franka Apache-2.0 / g1、ur BSD-3） |
| `pip install curobo` | **不行**。PyPI 上 `curobo` 0.2、`Nvidia-curobo` 0.1 都是第三方占位包（作者 fyl1991，935 B / 1003 B 的空包），不是 NVIDIA 的库 |
| 是否要编译 | v1（Isaac Lab 用的那版）**必须编译 CUDA 扩展**（nvcc，官方称约 20 分钟）；v2 仓库**已无 setup.py**，`.cu` 源码随包、运行时编译（推断，未确认） |
| Isaac Lab 官方集成 | 有，但**只有一条路径**：`isaaclab_mimic` 的 `motion_planners/curobo` + SkillGen。**没有** `scripts/tutorials/*/curobo`、**没有** `Isaac-Curobo-*` 任务、`controllers/` 里也**没有** cuRobo 封装 |
| 8 GB 可行性 | **SkillGen 全流程不可行**（官方实测 1 env 稳态 9.3–9.6 GB）；单独跑 cuRobo 规划可能可行（未确认，需实测） |
| 对「插接」的作用 | 只解决「初始位姿 → 预插入位姿」的无碰撞运动生成 + 批量 IK；**孔口接触后的力控/柔顺/微调完全不在能力范围** |
| 版本锁定建议 | 与 Isaac Lab 2.3.0 联用时锁 cuRobo commit `ebb7170`（v1 API），**不要用 v2** |

---

## 1. cuRobo 现状

### 1.1 版本谱系（GitHub Releases 实测）

| Tag | 日期 | 说明 |
|---|---|---|
| v0.6.0 | 2023-11 | 首个公开版 |
| v0.7.x | 2024 | 亚毫米 IK、ESDF voxel 碰撞、grasp API、re-timing |
| v0.7.7 | 2025-04-25 | Isaac Sim 4.5 支持（v1 线最后的「功能版」） |
| **v0.7.8** | — | **v1 最终 tag**，README 明确「若依赖 v1 API，请 pin 到 v0.7.8」 |
| **v0.8.0** | 2026-04-18 | **cuRoboV2 重写**（论文 arXiv:2603.05493），扁平化架构 |
| **v0.9.0** | 2026-11-05 | **cuRoboV2-Fast**（arXiv:2611.06913）：B 样条 TrajOpt、torque-limited TimeOpt、GPU 双向 RRT、Cartesian waypoint 规划、C++ runtime |

来源：<https://github.com/NVlabs/curobo/releases>、<https://github.com/NVlabs/curobo/blob/main/README.md>、<https://nvlabs.github.io/curobo/latest/index.html>

### 1.2 许可证

- 代码 Apache-2.0（`README.md` License 段、`pyproject.toml` `license = {text = "Apache-2.0"}`）。
- 资产单独许可：`franka_description` Apache-2.0；`g1`、`ur_description` BSD-3-Clause（`LICENSE_ASSETS`）。
- 对毕设：可直接改源码、可在论文中引用、可闭源分发自己的上层代码；但需保留版权头与 NOTICE，且**引用的机器人资产许可要单独核对**（Franka 资产是 Apache-2.0，商用前仍建议核一遍）。

### 1.3 依赖

| 项 | v1（v0.7.8，Isaac Lab 用） | v2（main / 0.9.0） |
|---|---|---|
| Python | 3.10+（Isaac Lab 用 3.11） | `requires-python >= 3.10`，> 3.13 未验证 |
| PyTorch | 由 Isaac Lab 提供（本机 2.7.0+cu128） | cu12 extra：`torch>=2.5`；cu13 extra：`torch>=2.9` |
| Warp | warp-lang（本机 1.19.0） | `warp-lang >= 0.10.0`（下限很松，1.19 满足） |
| CUDA 运行时 | 编译期需 nvcc（12.8，见 §2.4） | `cuda-core[cu12]>=0.7` + `nvidia-cuda-runtime-cu12>=12`（cu13 为 `cuda-bindings>=13.0.3`） |
| 驱动 | — | `>= 580.65.06`（本机 580.126.09 ✅） |
| GPU | Turing 以上 | Turing 以上，**≥ 4 GB VRAM**、CUDA 12+ |

来源：<https://github.com/NVlabs/curobo/blob/main/pyproject.toml>、<https://nvlabs.github.io/curobo/latest/getting-started/installation.html>

### 1.4 安装：`pip install curobo` 是错的

两个 PyPI 包都是垃圾包，**已验证**：

- <https://pypi.org/project/curobo/> → `curobo 0.2`，summary「A simple python package」，wheel 935 B，owner `fyl1991`
- <https://pypi.org/project/Nvidia-curobo/> → `Nvidia-curobo 0.1`，wheel 1003 B，同一 owner

官方分发名是 `nvidia-curobo`，但**只在源码树里**。正规路径两条：

```bash
# 路径 A：v2（当前主线，独立使用，不需要 Isaac Sim）
git clone https://github.com/NVlabs/curobo && cd curobo
uv venv --python 3.11 && source .venv/bin/activate
uv pip install ".[cu12]"          # 已装 torch 时
# 或 uv pip install ".[cu12-torch]"  # 连 torch 一起装

# 路径 B：v1（Isaac Lab 2.x 指定的 pinned commit，需 nvcc）
conda install -c nvidia cuda-toolkit=12.8 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.9+PTX"      # RTX 4060 = sm_89
pip install -e "git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8#egg=nvidia-curobo" --no-build-isolation
```

安装耗时：v1 官方写明「This will take 20 minutes to install」，其 `setup.py` 也会打印「20+ minutes」提示，并建议设 `TORCH_CUDA_ARCH_LIST` 加速（否则会为多架构编译）。v2 因为没有 `setup.py`、不需要 nvcc 构建，安装应显著更快，但**具体分钟数未确认**。

### 1.5 v2 的重大变化（对毕设的实际影响）

1. **API 全面破坏**：CHANGELOG 0.8.0 原文「Major refactor breaks most existing api」。
   - `curobo.wrap.reacher.motion_gen.MotionGen` → `curobo.motion_planner.MotionPlanner` + `MotionPlannerCfg`
   - IK：`curobo.wrap.reacher.ik_solver.IKSolver` → `curobo.inverse_kinematics.InverseKinematics` + `InverseKinematicsCfg`
   - 场景：`curobo.geom.types.WorldConfig` → `curobo.scene.Scene` / `Cuboid`
   - 目标位姿：`Pose` 仍可用，但多工具帧用 `GoalToolPose.from_poses({...}, num_goalset=...)`
2. **架构扁平化**：去继承，改组合；`use_cuda_graph` 从 mixin 变成构造参数；实现挪到 `curobo._src/`（`types/geom/motion/solver/rollout/optim/cost/perception/transition/graph_planner`）。
3. **`curobo.types` 还在，但内容变了**：`Pose / JointState / GoalToolPose / ContentPath / ToolPoseCriteria` 从 `curobo.types` 导出；v1 的 `curobo.types.base.TensorDeviceType`（Isaac Lab 里在用）在 v2 已不在同一位置。
4. **kernel 形态**：v2 根目录**没有 setup.py**，`curobo/_src/curobolib/` 下有 `kernels/`（`.cu/.cuh/.h`）且 `pyproject.toml` 把它们作为 package-data 打包，`cuda-core` 作为依赖 —— 结构上指向「运行时 JIT 编译 CUDA kernel」，替代 v1 的 `torch.utils.cpp_extension.CUDAExtension` 预编译 + Warp kernel 混合方案。**此点为仓库结构推断，官方文档未明说，标记未确认。**
5. **新增能力**（对插接任务有用）：B 样条轨迹优化、力矩受限 TimeOpt、GPU 双向 RRT、Cartesian 路径点规划 `plan_pose_waypoints()`、`plan_grasp()` 三相位（approach/grasp/lift）、C++ 部署 runtime。

v2 最小可用片段（来自官方 example，可直接抄）：

```python
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState

config = MotionPlannerCfg.create(
    robot="franka.yml",
    scene_model="collision_test.yml",
    graph_planner_algorithm="rrt",
)
planner = MotionPlanner(config)
planner.warmup(enable_graph=True, num_warmup_iterations=5)

q_start = JointState.from_position(
    planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
)
goal_pose = planner.compute_kinematics(q_start).tool_poses.as_goal()

result = planner.plan_pose(goal_pose, q_start, max_attempts=1, enable_graph_attempt=0)
print(result.success.any(), result.solve_time)
interp = result.get_interpolated_plan()      # [1, horizon, dof]，带 dt
```

来源：<https://github.com/NVlabs/curobo/blob/main/curobo/examples/getting_started/motion_planning.py>

---

## 2. Isaac Lab 官方集成

### 2.1 有哪些（本地 v2.3.0 实测存在）

| 路径 | 作用 |
|---|---|
| `source/isaaclab_mimic/isaaclab_mimic/motion_planners/motion_planner_base.py` | 规划器抽象基类（SkillGen 只认这个 API，可自己实现替换） |
| `source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner.py` | `CuroboPlanner`：世界同步、物体 attach/detach、`plan_motion()`（第 1053 行）、三相位接触规划 `_plan_to_contact_pose()`（1178 行）、线性重定时 `_linearly_retime_plan()`（1368 行） |
| `.../curobo/curobo_planner_cfg.py` | `CuroboPlannerCfg`：`num_trajopt_seeds=12`、`num_graph_seeds=12`、`trajopt_tsteps=32`、`collision_cache_size={"obb":150,"mesh":150}`、`interpolation_dt=0.05`、`approach_distance=0.05`、`retreat_distance=0.05`、`max_planning_attempts=15`、`from_task_name()` |
| `.../curobo/plan_visualizer.py` | Rerun 轨迹可视化（`visualize_plan=True`，仅 env 0） |
| `source/isaaclab_mimic/test/test_curobo_planner_franka.py`、`test_curobo_planner_cube_stack.py` | 官方用法参考（可当最小示例读） |
| `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py`（128–145 行） | `--use_skillgen` 入口，动态 import `CuroboPlanner` |
| `source/isaaclab_tasks/.../manipulation/stack/config/franka/__init__.py:112` | 任务 `Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0` |
| 文档 | <https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/skillgen.html> |

### 2.2 没有什么（必须澄清的四点）

1. **没有** `scripts/tutorials/.../curobo`：v2.3.0 的 `scripts/tutorials/` 只有 `00_sim / 01_assets / 02_scene / 03_envs / 04_sensors / 05_controllers`。
2. **没有** `Isaac-Curobo-*` 命名的任务；唯一的官方任务名是 SkillGen 那两个。
3. `isaaclab_tasks` 里除 SkillGen 的环境配置（`stack_ik_rel_env_cfg_skillgen.py`）外，**没有**任何 cuRobo 代码；该环境的 action 仍是 `DifferentialInverseKinematicsActionCfg(ik_method="dls")`。
4. `source/isaaclab/isaaclab/controllers/` 下**没有** cuRobo 封装，只有 `differential_ik.py`、`operational_space.py`、`joint_impedance.py`、`pink_ik/`、`rmp_flow.py`。

**因此：cuRobo 在 Isaac Lab 里不是「控制器后端」，不能替换 DifferentialIK 这个 action term。** 它是 Mimic 数据生成流水线里的「轨迹生成插件」：给它一个目标末端位姿，它返回一条带时间参数的关节轨迹，再由 SkillGen 逐 waypoint 回放。想让 RL 环境用 cuRobo 做 IK，只能自己在 action term 或 reset 事件里调 `CuroboPlanner.plan_motion()`，属于二次开发。

### 2.3 用法（v1，Isaac Lab 2.3.0）

```python
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

cfg = CuroboPlannerCfg.from_task_name("Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0")
cfg.num_trajopt_seeds = 4          # 省显存
cfg.collision_cache_size = {"obb": 30, "mesh": 20}
planner = CuroboPlanner(env, robot, cfg)   # 内部建 MotionGen（curobo_planner.py:242）

planner.update_world()             # 从 Isaac Lab 场景抽取障碍物到 cuRobo world
ok = planner.plan_motion(target_pose_4x4, step_size=0.01)   # 规划 + 线性重定时
if ok:
    while planner.has_next_waypoint():
        ee_pose = planner.get_next_waypoint_ee_pose()        # 逐点下发
```

SkillGen 数据生成（官方命令，一条都不改）：

```bash
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu --num_envs 1 --generation_num_trials 10 \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 --use_skillgen \
  --input_file ./datasets/annotated_dataset_skillgen.hdf5 \
  --output_file ./datasets/generated_small.hdf5
```

关键设计（对本毕设有借鉴价值）：`retreat → contact → approach` 三相位规划，**contact 相位可以关掉指定 link（手指）的碰撞**，相当于「允许贴上去但不允许撞穿」——这正是「接近到预插入位姿」需要的行为。

### 2.4 兼容矩阵（已核实的部分）

| Isaac Lab | Isaac Sim | cuRobo | 构建工具链 | 依据 |
|---|---|---|---|---|
| 2.0.x | 4.5 | v1（≤ v0.7.7 系列） | CUDA 12.x | release notes「Isaac Lab 2.0 … Isaac Sim 4.5」 |
| 2.2.x | 5.0（兼容 4.5） | v1 | CUDA 12.x | release notes 第 341 行 |
| **2.3.0（本机）** | **5.1.0** | **v1，pinned commit `ebb71702…`** | **conda cuda-toolkit 12.8** + `TORCH_CUDA_ARCH_LIST` | `docs/source/overview/imitation-learning/skillgen.rst` |
| main / 3.0.0 | 6.2.0（文档徽标） | 同上（**同一个 pinned commit**） | **CUDA 13.0** toolkit | 线上 SkillGen 文档 Step 2 |
| 任意 | 任意 | **v2（0.8/0.9）** | — | **不在官方矩阵内**：Isaac Lab 代码 import `curobo.wrap.reacher.motion_gen`、`curobo.geom.types`，v2 已删除/搬迁，必然 ImportError（高置信推断，未做实测） |

结论：**「Isaac Sim 5.1 + Isaac Lab 2.3 + cuRobo v1(ebb7170) + CUDA 12.8」是唯一被官方验证的组合**；想用 v2 的新算法（RRT / TimeOpt / B 样条），只能脱离 Isaac Lab 单独跑（v2 自带 Viser 可视化，够用）。

---

## 3. 显存与性能

### 3.1 官方性能数字（v0.9.0，均为官方 benchmark）

| 场景 | 硬件 | 数字 |
|---|---|---|
| Franka 运动规划（2590 题 MotionBenchMaker+MPINets，含失败） | RTX 5090 | **均值 3.797 ms**，成功率 99.73%，平均运动时长 1.189 s |
| 同上（`--disable-inverse-dynamics`） | RTX 5090 | 3.348 ms |
| 同上 | Jetson AGX Orin | 25.662 ms |
| Franka 位姿 IK（100 query/batch） | RTX 5090 | **0.556 ms/batch**，成功率 100% |
| Franka 无碰撞 IK | RTX 5090 | 0.563 ms/batch |
| cuRoboV2-Fast（3 kg 负载） | RTX 5090 | **4.4 ms** 均值，比 cuRoboV2 快 14.7× |

来源：<https://nvlabs.github.io/curobo/latest/reference/benchmarks.html>、<https://arxiv.org/abs/2611.06913>

注意：这些是 5090/Orin 的数据，**RTX 4060 Laptop（8 GB，算力 8.9，功耗受限）没有官方数字**。按算力与带宽粗估慢 3–6 倍，即单次规划 10–30 ms 量级 —— 这是**推算，未确认**，必须自己测（见 3.4）。

### 3.2 Isaac Sim + cuRobo 同卡峰值显存（官方实测，最硬的证据）

SkillGen 文档给了 RTX 6000 Ada 上的实测：

| 配置 | 稳态显存 |
|---|---|
| 1 env（立方体堆叠） | **9.3–9.6 GB**（初始化瞬间更高） |
| 5 envs | 21.8–22.2 GB |
| 1 env（窄箱自适应堆叠） | 9.3–9.6 GB |
| 官方建议 | `--num_envs` 1–2 需 **≥24 GB**；≤5 需 ≥48 GB |

同页还写：`--num_envs` 大于 1 时性能提升「not very significant」，**5 envs 是多数 GPU 的甜点**；生成 1000 条示范在 RTX 6000 Ada 上约 90–120 分钟（headless）。

### 3.3 8 GB 显存能不能跑？分三个层次回答

| 层次 | 可行性 | 依据 |
|---|---|---|
| A. SkillGen 端到端（Isaac Sim 渲染 + cuRobo + 数据生成） | **不可行**。1 env 就要 9.4 GB，8 GB 必 OOM | 官方 VRAM 表 |
| B. Isaac Lab 场景里跑 RL + 调 cuRobo 规划（headless，1 env） | **很可能不可行 / 勉强**。Isaac Sim 5.1 headless 本身约 2–3 GB（未确认），cuRobo `MotionGen` 的 world collision cache + CUDA graph 会再吃几 GB | 未确认，需实测 |
| C. 独立进程只跑 cuRobo（无渲染、无 Isaac Sim） | **可能可行**。官方最低要求仅 4 GB VRAM；Franka 单臂 + cuboid/mesh 场景，估计 1–3 GB（未确认） | 官方 install 页「at least 4 GB VRAM」 |

对毕设最实用的架构：**规划与仿真分进程**。cuRobo 独立进程（C 层）算轨迹 → 存 npz/共享内存 → Isaac Lab 进程回放。既避开显存争抢，也避开 §6 的 torch/warp 环境冲突。

### 3.4 降显存配置清单

v1（Isaac Lab 的 `CuroboPlannerCfg`，可直接改字段）：

| 参数 | 默认 | 降显存取值 | 代价 |
|---|---|---|---|
| `num_trajopt_seeds` | 12 | 4（甚至 2） | 成功率下降，复杂场景更明显 |
| `num_graph_seeds` | 12 | 4 | graph planner 更容易失败 |
| `trajopt_tsteps` | 32 | 24 / 16 | 轨迹分辨率与平滑度下降 |
| `collision_cache_size` | `{"obb":150,"mesh":150}` | `{"obb":30,"mesh":20}` | **cache 太小会丢障碍物→假阴性（误判无碰撞）**，障碍多时不要降 |
| `max_planning_attempts` | 15 | 3 | 失败率上升 |
| `enable_finetune_trajopt` | True | False | 精度略降 |
| `interpolation_dt` | 0.05 | 0.02 → 更密 → 更费；想省就调大 | 下发频率变粗 |
| `collision_checker_type` | MESH | PRIMITIVE（cuboid） | 精度降，速度/显存升 |
| `cuda_device` | 0 | 用 `CUDA_VISIBLE_DEVICES` 隔离 | — |

v2（`MotionPlannerCfg.create(...)`，字段已核对源码）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_batch_size` | 1 | 保持 1；>1 时 `num_ik_seeds` 从 32 降到 16、`num_trajopt_seeds` 从 4 降到 2（源码里的自适应） |
| `max_goalset` | 1 | 多候选抓取位姿才调大 |
| `interpolation_buffer_size` | None（按 `interpolation_dt` 自动） | 显式设小可省插值缓冲 |
| `collision_cache` | None | 显式给小字典 |
| `use_cuda_graph` | True | 省时间；显存是「一次性预留」，关掉不一定省，但能避开 §6.5 的崩溃 |
| `enable_inverse_dynamics` | True | 设 False 可省算力（不改显存），代价是失去力矩可行性保证 |

v1 还有一个「显存大杀器」：`torch.cuda.empty_cache()` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，能缓解碎片 OOM（治标）。

### 3.5 显存与耗时实测脚本（8 GB 上必做）

```python
import torch, time
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState

config = MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_test.yml")
planner = MotionPlanner(config)
planner.warmup(enable_graph=True, num_warmup_iterations=3)   # 必须 warmup，否则首次数百 ms

torch.cuda.reset_peak_memory_stats()
t0, n = time.perf_counter(), 50
for _ in range(n):
    planner.plan_pose(goal, q_start, max_attempts=1)
torch.cuda.synchronize()
print("mean ms:", (time.perf_counter() - t0) / n * 1e3)
print("peak MiB:", torch.cuda.max_memory_allocated() / 2**20)
```

把 `num_trajopt_seeds / trajopt_tsteps / collision_cache` 做 3×3 网格扫一遍，画「峰值显存–成功率–耗时」帕累托图，这就是毕设第 5 章一张很实在的表。

---

## 4. cuRobo 对「插接」到底有没有用

先把任务拆开，结论才站得住：

```
[初始位姿] --A:自由空间运动--> [预插入位姿] --B:接触逼近--> [孔口接触] --C:插入/柔顺--> [到位]
             ↑ cuRobo 的战场                              ↑ 力控/柔顺/RL 的战场
```

### 4.1 (a) 能贡献什么

1. **A 段的无碰撞运动生成**：这正是题目里「从初始位姿规划到预插入位姿」的标准解法，且带关节限位、速度/加速度/加加速度约束。
2. **批量 IK**：一次调用解 100 个位姿（Franka 0.556 ms/batch），可用于「预插入位姿可达性分析」「同一孔位多组接近姿态采样」「批量生成 RL 初始分布」。
3. **碰撞规避**：桌面、夹具、孔工件本体、夹爪都能进 world model；`_plan_to_contact_pose` 支持在接触相位禁用指定 link，等价于「贴上去但不穿模」。
4. **轨迹时间参数化**：输出带 `dt` 的插值轨迹，可直接下发到位置/速度控制器，省掉自己写梯形/SCurve 插值。
5. **可复现的强基线**：给定 seed 与同一初始位姿集，结果可复现，适合做论文对照组的「传统方法」。
6. **（v2）力矩受限 + payload**：插接工件有质量，torque-limited TimeOpt 能避免「规划出来但电机跟不上」。

### 4.2 (b) 不能贡献什么

1. **力控 / 柔顺 / 阻抗**：cuRobo 没有力反馈闭环，没有阻抗/导纳参数，没有接触力成本项。v2 的 torque-limited 是**动力学可行性约束**，不是力调节。
2. **亚毫米对位**：默认 `position_threshold=0.005 m`、`rotation_threshold=0.05 rad(≈2.9°)`；精密插接常要求 <0.5 mm / <0.5°，超出规划器语义。碰撞模型是**球体近似**（`surface_sphere_radius=0.005`），孔轴这种细长几何会被球化，精度天然受限。
3. **卡阻恢复**：spiral search、力螺旋搜索、微振动、退让重试——全不在范围内。
4. **不确定性下的适应**：孔位误差、摩擦变化、工件柔性，只能靠接触阶段的策略学习。
5. **接触力峰值控制**：规划器不建模接触刚度，无法保证「插入瞬间力不超 20 N」。

一句话：**cuRobo 管「最后一厘米之前」，接触之后必须换控制器。**

### 4.3 (c) 公平对照实验怎么设计

核心原则：**把「规划器」和「控制器」解耦，只在同一段比较同一类方法。**

**三段式流水线（统一骨架）**

| 段 | 内容 | 对照变量 |
|---|---|---|
| A | 初始位姿 → 预插入位姿（自由空间） | **规划器**：cuRobo / DifferentialIK / OSC / RMPflow / MoveIt+OMPL / Pinocchio |
| B | 预插入 → 接触（±2 mm 逼近） | **统一**用同一个阻抗控制器（固定刚度/阻尼），不让规划器差异泄漏 |
| C | 插入（接触 → 到位） | **控制器/策略**：纯位置 / 定阻抗 / 残差 RL（本课题主线） |

**必须锁死的控制变量**

- 同一 USD 场景、同一 Franka 模型、同一 TCP 偏移、同一夹爪开度；
- 同一初始位姿采样集（固定 `np.random.seed`，N ≥ 50 个位姿 × 3 档难度：无遮挡 / 单侧遮挡 / 窄通道）；
- 同一成功判据（孔轴偏差 < 0.5 mm 且 < 0.5°，插入深度 ≥ 目标深度的 95%，接触力峰值 < 阈值）；
- 同一控制频率（策略 60 Hz，力控内环 500 Hz~1 kHz）与同一物理步长（Isaac Lab `dt=1/120` 或 `1/200`，全文统一）；
- 每个格子 **≥30 次独立 trial**，成功率给 Wilson 95% 区间，耗时给 p50/p95 并用配对 Wilcoxon 检验（同一初始位姿集 → 配对样本）。

**实验矩阵建议（可直接当论文第 5 章表）**

| 编号 | 目的 | A 段规划器 | B 段 | C 段 | 关键指标 |
|---|---|---|---|---|---|
| E1 | 规划器基线对比 | cuRobo(v1, IK+TrajOpt) / DifferentialIK(DLS) / OSC / RMPflow / Pinocchio-IK | 统一阻抗 | 统一残差 RL | 规划成功率、p50/p95 耗时、末端位姿误差(mm/deg) |
| E2 | 碰撞能力消融 | cuRobo 开/关 world collision；有无 graph planner | 统一 | 统一 | 碰撞率、绕障成功率、耗时增量 |
| E3 | 控制器对比（A 段固定 cuRobo） | 固定最优 | 统一 | 纯位置 / 定阻抗 / 残差 RL / 直接 RL | 插接成功率、接触力峰值、样本效率 |
| E4 | 残差结构消融 | 固定 | 统一 | 残差输入 {仅位姿, 位姿+力, 位姿+力+历史} | 成功率、力峰值、收敛步数 |
| E5 | 泛化/鲁棒性 | 固定 | 统一 | 固定最优 | 孔位偏差 ±2/5/10 mm、间隙 0.1/0.2/0.5 mm、摩擦随机化下的成功率曲面 |
| E6 | 课程学习（可选，亮点） | cuRobo 批量 IK 生成初始分布 | 统一 | 残差 RL | 样本效率（达到 80% 成功率所需 episode 数） |

**怎么保证「能得出结论」**

- 指标要**正交**：规划类指标（成功率/耗时/位姿误差）只在 A 段测；接触类指标（力峰值/插接成功率/样本效率）只在 C 段测。混在一起就说不清是谁的贡献。
- 一定要报**失败模式分类**（规划无解 / 碰撞 / 超时 / 接触卡阻 / 侧向力过大），否则「成功率 62%」没有信息量。
- 耗时测量必须 `warmup + torch.cuda.synchronize()`，并区分「首帧 CUDA graph capture」与「稳态」；建议报 50 次均值 + p95。
- 若用 8 GB 卡，E1 的 cuRobo 与 E2/E3 的 RL **不要同时跑**（会 OOM 且互相拖慢），串行执行并在论文里说明。

### 4.4 给论文的一个「分工叙事」

> 用 cuRobo 做**批量 IK + 无碰撞接近轨迹**，生成「预插入位姿分布」与示范；用**残差 RL** 在接触阶段学习力/位姿修正；用**阻抗控制**保证接触刚度与力上限。cuRobo 是「先验/课程/示范来源」，不是竞争对象。

这样写，第 2 章相关工作、第 4 章方法、第 5 章实验能串起来，也不会被答辩问「你到底用 cuRobo 做了什么」。

---

## 5. 替代方案对比

| 方案 | 类型 | 速度 | 精度 | 碰撞 | 部署难度 | 8 GB 可行性 | 毕设适配 |
|---|---|---|---|---|---|---|---|
| **cuRobo v1**（Isaac Lab 集成） | GPU 批量 IK+TrajOpt+graph | 快（官方 0.5–4 ms 级，4060 上 10–30 ms 推算） | 位姿阈值 5 mm/2.9°（可调） | 强（mesh/obb/ESDF voxel） | **高**（pin commit + nvcc 编译 20 min + 环境冲突） | 独立进程可行；与 Isaac Sim 同进程 **不可行** | ★★★☆ 适合做「传统强基线」 |
| **cuRobo v2** | 同上 + RRT/TimeOpt/B 样条 | 最快（4.4 ms@5090） | 同上 | 强 | 中（无需编译，但 API 全新、无 Isaac Lab 集成） | 独立进程可行 | ★★★ 适合独立做规划基准 |
| **Isaac Lab DifferentialIK** | 雅可比迭代（DLS/伪逆/SVD） | 极快（亚毫秒，纯 GPU tensor） | 局部收敛，误差可 <1 mm（依赖增益） | **无** | **极低**（官方 action term，一行配置） | ✅ 与 Isaac Sim 共存无压力 | ★★★★★ 天然对照组，必选 |
| **OSC**（Operational Space Control） | 任务空间动力学控制 | 快 | 中（受动力学/增益影响） | 无 | 低（官方 `OperationalSpaceController`） | ✅ | ★★★★ 力/柔顺相关，适合做 B 段 |
| **RMPflow**（Isaac Sim LULA 封装） | 反应式运动策略 | 快 | 中 | 弱（依赖 LULA 场景描述） | 低（`RmpFlowControllerCfg`，需 enable lula/motion_generation 扩展） | ✅ | ★★★ 可作为「反应式」对照 |
| **MoveIt2 + OMPL** | CPU 采样规划 | 慢（10–1000 ms，CPU） | 高（IK 插件） | 强（FCL） | 中（ROS2 环境，与 Isaac Lab 异构） | ✅（CPU） | ★★ 跨栈集成成本高，本科不建议 |
| **Pinocchio IK** | CPU 解析雅可比 | 快（<1 ms/次，单点） | 高 | 弱（需自建碰撞） | 低（pip 可装） | ✅ | ★★★ 轻量基线，适合批量 IK 对照 |
| **直接 RL（端到端）** | 策略学习 | 推理快，训练慢 | 依赖奖励设计 | 隐式（学出来） | 中（Isaac Lab + skrl/rl_games） | ✅（小网络） | ★★★★ 主线对照组 |

补充：Isaac Lab 还有 **Pink IK**（`controllers/pink_ik`，基于 Pinocchio 的差分 IK，支持任务优先级），是 DifferentialIK 之外一个被低估的强基线，做「多约束（孔轴对齐 + 姿态约束）」时比 DLS 更稳。

---

## 6. 已知坑（按踩坑概率排序）

### 6.1 PyPI 假包（最容易中招）

`pip install curobo` 装到的是 935 字节的空包，`import curobo` 会报 `ModuleNotFoundError: No module named 'curobo.motion_planner'` 之类。**永远从 GitHub 源码装，包名 `nvidia-curobo`。** 证据：<https://pypi.org/project/curobo/>、<https://pypi.org/project/Nvidia-curobo/>

### 6.2 Isaac Sim 的 torch / warp / cudnn 抢 `sys.path`

Isaac Sim 的 `setup_python_env.sh` 会把 `pip_prebundle` 塞进 `PYTHONPATH`，里面有**旧版 torch、warp、nvidia-cudnn**，会 shadow 你 conda 环境里的新版本，表现为 CUDA runtime error 或 cuRobo kernel 编译/加载失败。Isaac Lab main 分支已在 `source/isaaclab/isaaclab/__init__.py` 里主动把这些路径 demote 到 `sys.path` 末尾（注释原文：「These contain older copies of packages like torch, warp, and nvidia-cudnn that shadow the versions installed by Isaac Lab, causing CUDA runtime errors」）。**v2.3.0 若无此逻辑，需自己保证 conda site-packages 优先。**

另一个官方警告：**source 过 Isaac Sim 脚本的 shell 里装 cuRobo 会崩**（`PYTHONHOME`/`PYTHONPATH` 让 conda 去 import Omniverse 的 `requests`/`urllib3`，报 `omni.kit.pip_archive` 相关 `TypeError`）。→ **另开一个干净 shell 装**。

证据：<https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/__init__.py>、<https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/skillgen.html>（warning 段）

### 6.3 warp 版本冲突

- Isaac Lab 2.3.0 用 **warp-lang 1.19.0**（本机 `import warp; warp.__version__` 实测 = 1.19.0），cuRobo v2 要求 `warp-lang>=0.10.0`，**宽松，不冲突**。
- 但 v1 的历史坑：`warp-lang < 1.2.1` 时 kernel module hash 不含 kernel 名，导致 Isaac Sim 里 kernel 串味/崩溃（cuRobo known issues）；`warp.Kernel.init()` 接口在旧 warp 上不兼容（issue #295）。
- 结论：**跟 Isaac Lab 的 warp 版本走，不要单独升级/降级 warp**。

证据：<https://nvlabs.github.io/curobo/v0.7.6/get_started/6_known_issues.html>、<https://github.com/NVlabs/curobo/issues/295>

### 6.4 编译失败（v1 特有，本机高危）

本机现状（已核实）：`which nvcc` 为空、`/usr/local/cuda` 不存在 → **没有 CUDA toolkit，直接 `pip install -e .` 必失败**。修法就是官方那条 `conda install -c nvidia cuda-toolkit=12.8`（只装 toolkit，不动驱动）。

高频失败点：
- `TORCH_CUDA_ARCH_LIST` 未设 → 为所有架构编译，40 分钟起步甚至爆磁盘；**4060 设 `8.9+PTX`**；
- 忘了 `--no-build-isolation` → 隔离环境里没有 torch，`from torch.utils.cpp_extension import ...` 直接 ImportError；
- gcc 版本过高（>11 在旧 CUDA 上）→ `-allow-unsupported-compiler` 或换 gcc；
- `setuptools` 升级过高（本机已 pin 80.9.0，**不要升**）；
- 磁盘：编译中间产物 + Isaac Sim 资产，`/` 与 `/home` 都要留 ≥20 GB。

### 6.5 CUDA graph 与 Isaac Sim 的冲突

- 已知崩溃：场景里**多个不同 DoF 的机器人 + `use_cuda_graph=True`** 会崩；关掉 `use_cuda_graph` 即正常（NVIDIA 论坛案例，cuRobo 定制 Kit 扩展）。
- v1 的 `MotionGenConfig.load_from_robot_config()` **默认 `use_cuda_graph=True`**（0.7.0 changelog），而 Isaac Lab 的 `CuroboPlannerCfg` **没有暴露这个开关**（grep 无 `use_cuda_graph`）→ 想关只能改 `curobo_planner.py:242` 那段。
- 换问题类型（如先 `plan_pose` 再 `plan_goalset`）需要重捕 graph，可试 `export CUROBO_TORCH_CUDA_GRAPH_RESET=1`（0.7.4 引入的实验特性）。

证据：<https://forums.developer.nvidia.com/t/cuda-crash-when-using-multiple-robots-with-different-dof-and-use-cuda-graph-true/334755>、<https://github.com/NVlabs/curobo/blob/main/CHANGELOG.md>

### 6.6 Docker / CI 也会踩

Isaac Lab 的 curobo dockerfile 曾因「与 Isaac Sim 自带 pytorch 冲突」而挂掉，官方用 hack 修（PR #4462）。说明这不是个人环境问题，是结构性问题。证据：<https://github.com/isaac-sim/IsaacLab/pull/4462>

### 6.7 版本漂移

Isaac Lab 锁的是 **commit `ebb71702f3f70e767f40fd8e050674af0288abe8`**（v1 API，含 quad face mesh 三角化，用于解析 USD 碰撞体）。**不要在 `requirements` 里写分支名**，写 commit；否则某天上游切 v2，你的代码全废。

### 6.8 概念误用

「cuRobo 能替换 DifferentialIK」是错的（§2.2）。它是**离线/半离线轨迹生成器**，规划一次几十毫秒，不适合 60 Hz 闭环里的逐帧 IK。要用在闭环，只能用它的 IK（`InverseKinematics.solve_pose`，批量）而不是 `plan_pose`。

---

## 7. 针对本机（8 GB / Sim 5.1 / Lab 2.3.0）的落地路线

```bash
# 0) 干净 shell（不要 source 任何 Isaac Sim 脚本）
conda activate isaaclab

# 1) 装 CUDA toolkit（只装 toolkit，不动驱动）
conda install -c nvidia cuda-toolkit=12.8 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.9+PTX"

# 2) 装 v1（Isaac Lab 认证 commit）
cd ~/graduation_design/RL
git clone https://github.com/NVlabs/curobo.git src/nvidia-curobo
git -C src/nvidia-curobo checkout ebb71702f3f70e767f40fd8e050674af0288abe8
pip install -e ./src/nvidia-curobo --no-build-isolation   # ~20 min

# 3) 验证（不启 Isaac Sim，最省显存）
python -c "import curobo, torch; print(curobo.__version__, torch.__version__, torch.cuda.is_available())"

# 4) 跑独立规划基准，测峰值显存（§3.5 脚本）

# 5) 再试 SkillGen 单 env（预期 OOM，作为负结果写进论文"平台限制"一节）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu --num_envs 1 --generation_num_trials 5 \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0 --use_skillgen \
  --input_file ./datasets/annotated_dataset_skillgen.hdf5 \
  --output_file ./datasets/try5.hdf5
```

如果第 5 步 OOM（大概率），论文里这样写是**加分的**：「受 8 GB 显存限制，SkillGen 全流程（官方实测 1 env 稳态 9.3–9.6 GB）无法在本机运行，故采用 cuRobo 独立进程规划 + Isaac Lab 回放的解耦方案」——有官方数字支撑，答辩很好讲。

---

## 8. 未确认清单（写论文时注意措辞）

1. cuRobo v2 是否真的用 `cuda.core` 在运行时 JIT 编译 kernel（依据：无 `setup.py`、`kernels/*.cu` 随包、依赖 `cuda-core`）。**官方文档未明说。**
2. v2 安装耗时（分钟）。仅知 v1 官方 ~20 min。
3. RTX 4060 Laptop 上 cuRobo 单次规划耗时与峰值显存 —— 无公开数据，必须自测。
4. Isaac Lab 2.1.x 对应的 Isaac Sim 版本（本文只核实了 2.0↔4.5、2.2↔5.0/4.5、2.3↔5.1、3.0↔6.2）。
5. 「cuRobo v2 与 Isaac Lab 必然 ImportError」是基于 import 路径的推断，未实测（无 v2 环境）。
6. Isaac Sim 5.1 headless 自身的显存占用（本文按 2–3 GB 估计）。
7. Isaac Lab 是否计划在 3.x 迁移到 cuRobo v2 —— 当前 main 仍锁 v1 commit，无公开 roadmap。

---

## 9. 参考链接

**cuRobo 本体**
- 仓库：<https://github.com/NVlabs/curobo>
- README（v1/v2 声明、许可）：<https://github.com/NVlabs/curobo/blob/main/README.md>
- Releases：<https://github.com/NVlabs/curobo/releases>（v0.9.0 / v0.8.0 / v0.7.7）
- CHANGELOG（v2 破坏性变更明细）：<https://github.com/NVlabs/curobo/blob/main/CHANGELOG.md>
- 依赖声明：<https://github.com/NVlabs/curobo/blob/main/pyproject.toml>
- v1 编译脚本：<https://github.com/NVlabs/curobo/blob/v0.7.8/setup.py>
- 安装文档：<https://nvlabs.github.io/curobo/latest/getting-started/installation.html>
- 性能基准：<https://nvlabs.github.io/curobo/latest/reference/benchmarks.html>
- 文档首页（版本更新日志）：<https://nvlabs.github.io/curobo/latest/index.html>
- 规划示例：<https://github.com/NVlabs/curobo/blob/main/curobo/examples/getting_started/motion_planning.py>
- IK 示例：<https://github.com/NVlabs/curobo/blob/main/curobo/examples/getting_started/inverse_kinematics.py>
- v1 安装说明（20 分钟）：<https://nvlabs.github.io/curobo/v0.7.6/get_started/1_install_instructions.html>
- v1 known issues（warp）：<https://nvlabs.github.io/curobo/v0.7.6/get_started/6_known_issues.html>
- warp.Kernel 兼容 issue：<https://github.com/NVlabs/curobo/issues/295>

**论文**
- cuRoboV2-Fast：<https://arxiv.org/abs/2611.06913>
- cuRoboV2：<https://arxiv.org/abs/2603.05493>
- Residual RL：<https://arxiv.org/abs/1812.03201>
- 高精度装配 RL（Inoue et al.）：<https://arxiv.org/abs/1708.04033>
- IndustReal（Isaac 装配 sim2real）：<https://arxiv.org/abs/2305.17110>

**Isaac Lab / Isaac Sim**
- SkillGen 文档（安装、VRAM 表、任务名）：<https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/skillgen.html>
- 2.3.0 对应文档：<https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/overview/imitation-learning/skillgen.html>
- Release notes（版本↔Sim 对应）：<https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html>
- pip 安装（Sim 5.1 + torch 2.7.0+cu128）：<https://github.com/isaac-sim/IsaacLab/blob/main/docs/source/setup/installation/pip_installation.rst>
- `__init__.py` 的 pip_prebundle 处理：<https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/__init__.py>
- curobo docker 冲突 PR：<https://github.com/isaac-sim/IsaacLab/pull/4462>
- 控制器 API：<https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.controllers.html>
- CUDA graph 多机器人崩溃案例：<https://forums.developer.nvidia.com/t/cuda-crash-when-using-multiple-robots-with-different-dof-and-use-cuda-graph-true/334755>

**其它**
- PyPI 假包：<https://pypi.org/project/curobo/>、<https://pypi.org/project/Nvidia-curobo/>
- MoveIt2：<https://moveit.picknik.ai/>
- Pinocchio：<https://github.com/stack-of-tasks/pinocchio>

**本仓库内证据文件**
- `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner.py`（1053 行 `plan_motion`、242 行 `MotionGenConfig.load_from_robot_config`）
- `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner_cfg.py`（86 行 seeds、95 行 cache）
- `IsaacLab/docs/source/overview/imitation-learning/skillgen.rst`（pinned commit、cuda-toolkit 12.8）
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/__init__.py:112`（SkillGen 任务注册）
