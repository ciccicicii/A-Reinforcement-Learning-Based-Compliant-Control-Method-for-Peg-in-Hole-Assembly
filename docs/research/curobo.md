# cuRobo 技术调研：版本谱系、8 GB 可行性与插接对照组实验设计

> 面向本科毕设《基于残差强化学习与柔顺控制的机器人精密插接方法研究》
> 调研日期 2026-11-22。本机事实：Isaac Sim 5.1.0 / Isaac Lab 2.3.0 / Python 3.11 / torch 2.7.0+cu128 / warp-lang 1.19.0 / RTX 4060 Laptop 8188 MB / 驱动 580.126.09。
> 每条结论附可核验链接；官方无公开数据的一律标「未确认/估算」，性能数字只抄官方原值，不做二次编造。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 最新 release | **v0.9.0（2026-11-05，cuRoboV2-Fast）**；v0.8.0（2026-04-18）为 v2 重写；**v1 末 tag = v0.7.8** |
| 许可证 | 代码 **Apache-2.0**（`pyproject.toml` 的 `license` 字段 + README 顶栏） |
| `pip install curobo` | **不能**：PyPI 上 `curobo`（935 B）/`Nvidia-curobo`（1003 B）是 2024-01-23 的占位包 |
| 锁的 commit | `ebb71702f3…`（**2025-08-04**，v1 线，落在 v0.7.7 与 v0.7.8 之间，落后 v0.7.8 四个提交） |
| 有无官方更新 commit | **没有**：2.3.0/2.3.1/2.3.2 与 main、3.0.0 全写同一 hash（后者只换 toolkit 为 CUDA 13.0） |
| 4060（sm_89）的 arch | `TORCH_CUDA_ARCH_LIST="8.9+PTX"`（官方命令里写的 `8.0+PTX` 是给 A100 的通用值） |
| 8 GB 共存 | **SkillGen 不行**（官方 1 env 稳态 9.3–9.6 GB）；**独立进程规划可行**（官方门槛 ≥4 GB）；同进程 RL+规划「未确认」 |
| 对插接的价值 | 只覆盖「初始位姿 → 预插入位姿」的 free-space 段（+ 批量 IK 采样）；**接触之后帮不上忙** |

---

## 1. cuRobo v1 与 v2 的差异、版本、许可、安装

### 1.1 版本谱系（GitHub Releases API 实测，非记忆）

| Tag | 发布日期 | 说明 |
|---|---|---|
| v0.6.0 | 2023-11-04 | 首个公开版 |
| v0.7.x | 2024 | ESDF voxel、grasp API、re-timing；v0.7.6 称位姿精度提升 10× |
| v0.7.7 | 2025-04-25 | Isaac Sim 4.5 支持，v1 线最后的「功能版」 |
| **v0.7.8** | tag 指向 `d64c4b05`（2026-03-02） | **v1 最终版**；README 明确「若依赖 v1 API，pin 到 v0.7.8」 |
| **v0.8.0** | 2026-04-18 | cuRoboV2 重写（论文 arXiv:2603.05493） |
| **v0.9.0** | 2026-11-05 | cuRoboV2-Fast（arXiv:2611.06913）：B 样条 TrajOpt、力矩受限 TimeOpt、GPU 双向 RRT、C++ runtime |

来源：<https://github.com/NVlabs/curobo/releases>、<https://github.com/NVlabs/curobo/blob/main/README.md>。v1 论文为 <https://arxiv.org/abs/2310.17274>「cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation」。

### 1.2 两代的关键差异

| 维度 | v1（≤ v0.7.8，Isaac Lab 用的那代） | v2（v0.8.0 / v0.9.0） |
|---|---|---|
| 架构 | 继承式（mixin） | **扁平无继承**，实现在 `curobo._src/` |
| 规划入口 | `curobo.wrap.reacher.motion_gen.MotionGen` | `curobo.motion_planner.MotionPlanner` |
| IK 入口 | `curobo.wrap.reacher.ik_solver.IKSolver` | `curobo.inverse_kinematics.InverseKinematics` |
| 场景 | `curobo.geom.types.WorldConfig` | `curobo.scene.Scene` |
| seed 参数 | `num_trajopt_seeds`/`num_graph_seeds` | `num_ik_seeds`/`num_trajopt_seeds`（None 时自适应：batch=1→32/4，>1→16/2） |
| cuda graph | mixin，默认 `True` | 构造参数，默认 `True` |
| kernel 编译 | `setup.py` 的 `CUDAExtension` **预编译**，需 nvcc | 无 `setup.py`，`.cu` 随包，**运行时 `cuda.core` 编译**（pyproject 注释原文如此） |
| 新算法 | — | B 样条 TrajOpt、`plan_pose_waypoints()`、GPU 双向 RRT、TimeOpt、C++ runtime、`servo`/MPC |
| 依赖 | torch + warp-lang | `warp-lang>=0.10.0`、`cuda-core[cu12]>=0.7`、`nvidia-cuda-runtime-cu12>=12` |
| Python | 官方写 3.8–3.10，3.11「might work but is not tested」 | `>=3.10`，>3.13 未验证 |

来源：`CHANGELOG.md` 0.8.0 段（「Major refactor to a flat, inheritance-free architecture」）<https://github.com/NVlabs/curobo/blob/main/CHANGELOG.md>；`pyproject.toml` <https://github.com/NVlabs/curobo/blob/main/pyproject.toml>；v2 配置默认值 <https://github.com/NVlabs/curobo/blob/main/curobo/_src/motion/motion_planner_cfg.py>。

### 1.3 许可证

代码 Apache-2.0（`pyproject.toml` 的 `license` 字段 + 文档站顶栏）。含义：可改源码、可引用、上层代码可闭源，保留版权头即可。内置资产另有 `LICENSE_ASSETS`，**条目本次未逐条核对（未确认）**，写论文前需自查。

### 1.4 安装方式与编译耗时

**PyPI 假包（已核验）**：`curobo` 0.2 与 `Nvidia-curobo` 0.1 均 2024-01-23 上传、wheel 仅 935 B / 1003 B，summary 都是「A simple python package」（<https://pypi.org/project/curobo/>、<https://pypi.org/project/Nvidia-curobo/>）。**只从 GitHub 装。**

```bash
# 路径 A：v1（Isaac Lab 2.3 指定的 pinned commit，需 nvcc）
conda install -c nvidia cuda-toolkit=12.8 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.9+PTX"        # 4060 Laptop = sm_89
# 关键：conda 的 cuda-toolkit 会带进 conda GCC 14，而 CUDA 12.8 的 nvcc 要求 GCC < 14
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
pip install -e "git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8#egg=nvidia-curobo" --no-build-isolation

# 路径 B：v2（独立使用，不需要 nvcc、不需要 Isaac Sim）
git clone https://github.com/NVlabs/curobo && cd curobo
uv venv --python 3.11 && source .venv/bin/activate
uv pip install ".[cu12]"                     # 已装 torch 时；连 torch 一起装用 ".[cu12-torch]"
```

编译耗时：**v1 官方原文「This will take 20 minutes to install」**（<https://nvlabs.github.io/curobo/v0.7.6/get_started/1_install_instructions.html>），且随 `TORCH_CUDA_ARCH_LIST` 里的架构数增加而线性增长；只编一个架构到底几分钟**未确认**。v2 因为没有 nvcc 预编译，安装应接近即时，但**首次运行会有 JIT 编译开销，具体秒数未确认**。

### 1.5 4060 该写什么 arch

4060 Laptop 是 Ada，算力 **8.9** → `TORCH_CUDA_ARCH_LIST="8.9+PTX"`。官方脚本写 `8.0+PTX`，同页 note 却说「`8.9` for RTX 4090」，**文档自相矛盾**：写 8.0 能跑，但这张卡只会加载 PTX 并在首次调用时 JIT，首跑明显更慢。建议按 8.9 写，并在论文「实验设置」里记录（它影响首跑耗时）。

---

## 2. 固定 commit 归属与「有没有更新 commit」

### 2.1 pinned commit 到底是哪个版本

`ebb71702f3…` 的提交日期是 **2025-08-04**，信息「Convert all LFS files to normal Git files」。用 compare API 做两次比较（可复现）：

- `v0.7.7...ebb7170` → **ahead 5**，其中包含 `add quad mesh support to usd parser`——正是 SkillGen 文档所说的「quad face mesh triangulation」；
- `ebb7170...v0.7.8` → **ahead 4**，内容是：URDF 导入模块适配新版 Isaac Sim（`49b5a50`）、`multi_arm_reacher` 迭代器 bug 修复（`0db44e5`）、以及两个 PR 合并（#591、#587）。

**结论：它属 v1 线、落在 v0.7.7 与 v0.7.8 之间，等价于「v0.7.8 往前 4 个提交」的快照**；版本号语义上是 0.7.x（setuptools_scm 派生，无独立 `version.py`）。

### 2.2 是否有与 Isaac Lab 2.3 兼容的更新 commit

我逐个 tag 抓取 `docs/source/overview/imitation-learning/skillgen.rst` 对比安装命令：

| Isaac Lab 版本 | pinned cuRobo commit | toolkit |
|---|---|---|
| v2.3.0（本机） | `ebb7170…` | cuda-toolkit **12.8** |
| v2.3.1（2025-12-04） | `ebb7170…`（**同一 hash**） | 12.8 |
| v2.3.2（2026-02-02） | `ebb7170…`（**同一 hash**） | 12.8 |
| v3.0.0（2026-10-20）/ main | `ebb7170…`（**同一 hash**） | **13.0** |

所以**官方从未把这个 commit 往前推过**：官方文档里不存在「更新的、与 2.3 兼容的 commit」。唯一候选是同线 `v0.7.8`（`d64c4b05`，只多 4 个上游提交、API 未变，理论上可直接替换），但 **NVIDIA 未声明测过 → 记「未确认」**。实操：**锁 commit 不锁分支名**（分支某天切 v2 会立刻 ImportError）。

### 2.3 一个必须知道的上游 bug（对插接场景有实际影响）

Isaac Lab PR **#4212**（2026-09-26 合入 `develop`，<https://github.com/isaac-sim/IsaacLab/pull/4212>）修的是：**机器人 base 不在世界原点时，动态障碍物位姿没换算到机器人坐标系**，碰撞世界里的障碍物会整体偏移。它基于 3.0 的四元数约定与 `ProxyArray`，**2.3.x 不含此修复**。

本仓库 2.3.0 的 `curobo_planner.py` 里两套基准并存：`_sync_object_poses_with_isaaclab()` 用 `root_pos_w - env_origin`（`:626`、`:1801`），另一处用世界系 `root_pos_w`（`:1692`）。**只要 `env_origin` 非零（多环境 grid 布局就是非零），孔件在 cuRobo 世界里就可能整体偏掉**，表现为「没碰却报碰撞」或「撞了不报」。规避：① 单环境 + `env_origin=(0,0,0)`；② 上 v3.0.0（需 Isaac Sim 6.2）。写作上按「平台限制」交代。

---

## 3. 显存、配置与耗时

### 3.1 官方性能数字（只有 v2 有公开 benchmark）

| 场景 | 硬件 | 数字 |
|---|---|---|
| Franka 运动规划（2590 题 MotionBenchMaker+MPINets，含失败） | RTX 5090 | 均值 **3.797 ms**，成功 2583/2590（99.73%），运动时长均值 1.189 s |
| 同上 `--disable-inverse-dynamics` | RTX 5090 | 3.348 ms |
| 同上 | Jetson AGX Orin | 25.662 ms |
| Franka 位姿 IK（100 query/batch） | RTX 5090 | **0.556 ms/batch**，成功率 100% |
| Franka 无碰撞 IK | RTX 5090 | 0.563 ms/batch，100% |

来源：<https://nvlabs.github.io/curobo/latest/reference/benchmarks.html>（原文注明 warmed C++ Release build + CUDA graphs，计时为同步后的 host wall time）。**没有任何 4060 的官方数字，也没有 v1 的同口径数字。**

### 3.2 8 GB 的显存账

| 项 | 数值 | 性质 |
|---|---|---|
| cuRobo 官方最低显存 | ≥4 GB VRAM（v1、v2 安装页均如此写） | 官方 |
| SkillGen 1 env 稳态（堆叠任务） | 9.3–9.6 GB；官方建议 ≥24 GB 才跑 1–2 env | 官方实测，见本仓库 `IsaacLab/docs/source/overview/imitation-learning/skillgen.rst:369-375` |
| 本机空场景 GUI | ≈2.2 GB | 本机实测（`依赖安装与排错记录.md`） |
| 场景 + 资产（Franka + 孔件 + 桌面） | 0.5–1.5 GB | 估算 |
| torch/CUDA 上下文 | 0.8–1.2 GB | 估算 |
| cuRobo（Franka，warm cache + cuda graph 预留） | 1–3 GB | **估算，未实测** |

三条结论：① **SkillGen 端到端在 8 GB 上不可行**（1 env 就要 9.4 GB），与代码写得好坏无关；② **独立进程跑规划大概率可行**（门槛 4 GB，Franka 单臂估 1–3 GB），必须自测；③ **同进程「RL + 调规划」最危险**，不建议作主路线。

正确架构是**规划与仿真分进程**：cuRobo 进程算轨迹 → 落 `npz` → Isaac Lab 回放。既避开显存争抢，也避开 torch/warp 的 `sys.path` 冲突（§6）。

### 3.3 最省显存的配置清单

v1（改 `CuroboPlannerCfg` 字段即可，`curobo_planner_cfg.py`）：

| 参数 | 默认 | 省显存取值 | 代价 |
|---|---|---|---|
| `num_trajopt_seeds` | 12 | 4（极限 2） | 成功率下降 |
| `num_graph_seeds` | 12 | 4；或 `enable_graph=False` 直接关图搜索 | 窄通道场景失败率上升 |
| `trajopt_tsteps` | 32 | 24 | 轨迹分辨率下降 |
| `collision_cache_size` | `{"obb":150,"mesh":150}` | `{"obb":30,"mesh":20}` | **cache 溢出会静默丢障碍物→假阴性**，障碍多时不能降 |
| `collision_checker_type` | `MESH` | `PRIMITIVE`（BLOX/cuboid） | 几何精度降、速度升；枚举值为 `PRIMITIVE/BLOX/MESH/VOXEL` |
| `max_planning_attempts` | 15 | 3 | 失败率上升 |
| `enable_finetune_trajopt` | True | False | 到位精度降 |
| `extra_collision_spheres["attached_object"]` | 100 | 50 | 附着物碰撞检测变粗 |

两个 v1 的硬限制要特别注意：
- **`use_cuda_graph` 在 `CuroboPlannerCfg` 里没有暴露**，`curobo_planner.py:242` 调 `MotionGenConfig.load_from_robot_config(...)` 时未传该参数 → 实际用默认 `True`。想关只能改这一行，或自己建 `MotionGen`。
- `self.tensor_args` 被强制钉在 `cuda:{self.config.cuda_device}`（默认 0），多卡/显存隔离要用 `CUDA_VISIBLE_DEVICES`。

v2（`MotionPlannerCfg.create`）：`max_batch_size=1`（保持默认，>1 会把 `num_ik_seeds` 降到 16、`num_trajopt_seeds` 降到 2）；`use_cuda_graph=True` 是默认值（省时间，显存是一次性预留，关掉未必省）；`interpolation_dt` 默认 0.025。

### 3.4 单次 IK / 规划耗时量级

- 官方口径（5090）：规划 ~3.8 ms、IK 0.556 ms/100 位姿批。
- 4060 与 5090 的 SM/带宽差一个数量级，粗推 **单次规划 15–40 ms、100 位姿批量 IK 3–8 ms**——**纯估算、未确认、必须自测**。
- v1 与 v2 是不同算法栈，**不能假设 v1 同量级**（v1 无公开同口径数字）。
- 计时必须 `warmup()` → `synchronize()` → ≥50 次 → 报 p50/p95，并**剔除首次 CUDA graph capture**。

```python
import time, torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

planner = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_test.yml"))
planner.warmup(enable_graph=True, num_warmup_iterations=5)   # 不做这一步首次数百 ms
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50):
    planner.plan_pose(goal, q_start, max_attempts=1)
torch.cuda.synchronize()
print("p50-ish ms:", (time.perf_counter() - t0) / 50 * 1e3)
print("peak MiB:", torch.cuda.max_memory_allocated() / 2**20)
```

### 3.5 扫参前必读：CUDA graph 把张量尺寸锁死了

v1 known issues 原文：**「Once a cuda graph is generated, we cannot change the dimensions of any of the tensors (e.g., dof, timesteps, number of seeds)」**，且 `CUDAGraph.reset()` 在 PyTorch 下会崩（<https://nvlabs.github.io/curobo/v0.7.6/get_started/6_known_issues.html>）。工程后果：**每换一组 seeds/tsteps/batch 必须重启进程**，不能同进程扫参。缓解是实验特性 `export CUROBO_TORCH_CUDA_GRAPH_RESET=1`（v0.7.8 CHANGELOG 原文「allow for changing the problem type in `motion_gen` and `ik_solver` without requiring recreation of the class」）——实验性，别拿它撑主结果。

---

## 4. 实验设计：论证「cuRobo 对插接有没有价值」

### 4.1 分段模型与三个被测对象

```
[初始位姿] --A: free-space--> [预插入位姿 handoff，孔口上方 3 mm]--B--> [接触] --C: 插入--> [到位]
             ↑ 规划类指标                                ↑ 交接点           ↑ 插接类指标
```

本仓库已有可复用的分段定义（`forge_ext/base_policy.py`）：`z_start=0.047 m`、`z_press_start=0.003 m`（自由空间终点 = handoff）、`z_min=-0.027`、成功判据 z < −0.024；控制 15 Hz（120 Hz 物理 / decimation 8）。**handoff 取「z 指令到 3 mm 的那一刻」**，A/C 段由此有可测分界。

| # | 被测对象 | A 段实现 | 输入 | 输出 | 代码 |
|---|---|---|---|---|---|
| O1 | **差分 IK** | `DifferentialIKController(ik_method="dls")` 逐帧跟踪笛卡尔直线 | 目标末端位姿序列（4×4）+ 当前关节角 | 关节目标（每帧） | `isaaclab/controllers/differential_ik.py`；`demos/01_franka_hold.py` 已有误差统计框架 |
| O2 | **cuRobo 规划** | `CuroboPlanner.plan_motion(target_pose_4x4)`（内部 TrajOpt + graph，带 world collision） | 单个目标位姿 4×4 + 当前 `JointState` + 世界模型 | 带 dt 的关节轨迹 `[1,H,7]` + 成功标志 + `solve_time` | `isaaclab_mimic/.../curobo_planner.py:1053`；建议独立进程跑，落 `npz` |
| O3 | **解析柔顺 + 残差 RL** | 不含 free-space 规划，直接由脚本基线下降段 + 2D 残差策略 | 19 维观测（含 `ft_force`）+ 2 维残差动作 | 7 维任务空间动作 | `forge_ext/residual_env.py`、`demos/03/04/05` |

**关键立场（必写进论文）**：O2 只在 **A 段**与 O1/O3 可比；**B、C 段 cuRobo 完全不参与**——规划器语义里没有接触力、没有阻抗、没有力反馈闭环。「cuRobo 有没有价值」只能拆成两个可证伪命题：
- **H1（free-space 价值）**：在带障碍/窄通道的位姿分布上，O2 的成功率与 handoff 误差优于 O1（后者无碰撞规避），耗时仍可接受；
- **H2（下游增益）**：把 O2 的 handoff 分布喂给同一接触段控制器，插接成功率**不低于**「直接以目标位姿下令」的基线，且减少「撞孔壁/工件侧面」类失败。

H2 最容易被答辩追问：若无遮挡场景下 A 段换谁都一样，cuRobo 的价值就只剩「批量 IK 采样」与「无碰撞」，必须诚实画出适用边界。

### 4.2 控制变量（不控住这些，任何对比都无效）

1. 同一 USD 场景、同一 Franka 资产、同一 TCP 偏移、同一夹爪开度；
2. **同一批初始位姿**：N=50，固定 seed，落盘 `poses_seed{0..4}.npz`，三个对象读同一文件；
3. **同一随机种子**：`--seed` 同时控物理随机化（摩擦/质量/孔位抖动）与采样；每格 ≥20 trials，报 Wilson 95% 区间；
4. **同一评估脚本**：接触段一律 `demos/05_eval_success.py`（同判据、同力峰值口径），free-space 段一律新建的 `demos/06_plan_free_space.py`（同计时口径）；
5. 同一控制频率与物理步长（15 Hz 策略 / 120 Hz 物理），全文统一；
6. **handoff 判据一致**：位置误差 < 1 mm 且姿态误差 < 1° 记为「到达」，超时 5 s 记失败；
7. **不让规划器差异泄漏进接触段**：B 段固定同一组阻抗/导纳参数（`InsertBaseCfg.hold_force=30 N`）。

### 4.3 指标定义（口径写清，别只报「成功率 62%」）

| 指标 | 定义 | 归属段 |
|---|---|---|
| 规划成功率 | 成功返回轨迹的请求数 / 总请求数；超时（5 s）与无解都记失败 | A |
| 单次耗时 | p50 / p95（ms），warmup 后、`synchronize` 后测，剔除首次 capture | A |
| 末端到位误差 | handoff 时刻 `e_pos = ‖p_ee − p_target‖`（mm）、`e_rot = 2·arccos(|⟨q_ee,q_target⟩|)`（°） | A（交接点） |
| 碰撞事件率 | 规划轨迹回放期间 Isaac Lab 接触对里出现「机器人-环境」非预期接触的比例 | A |
| 插接成功率 | FORGE 判据（z < −0.024 且保持若干步） | C |
| 接触力峰值 | `max_t ‖f_t‖`（N），并单独记 z 向峰值 | C |
| 失败模式 | 无解 / 超时 / 碰撞 / 卡阻 / 侧向力过大 / 过冲 | A+C |

### 4.4 与「孔位估计误差」解耦（本题最容易写糊的地方）

核心手法：把成功率按**全概率**拆开，别把两个误差源混进一个数字：

```
P(success) = ∫ P(success | e) · p_planner(e) de
                    ↑ 控制器能力            ↑ 规划器引入的 handoff 误差分布
```

三步：① **注入**——在「估计孔位」与真实孔位间加已知偏置 δ ∈ {0,2,5} mm（模拟视觉/标定误差），δ 只作用在**下发的目标位姿**上、不改仿真里的真实孔位；② **测量**——每个 handoff 记 `e_pos/e_rot`（含 δ），得各对象的 `p_planner(e)`；③ **分层报告**——固定 O3 的接触段控制器，在 e 网格（0.2/1/3 mm）各跑 20 trials 得 `P(success|e)`，再把各 `p_planner(e)` 代进去，把「规划器差异」归因到「下游成功率差异」，而不是一句「cuRobo 更好」。副产品是这张曲线能定量回答「handoff 误差多大时残差 RL 就救不回来」。

### 4.5 实验矩阵（第 5 章表格骨架）

| 编号 | 目的 | A 段 | B 段 | C 段 | 主要指标 | 规模 |
|---|---|---|---|---|---|---|
| E1 | free-space 规划器对比（H1） | O1 / O2 / 纯笛卡尔直线 | — | — | 规划成功率、p50/p95、e_pos/e_rot、碰撞率 | 50 位姿 × 3 难度 × 3 对象 |
| E2 | cuRobo 碰撞能力消融 | O2 开/关 world collision、开/关 graph | — | — | 碰撞率、成功率、耗时增量 | 50 × 4 配置 |
| E3 | 参数扫描（显存-耗时-成功率帕累托） | O2，`num_trajopt_seeds ∈ {2,4,12}` × `trajopt_tsteps ∈ {16,24,32}` | — | — | 峰值显存、p95、成功率 | 每配置独立进程 |
| E4 | 下游增益（H2） | O1 / O2 的 handoff | 统一阻抗 | O3 | 插接成功率、力峰值 | 50 × 2，每格 20 trials |
| E5 | 孔位误差解耦 | 固定 O2 | 统一 | O3 | `P(success\|e)` 曲线、δ 分层表 | 3×3 网格 × 20 |
| E6 | 残差结构消融（对照系，非 cuRobo 主线） | 固定 O2 | 统一 | O3 观测 {位姿} / {位姿+力} / {位姿+力+历史} | 成功率、力峰值、收敛步数 | 3 × 20 |
| E7 | 批量 IK 用途（加分项） | O2 的 IK，批量 100 位姿 | — | — | 可达率、单批耗时、采样覆盖度 | 5 批 × 3 场景 |

E1/E2/E3/E7 **不需要 Isaac Sim 渲染**（甚至不需要 Sim），可在独立进程跑；E4/E5/E6 必须进物理仿真。**8 GB 上必须串行**：先跑完 E1–E3 把 `npz` 落盘，再进 E4–E6。

### 4.6 建议的表图数量

**7 表**：T1 平台与版本核对（Sim/Lab/torch/warp/commit/arch，必写）；T2 三对象 I/O 与超参；T3 指标口径；T4 E1 主结果；T5 E3 参数扫描；T6 E5 孔位误差分层；T7 失败模式计数。

**6 图**：F1 三段时间轴 + 系统框图；F2 handoff 误差分布（三对象叠一张）；F3 耗时 p50/p95；F4 `P(success|e)` 曲线（核心图）；F5 接触力时域曲线（成功 vs 卡阻）；F6 显存-耗时-成功率帕累托（E3）。

### 4.7 8 GB 上的执行顺序

1. 独立 venv 装 v1（`8.9+PTX`）→ `pytest --pyargs curobo.tests` 或最小 `plan_pose` 冒烟；
2. 跑 §3.4 脚本拿 4060 真实数字（产物直接进 T1/T5）；
3. 新建 `demos/06_plan_free_space.py`：读 `poses_seed*.npz`，用 O1（Isaac Lab 内）与 O2（独立进程）跑 A 段，输出 `handoff.npz` + `plan_metrics.csv`；
4. 给 `demos/05_eval_success.py` 加 `--handoff_set`，读 `handoff.npz` 从该位姿起步跑 C 段；
5. 试 SkillGen 预期 OOM，**作为「平台限制」负结果写进论文**（有官方 9.3–9.6 GB 支撑）。

---

## 5. 替代方案对比

| 方案 | 类型 | 速度 | 精度 | 碰撞规避 | 部署复杂度 | 8 GB 可行性 | 离线/在线 |
|---|---|---|---|---|---|---|---|
| **cuRobo v1**（`ebb7170`，Isaac Lab 集成） | GPU 批量 IK + TrajOpt + graph | 官方无 4060 数据；按 5090 数据粗推 15–40 ms/次（估算） | 阈值 5 mm / 0.05 rad 可调；碰撞体为球体近似（`surface_sphere_radius=0.005`），亚毫米对位超纲 | **强**（MESH/OBB/VOXEL 世界） | **高**（pin commit + nvcc 编译 + 环境冲突 + 2.3.x 有 base 偏移 bug） | 独立进程可行；同进程未确认 | 离线/半离线（毫秒级，不宜进 15 Hz 闭环） |
| **cuRobo v2**（0.9.0） | 同上 + RRT/TimeOpt/B 样条/C++ runtime | 3.797 ms（含逆动力学）/ 3.348 ms（关闭）@5090，Orin 25.7 ms（官方） | 同上 | 强 | 中（`uv pip install ".[cu12]"`，无需 nvcc；但 API 全新、无 Isaac Lab 集成） | 独立进程可行（官方门槛 4 GB） | 离线/半离线 |
| **Isaac Lab `DifferentialIKController`** | 雅可比迭代（`pinv/svd/trans/dls`） | 极快（纯张量，亚毫秒级） | 局部收敛，取决于增益；稳态可到亚毫米（需实测） | **无** | **最低**（一个 action term 配置） | ✅ 与 Sim 共存无压力 | **在线**（15–60 Hz 逐帧） |
| **`OperationalSpaceController`** | 任务空间动力学 + 阻抗（`impedance_mode ∈ {fixed, variable, variable_kp}`） | 快 | 中（受增益与动力学影响） | 无 | 低（官方 action term） | ✅ | **在线**，且是接触段柔顺的天然载体 |
| **RMPflow**（Isaac Sim LULA 封装） | 反应式运动策略 | 快 | 中 | 弱（依赖 LULA 的 `collision_file` 场景描述） | 低但需 `enable_extension("isaacsim.robot_motion.lula"/"…motion_generation")` + URDF + 碰撞描述 | ✅（CPU 侧计算） | 在线 |
| **MoveIt2 + OMPL** | CPU 采样规划 | 慢（10–1000 ms 量级，CPU） | 高（IK 插件） | 强（FCL） | 中高（ROS2 栈，与 Isaac Lab 异构） | ✅（不占显存） | 离线 |
| **Pinocchio / Pink IK** | CPU 解析雅可比 / 差分 IK（支持任务优先级） | 快 | 高 | 需自建（Pinocchio 无碰撞） | 低（pip 可装）；Isaac Lab 已有 `controllers/pink_ik/` | ✅ | 在线 |

两条与课题强相关的判断：① **RMPflow** 在 Isaac Lab 里是「反应式避障」而非全局规划，且必须提供 LULA 的 `collision_file`，几何准备成本不低；② **Pink IK 是被低估的强基线**——`controllers/pink_ik/` 支持 null-space posture 与 local-frame task，做「孔轴对齐 + 姿态约束」比 DLS 稳且不占显存。

---

## 6. 安装与运行踩坑清单

1. **PyPI 假包**：`pip install curobo` 装到 935 B 空包，`import curobo.motion_planner` 直接 `ModuleNotFoundError`。从 GitHub 装，包名 `nvidia-curobo`。
2. **torch 冲突**：v1 的 `setup.py` 在 build 阶段 `from torch.utils.cpp_extension import CUDAExtension`，**必须 `--no-build-isolation`**；隔离环境里没 torch 立刻 ImportError。而且 build backend 把 `torch` 列为 build requirement，pip 可能另装一个 torch——这就是要关隔离的原因。
3. **`CUDA_HOME`/nvcc**：本机原先 `which nvcc` 为空、`/usr/local/cuda` 不存在（见 `依赖安装与排错记录.md`），必须先装 `cuda-toolkit=12.8` 并导出 `CUDA_HOME=$CONDA_PREFIX`，否则 `CUDAExtension` 直接失败。3.0.0/main 文档用 13.0 toolkit，**别和 torch 2.7.0+cu128 混用**。
4. **`PYTHONHOME`/`PYTHONPATH` 污染**：source 过 Isaac Sim 的 `setup_conda_env.sh` 后，Kit 的 `pip_prebundle` 会把旧 torch/warp/nvidia-cudnn 注入 `PYTHONPATH`，conda 装包时 import 到 Omniverse 的 `requests`/`urllib3` 而崩（报错常见 `omni.kit.pip_archive`）。**装 cuRobo 用干净 shell**，或临时 `unset PYTHONPATH PYTHONHOME`；运行时让 conda site-packages 优先。
5. **warp 版本**：跟 Isaac Lab 走（本机 1.19.0），别单独升降；v2 只要 `warp-lang>=0.10.0`。v1 旧坑：warp 太老时 kernel module hash 不含 kernel 名，Sim 里会 kernel 串味。
6. **编译失败的其他高频点**：
   - `TORCH_CUDA_ARCH_LIST` 未设 → 为全部架构编译，40 分钟起步甚至爆盘（4060 写 `8.9+PTX`）；
   - **GCC 版本**（官方已确认，本机 2.3.0 文档尚无此修正）：`conda install -c nvidia cuda-toolkit=12.8` 会把 conda 的 **GCC 14** 带进环境（defaults 频道从 GCC 11 跳到了 14），而 **CUDA 12.8 的 nvcc 要求 GCC < 14**。修法就是 Isaac Lab PR #7649 加的那行：`export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++`，强制用系统编译器（Ubuntu 22.04 是 GCC 11、24.04 是 GCC 13，均兼容）。见 <https://github.com/isaac-sim/IsaacLab/pull/7649>；
   - `setuptools` 被升到很新版本常常连带失败；磁盘需留 ≥20 GB。
7. **CUDA graph 与 Kit 冲突**：① graph 一旦 capture，`num_seeds`/`tsteps`/dof 全锁死，`CUDAGraph.reset()` 在 PyTorch 下会崩；② `CuroboPlannerCfg` 未暴露 `use_cuda_graph`（默认 `True`），要关得改 `curobo_planner.py:242`；③ 换问题类型可能需重捕 graph，可试 `CUROBO_TORCH_CUDA_GRAPH_RESET=1`。
8. **2.3.x 的碰撞世界同步 bug**（§2.3）：`env_origin` 非零时障碍物位姿偏移。单环境 + 原点布局可规避。
9. **显存碎片**：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，配合每次实验前重启进程（也顺带解决 graph 尺寸锁死）。
10. **别把 cuRobo 当控制器**：`plan_motion` 是一次算一条带 dt 的轨迹（十几到几十毫秒），15 Hz 闭环逐帧调用它既慢又撞 graph 尺寸限制；闭环里只考虑批量 IK，不用 `plan_pose`。

---

## 7. 未确认清单（写论文时注意措辞）

1. 4060 上 cuRobo 的**单次耗时与峰值显存**：无公开数据，本文 15–40 ms、1–3 GB 均为估算。
2. `v0.7.8` 与 Isaac Lab 2.3.x 的兼容性：只做了提交级 diff（4 个提交、API 未变），**未做官方级验证**。
3. v2 首次运行的 JIT 编译耗时；以及 v2 与 Isaac Lab 2.3 能否共存（其 `curobo.scene`/`MotionPlanner` 与 Isaac Lab import 路径完全不同，必然要改代码，**未实测**）。
4. SkillGen 在 8 GB 上一定 OOM 吗：官方 9.3–9.6 GB 是 RTX 6000 Ada 的数，本机未实测（预期 OOM）。
5. 内置机器人资产的逐条许可（`LICENSE_ASSETS` 未逐条核对）。
6. v1 在 Python 3.11 上的官方支持：文档写「might work but is not tested」，但 Isaac Lab 2.3 实际就是 3.11 + v1（事实可用，非官方承诺）。

---

## 8. 参考链接

**cuRobo 本体**
- 仓库：<https://github.com/NVlabs/curobo>
- README（v1/v2 声明）：<https://github.com/NVlabs/curobo/blob/main/README.md>
- Releases：<https://github.com/NVlabs/curobo/releases>
- CHANGELOG（0.8.0 破坏性变更）：<https://github.com/NVlabs/curobo/blob/main/CHANGELOG.md>（v1 线：<https://github.com/NVlabs/curobo/blob/v0.7.8/CHANGELOG.md>）
- `pyproject.toml`（license / 依赖 / 运行时编译注释）：<https://github.com/NVlabs/curobo/blob/main/pyproject.toml>
- v1 `setup.py`（CUDAExtension 清单）：<https://github.com/NVlabs/curobo/blob/ebb71702f3f70e767f40fd8e050674af0288abe8/setup.py>
- v1 `MotionGenConfig` 默认值：<https://github.com/NVlabs/curobo/blob/ebb71702f3f70e767f40fd8e050674af0288abe8/src/curobo/wrap/reacher/motion_gen.py>
- v2 `MotionPlannerCfg` 默认值：<https://github.com/NVlabs/curobo/blob/main/curobo/_src/motion/motion_planner_cfg.py>
- 安装文档（v2）：<https://nvlabs.github.io/curobo/latest/getting-started/installation.html>
- 安装文档（v1，「20 minutes」）：<https://nvlabs.github.io/curobo/v0.7.6/get_started/1_install_instructions.html>
- Known issues（cuda graph 尺寸锁死）：<https://nvlabs.github.io/curobo/v0.7.6/get_started/6_known_issues.html>
- Benchmarks（5090/Orin 数字）：<https://nvlabs.github.io/curobo/latest/reference/benchmarks.html>

**论文**
- cuRobo v1：<https://arxiv.org/abs/2310.17274>
- cuRoboV2：<https://arxiv.org/abs/2603.05493>
- cuRoboV2-Fast：<https://arxiv.org/abs/2611.06913>

**Isaac Lab / Isaac Sim**
- SkillGen 文档（安装 + VRAM 表）：<https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/skillgen.html>
- 2.3.0 对应页：<https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/overview/imitation-learning/skillgen.html>
- 碰撞同步修复 PR #4212：<https://github.com/isaac-sim/IsaacLab/pull/4212>
- SkillGen 引入 PR #3303：<https://github.com/isaac-sim/IsaacLab/pull/3303>
- 控制器 API：<https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.controllers.html>
- Releases（版本↔Sim 对应）：<https://github.com/isaac-sim/IsaacLab/releases>

**本仓库内证据**
- `IsaacLab/docs/source/overview/imitation-learning/skillgen.rst`（pinned commit、12.8 toolkit、VRAM 表在 369–375 行）
- `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner.py`（`:242` MotionGen 创建、`:626/:1692/:1801` 坐标基准、`:1053` `plan_motion`）
- `IsaacLab/source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner_cfg.py`（seeds 12/12、cache 150/150、`collision_checker_type`）
- `IsaacLab/source/isaaclab/isaaclab/controllers/{differential_ik,operational_space,rmp_flow}.py`、`controllers/pink_ik/`
- `forge_ext/base_policy.py`、`forge_ext/residual_env.py`、`demos/01_franka_hold.py`、`demos/03_scripted_insert.py`、`demos/05_eval_success.py`
- `依赖安装与排错记录.md`（8 GB / 驱动 580.126.09 / warp 1.19.0 / 空场景 2.2 GB）
