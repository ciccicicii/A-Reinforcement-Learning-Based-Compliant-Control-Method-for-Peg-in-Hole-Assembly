# RLinf 技术调研报告

> 面向本科毕设《基于残差强化学习与柔顺控制的机器人精密插接方法研究》
> 调研时间：2026-11-22；代码基线：`RLinf/RLinf` main 分支 commit `dcf00fd`（2026-11-21），最新 release tag `v0.4`（2026-10-20）
> 全文事实均给可核验链接；无法从公开资料确证之处标注 **未确认**。

---

## 0. 结论速览

| 问题 | 结论 | 关键证据 |
|---|---|---|
| RLinf 是什么 | 面向具身智能/Agentic 的分布式 RL **训练基础设施**（系统论文，非算法库） | [arXiv:2509.15965](https://arxiv.org/abs/2509.15965)、[OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/yu-chao) |
| 是否官方支持 Isaac Lab | **是，官方支持**（有独立 env 包、文档、配置、上游背书） | [rlinf/envs/sim/isaaclab](https://github.com/RLinf/RLinf/tree/main/rlinf/envs/sim/isaaclab)、[Isaac Lab v3.0.0 文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/concepts/reinforcement_learning.html#rlinf-vla-post-training) |
| 官方 Isaac Lab 示例的硬件口径 | **1 节点 8 GPU** | [isaaclab.rst](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/isaaclab.html) |
| 8GB / 14GB 能否跑 | 只能跑 **MLP 级小示例且需改配置**；VLA 与 Isaac Lab 均不可行 | 官方测试配置 8×H100/1.8TB、Isaac Sim 最低 32GB RAM/16GB VRAM |
| 有无 residual RL | **无原生支持**；最接近的是 DSRL（冻结 π₀ + 噪声空间 steering） | `grep -r residual rlinf/` 仅命中 RTC 表述与 DSRL |
| 毕设建议 | 主线用 MuJoCo/ManiSkill 自写残差 PPO；Isaac Lab 仅作高保真验证（走 rsl_rl/skrl）；RLinf 定位"系统层对照" | 见第 5 节 |

---

## 1. RLinf 是什么

### 1.1 论文与出处

- 标题：**RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation**
- arXiv：**2509.15965**（v1 2025-09-19，v2 2025-12-29）→ <https://arxiv.org/abs/2509.15965>
- 作者：Chao Yu, Yuanqing Wang, Zhen Guo, Hao Lin, Si Xu, Hongzhi Zang, Quanlu Zhang, Yongji Wu, ... Bo Dai, ... Zhi Yang, Guohao Dai, Yu Wang（共 29 人）
- 单位（据论文首页署名）：**清华大学（Tsinghua University）**、**无问芯穹（Infinigence AI）**、**北京大学**、**UC Berkeley**、**中关村学院（Zhongguancun Academy）** 等
  → 核验：<https://ar5iv.labs.arxiv.org/html/2509.15965>（正文首段即列出逐人 affiliation）
- 发表：**OSDI 2026**（<https://www.usenix.org/conference/osdi26/presentation/yu-chao>）；姊妹工作 **RLinf-VLA** 见 arXiv:2510.06710、RSS 2026（<https://roboticsproceedings.org/rss22/p089.html>）
- 仓库：<https://github.com/RLinf/RLinf>，Apache-2.0，约 5.7k stars（2026-11-22 取数）

### 1.2 核心设计：M2Flow + Worker/Runner/Scheduler + Ray

论文的立论是"RL 训练慢的根因是**系统灵活性不足**"，因此提出 **macro-to-micro flow transformation（M2Flow）**：

1. 用户按**宏逻辑流（macro logical flow）**写训练流程（如"采样—推理—更新—同步"），它易组合但粒度粗；
2. 系统自动在**时间维度**（把串行阶段切成可重叠的微步）与**空间维度**（把组件切分到不同进程/卡）把宏流拆成**微执行流（micro execution flow）**；
3. 依靠 RLinf Worker 的**自适应通信能力**实现 **context switching（上下文切换）** 与 **elastic pipelining（弹性流水）**；
4. 用 **profiling-guided scheduling** 搜索/生成执行计划（placement + pipeline stage）。

对应到代码，三大角色分别落在：

```
rlinf/scheduler/worker/          # Worker：统一抽象，支持跨节点弹性扩缩、异构设备
rlinf/scheduler/manager/         # Scheduler：资源探测与调度
rlinf/scheduler/placement/       # ComponentPlacement：组件→节点/卡 的放置
rlinf/scheduler/collective/      # 自建通信组（NCCL/Gloo），非纯依赖 torch.distributed
rlinf/scheduler/channel/         # 组件间数据通道
rlinf/scheduler/dynamic_scheduler/
rlinf/workers/{actor,rollout,env,critic,reward,inference}/   # 具体 Worker 实现
```

**Ray 的角色**：Ray 只做集群与资源管理、进程编排，通信与调度策略由 RLinf 自己实现（`pyproject.toml` 里 `ray[default]>=2.47.0`）。这也是它能做"context switching + elastic pipeline"的前提——纯 Ray actor 模型做不到组件级共卡切换。

**actor / rollout 分离**是它的关键工程特征：

- `actor`：训练引擎，`training_backend` 支持 `fsdp` / `megatron`（`rlinf/workers/actor/__init__.py`）；
- `rollout`：推理引擎，`rollout_backend` 支持 `huggingface` / `sglang` / `vllm` / `phyai` / `apxinf`；
- 两者之间靠 **weight syncer**（`examples/embodiment/config/weight_syncer/{patch_syncer,bucket_syncer}.yaml`）增量同步权重，避免每轮全量拷贝；
- 是否共卡由 YAML 决定：`cluster.component_placement: {actor,env,rollout: all}`（共置）或 `{actor: 0-3, env: 0-7, rollout: 0-7}`（分离）。
  文档：<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/concepts/placement.html>

性能口径：论文报 **1.07×–2.43×** 端到端吞吐提升；README 单独强调具身 RL 混合执行模式 **2.434×**。

### 1.3 版本现状

| 版本 | 发布日期 | 备注 |
|---|---|---|
| v0.1 | 2025-12-17 | 首个 release |
| v0.2 | 2026-03-26 | 真机 RL + 多智能体 |
| v0.3 | 2026-07-15 | 真机全流程（采集→SFT→RL→部署） |
| **v0.4（最新）** | **2026-10-20** | 默认栈升级到 **PyTorch 2.11**；新增 `rlinf.robotics` 模块化真机系统、RLT/STEAM/OPD/RTC、8 类加速卡后端 |

- v0.4 release notes：<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/resources/release_v0.4.html>
  （原文警告：**"不要复用 v0.3 虚拟环境"**，默认栈已换到 torch 2.11）
- main 分支持续开发中：最新提交 2026-11-21（`feat(envs): support heterogeneous envs in one run (#1796)`），说明"异构环境混跑"是当前活跃方向。

---

## 2. 支持的环境

### 2.1 环境总表（`env_type` 枚举 ↔ 示例配置）

枚举定义在 `rlinf/envs/__init__.py` 的 `SupportedEnvType`，分派在 `get_env_cls()`：
<https://github.com/RLinf/RLinf/blob/main/rlinf/envs/__init__.py>

| `env_type` | 对应 example 配置（`examples/embodiment/config/`） | 说明 |
|---|---|---|
| `maniskill` | `maniskill_ppo_mlp.yaml`、`maniskill_sac_mlp.yaml`、`maniskill_ppo_openpi.yaml`、`maniskill_async_ppo_*.yaml` | 主力桌面操作基准，状态/视觉两路 |
| `maniskill_rlt` | `maniskill_rlt_stage2_{ac,td3}_mlp.yaml` | RLT 在线微调 |
| `libero` | `libero_{spatial,goal,object,10,90,130}_ppo/grpo_*.yaml`（30+ 个） | VLA 后训练主战场 |
| `libero`（Mem） | `env/libero_mem.yaml` | 长程记忆基准 |
| `robocasa` / `robocasa365` | `robocasa_closedrawer_ppo_openpi.yaml`、`robocasa365_opendrawer_ppo_openpi.yaml` | 厨房长程 |
| `robotwin` | `robotwin_*_ppo/grpo_*.yaml`（10+ 个） | 双臂 |
| `robodojo` | `env/robodojo_put_bottles_into_dustbin.yaml` | **Isaac Sim** 系双臂基准（评估为主） |
| **`isaaclab`** | **`isaaclab_franka_stack_cube_ppo_{gr00t,openpi_pi05}.yaml`** | 见 2.2 |
| `metaworld` | `metaworld_45/50_*.yaml` | 45/50 任务 |
| `behavior` | `behavior_ppo_openpi*.yaml` | 大规模家务（需 Python 3.10） |
| `calvin` | `calvin_{d,abc,abcd}_ppo_*.yaml` | 语言条件长程 |
| `frankasim` | `frankasim_ppo_mlp.yaml`、`frankasim_sac_cnn_async.yaml` | Franka 轻量仿真 |
| `genesis` / `embodichain` | `genesis_cubepick_ppo_mlp.yaml`、`embodichain_ppo_cart_pole.yaml` | 通用物理引擎，**CartPole 是最小示例** |
| `mujoco_warp` | `mujoco_warp_ppo_mlp.yaml` | GPU 批量化 locomotion（5 个经典任务） |
| `geniesim` / `polaris` / `roboverse` / `simple` / `habitat` / `d4rl` / `diffusion` / `world_model` / `real` / `robolab` | 同名配置文件 | 仿真、真机、离线、扩散、世界模型 |

真机（`real`）侧已有：单/双臂 Franka、ARX X5、Piper、SO-101、DOS-W1、GimArm、LeKiwi、Robbyant R2（`rlinf/envs/real/`）——注意 **`realworld_peg_insertion*` 系列配置已经存在**，对"精密插接"选题是可复用的真机实验脚本（RLPD / async PPO / VLM reward 三种配方）。

### 2.2 关键问题：RLinf 是否官方支持 Isaac Lab / Isaac Sim？

**答：支持，且是"一等公民"级别，但支持面很窄（1 个任务、只配 VLA 后训练、8 卡口径）。**

**四条独立证据链：**

1. **代码**：存在完整 env 包
   ```
   rlinf/envs/sim/isaaclab/
   ├── isaaclab_env.py          # IsaaclabBaseEnv(gym.Env)：观测/奖励/终止/分块 step
   ├── venv.py                  # SubProcIsaacLabEnv：把 Kit 应用关进独立子进程
   ├── utils.py                 # quat2axisangle_torch、CloudpickleWrapper
   ├── tasks/stack_cube.py      # IsaaclabStackCubeEnv
   └── __init__.py              # REGISTER_ISAACLAB_ENVS（task_id → 类）
   ```
   `SupportedEnvType.ISAACLAB = "isaaclab"`，`get_env_cls()` 按 `init_params.id` 查 `REGISTER_ISAACLAB_ENVS`。
2. **文档**：<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/isaaclab.html>（中英双语，含安装、模型下载、运行、指标、成功率对照表）。
3. **配置**：
   - `examples/embodiment/config/env/isaaclab_stack_cube.yaml`（环境参数）
   - `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml`（π₀.₅）
   - `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml`（GR00T N1.5）
4. **上游背书**：
   - **Isaac Lab v3.0.0 官方文档把 RLinf 列为 VLA post-training 的 RL 基础设施**（`#rlinf-vla-post-training`）；
   - NVIDIA 的 **Isaac RL Post-Training（`nvidia-isaac/isaac-rlpt`）** 采用 RLinf 作分布式 rollout/training runtime（<https://github.com/nvidia-isaac/isaac-rlpt>）；
   - README 2026/08 条目："Isaac Lab v3.0.0 officially adopts RLinf as its RL training infrastructure"。

**支持的具体任务（唯一一个）**

```
id: Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0
任务：红方块叠蓝方块，再绿方块叠红方块
观测：table_cam + wrist_cam RGB（默认 256×256）+ 本体状态（eef_pos, eef_quat→axis-angle, gripper_pos）
动作：7 维连续（x,y,z + roll,pitch,yaw + gripper）
奖励：稀疏 0/1（配置里 use_rel_reward: True → 实际喂给算法的是 reward 差分）
```

**启动方式（官方原文命令）**

```bash
# 1) 推荐：直接用官方镜像（省掉 Isaac Sim + torch 版本地狱）
docker run -it --rm --gpus all --shm-size 32g --network host --name rlinf \
  -v .:/workspace/RLinf rlinf/rlinf:agentic-rlinf0.4-isaaclab
source switch_env gr00t      # 或 source switch_env openpi

# 2) 裸机安装
bash requirements/install.sh embodied --model openpi --env isaaclab
source .venv/bin/activate

# 3) 下载 Isaac Sim（文档指定 5.1.0）
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip && source ./setup_conda_env.sh   # 每个新终端都要 source

# 4) 下载 SFT 起点（必须，否则从零 PPO 成功率 0.000）
hf download RLinf/RLinf-Gr00t-SFT-Stack-cube --local-dir <path>
# 或 YifWRobotics/RLinf-pi05-SFT-Stack-cube

# 5) 训练
bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_openpi_pi05
tensorboard --logdir ../results --port 6006      # 关键指标 env/success_once
```

**版本要求**
- Isaac Sim：**5.1.0**（文档硬编码下载链接）；
- Isaac Lab：`install.sh` 的 `install_isaaclab_env()` 克隆 **`https://github.com/RLinf/IsaacLab`**（fork），其 `VERSION` 文件为 **2.3.0**（2026-11-22 核对）；这与"上游 v3.0.0 文档收录 RLinf"是两个不同集成点，**fork 是否已跟进 3.0.0 未确认**；
- 硬件：文档卡片写 **"1 节点 · 8 GPUs"**；
- 该 example 只支持 **VLA**（GR00T N1.5 / π₀.₅），**没有 MLP/状态型 Isaac Lab 示例**（`ls examples/embodiment/config | grep isaaclab` 只有这两个）。

**若要用自己的 Isaac Lab 任务**，文档给的三步（原文）：

> 在 `rlinf/envs/sim/isaaclab/tasks/` 下实现任务 → 在 `rlinf/envs/sim/isaaclab/__init__.py` 注册 → 把 `examples/embodiment/config/env/isaaclab_stack_cube.yaml` 的 `init_params.id` 指向新 task id。

### 2.3 环境接口：接入新仿真器要写什么

RLinf 的环境接口是 **`gymnasium.Env`（不是 gym 旧 API，也不是自定义基类）**，另有可选 `EnvOffloadMixin`。文档：<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/extending/new_env.html>

必需实现（签名照抄官方文档 / `IsaaclabBaseEnv`）：

```python
class YourEnv(gymnasium.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info): ...
    @property
    def device(self): ...                 # 张量所在设备（IsaacLab 是 cuda）
    @property
    def total_num_group_envs(self): ...   # GRPO 分组用；PPO-only 场景可不实现（IsaacLab 实现里就没有）
    def reset(self, seed=None, env_ids=None): return obs, {}          # obs 为 dict，info 为 dict
    def step(self, actions=None, auto_reset=True):
        return obs, reward, terminations, truncations, infos           # 五元组，张量
    def chunk_step(self, chunk_actions):  # [num_envs, chunk, action_dim]
        return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list
    def close(self): ...
    def update_reset_state_ids(self): ...
```
（`num_envs` 在 IsaacLab 实现里是**普通属性**而非 property，文档示例写成 property 只是建议；照 `IsaaclabBaseEnv.__init__` 的写法最稳。）

可选（要支持 env offload / 断点续跑时）：

```python
class YourEnv(gymnasium.Env, EnvOffloadMixin):   # rlinf.envs.env_offload_wrapper
    def get_state(self) -> bytes: ...
    def load_state(self, state_buffer: bytes): ...
```

注册三件套：
1. `rlinf/envs/__init__.py`：加 `SupportedEnvType` 枚举值 + `get_env_cls()` 分支；
2. 包内 `__init__.py`：`REGISTER_XXX_ENVS = {"task-id": YourEnv}`（Isaac Lab 就是这个模式）；
3. `rlinf/envs/action_utils.py`：加 `prepare_actions_for_your_env(raw_chunk_actions, ...)` 并在 `prepare_actions()` 里分发（**动作缩放/夹爪符号/维度裁剪都在这里，是天然的"动作包装层"**）。

配置文件两条：`examples/embodiment/config/env/<name>.yaml`（`env_type`、`total_num_envs`、`max_episode_steps`、`seed`、`reward_coef`、`init_params`）+ 主实验 yaml 的 `defaults: - env/<name>@env.train`。

---

## 3. 算法支持

### 3.1 算法 ↔ 配置开关 ↔ 代表文件

算法不是"类"，而是 `adv_type`（优势估计）+ `loss_type`（策略损失）的组合，注册在 `rlinf/algorithms/{registry,advantages,losses}.py`：

| 算法 | `algorithm.adv_type` | `algorithm.loss_type` | 代表配置 |
|---|---|---|---|
| PPO | `gae` | `actor_critic` | `maniskill_ppo_mlp.yaml`、`isaaclab_*_ppo_*` |
| Async PPO | `gae` | `actor_critic` | `maniskill_async_ppo_mlp.yaml` |
| GRPO | `grpo` | `actor` | `libero_spatial_grpo_openpi_pi05.yaml`（`group_size: 8`） |
| DAPO | `grpo_dynamic` | `actor` | 见 `docs/.../reference/algorithms/dapo.rst` |
| Reinforce++ | `reinpp` | `actor` | 同上 |
| SAC | `embodied_sac` | `embodied_sac` | `maniskill_sac_mlp.yaml`、`realworld_*_sac_cnn*.yaml` |
| CrossQ | — | — | `maniskill_crossq_mlp.yaml` |
| RLPD | — | — | `realworld_peginsertion_rlpd_cnn_async.yaml` |
| SAC-Flow / DSRL | `embodied_sac` 变体 | — | `maniskill_sac_flow_state.yaml`、`libero_spatial_dsrl_openpi.yaml` |
| IQL（离线） | — | — | `d4rl_iql_mujoco.yaml` |
| DAgger / HG-DAgger | — | `embodied_dagger` | `libero_spatial_dagger_openpi.yaml` |
| RECAP / STEAM / RLT / OPD | `opd` 等 | `rlt_ac`、`rlt_td3` | `maniskill_rlt_stage2_ac_mlp.yaml` |

文档索引（每个算法一页超参说明）：<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/reference/algorithms/index.html>
完整清单（README 表格）：GRPO、PPO、Async PPO、DAPO、Reinforce++、SAC、CrossQ、RLPD、SAC-Flow、DSRL、IQL、RECAP、STEAM、RLT、OPD、DiffusionART、ReFL。

**结论：PPO / SAC / GRPO 全部有，且都支持连续动作；但没有 model-free 的 "residual PPO" 这类现成配方。**

### 3.2 连续动作怎么指定

```yaml
actor:
  model:
    model_type: "mlp_policy"
    policy_setup: "panda-ee-dpos"   # 决定动作语义（末端 delta 位姿 / 关节）
    action_dim: 4                   # 动作维度，必须与环境对齐
    num_action_chunks: 1            # 一次预测多少步动作（动作分块）
    obs_dim: 42
    hidden_dim: 256
    precision: "32"
    add_value_head: False           # PPO 需要 True；SAC 用 add_q_head
```
（`examples/embodiment/config/model/mlp_policy.yaml`、`maniskill_sac_mlp.yaml`）
VLA 侧：`model/pi0_5.yaml` → `action_dim: 7`、`num_action_chunks: 10`、`openpi.task: rl`。
动作进入仿真的最后一道加工在 `rlinf/envs/action_utils.py:prepare_actions(env_type, raw_chunk_actions, num_action_chunks, action_dim, action_scale, policy)`——`action_scale` 就是动作缩放入口。

### 3.3 奖励缩放 / 奖励整形

| 机制 | 位置 | 作用 |
|---|---|---|
| `env.train.reward_coef` | 环境配置 | 环境侧奖励乘子 |
| `env.use_rel_reward` | 环境配置 | True 时只喂**奖励差分**（稀疏 0/1 任务必开，Isaac Lab 默认 True） |
| `algorithm.reward_coef` | 主配置 | 算法侧再乘一次 |
| `algorithm.reward_type` | 主配置 | `chunk_level`（整段动作一个奖励，VLA/IsaacLab）或 `action_level`（每步，MLP/PPO） |
| `algorithm.filter_rewards` + `rewards_lower_bound/upper_bound` | 主配置 | 奖励裁剪（IsaacLab 配置里是 0.1–0.9） |
| `reward.use_reward_model` | 主配置 | 接 VLM/ResNet 奖励模型（`maniskill_ppo_mlp_resnet_reward.yaml`、`*_vlm_trend_reward.yaml`） |

### 3.4 日志

```yaml
runner:
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "isaaclab_ppo_openpi_pi05"
    logger_backends: ["tensorboard"]      # 可选 "wandb" / "swanlab"
```
目录约定：`<log_path>/{tensorboard,wandb,swanlab}/`、`checkpoints/`（文档 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/guides/logger.html>）。W&B 需先 `wandb login`；`pyproject.toml` 锁 `wandb<0.25.1`。

### 3.5 加载预训练策略 / 断点

```yaml
actor:
  model:
    model_path: "/path/to/RLinf-Gr00t-SFT-Stack-cube"   # 预训练权重目录（HF 下载）
runner:
  resume_dir: null    # 恢复训练：指向 checkpoints/global_step_10（含优化器/RNG）
  ckpt_path: null     # 只加载权重做 eval：指向 .pt（实例化后加载）
  only_eval: False
```
（`examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml:130`；恢复机制见 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/guides/resume.html>）

### 3.6 有没有 residual RL / 分层 / 动作包装？

**逐项核实（`grep -rIl "residual"` 全仓库）：**

| 能力 | 有无 | 事实 |
|---|---|---|
| Residual RL（残差策略叠加基座策略） | **无** | 全仓库无 `residual` 相关的策略/损失/配置；`methods_index.rst` 里 "residual / noise-space policy steering" 指的就是 DSRL |
| 分层策略（hierarchical policy / manager-worker） | **无** | 仅 `concepts/robotics.rst` 讲硬件层级，与策略分层无关 |
| 动作包装（action wrapper） | **有（间接）** | `rlinf/envs/action_utils.py:prepare_actions_*`（每环境一套缩放/符号/维度规则）+ `rlinf/envs/wrappers/{insert_delay,record_video,collect_episode}.py` |
| 动作分块（action chunk） | **有** | `num_action_chunks` + 环境侧 `chunk_step()`；`chunk_actions: [num_envs, chunk, action_dim]`，奖励/终止按 chunk 聚合 |
| 噪声空间 steering（最接近残差思想） | **有** | **DSRL**：冻结 π₀ 扩散策略，用轻量 SAC 在**噪声空间**输出 steering 量（<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dsrl.html>），配置 `libero_spatial_dsrl_openpi.yaml` |
| 实时分块（RTC） | **有** | `guides/rtc.rst`，real-time chunking，用于降低推理延迟 |

**对毕设的含义**：RLinf 里做"残差强化学习"，必须自己实现。两条低成本路径：
1. **模型层**：写一个 `model_type: residual_mlp`（`rlinf/models/embodiment/` 下新增），forward 里 `a = a_base(s) + a_res(s)`，基座策略冻结、只训残差头；PPO 侧 `adv_type: gae` + `loss_type: actor_critic` 不用改。
2. **环境层**：在 `action_utils.prepare_actions_for_<env>` 里做 `a_final = clip(a_base + α·a_res)`，把"残差"藏进动作包装——改动最小，但梯度语义不如方案 1 干净（不推荐用于论文方法章节）。

---

## 4. 硬件要求：8GB RTX 4060 Laptop + 14GB 内存够不够

### 4.1 官方口径（三处，都不是"最低配置"）

| 来源 | 配置 | 性质 |
|---|---|---|
| 安装文档"环境要求" | **每节点 8×H100 / 192 核 / 1.8TB 内存 / 3.2Tbps** | 明确写"以下是**经过充分测试**的配置"，非硬下限（<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/requirements.html>） |
| Isaac Lab 示例卡片 | **1 节点 · 8 GPU** | 官方示例的默认口径 |
| Isaac Sim 5.1 系统需求 | **RAM 32GB 起、VRAM 16GB 起**（recommended 64GB/48GB） | 硬下限（<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>） |

### 4.2 逐档判定

| 目标 | 8GB VRAM | 14GB RAM | 判定 |
|---|---|---|---|
| Isaac Lab 示例（π₀.₅/GR00T + Kit） | ✗（Isaac Sim 下限 16GB） | ✗（下限 32GB） | **不可行** |
| VLA 类（OpenVLA-7B / π₀.₅ 3B） | ✗（bf16 权重 + 激活即超 8GB） | ✗ | **不可行** |
| MLP + PPO 状态型（ManiSkill / Genesis / EmbodiChain / MuJoCo Warp） | ✓（改配置后） | △（紧，需压 env 数） | **可行，但必须改配置** |

**最小可跑的三个示例（按"改动量"排序）**

1. `embodichain_ppo_cart_pole.yaml` —— 纯 CartPole + MLP，最轻，用来验证"Ray + FSDP + 训练循环"能起来；
2. `mujoco_warp_ppo_mlp.yaml` —— GPU 批量物理，5 个 locomotion 任务，显存占用可控；
3. `maniskill_ppo_mlp.yaml` —— **真正有操作语义**（pick cube 状态版），也是毕设最该先跑通的一个。

必须改的四处（`maniskill_ppo_mlp.yaml` 默认要 1024 个并行环境，8GB 直接爆）：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: 0        # 原来就是 0，单卡共置
env:
  train:
    total_num_envs: 64          # 1024 → 64（先跑通再往上加）
    max_episode_steps: 50
  eval:
    total_num_envs: 8           # 32 → 8
    video_cfg: {save_video: False}   # 关视频，省显存与磁盘
actor:
  micro_batch_size: 512         # 6400 → 512
  global_batch_size: 512
  enable_offload: False
rollout:
  enable_torch_compile: False   # 首次编译吃内存，排障期先关
  enable_cuda_graph: False
```

### 4.3 是否需要多卡/多机

**不需要，但官方明确口径是 8 卡 80GB。** 依据：
- 配置里 `component_placement` 支持把 `actor,env,rollout` 全放 `0` 号卡（`maniskill_ppo_mlp.yaml` 默认就是 `env,rollout,actor: 0`），说明单卡是**支持的拓扑**；
- **维护者原话**（issue #1210，问"isaaclab 那个配置是不是 8 卡 4090"）：
  > "RLinf 默认配置是 **8 卡 80GB A100/A800/H100，全量**。" —— @zoeyuchao（2026-06-09）
  → <https://github.com/RLinf/RLinf/issues/1210>
  即：默认配置按 8×80GB 设计，**低配没有任何官方承诺**；
- issue #1384 报告单卡 IsaacLab 挂死（见 6.4）。

**14GB 内存才是真瓶颈**：Ray（每 worker 一个 Python 进程）+ torch + FSDP + 仿真器，光空载就 6–8GB（本机实测：14GB 总量、6GB 已用）。建议 32GB swap + 关闭 `record_video` + `--num-envs` 从 8 起步。

---

## 5. 接入 Isaac Lab 的现实方案（两条路线）

先纠正一个前提：**不需要"写 Isaac Lab Env Adapter"——官方已经写好了**（`IsaaclabBaseEnv` + `SubProcIsaacLabEnv`）。真正要写的只是"任务子类"。所以两条路线的真实对比是：

### 路线 A：在 RLinf 内接自己的 Isaac Lab 任务（复用官方 adapter）

```python
# rlinf/envs/sim/isaaclab/tasks/peg_insert.py
class IsaaclabPegInsertEnv(IsaaclabBaseEnv):
    def _make_env_function(self):
        def make():
            os.environ.pop("DISPLAY", None)                  # 强制 headless，避免 GLX 报错
            from isaaclab.app import AppLauncher
            sim_app = AppLauncher(headless=True, enable_cameras=True).app
            from isaaclab_tasks.utils import load_cfg_from_registry
            cfg = load_cfg_from_registry(self.isaaclab_env_id, "env_cfg_entry_point")
            cfg.seed = self.seed
            cfg.scene.num_envs = self.cfg.init_params.num_envs
            env = gym.make(self.isaaclab_env_id, cfg=cfg, render_mode="rgb_array").unwrapped
            return env, sim_app
        return make
    def _wrap_obs(self, obs):        # IsaacLab 观测 → RLinf 统一 dict
        ...
```
再注册 `REGISTER_ISAACLAB_ENVS["My-Peg-Insert-v0"] = IsaaclabPegInsertEnv`，写 `config/env/isaaclab_peg_insert.yaml`。

**它已经替你解决/必须自己面对的语义问题：**

| 问题 | 官方怎么处理 | 你还得做什么 |
|---|---|---|
| Isaac Sim 必须独占主进程 | `SubProcIsaacLabEnv` 用 `multiprocessing.Process` 起 Kit，父进程只发 RPC（`venv.py` 的 `_worker` 循环：`step/reset/close/getattr/setattr`） | 子进程里**不能再 import torch 训练代码**；`cloudpickle` 序列化 env_fn，闭包不能带不可序列化对象 |
| 观测 numpy↔torch | Isaac Lab 直接返回 **cuda tensor**，`_wrap_obs` 不做 numpy 转换（`IsaaclabBaseEnv` 注释明说"output of isaaclab is all tensor on cuda"） | 你的 `_calc_step_reward`/metrics 必须**全程 torch on cuda**，混 numpy 会隐式同步变慢 |
| env 并行 | `Isaac Lab` 内部 `scene.num_envs` 做批量化；RLinf 侧 `total_num_envs` 再按 env worker 数均分（`total_num_processes=assignment.world_size * stage_num`） | 两边数量要整除，否则某 worker 拿 0 个环境 |
| reset/step 语义 | 5 元组 + `auto_reset=False`（IsaacLab 强制关自动 reset，由 `ignore_terminations` 控制） | 终止后必须显式 `reset(env_ids=...)`，否则 episode 无限长 |
| seed | 双重：`cfg.seed = self.seed` 给 Isaac 侧，`self.seed = cfg.seed + seed_offset` 给 RLinf 侧 | 想做可复现对比实验，`use_fixed_reset_state_ids: True` |
| 显存共享 | actor/rollout/env 若共卡，显存由三者共享，靠 `actor.enable_offload`、`rollout.pipeline_stage_num` 调 | 8GB 上这三角不可能共存 |

**工作量**：写任务子类 1–2 人日；把 Isaac Lab 自己的任务（如 `Isaac-Lift-Cube-Franka-v0`）改造成"有相机 + 有 0/1 奖励"的可 RL 任务 2–4 人日；调通分布式启动 + 规避 #1384 的静默挂起 3–5 人日 → **合计 6–11 人日**，且**大概率卡在 #1384 上无法交付**（该缺陷截至 2026-11-06 仍未合入 main）。

### 路线 B：用 Isaac Lab 原生 RL 库训练，RLinf 只作对照框架

`rsl_rl`（Isaac Lab 官方自带 PPO）、`skrl`、`stable-baselines3` 都在 Isaac Lab 生态里现成可用；Isaac Lab v3.0.0 文档本身就把 rsl_rl / skrl / sb3 / RLinf 并列列出（`rl_existing_scripts.html`）。

- **优点**：单卡也能跑（Isaac Lab 自带 `train.py --num_envs 64 --headless`），残差结构改造自由（rsl_rl 的 policy 就是一个 MLP，加个 `residual` 分支 30 行）；
- **缺点**：**丢掉 RLinf 的全部价值**（M2Flow 调度、actor/rollout 分离、VLA 后训练、多机弹性）；且 Isaac Sim 自身的 32GB RAM / 16GB VRAM 下限仍在，8GB 本机依然跑不动。

### 推荐方案（面向本科毕设 + 8GB/14GB 硬件）

> **主线不做 Isaac Lab，也不把 RLinf 当训练框架。**
>
> 1. **方法主体**：ManiSkill / MuJoCo（`sim_backend: cpu` 可把 PhysX 放 CPU，见 `_maniskill_non_cuda.rst`）搭"轴孔插接"任务，自写**残差 PPO**（基座=阻抗/导纳控制器或预训练策略，残差=小 MLP，动作 `a = a_base + α·a_res`，α 退火）。这条在 8GB 上能跑，且与"柔顺控制"结合最自然（残差输出可直接叠加到阻抗参数/位姿修正量上）。
> 2. **RLinf 的用法**：作为**系统层对照与展望**——在论文"相关系统"章节引用 M2Flow/OSDII 2026，说明"若把该插接任务扩展到 VLA 后训练/多机，可用 RLinf 承载"，并给出官方 `isaaclab_franka_stack_cube` 的成功率数据（GR00T 0.654→0.897、π₀.₅ 0.859→0.953）作为"RL 后训练有效性"的旁证。
> 3. **Isaac Lab**：只在**实验室有 ≥32GB RAM / ≥16GB VRAM 的机器**时做"高保真复现 + 答辩视频"，走路线 B（rsl_rl + 自写 residual wrapper），**不要**去趟 RLinf×IsaacLab 的单卡静默挂起（#1384）。
> 4. 若导师坚持"必须用 RLinf 跑 Isaac Lab"：直接申请 8 卡机器或用官方 `rlinf/rlinf:agentic-rlinf0.4-isaaclab` 镜像，**不要试图在 4060 上复现**。

### 工作量对比

| 路线 | 内容 | 估计 | 风险 |
|---|---|---|---|
| A（RLinf + IsaacLab 自定义任务） | 任务子类 + 注册 + 配置 + 分布式调通 | 6–11 人日 | **高**（#1384 未修、官方口径 8×80GB） |
| B（Isaac Lab + rsl_rl/skrl + 残差改造） | 任务 + residual policy + PPO 调参 | 4–7 人日 | 中（Isaac Sim 硬件下限） |
| **C（推荐）ManiSkill/MuJoCo + 残差 PPO** | 插接任务建模 + 阻抗基座 + 残差策略 + 对照实验 | 8–12 人日 | **低**（8GB 可跑、可复现） |

---

## 6. 已知坑

### 6.1 Python / torch / 依赖版本

| 项 | 事实 | 来源 |
|---|---|---|
| Python | `requires-python >= 3.10`；`install.sh` 默认 **3.11.14**；`agentic` 目标在 torch 2.11 栈上用 **3.12.12**；`behavior`/`d4rl` 需 **3.10** | `pyproject.toml`、`install.sh --help` |
| torch | **v0.4 默认栈 torch 2.11.0 / torchvision 0.26.0**（v0.3 是更早栈，**禁止复用旧 venv**） | v0.4 release notes |
| torch（IsaacLab 分支） | `install_isaaclab_env()` 注释写明 **"Torch is modified in Isaac Lab"**：`isaaclab.sh --install` 会覆盖 torch，所以 `install_flash_attn` 必须放在**之后**；随后强制 `numpydantic==1.7.0 pydantic==2.11.7 numpy==1.26.0` | `requirements/install.sh` |
| Isaac Sim 5.1 自带 torch | 普遍为 2.7.x（与 RLinf 默认 2.11 冲突，这正是"必须装在独立 venv + env 走子进程"的根因） | **未确认**（未在 RLinf 文档中明示具体小版本） |
| Ray | `ray[default]>=2.47.0` | `pyproject.toml` |
| flash-attn | `install.sh` 默认装；无 CUDA 编译链用 `--no-flash-attn` 跳过；平台无预编译 wheel 时走源码编译（慢，易失败）；另需 `cmake<4`（egl-probe/robomimic 原生构建） | `install.sh` 参数表 |
| wandb | 锁 `wandb<0.25.1` | `pyproject.toml` |

### 6.2 安装形态

- 强烈建议 **Docker**（`rlinf/rlinf:agentic-rlinf0.4-<env>`），国内可用 `infinigence-ai-registry.cn-beijing.cr.aliyuncs.com` 镜像；
- 裸机：`bash requirements/install.sh embodied --model <m> --env <e>`，`--use-mirror` 加速；`--install-rlinf` 才会装 RLinf 本体；
- 容器内**不要覆盖 `/root`、`/opt`**，平台改 `$HOME` 时先跑 `link_assets`；
- **每次新终端启动 Isaac Lab 前都要 `source ./setup_conda_env.sh`**（文档 warning 原文）。

### 6.3 Hydra

所有 embodied 配置的 `hydra.searchpath` 依赖环境变量 `EMBODIED_PATH`（`file://${oc.env:EMBODIED_PATH}/config/`），直接 `python` 跑会报 `InterpolationKeyError`。必须走 `bash examples/embodiment/run_embodiment.sh <cfg_suffix>`（脚本里已 export）。

### 6.4 未修复的运行时坑（对本毕设影响最大）

| Issue | 状态 | 内容 |
|---|---|---|
| [#1384](https://github.com/RLinf/RLinf/issues/1384) | **open**（2026-07-15，**截至 2026-11-06 仍未合入 main**） | **单卡 IsaacLab 训练挂死（静默）**：`EnvWorker` 子进程变僵尸，父进程阻塞在 `multiprocessing.queues.get`；`CollectiveManager` 与 `EmbodiedFSDPActor._atomic_recv` 互相等待。作者用**官方 example 改 placement=0 + num_envs=2 复现**，排除了自定义任务因素。已定位根因（维护者 yangzhongii 2026-08-01）：`venv.py` 里 `env_fn()`（拉起 IsaacSim）写在 `try:` **之外**，IsaacSim 启动失败时子进程静默死亡；`obs_queue.get()` **无 timeout** → 父进程永久阻塞；`TIMEOUT_WARN_TIME=3600000` 使 1 小时内无告警。修复分支 `yangzhongii/RLinf:fix/issue-1384-deadlock-zombie` 已给出但**未合并**；社区成员 2026-10-29 复核 main 仍是旧代码 |
| [#1026](https://github.com/RLinf/RLinf/issues/1026) | open（2026-04-12） | 自定义 IsaacLab 任务（SFT+RL）全流程**无官方答复**（仅提问者自己 bump 一次）；IsaacLab 遥操作 HDF5 → RLinf 数据格式**无内置转换器**，需自写 parser |
| [#1210](https://github.com/RLinf/RLinf/issues/1210) | open（2026-05-27） | 维护者答复"默认配置是 8 卡 80GB A100/A800/H100，全量" → **官方对低配硬件零承诺**（这是全仓库关于硬件下限最直接的一句话） |

> 对毕设的解读：#1384 说明"IsaacSim 启动失败 → 静默 hang"是**结构性缺陷**而不是偶发 bug。在 8GB 显存上 IsaacSim 很可能启动失败，而失败的表现就是**无限挂起且无日志**——这正是"看起来像死锁、实际是启动失败"的典型陷阱。若一定要试，先手工验证 `AppLauncher(headless=True)` 能起来。

### 6.5 本机（14GB RAM / 8GB VRAM）额外注意

- `total_num_envs` 是**全局**数量，会被均分到 `world_size × stage_num` 个 env worker，设成小于进程数会直接 assert；
- `component_placement: all` 表示 actor/env/rollout **同卡同进程组**，8GB 下应显式分开写（`actor: 0` / `env: 0` / `rollout: 0` 等价，但便于后续拆卡）；
- `rollout.enable_torch_compile: True` 首次编译会吃 2–4GB 内存，排障期先关；
- `save_video: True`（ManiSkill 默认 eval 开）会额外驻留渲染上下文，8GB 上务必关。

---

## 7. 核验清单

| 事实 | URL |
|---|---|
| 论文（M2Flow） | https://arxiv.org/abs/2509.15965 |
| 作者单位（逐人） | https://ar5iv.labs.arxiv.org/html/2509.15965 |
| OSDI 2026 | https://www.usenix.org/conference/osdi26/presentation/yu-chao |
| RLinf-VLA（RSS 2026） | https://arxiv.org/abs/2510.06710 |
| 仓库 / README | https://github.com/RLinf/RLinf |
| 版本 release | https://github.com/RLinf/RLinf/releases/tag/v0.4 |
| v0.4 release notes | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/resources/release_v0.4.html |
| IsaacLab 官方示例文档 | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/isaaclab.html |
| IsaacLab env 代码 | https://github.com/RLinf/RLinf/tree/main/rlinf/envs/sim/isaaclab |
| IsaacLab 示例配置 | https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml |
| IsaacLab 环境配置 | https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/env/isaaclab_stack_cube.yaml |
| RLinf 的 IsaacLab fork（VERSION 2.3.0） | https://github.com/RLinf/IsaacLab |
| Isaac Lab v3.0.0 收录 RLinf | https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/concepts/reinforcement_learning.html#rlinf-vla-post-training |
| NVIDIA Isaac RL Post-Training | https://github.com/nvidia-isaac/isaac-rlpt |
| 环境枚举与分派 | https://github.com/RLinf/RLinf/blob/main/rlinf/envs/__init__.py |
| 新增环境指南 | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/extending/new_env.html |
| 动作包装层 | https://github.com/RLinf/RLinf/blob/main/rlinf/envs/action_utils.py |
| 算法索引 | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/reference/algorithms/index.html |
| 日志配置 | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/guides/logger.html |
| 断点恢复 | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/guides/resume.html |
| 放置策略 | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/concepts/placement.html |
| 环境要求（8×H100 口径） | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/requirements.html |
| 安装 | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/installation.html |
| install.sh | https://github.com/RLinf/RLinf/blob/main/requirements/install.sh |
| pyproject（ray/torch/wandb） | https://github.com/RLinf/RLinf/blob/main/pyproject.toml |
| Isaac Sim 5.1 系统需求 | https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html |
| DSRL（噪声空间 steering） | https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dsrl.html |
| 单卡 IsaacLab 挂死 issue | https://github.com/RLinf/RLinf/issues/1384 |

## 8. 明确"未确认"的项

1. Isaac Sim 5.1 随附 torch 的**确切小版本**（RLinf 文档未写，`install.sh` 仅注释"Torch is modified"）；
2. `RLinf/IsaacLab` fork（VERSION 2.3.0）与上游 Isaac Lab **v3.0.0-EA** 的差异，以及 fork 是否计划跟进 3.0.0；
3. 8GB 显存下 MLP+PPO 的**实测**显存占用与可支撑的最大 `total_num_envs`（本文给的是保守起手值，非实测值）；
4. 官方是否在**任何**低配硬件（≤1 GPU）上验证过 embodied 训练——维护者只给了"8 卡 80GB 全量"的默认口径（issue #1210），未提及验证过小卡；
5. `maniskill_ppo_mlp.yaml` 在 14GB RAM 上能否稳定跑满 1000 epoch（未实测，仅按组件内存估算）。
