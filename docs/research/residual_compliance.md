# 残差强化学习 × 柔顺控制：面向 FORGE PegInsert 精密插接的技术调研

> 对象：本科毕设《基于残差强化学习与柔顺控制的机器人精密插接方法研究》
>
> 方案：解析柔顺基线（自由下降 + 恒压 + 螺旋搜索）+ 二维横向残差策略 $(\Delta x,\Delta y)$
>
> 环境：Isaac Lab **v2.3.0**（本地 `/home/mi/graduation_design/RL/IsaacLab`，`git describe --tags` = `v2.3.0`），FORGE PegInsert，Franka，销 7.986 mm / 孔 8.1 mm
>
> 调研日期：2026-11-22。源码结论均按本地 v2.3.0 逐文件核对，链接指向 GitHub `v2.3.0` tag 下同名路径。
>
> 约定：**未确认** = 原文/源码未直接给出，或仅为本文推算。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 最稳的残差形式 | 在**参考位姿**上叠加有界 $\Delta$pose，而不是力矩/关节加速度；与本项目 $\Delta x,\Delta y$ 一致 |
| base 是否冻结 | 上述工作全部冻结（或离线）base、残差不回传梯度；但 base 的**内部状态机**会与残差"抢控制权"（见 A.3 的 RFL） |
| 如何防残差接管 | 三重：硬限幅（$\pm$3 mm）+ 幅值/变化率惩罚 + 阶段门控/残差 dropout；监控残差饱和度 |
| 奖励必备正则 | 残差幅值、残差变化率、接触力超限、穿透/挤压、成功项；FORGE 已有 keypoint 整形与力惩罚 |
| 柔顺结构 | FORGE = **外环导纳（力→z 冻结）+ 内环阻抗（565 N/m 任务空间 PD）**，正是经典双层结构 |
| 0.057 mm 间隙意味着什么 | 无倒角时横向容差仅 $\pm$0.057 mm；倒角是唯一"漏斗"，因此**螺旋搜索不可省**，残差只做最后几十~几百 µm 的修正 |
| Isaac Lab 组件 | `OperationalSpaceController` 的 `variable_kp` 可直接做变阻抗；但 FORGE 走自定义 `compute_dof_torque`，最小改法是**把 kp 拼进动作**（该函数已接收 per-env 张量） |
| 最大工程风险 | 15 Hz 策略 + 120 Hz 力矩环 + ~5 Hz 力滤波 → 力反馈相位滞后；死区 5 N 等效 8.8 mm 位置死区，远大于几何容差（推算，见 §1） |

---

## 1. 先把事实钉住：本项目的控制栈

| 事实 | 数值 | 出处 |
|---|---|---|
| 官方 FORGE PegInsert 几何 | 8 mm 销，0.5 mm 直径间隙 | FORGE 论文 [arXiv:2408.04587](https://arxiv.org/abs/2408.04587) §II-A |
| 本项目几何 | 7.986 / 8.1 mm → 直径间隙 0.114 mm，单边 0.057 mm | 项目设定；Isaac Lab 源码无此常数（销径来自 USD 资产）**未在源码中复核** |
| 任务空间刚度 | $k_p=[565,565,565,28,28,28]$（N/m, Nm/rad） | [`forge_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env_cfg.py) L24 |
| 阻尼 | $k_d=2\sqrt{k_p}$（临界阻尼）；reset 时转动分支再除以 10 | [`factory_utils.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_utils.py) `get_deriv_gains` |
| 执行器死区 | 平动 5 N、转动 1 Nm，每回合 $U[0,\text{default}]$ 随机 | `forge_env_cfg.py` L28；`factory_control.py` 中 `dead_zone_thresholds` |
| 动作 EMA | 因子 0.025–0.1（很重） | `forge_env_cfg.py` L23 |
| 力 EMA | $\alpha=0.25$ → 截止频率约 $0.25\times120/(2\pi)\approx4.8$ Hz | [`forge_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env_cfg.py) L100、`forge_env.py` L99 |
| 频率 | 物理 120 Hz，策略 15 Hz（`decimation=8`） | `factory_env_cfg.py` L72 |
| 力矩环刷新 | **每个物理步都重算**（`_apply_action` 在 decimation 循环内） | [`direct_rl_env.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/envs/direct_rl_env.py) `step()` L350–357 |
| 力源 | `get_link_incoming_joint_force()`，父体坐标系 6D wrench | [`articulation_data.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/assets/articulation/articulation_data.py) `body_incoming_joint_wrench_b` |

**两个由源码数值推算的结论（未实测）**：

1. 死区在 wrench 域做"符号-减法"（`w.sign()*(|w|-dz)`），等价于库仑静摩擦。平动 5 N / 565 N·m⁻¹ ≈ **8.8 mm** 的等效位置死区；转动（上限）1 Nm / 28 Nm·rad⁻¹ ≈ **2.05°** 的等效角度死区（每回合按 $U[0,\text{default}]$ 缩放，部分环境可能接近 0）。二者都**远大于** 0.057 mm / 0.65°（§B.2）的几何容差。含义：FORGE 靠 15 Hz 的大步长指令与力阈值"越过"死区，而本项目的精密段应把死区降到 0.2–1 N / 0.1–0.3 Nm，或改用显式摩擦补偿。
2. 15 Hz 决策 + ~5 Hz 力滤波 → 力通道相位滞后约 1 个策略步（67 ms）；若残差用力做闭环，等效延迟 100–200 ms，训练时应把"力观测延迟 1 步"显式建模。

---

## A. 残差 RL 文献

### A.1 Johannink et al., *Residual Reinforcement Learning for Robot Control* (ICRA 2019)

- arXiv: <https://arxiv.org/abs/1812.03201>；DOI: <https://doi.org/10.1109/ICRA.2019.8794127>（pp. 6023–6029）。
- **残差形式**：$u_t=\pi_H(s_{m,t})+\pi_\theta(s_{m,t},s_{o,t})$。$\pi_H$ 是人工控制器，$\pi_\theta$ 是可学习残差；作者把状态拆成机器人 $s_m$ 与物体 $s_o$，并论证 $\pi_H$ 负责 $f(s_m)$（几何/轨迹项），RL 只负责 $g(s_o)$（接触、物体姿态项）。
- **训练细节**：仿真用 MuJoCo + Sawyer，**笛卡尔位置控制器**；真机换成"平滑、耐接触的关节空间阻抗控制器"。算法为 TD3（rlkit 实现），带探索噪声 $\mathcal{N}_t$；真机只观测 z 向力（xy 力测不到），观测由相机跟踪给出。
- **装配实验**：任务是"把方块插入两块竖立方块之间"（**不是圆销-圆孔**），两块立块可滑动/倾倒。奖励为
  $$r_t=-\|x_g-x_t\|_2-\lambda(\|\theta_l\|_1+\|\theta_r\|_1)$$
  真机版再加末端位置项与立块姿态项（原文 Eq. 6/7）。
- **结论**：① 同样样本下残差 RL 的最终性能与样本效率均优于纯 RL（纯 RL 要从零学"位置控制结构"）；② 初始姿态对齐时残差 20/20、不对齐时 15/20，而不对齐时人工控制器仅 2/20；③ 对**有偏**控制噪声（$\mu$ 最大 0.2）仍能学出反向补偿；④ 真机约 **8000 样本 / 3 小时**学会；⑤ 用仿真学到的策略做 $\pi_H$ 初始化，可 sim-to-real。
- **未确认**：原文未给残差项的显式缩放/限幅（只给探索噪声），也未报告插入间隙数值。

### A.2 Silver et al., *Residual Policy Learning* (arXiv 2018)

- arXiv: <https://arxiv.org/abs/1812.06298>（预印本）。
- **核心主张**：在**不可微**的基策略上做无模型 RL，$a=a_{\text{base}}(s)+\pi_\theta(s)$；基策略可以是手写控制器，也可以是 MPC（模型已知或已学）。六个 MuJoCo 任务（Push、PickAndPlace、SlipperyPush、NoisyHook、ComplexHook、MBRLPusher），动作归一化到 $[-1,1]^D$，用 HER 处理稀疏奖励。
- **与 A.1 的区别**：

  | 维度 | Johannink 2019 | Silver 2018 |
  |---|---|---|
  | 动机 | 制造场景接触/摩擦难以建模，用残差补足 | 通用地"改进一个不完美但可用的控制器" |
  | base | 人工控制器 **或** 仿真学到的策略 | 手写策略或 MPC，强调**不可微** |
  | 任务 | 真机方块装配 + MuJoCo | 纯 MuJoCo，无装配/插接 |
  | 算法要点 | TD3、探索噪声、特权状态 | 先**只训 critic**（burn-in，阈值 $\beta=1.0$）再放开 actor；POMDP 用历史长度 1 |
  | 验证 | 真机 3 小时 | 仿真六任务，无真机 |

  两者共同的隐含假设：**base 参数冻结**、残差是小量修正、奖励仍是任务奖励（无需为 base 单独设计）。

### A.3 接触-rich 装配/插接上的残差 RL（5 篇）

| 文献 | 方法要点 | 关键结论 |
|---|---|---|
| Ranjbar et al. 2021, *Residual Feedback Learning*，arXiv:2106.04306 | 在**笛卡尔阻抗控制器（黑盒）**上做残差；指出标准 RPL 会被控制器"当成扰动抵抗"，改为同时修改**反馈信号**（residual feedback），并配自适应课程 | 插接任务在位置/姿态不确定性下显著优于标准 RPL；"残差作用 vs 残差反馈"各有适用区间 |
| Davchev et al. 2020, *Residual Learning from Demonstration*，arXiv:2008.07682 | DMP 基座 + 任务空间**全位姿**残差修正；销/齿轮/插头插入，仿真 + 真机 | 残差加在任务空间（而非关节）效果最好；"对关节温和"，并支持少样本迁移到新几何/摩擦 |
| Marougkas et al. 2025, *Integrating Model-based Control and RL for Sim2Real Tight Insertion*，arXiv:2505.11858 | 势场（PF）控制器 + 残差 SE(3) 动作：$a^T_t=a^{PF}_t+\beta a^{RL}_t,\ \beta\in[0,1]$；稀疏奖励 + 穿透惩罚；对观测噪声与 $\beta$ 做课程；PPO + 非对称 actor-critic（actor 带 LSTM） | **零样本** sim-to-real 插入（<1 mm 公差），优于包括 IndustReal 类方法在内的多个基线；$\beta$ 课程让残差"随不确定性增大而接管" |
| Guo et al. 2026, *SPARR*，arXiv:2602.23253 | 仿真 base（低层状态 + 稠密奖励）+ **真机视觉残差**（稀疏奖励） | 相比零样本 sim-to-real，成功率 +38.4%、节拍 −29.7%，且无需人工干预 |
| Liu et al. 2026, *Hybrid Residual RL for Book Insertion*，arXiv:2609.19962 | 名义任务空间控制器 + **有界**残差 PPO（局部修正 + 决定何时松手） | 仿真 98.50% vs 名义 37.89%；真机 63.3% vs 26.7%；初始扰动 1.5× 仍 >87%；**间隙过小时局部修正触及几何极限** |

补充：Abuibaid et al. 2026（arXiv:2604.06949）用"复合技能 + RRL"做 UR5e 销孔装配，把残差限制在技能内部的接触段，用 SAC/JAX 训练，强调**位置控制机械臂**上的可复用性。

### A.4 归纳：残差的形式、缩放、冻结与正则

**(1) 残差形式**

| 形式 | 表达 | 适用 | 风险 |
|---|---|---|---|
| $\Delta$pose（推荐） | $x^{\text{ref}}\leftarrow x^{\text{ref}}+\beta\,s\,\pi_\theta(o)$ | 位置/阻抗内环的插接 | 与 base 的搜索相位耦合 |
| $\Delta$velocity | $v^{\text{ref}}\leftarrow v^{\text{ref}}+\Delta v$ | 速度接口、遥操作 | 积分漂移 |
| $\Delta$force/torque | $F^{\text{ref}}\leftarrow F^{\text{ref}}+\Delta F$ | 力控内环、柔顺装配 | 接触刚度未知 → 易失稳 |
| $\Delta$joint | $q^{\text{ref}}\leftarrow q^{\text{ref}}+\Delta q$ | 关节阻抗、无 IK 场景 | 任务空间语义弱、易碰撞 |
| $\Delta$gain | $k_p\leftarrow k_{p0}e^{\Delta k}$ | 变阻抗（§B.3） | 需限幅 + 变化率约束 |

**(2) 缩放与限幅**：三种主流做法——硬裁剪 $\|\Delta\|_\infty\le\delta_{\max}$（本项目 $\pm3$ mm）、$\beta$ 从 0 退火到 1（2505.11858）、以及"动作幅值课程"（先小后大）。缩放系数应满足 $\delta_{\max}\ll$ 倒角漏斗尺度，否则残差会自己变成"第二个搜索器"。

**(3) base 冻结**：所有工作都冻结 base 参数；梯度不穿过 base（也不要求可微）。但"冻结参数"≠"冻结行为"：Ranjbar 指出内环会把残差当外扰并**反向抵抗**，造成"两个控制器打架"。在本项目里，内环 PD 跟踪的是绝对目标，不会抵抗；真正会抵抗的是 **base 的螺旋搜索**——残差把销推离当前螺旋点后，力变化会改变 base 下一步的 z 冻结/搜索状态。建议：残差训练时冻结螺旋相位与半径增长（或让残差只作用于 base 输出之后、且不回写 base 状态）。

**(4) 防"接管"的工程手段**：① 硬限幅 + 输出 tanh；② 奖励里加 $\lambda_1\|\Delta\|^2+\lambda_2\|\Delta_t-\Delta_{t-1}\|^2$；③ **残差 dropout**（每步 5–10% 置零），强迫 base 单独可完成任务；④ 阶段门控（自由空间段残差置零，接触后才开启）；⑤ 监控"残差饱和度"（$|\Delta|>\ 0.9\delta_{\max}$ 的比例）与"base/残差贡献比"，接管时该比例会单调上升。

**(5) 奖励正则**（在 FORGE 的 keypoint 整形 + 力惩罚之上）：
$$r=r_{\text{FORGE}}-\lambda_c\,\mathrm{ReLU}(F-F_{\text{th}})-\lambda_1\|\Delta\|^2-\lambda_2\|\Delta_t-\Delta_{t-1}\|^2+\lambda_s\,\mathbb{1}[\text{success}]$$
其中 $F$ 用 EMA 后的力范数；穿透/挤压项在仿真里必须加（2505.11858 明确指出仿真允许互穿，不加会被 RL 利用）。

**(6) 为什么样本效率高**：base 已经解析地解决了 $f(s_m)$（下降、恒压、搜索），残差只需学 $g(s_o)$ 的小修正；有效动作维度从 7 降到 2（本项目），探索空间缩小几个数量级；同时稠密整形奖励保留，使 PPO 在 15 Hz、64 env 的设置下可行。Johannink 的对照（残差 vs 纯 RL）与 SPARR 的 +38.4% 是这条论断最直接的证据。

---

## B. 柔顺控制

### B.1 阻抗 / 导纳 / 力位混合

**阻抗控制**（ torque-controlled，关节/末端力矩为输入）把机器人做成"质量-弹簧-阻尼"：
$$M_d(\ddot x-\ddot x_d)+D_d(\dot x-\dot x_d)+K_d(x-x_d)=F_{\text{ext}}$$
即用位置偏差"换"接触力；环境越硬，等效刚度越高，越容易失稳。适用：有力矩接口、需要与环境"共融"的接触任务。

**导纳控制**（位置控制机器人 + 力传感器）是上式的位置形式（把力当输入、位置当输出）：
$$\Delta x=(M_d s^2+D_d s+K_d)^{-1}F_{\text{ext}}\ \Rightarrow\ x_{\text{cmd}}=x_r+\Delta x$$
准静态近似下 $\Delta x\approx F_{\text{ext}}/K_d$。它把力偏差转成位置修正，串在机器人的位置环外面，因此**特别适合工业位置控制臂**（本项目 Franka 的力矩接口在仿真里可用，但真机常以位置/速度接口暴露）。

**力位混合控制**用选择矩阵 $S=\mathrm{diag}(s_1,\dots,s_6)$ 沿任务方向分工：
$$\tau=J^\top\big[S F_d+(I-S)\big(K_p(x_d-x)+K_d(\dot x_d-\dot x)\big)\big]$$
适用：约束方向已知（插接的 z 向力控、xy 位控）。缺点：方向切换处易跳变，需平滑 $S$。

**导纳在位置控制臂上的实现步骤**：① 力传感器读数 → 重力/工具偏置补偿 → 低通（截止 5–20 Hz）；② 在**柔顺坐标系**（孔口系）里解算 $\Delta x$；③ 二阶导纳滤波器离散积分（或用准静态 $F/K$）；④ 限幅 $\|\Delta x\|\le\Delta_{\max}$、限速；⑤ 叠加到参考位姿，送 IK/位置环；⑥ 接触丢失时把 $\Delta x$ 平滑回零（否则会"弹开"）。稳定性要点：环境刚度 $K_e$ 与导纳刚度 $K_d$ 之比决定稳定域，$K_d$ 不能太小、采样率不能太低，力微分噪声必须滤掉。

本项目的 `ScriptedInsertPolicy` 的"力超 30 N 冻结 z"就是一个**只有轴向的导纳环**（力→位置），与 FORGE 的 565 N/m 任务空间 PD（阻抗环）串联，构成"外环导纳 + 内环阻抗"。

### B.2 经典销孔接触力学：RCC、jamming/wedging、倒角与间隙

- **RCC（Remote Center of Compliance）**：Whitney, *Quasi-Static Assembly of Compliantly Supported Rigid Parts*, ASME J. Dyn. Sys. Meas. Control 104(1):65–77, 1982，DOI: <https://doi.org/10.1115/1.3149634>。核心思想：把柔顺中心放在销尖，使横向力产生"绕孔口旋转"的自动对准，而轴向保持高刚度以传力；工程上常用柔性簧片/柔性梁实现"低横向刚度 + 低转动刚度 + 高轴向刚度"（具体结构因实现而异）。
- **准静态判据**：把销视为刚体，接触力只能落在摩擦锥内（半角 $\varphi=\arctan\mu$）。**Jamming** 的定义是存在一组非负接触力（每个都在摩擦锥内）与外力平衡：
  $$\exists\,\lambda_i\ge0:\ F_{\text{app}}+\sum_i\lambda_i\,(n_i+\mu t_i)=0,\quad t_i\perp n_i,\ |t_i|=1$$
  此时销"卡住"、不滑动也不自对齐；若不存在这样的解，销就会滑移/转动进入对齐。**Wedging** 是两点对称接触的特例（倒角/锥面），自锁条件为
  $$\mu\ge\tan\alpha$$
  $\alpha$ 为楔面半角：倒角越"尖"（$\alpha$ 越小）越容易楔死；45° 倒角（$\alpha=45^\circ$）几乎不会自锁。具体 jamming 边界随接触几何与倒角角定义而变，闭式系数**未确认**，工程上以摩擦锥判据 + 仿真为准。
- **倒角/间隙的定量影响**：无倒角时销要"直插"成功必须满足横向误差 $|e|<c_r$（$c_r$ 为单边间隙）；倒角提供漏斗，容差约 $|e|<c_r+w\tan\alpha_c$（$w$ 为倒角轴向长度）。倾斜容差（小角近似）为
  $$\theta_{\max}\approx\arctan\!\big(c_d/h\big)$$
  $c_d$ 为直径间隙、$h$ 为已插入深度。对本项目：$c_r=0.057$ mm，若取 45°、0.3 mm 倒角，漏斗容差约 **0.36 mm**（推算）；$c_d=0.114$ mm、$h=10$ mm 时 $\theta_{\max}\approx0.65^\circ$（推算）。这解释了为什么 0.057 mm 的项目必须靠**螺旋搜索**把误差压进漏斗，而残差只需承担最后几十~几百 µm。

### B.3 Isaac Lab v2.3.0 可用组件（已在本地源码核对）

| 组件 | 文件（GitHub v2.3.0） | 关键参数 | 用法/坑 |
|---|---|---|---|
| `OperationalSpaceController` | [`controllers/operational_space.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/controllers/operational_space.py) | `target_types`（`pose_abs`/`pose_rel`/`wrench_abs`）、`motion_control_axes_task`、`contact_wrench_control_axes_task`、`impedance_mode`（`fixed`/`variable_kp`/`variable`）、`motion_stiffness_task`、`motion_damping_ratio_task`、`motion_stiffness_limits_task`、`contact_wrench_stiffness_task`、`nullspace_control` | 纯计算类，需自己喂 Jacobian/质量阵/位姿/力；`action_dim` = 目标维 + 6（`variable_kp`）或 +12（`variable`）；接触力闭环只用前 3 维（转动不可测） |
| `JointImpedanceController` | [`controllers/joint_impedance.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/controllers/joint_impedance.py) | `command_type`（`p_abs`/`p_rel`）、`stiffness`、`damping_ratio`、`stiffness_limits`（默认 0–300）、`gravity_compensation` | 关节级，适合"柔顺外壳"，但任务空间语义弱 |
| `DifferentialIKController` | [`controllers/differential_ik.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/controllers/differential_ik.py) | `command_type="pose"`、`use_relative_mode`、`ik_method`（`dls` 等） | 输出关节目标给位置环；常与导纳外环配合 |
| `ContactSensor` | [`sensors/contact_sensor/contact_sensor.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py) | `track_pose`、`track_contact_points`、`force_threshold`、`filter_prim_paths_expr`；数据 `net_forces_w`、`force_matrix_w` | **只报法向力**；需在 spawner 开 `activate_contact_sensors`；过滤是"一对多"，多体需建多个 sensor |
| `RigidContactView` | —（**不是 Isaac Lab 类**） | 由 `ContactSensor.contact_physx_view` 暴露，类型是 `omni.physics.tensors.impl.api.RigidContactView` | 底层 PhysX 视图，需按 Isaac Sim API 处理张量；一般直接用 `ContactSensorData` 即可 |
| `root_physx_view.get_link_incoming_joint_force()` | [`articulation_data.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/assets/articulation/articulation_data.py) `body_incoming_joint_wrench_b` | 返回 `(N, bodies, 6)`，**父体坐标系**的关节反力 | FORGE 就是用它取 6D wrench，再 `change_FT_frame` 转到孔口系并 EMA；注意含重力/惯性项，需减基线 |

**最小变阻抗改法（两条路）**：

1. **OSC 路线**（新环境/新任务）：`impedance_mode="variable_kp"`，`motion_stiffness_limits_task=(100,2000)`，动作 = `[pose_target(7), kp(6)]`（`action_dim=13`），调用 `osc.set_command(cmd)`；`motion_damping_ratio_task` 保持 1.0 即自动临界阻尼（[官方教程](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/docs/source/tutorials/05_controllers/run_osc.rst) 即此用法，含 `wrench_abs` 的 z 向力控示例）。
2. **FORGE 直连路线（推荐，改动最小）**：FORGE 的 [`factory_control.compute_dof_torque`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py) 已经接收 **per-env 的 `task_prop_gains` 张量**，且 `forge_env.py` 每回合对它做 ±41% 随机化。因此只需：动作从 2 维扩到 8 维（$\Delta x,\Delta y,\Delta\log k_p^{(6)}$）→ `kp = kp0*exp(clip(Δlog kp, -0.7, 0.7))` 并 clip 到 `[200,1500]` → `kd = 2*sqrt(kp)` → 传给 `compute_dof_torque`。注意：死区是在 **wrench 域**做的，kp 变了等效位置死区也变（$dz/k_p$），变阻抗训练时必须同步随机化死区，否则策略会学到"靠调 kp 绕过死区"的退化解。

### B.4 参数整定经验

| 项 | 合理量级 | 说明 |
|---|---|---|
| 平动刚度 $k_p$ | 300–1500 N/m（FORGE 565） | 精密插接偏低刚度更安全，但太低会被死区/摩擦吞掉 |
| 转动刚度 | 20–60 Nm/rad（FORGE 28） | 与销长、倒角容差匹配；过大易 jamming |
| 阻尼比 $\zeta$ | 0.7–1.0（临界） | $k_d=2\sqrt{k_p}\,\zeta$；FORGE 取 $\zeta=1$ |
| 控制频率 | 策略 10–50 Hz，力矩/阻抗环 ≥ 100 Hz | 本项目 15/120 Hz；再低会让接触力"打夯" |
| 力低通 | 5–20 Hz（一阶 EMA，$\alpha$ 0.25–0.75 @120 Hz；FORGE 取 0.25≈5 Hz） | 太低→相位滞后，太高→力噪声进策略 |
| 死区 | 0.2–1 N / 0.1–0.3 Nm（精密段） | FORGE 默认 5 N/1 Nm 对 0.057 mm 太大（§1） |
| 摩擦补偿 | 库仑 0.5–2 N + 黏滞项，按关节辨识 | 真机上比"加大 kp"更有效；仿真里可用死区近似 |
| 延迟 | 1–2 个策略步 | 训练时随机化，评估时固定 |

**sim-to-real 迁移**：① 先在真机辨识 $k_p,k_d$、死区、摩擦与力偏置（静态加载/自由振动）；② 训练时对 $k_p$ 做 ±30–50% 随机化、死区 $U[0,dz]$、EMA 因子与力噪声（FORGE 的做法，见 `forge_env_cfg.py`）；③ 力通道做零偏标定与温度漂移补偿；④ 把"策略→执行"的延迟显式加进仿真。FORGE 的增益随机化实验（[arXiv:2408.04587](https://arxiv.org/abs/2408.04587) Fig. 5）表明：同一策略在不同部署增益下仍能靠力反馈调制动作，这正是变阻抗训练最值得复用的鲁棒性来源。

### B.5 本项目实现清单（按依赖顺序，14 条）

1. **打通力通道**：`get_link_incoming_joint_force()` → 减空载基线 → `change_FT_frame` 转到孔口系 → EMA($\alpha$=0.2–0.3)。验收：空载 $\|F\|<0.5$ N，轻触时单调。
2. **安全层**：力上限（如 60 N）触发 z 冻结 + 残差清零 + 回合终止；记录峰值力。验收：压溃试验不穿透孔底。
3. **标定内环**：确认 565/28 N/m、$k_d=2\sqrt{k_p}$；把死区改成可配置（默认 5/1，精密段 1/0.2）。验收：阶跃响应无持续振荡。
4. **解析基线单测**：自由下降 25 步、恒压 30 N、螺旋 2 µm→2 mm / 18 µm 步进；无残差下测成功率 vs 初始横向误差曲线。验收：$|e|<0.3$ mm 时 >90%。
5. **残差接口**：2 维、`clamp(-1,1)`、$s=3$ mm、不经过动作 EMA、每回合清零；观测加 `residual_action`。验收：$\Delta=0$ 时与官方 FORGE 行为逐位一致。
6. **奖励与指标**：FORGE keypoint + 力惩罚 + $\lambda_1\|\Delta\|^2+\lambda_2\|\Delta_t-\Delta_{t-1}\|^2$；日志加残差饱和度、力峰值、成功率。验收：TensorBoard 能看到"残差→0 但成功率上升"。
7. **固定阻抗残差 PPO**：64 env、1500 iter，先在小间隙随机化（±0.1 mm）上训。验收：成功率 > 基线 10 个百分点。
8. **域随机化**：$k_p$ ±41%、死区 $U[0,dz]$、EMA 0.025–0.1、力噪声 1 N、观测噪声（沿用 FORGE）。验收：随机化后成功率下降 <5 个百分点。
9. **消融**：base-only / residual-only / residual+正则 / 无 dropout。验收：证明每一项都有贡献。
10. **变阻抗（推理期手调）**：固定策略，按阶段调度 $k_p$（接近 300 → 接触 565 → 插入后 800），观察成功率与峰值力。验收：峰值力下降 ≥20% 且成功率不降。
11. **把 kp 拼进动作**：动作 2→8 维（$\Delta\log k_p$），限幅 ±0.7、clip `[200,1500]`、$k_d=2\sqrt{k_p}$、同步随机化死区。验收：训练稳定，kp 轨迹可解释（接触段降、插入段升）。
12. **联合训练**：残差 + $\Delta\log k_p$ 同时学，加 kp 变化率惩罚与"低 kp 早停"保护。验收：成功率 ≥ 固定阻抗版，峰值力更低。
13. **sim-to-real**：真机辨识 $k_p,k_d$、力偏置、延迟；把辨识值写进部署配置，仿真侧随机化包住它。验收：真机 30 次试验成功率 ≥ 仿真 −15 个百分点。
14. **写论文图表**：成功率 vs 间隙（0.057/0.114/0.5 mm）、vs 初始误差、vs 力峰值；残差幅值热图；变阻抗 kp 曲线。验收：三张主图 + 两张消融图。

---

## 参考文献

1. Johannink T., Bahl S., Nair A., Luo J., Kumar A., Loskyll M., Ojea J.A., Solowjow E., Levine S. *Residual Reinforcement Learning for Robot Control.* ICRA 2019. arXiv: <https://arxiv.org/abs/1812.03201>；DOI: <https://doi.org/10.1109/ICRA.2019.8794127>
2. Silver T., Allen K., Tenenbaum J., Kaelbling L. *Residual Policy Learning.* arXiv: <https://arxiv.org/abs/1812.06298>
3. Ranjbar A., Vien N.A., Ziesche H., Boedecker J., Neumann G. *Residual Feedback Learning for Contact-Rich Manipulation Tasks with Uncertainty.* arXiv: <https://arxiv.org/abs/2106.04306>
4. Davchev T., Luck K.S., Burke M., Meier F., Schaal S. *Residual Learning from Demonstration: Adapting DMPs for Contact-rich Manipulation.* arXiv: <https://arxiv.org/abs/2008.07682>
5. Marougkas I., Ramesh D.M., Doerr J.H., Granados E., Sivaramakrishnan A., Boularias A., Bekris K.E. *Integrating Model-based Control and RL for Sim2Real Transfer of Tight Insertion Policies.* arXiv: <https://arxiv.org/abs/2505.11858>
6. Guo Y., Akinola I., Johannsmeier L., Hadfield H., Gupta A. *SPARR: Simulation-based Policies with Asymmetric Real-world Residuals for Assembly.* arXiv: <https://arxiv.org/abs/2602.23253>
7. Liu T., Patamia R.A., Champion B., Cosgun A., Dazeley R. *Hybrid Residual Reinforcement Learning for Contact-Rich Robotic Book Insertion.* arXiv: <https://arxiv.org/abs/2609.19962>
8. Abuibaid K., Sidorenko A., Wagner A., Ruskowski M. *Learning-Based Strategy for Composite Robot Assembly Skill Adaptation.* arXiv: <https://arxiv.org/abs/2604.06949>
9. Tang B., Lin M.A., Akinola I., et al. *IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality.* arXiv: <https://arxiv.org/abs/2305.17110>
10. Whitney D.E. *Quasi-Static Assembly of Compliantly Supported Rigid Parts.* ASME J. Dyn. Sys. Meas. Control, 104(1):65–77, 1982. DOI: <https://doi.org/10.1115/1.3149634>
11. Pitchandi S., Subramanian S., Irulappan M. *Insertion force analysis of compliantly supported peg-in-hole assembly.* Assembly Automation, 37(3):285–295, 2017. DOI: <https://doi.org/10.1108/AA-12-2016-167>
12. Noseworthy M., Tang B., Wen B., Handa A., Kessens C., Roy N., Fox D., Ramos F., Narang Y., Akinola I. *FORGE: Force-Guided Exploration for Robust Contact-Rich Manipulation under Uncertainty.* arXiv: <https://arxiv.org/abs/2408.04587>
13. Isaac Lab v2.3.0 源码：<https://github.com/isaac-sim/IsaacLab/tree/v2.3.0>；OSC 教程：<https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/docs/source/tutorials/05_controllers/run_osc.rst>
