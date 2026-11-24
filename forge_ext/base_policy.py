# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""FORGE PegInsert 的解析柔顺基座策略（Scripted Compliance Baseline）。

设计要点
--------
1. 动作语义与 FORGE 一致：``a[0:3]`` 是"相对孔口（fixed asset tip）的绝对目标位置"
   ，归一化后乘以 ``ctrl.pos_action_bounds``（0.05 m）；``a[5]`` 控制 yaw。
2. 分两段：自由空间快速下降 → 接触后恒压 + 螺旋搜索。
3. 恒压通过"力超阈值则冻结 z 指令"实现，等价于一个只有轴向的导纳控制
   （admittance：力 → 位置修正），与工厂环境里的任务空间 PD（阻抗环）串联，
   构成"外环导纳 + 内环阻抗"的经典双层柔顺结构。
4. 该策略同时是阶段三残差 RL 的 base policy：残差只修正 xy 两个方向。

单位约定：位置 m，力 N，角度 rad。控制频率 15 Hz（120 Hz 物理 / decimation 8）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InsertBaseCfg:
    """解析插入基线的参数（可按任务微调，见教程"参数整定"一节）。"""

    # --- 几何 ---
    z_start: float = 0.047
    """起始高度（相对孔口顶面）。等于 PegInsert.hand_init_pos[2]。"""

    z_press_start: float = 0.003
    """自由空间下降的终点高度，再往下就进入接触阶段。"""

    z_min: float = -0.027
    """允许的最低 z 指令。孔深 25 mm，-0.027 保证 peg 底部低于孔底 2 mm（成功判据要求 z < -0.024）。"""

    approach_steps: int = 25
    """自由空间下降占用的策略步数。"""

    press_step: float = 5.0e-4
    """接触阶段每步 z 指令增量（m）。15 Hz 下约 7.5 mm/s。"""

    # --- 力阈值 ---
    contact_force: float = 4.0
    """判定"已接触"的力范数阈值（N）。用于触发螺旋搜索。"""

    hold_force: float = 30.0
    """力范数超过该值就冻结 z 指令（防止压溃），即轴向力限幅。"""

    # --- 螺旋搜索 ---
    search_radius_min: float = 2.0e-4
    search_radius_max: float = 2.0e-3
    search_radial_steps: int = 100
    """半径从 min 线性长到 max 所需的步数（=> 每步约 18 µm，必须小于倒角量级）。"""

    search_phase_step: float = 0.5
    """每步极角增量（rad）。0.5 rad/step @15 Hz ≈ 1.2 Hz 旋转。"""

    # --- 动作归一化 ---
    pos_bound: float = 0.05
    """FORGE 的 pos_action_bounds，用于把米转换为 [-1, 1] 归一化动作。"""

    yaw_action: float = 1.0 / 3.0
    """对应 yaw=0 的归一化取值（见 ForgeEnv._apply_action 的映射：a5=1/3 → yaw 0）。"""

    # --- 基座自身随机化（做残差时必开，做阶段二基线时保持 0） ---
    phase_noise: float = 0.0
    """螺旋初相位的均匀随机幅度（rad）。0 = 确定性，π = 全相位随机。

    为什么重要：螺旋是**时间确定性**的，如果每个环境相位都一样，残差策略会过拟合
    "第 k 步该往哪边推"，而不是学"看到力就往哪边修正"。给相位随机化后，
    基座变成一族不同相位的搜索，迫使残差学到相位无关的修正律。"""

    press_step_rand: float = 0.0
    """下压速度的相对随机幅度（0 = 不加噪声，0.3 = ±30% 均匀分布）。"""

    radius_max_rand: float = 0.0
    """螺旋最大半径的相对随机幅度（0 = 固定，0.5 = ±50% 均匀分布）。"""


class ScriptedInsertPolicy:
    """带状态的解析插入策略，可向量化到任意 num_envs。

    用法::

        policy = ScriptedInsertPolicy(cfg, num_envs=128, device="cuda:0")
        # 每个策略步：
        action = policy.act(force_norm, )          # [N, 7] 归一化动作
        env.step(action)
        # 环境 reset 后（或只 reset 部分 env）：
        policy.reset(env_ids)
    """

    def __init__(self, cfg: InsertBaseCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.t = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.z_cmd = torch.full((num_envs,), cfg.z_start, device=device)
        self.searching = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.search_t = torch.zeros(num_envs, dtype=torch.long, device=device)
        # 每环境随机化（0 = 关闭），见 InsertBaseCfg 中的说明
        self.phase_off = torch.zeros(num_envs, device=device)
        self.press_scale = torch.ones(num_envs, device=device)
        self.radius_scale = torch.ones(num_envs, device=device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None):
        """复位内部状态；``env_ids=None`` 表示全部复位。"""
        cfg = self.cfg
        idx = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else int(env_ids.numel())

        def _rand_scale(amplitude: float) -> torch.Tensor:
            if amplitude <= 0.0:
                return torch.ones(n, device=self.device)
            return 1.0 + amplitude * (2.0 * torch.rand(n, device=self.device) - 1.0)

        self.t[idx] = 0
        self.z_cmd[idx] = cfg.z_start
        self.searching[idx] = False
        self.search_t[idx] = 0
        self.press_scale[idx] = _rand_scale(cfg.press_step_rand) * cfg.press_step
        self.radius_scale[idx] = _rand_scale(cfg.radius_max_rand)
        if cfg.phase_noise > 0.0:
            self.phase_off[idx] = cfg.phase_noise * (2.0 * torch.rand(n, device=self.device) - 1.0)
        else:
            self.phase_off[idx] = 0.0

    @torch.no_grad()
    def act(self, force_norm: torch.Tensor) -> torch.Tensor:
        """根据当前接触力范数计算归一化动作。

        Args:
            force_norm: 形状 ``[N]``，末端力范数（N）。

        Returns:
            形状 ``[N, 7]`` 的归一化动作，可直接喂给 FORGE 环境。
        """
        cfg = self.cfg
        t = self.t.float()

        # --- 1) z 指令：自由空间线性下降，之后按 press_step 递减 ---
        approach_frac = (t / max(cfg.approach_steps, 1)).clamp(max=1.0)
        z_approach = cfg.z_start + (cfg.z_press_start - cfg.z_start) * approach_frac
        z_press = (self.z_cmd - self.press_scale).clamp(min=cfg.z_min, max=cfg.z_start)
        z_target = torch.where(t < cfg.approach_steps, z_approach, z_press)

        # --- 2) 力限幅：力过大则冻结 z（轴向导纳） ---
        freeze = force_norm > cfg.hold_force
        z_target = torch.where(freeze, self.z_cmd, z_target)
        self.z_cmd = z_target.clamp(min=cfg.z_min, max=cfg.z_start)

        # --- 3) 螺旋搜索：一旦接触就常开，半径随步数缓慢长大 ---
        contact = force_norm > cfg.contact_force
        self.searching = self.searching | contact
        self.search_t = torch.where(self.searching, self.search_t + 1, torch.zeros_like(self.search_t))

        frac = (self.search_t.float() / max(cfg.search_radial_steps, 1)).clamp(max=1.0)
        # 每环境随机化最大半径；注意 radius_min 是"起点"，不缩放
        radius_max = cfg.search_radius_max * self.radius_scale
        radius = cfg.search_radius_min + (radius_max - cfg.search_radius_min) * frac
        theta = self.search_t.float() * cfg.search_phase_step + self.phase_off
        xy = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
        xy = xy * self.searching.float().unsqueeze(-1)

        # --- 4) 组装归一化动作 ---
        action = torch.zeros((self.num_envs, 7), device=self.device)
        action[:, 0:2] = xy / cfg.pos_bound
        action[:, 2] = self.z_cmd / cfg.pos_bound
        action[:, 5] = cfg.yaw_action
        self.t += 1
        return action
