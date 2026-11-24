# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""残差 FORGE 插接任务：解析柔顺基线 + 二维横向残差策略。

与官方 ``Isaac-Forge-PegInsert-Direct-v0`` 的差别只有三点：

1. 策略动作从 7 维变成 2 维（xy 横向偏移，单位 m，经 ``residual_pos_scale`` 缩放）；
2. 实际下发给机器人的动作 = 脚本基座策略（``ScriptedInsertPolicy``）+ 残差；
3. 奖励在 FORGE 原有奖励上增加两项正则：残差幅值、残差变化率。

其余（场景、随机化、任务空间 PD 控制器、成功判据、力观测）完全复用官方实现，
因此可以直接与官方 FORGE 训练结果做对照。

已知上游问题与本文处理
----------------------
1. **速度观测 bug（IsaacLab issue #8874，v2.3.0 存在）**：`FactoryEnv._compute_intermediate_values`
   先用真值位置算速度并覆盖 `prev_fingertip_pos`，`ForgeEnv` 随后再用含噪位置差分，结果
   `ee_linvel`/`ee_angvel` 实际是"噪声 / dt"，24 维观测里有 6 维是废的。本实现在
   `cfg.fix_velocity_obs=True`（默认）时用真值历史重新差分，修掉这 6 维；
   设为 False 可复现上游原始行为，便于做对照。
2. **力坐标系（issue #3513）**：`change_FT_frame` 传入的是单位四元数，只有平移项生效，
   因此 `ft_force` 实际仍在**世界系**（不是孔轴系）。圆孔绕 z 轴对称、且固定件只随机 yaw，
   所以对本任务无实质影响，但写论文时不要声称"力已转到孔口系"。
"""

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg

from .base_policy import InsertBaseCfg, ScriptedInsertPolicy


@configclass
class ForgeResidualCfg(ForgeTaskPegInsertCfg):
    """残差任务的配置。"""

    # --- 残差接口 ---
    residual_dim: int = 2
    """残差动作维度：2 = 只在孔口坐标系下修正 x / y。"""

    residual_pos_scale: float = 0.003
    """每单位残差对应的横向位移（m）。裁剪到 [-1,1] 后即 ±3 mm。"""

    residual_penalty_scale: float = 0.05
    """残差幅值惩罚系数（鼓励残差尽量小，让基线完成大部分工作）。"""

    residual_smooth_scale: float = 0.2
    """残差变化率惩罚系数（抑制高频抖动，利于 sim-to-real）。"""

    # --- 基座策略参数（残差训练时给基座加随机化，避免残差过拟合螺旋相位） ---
    base: InsertBaseCfg = InsertBaseCfg(
        phase_noise=3.141592653589793,  # 初相位全域随机
        press_step_rand=0.3,            # 下压速度 ±30%
        radius_max_rand=0.5,            # 螺旋最大半径 ±50%
    )

    # --- 已知问题修复开关 ---
    fix_velocity_obs: bool = True
    """True = 用真值位置重新差分速度（修 IsaacLab issue #8874 的 6 维废观测）。"""

    # --- gym 空间（仅用于打印信息；训练按实际张量维度） ---
    action_space: int = 2
    observation_space: int = 19
    state_space: int = 56

    # --- 观测：把 prev_actions 换成 2 维残差 ---
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
        "residual_action",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "force_threshold",
        "pos_threshold",
        "rot_threshold",
        "residual_action",
    ]


class ForgeResidualEnv(ForgeEnv):
    """解析基线 + 残差策略的 FORGE 插接环境。"""

    cfg: ForgeResidualCfg

    def __init__(self, cfg: ForgeResidualCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.base_policy = ScriptedInsertPolicy(cfg.base, self.num_envs, self.device)
        self.residual = torch.zeros((self.num_envs, cfg.residual_dim), device=self.device)
        self.prev_residual = torch.zeros_like(self.residual)
        self.base_action = torch.zeros((self.num_envs, 7), device=self.device)

        # 记录量，供评估脚本与 TensorBoard 使用
        self.force_norm = torch.zeros(self.num_envs, device=self.device)
        self.peak_force = torch.zeros(self.num_envs, device=self.device)

        # 真值位姿历史（用于修正速度观测，见模块 docstring）
        self._true_prev_pos: torch.Tensor | None = None
        self._true_prev_quat: torch.Tensor | None = None

    # ---------------------------------------------------------------- 速度观测
    def _compute_intermediate_values(self, dt):
        """修 IsaacLab #8874：用真值位姿差分速度，而不是"含噪位姿 / dt"。"""
        super()._compute_intermediate_values(dt)

        if not self.cfg.fix_velocity_obs:
            return

        if self._true_prev_pos is None:
            self._true_prev_pos = self.fingertip_midpoint_pos.clone()
            self._true_prev_quat = self.fingertip_midpoint_quat.clone()
            return

        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self._true_prev_pos) / dt

        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self._true_prev_quat)
        )
        rot_diff_quat = rot_diff_quat * torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        self.ee_angvel_fd = math_utils.axis_angle_from_quat(rot_diff_quat) / dt
        self.ee_angvel_fd[:, 0:2] = 0.0  # 与 FORGE 一致：只有 yaw 受控

        self._true_prev_pos = self.fingertip_midpoint_pos.clone()
        self._true_prev_quat = self.fingertip_midpoint_quat.clone()

    # ---------------------------------------------------------------- 动作
    def _pre_physics_step(self, action: torch.Tensor):
        """把 2 维残差叠加到脚本基座动作上，得到 7 维 FORGE 动作。"""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # 1) 基座动作：用上一步测得的力做导纳/限幅判断
        self.force_norm[:] = torch.linalg.norm(self.force_sensor_smooth[:, 0:3], dim=-1)
        base = self.base_policy.act(self.force_norm)

        # 2) EMA 平滑（沿用 FORGE 的动作平滑，只作用于基座，避免基座抖动）
        self.base_action = self.ema_factor * base + (1.0 - self.ema_factor) * self.base_action

        # 3) 叠加残差（残差不经过 EMA，响应更快）
        residual = torch.clamp(action.to(self.device), -1.0, 1.0)
        self.residual = residual
        self.actions = self.base_action.clone()
        scale = self.cfg.residual_pos_scale / self.cfg.ctrl.pos_action_bounds[0]
        self.actions[:, 0:2] = torch.clamp(
            self.actions[:, 0:2] + residual * scale, -1.0, 1.0
        )

        # 4) 统计
        self.peak_force[:] = torch.maximum(self.peak_force, self.force_norm)

    # ---------------------------------------------------------------- 观测
    def _get_observations(self):
        obs_dict, state_dict = self._get_factory_obs_state_dict()
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        obs_dict.update(
            {
                "fingertip_pos": self.noisy_fingertip_pos,
                "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
                "fingertip_quat": self.noisy_fingertip_quat,
                "ft_force": self.noisy_force,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "residual_action": self.residual,
            }
        )
        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "ft_force": self.force_sensor_smooth[:, 0:3],
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "residual_action": self.residual,
            }
        )

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order)
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order)
        return {"policy": obs_tensors, "critic": state_tensors}

    # ---------------------------------------------------------------- 奖励
    def _get_rewards(self) -> torch.Tensor:
        """FORGE 的关键点整形 + 接触力惩罚 + 残差正则。"""
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=False
        )
        rew_dict, _ = self._get_factory_rew_dict(curr_successes)

        rew = (
            rew_dict["kp_baseline"]
            + rew_dict["kp_coarse"]
            + rew_dict["kp_fine"]
            + rew_dict["curr_engaged"]
            + rew_dict["curr_success"]
        )

        # 接触力惩罚（与 FORGE 一致：只惩罚超过每环境随机阈值的部分）
        contact_penalty = torch.nn.functional.relu(self.force_norm - self.contact_penalty_thresholds)
        rew = rew - self.cfg_task.contact_penalty_scale * contact_penalty

        # 残差正则
        res_sq = self.residual.pow(2).sum(dim=-1)
        res_delta = (self.residual - self.prev_residual).pow(2).sum(dim=-1)
        rew = rew - self.cfg.residual_penalty_scale * res_sq - self.cfg.residual_smooth_scale * res_delta
        self.prev_residual.copy_(self.residual)

        # 日志
        self.extras["logs_rew_residual_abs"] = res_sq.mean()
        self.extras["logs_rew_contact_penalty"] = contact_penalty.mean()
        self.extras["logs_force_norm"] = self.force_norm.mean()
        self._log_factory_metrics(rew_dict, curr_successes)
        self.prev_actions = self.actions.clone()
        return rew

    # ---------------------------------------------------------------- 复位
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.base_policy.reset(env_ids)
        self.residual[env_ids] = 0.0
        self.prev_residual[env_ids] = 0.0
        self.base_action[env_ids] = 0.0
        self.force_norm[env_ids] = 0.0
        self.peak_force[env_ids] = 0.0
        # 复位后位姿会跳变，清空速度差分历史，避免产生一次虚假的巨大速度
        if self._true_prev_pos is not None:
            self._true_prev_pos[env_ids] = self.fingertip_midpoint_pos[env_ids]
            self._true_prev_quat[env_ids] = self.fingertip_midpoint_quat[env_ids]
