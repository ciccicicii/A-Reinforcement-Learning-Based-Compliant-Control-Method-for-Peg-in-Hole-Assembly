# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""把 FORGE 残差插接任务接进 RLinf（阶段三 3.5 节）。

RLinf 官方的 Isaac Lab 适配层由三部分组成：

  * ``rlinf/envs/sim/isaaclab/isaaclab_env.py``：``IsaaclabBaseEnv``，负责指标统计、
    动作分块、auto-reset 语义；
  * ``rlinf/envs/sim/isaaclab/venv.py``：``SubProcIsaacLabEnv``，把 Isaac Sim 放在
    **独立子进程**里跑（spawn），父进程只通过 pipe/queue 收发张量；
  * ``rlinf/envs/sim/isaaclab/tasks/*.py``：具体任务，负责建环境（``_make_env_function``）
    和把 Isaac Lab 观测包装成 RLinf 的 ``{"states": ..., "main_images": ...}`` 字典。

官方只注册了 ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0`` 一个任务，且
``_wrap_obs`` 依赖两路相机。本文件为 FORGE 残差任务补一个**纯状态**任务类：RLinf 的
``MLPPolicy`` 只读 ``env_obs["states"]``（见 ``rlinf/models/embodiment/mlp_policy/mlp_policy.py``
的 ``preprocess_env_obs``），所以不需要相机、不需要 VLA 权重。

放置方法（在 RLinf 仓库根目录）::

    cp /path/to/RL/rlinf_adapter/forge_peginsert_env.py \
       rlinf/envs/sim/isaaclab/tasks/forge_peginsert.py

然后在 ``rlinf/envs/sim/isaaclab/__init__.py`` 里加两行::

    from .tasks.forge_peginsert import IsaaclabForgeResidualEnv
    REGISTER_ISAACLAB_ENVS = {
        ...
        "Isaac-Forge-PegInsert-Residual-v0": IsaaclabForgeResidualEnv,
    }

注意：本文件**未在 RLinf v0.4 上实测**（见教程 3.5 节的验证步骤与降级方案）。
"""

from __future__ import annotations

import gymnasium as gym
import torch

from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv


class IsaaclabForgeResidualEnv(IsaaclabBaseEnv):
    """FORGE PegInsert 残差任务的 RLinf 适配器（state-only）。"""

    # ------------------------------------------------------------------ 建环境
    def _make_env_function(self):
        """返回一个在子进程里调用的闭包：启动 Isaac Sim + 创建 Isaac Lab 环境。"""

        def make_env_isaaclab():
            import os

            # 子进程不要尝试连 X11，否则会报 GLX 错误
            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            # 纯状态任务：不需要相机，省显存（8 GB 卡必须这样）
            sim_app = AppLauncher(headless=True, enable_cameras=False).app

            import isaaclab_tasks  # noqa: F401  注册官方任务
            import forge_ext  # noqa: F401  注册 Isaac-Forge-PegInsert-Residual-v0

            from isaaclab_tasks.utils import load_cfg_from_registry

            env_cfg = load_cfg_from_registry(self.isaaclab_env_id, "env_cfg_entry_point")
            env_cfg.seed = self.seed
            env_cfg.scene.num_envs = self.cfg.init_params.num_envs

            env = gym.make(self.isaaclab_env_id, cfg=env_cfg).unwrapped
            return env, sim_app

        return make_env_isaaclab

    # ------------------------------------------------------------------ 观测包装
    def _wrap_obs(self, obs):
        """把 Isaac Lab 的 ``{"policy": [...], "critic": [...]}`` 映射成 RLinf 的观测字典。

        ``states`` 直接就是残差策略的 19 维输入：
        ``[fingertip_pos_rel_fixed(3), fingertip_quat(4), ee_linvel(3), ee_angvel(3),
        ft_force(3), force_threshold(1), residual_action(2)]``
        """
        states = obs["policy"].to(torch.float32)
        env_obs = {
            "states": states,
            # 纯状态任务不产生图像；如后续 RLinf 版本强依赖这两个键，给占位即可
            "task_descriptions": [self.task_description] * self.num_envs,
        }
        return env_obs
