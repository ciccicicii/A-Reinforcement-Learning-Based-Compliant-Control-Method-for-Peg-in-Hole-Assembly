# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""FORGE 扩展任务包（残差插接）。

用法：在 Isaac Sim 启动之后 ``import forge_ext``，即可通过 ``gym.make`` 创建：

  - ``Isaac-Forge-PegInsert-Residual-v0``：解析柔顺基线 + 二维残差策略

本包不修改 IsaacLab 源码，放在毕设仓库里，靠 ``sys.path`` 引入。
"""

import gymnasium as gym

_REGISTERED = False


def register():
    """注册残差任务（幂等）。"""
    global _REGISTERED
    if _REGISTERED:
        return

    from .logged_env import ForgeLoggedEnv
    from .residual_env import ForgeResidualCfg, ForgeResidualEnv

    from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg

    # 与官方任务等价、只是把成功率写进 extras["log"]（rl_games 才会记录）
    gym.register(
        id="Isaac-Forge-PegInsert-Logged-v0",
        entry_point="forge_ext.logged_env:ForgeLoggedEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": ForgeTaskPegInsertCfg,
            "rl_games_cfg_entry_point": "isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml",
        },
    )

    gym.register(
        id="Isaac-Forge-PegInsert-Residual-v0",
        entry_point="forge_ext.residual_env:ForgeResidualEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": ForgeResidualCfg,
            "rsl_rl_cfg_entry_point": "forge_ext.agents:ForgeResidualPPORunnerCfg",
        },
    )
    _REGISTERED = True


register()

__all__ = ["register", "ForgeResidualEnv"]
