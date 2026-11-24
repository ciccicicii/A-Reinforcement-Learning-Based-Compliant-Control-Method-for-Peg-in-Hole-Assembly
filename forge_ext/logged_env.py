# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""给官方 FORGE 任务加一个"能看见成功率"的壳（仅补日志，不改动力学）。

为什么需要它
------------
Factory/FORGE 把指标写在 ``extras["successes"]``、``extras["logs_rew_*"]`` 这一层，
而 rl_games 只记录 ``extras["episode"]``。Isaac Lab 的 rl_games 封装里有一句

    if "log" in extras: extras["episode"] = extras.pop("log")

也就是说：**官方 FORGE 任务用 rl_games 训练时，成功率不会出现在 TensorBoard 里**，
只能靠 ``play.py`` 肉眼看，或用 ``demos/05_eval_success.py`` 定期评估。

本文件用子类把这些标量转存到 ``extras["log"]``，不改任何物理参数、奖励或随机化，
因此训练出的策略与官方任务**完全等价**，只是曲线可见。

注册 id：``Isaac-Forge-PegInsert-Logged-v0``。
"""

import torch

from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

_SCALAR_PREFIXES = ("logs_rew_", "early_term_", "successes", "success_times", "force")


class ForgeLoggedEnv(ForgeEnv):
    """与 ``ForgeEnv`` 完全相同，只额外把指标写进 ``extras["log"]``。"""

    def _log_factory_metrics(self, rew_dict, curr_successes):
        super()._log_factory_metrics(rew_dict, curr_successes)

        log = self.extras.setdefault("log", {})
        for key, value in list(self.extras.items()):
            if not key.startswith(_SCALAR_PREFIXES):
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    log[key] = value.reshape(())
            elif isinstance(value, (int, float)):
                log[key] = torch.tensor(float(value), device=self.device)
