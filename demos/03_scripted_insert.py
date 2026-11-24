# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""解析柔顺基线：脚本化插接（阶段二第 2 步，也是阶段三残差 RL 的 base policy）。

在**官方** ``Isaac-Forge-PegInsert-Direct-v0`` 上跑，不加载任何策略网络：
  - 自由空间：从 47 mm 高度线性下降到孔口上方 3 mm；
  - 接触阶段：恒速下压 + 力超阈值冻结 z（轴向导纳）；
  - 一旦测到接触力，叠加慢速增长的螺旋搜索（横向 ±2 mm）。

产出的成功率就是"不训练能到多少"的基线，后面残差 RL 的增益以此为参照。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab
    ./isaaclab.sh -p ../demos/03_scripted_insert.py --num_envs 32 --episodes 10
    ./isaaclab.sh -p ../demos/03_scripted_insert.py --num_envs 64 --episodes 20 --headless

注意：螺旋搜索的默认参数偏保守（成功率不会很高），这正是残差策略要改进的空间。
"""

import argparse
import csv
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FORGE PegInsert 解析柔顺基线")
parser.add_argument("--num_envs", type=int, default=32, help="并行环境数")
parser.add_argument("--episodes", type=int, default=10, help="评估回合数（每回合所有环境同步重置）")
parser.add_argument("--out_dir", type=str, default=None, help="CSV 输出目录")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.utils import parse_env_cfg

# 让 `forge_ext` 可被导入（脚本位于 RL/demos/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge_ext.base_policy import InsertBaseCfg, ScriptedInsertPolicy  # noqa: E402

TASK_ID = "Isaac-Forge-PegInsert-Direct-v0"


def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK_ID, cfg=env_cfg)
    u = env.unwrapped

    base_cfg = InsertBaseCfg()
    policy = ScriptedInsertPolicy(base_cfg, u.num_envs, u.device)

    print(f"[INFO] 控制频率 {1.0 / (u.cfg.sim.dt * u.cfg.decimation):.1f} Hz, 每回合 {u.max_episode_length} 步")
    print(f"[INFO] 基线参数 {base_cfg}")

    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"scripted_baseline_{datetime.now():%Y%m%d_%H%M%S}.csv")
    fh = open(csv_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["episode", "env_id", "success", "first_success_step", "final_z_mm", "peak_force_N"])

    env.reset()
    policy.reset()

    all_success = []
    all_success_step = []
    with torch.inference_mode():
        for ep in range(args_cli.episodes):
            success_any = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
            first_step = torch.full((u.num_envs,), -1, dtype=torch.long, device=u.device)
            peak = torch.zeros(u.num_envs, device=u.device)
            final_z = torch.zeros(u.num_envs, device=u.device)

            for step in range(u.max_episode_length):
                force = u.force_sensor_smooth[:, 0:3]
                fnorm = torch.linalg.norm(force, dim=-1)
                peak = torch.maximum(peak, fnorm)

                actions = policy.act(fnorm)
                env.step(actions)

                held_pos, _ = factory_utils.get_held_base_pose(
                    u.held_pos, u.held_quat, u.cfg_task.name, u.cfg_task.fixed_asset_cfg, u.num_envs, u.device
                )
                target_pos, _ = factory_utils.get_target_held_base_pose(
                    u.fixed_pos, u.fixed_quat, u.cfg_task.name, u.cfg_task.fixed_asset_cfg, u.num_envs, u.device
                )
                final_z = held_pos[:, 2] - target_pos[:, 2]
                ok = u._get_curr_successes(success_threshold=u.cfg_task.success_threshold, check_rot=False)

                newly = ok & (~success_any)
                first_step[newly] = step
                success_any |= ok

            rate = success_any.float().mean().item()
            all_success.append(rate)
            med_step = first_step[success_any].float().mean().item() if success_any.any() else float("nan")
            all_success_step.append(med_step)
            print(
                f"[ep {ep:02d}] 成功率 {rate:.3f} | 平均首次成功步 {med_step:.1f} | "
                f"力峰值均值 {peak.mean().item():.2f} N | 末态 z {final_z.mean().item() * 1000:.2f} mm"
            )
            for i in range(u.num_envs):
                writer.writerow(
                    [
                        ep,
                        i,
                        int(success_any[i].item()),
                        int(first_step[i].item()),
                        f"{final_z[i].item() * 1000:.3f}",
                        f"{peak[i].item():.3f}",
                    ]
                )
            policy.reset()

    fh.close()
    print("=" * 78)
    print(f"[RESULT] 解析基线平均成功率 : {sum(all_success) / len(all_success):.3f}  (n={len(all_success)} 回合)")
    print(f"[RESULT] 平均首次成功步数   : {sum(all_success_step) / len(all_success_step):.1f}")
    print(f"[RESULT] CSV               : {csv_path}")
    print("[HINT] 若成功率接近 0，先调 base 参数：hold_force↑、press_step↓、search_radius_max↑")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
