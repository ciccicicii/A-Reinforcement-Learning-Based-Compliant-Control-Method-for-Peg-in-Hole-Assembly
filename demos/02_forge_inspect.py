# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""FORGE PegInsert 环境体检（阶段二第 1 步）。

回答的问题：
  1. 观测 / 状态 / 动作的真实维度是多少（不是 cfg 里写的那个数）；
  2. 控制频率、每回合步数；
  3. 力传感器读数是什么量级、正负号约定；
  4. 零动作下 peg 会不会自己往下掉、成功率是否恒为 0；
  5. 成功判据的三个量（xy 误差、z 位移、成功标志）怎么读出来。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab
    ./isaaclab.sh -p ../demos/02_forge_inspect.py --num_envs 4 --steps 150
    ./isaaclab.sh -p ../demos/02_forge_inspect.py --num_envs 4 --steps 150 --headless

输出：终端摘要 + ``RL/outputs/forge_inspect_<时间戳>.csv``
"""

import argparse
import csv
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FORGE PegInsert 环境体检")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数，8GB 显存建议 4-16")
parser.add_argument("--steps", type=int, default=150, help="运行步数（150 步 = 1 个回合）")
parser.add_argument(
    "--action",
    choices=["zero", "random", "down"],
    default="zero",
    help="zero=全零；random=均匀随机；down=恒定向下压",
)
parser.add_argument("--out_dir", type=str, default=None, help="CSV 输出目录")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下必须在 AppLauncher 之后 import。"""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.utils import parse_env_cfg

TASK_ID = "Isaac-Forge-PegInsert-Direct-v0"


def _task_metrics(env):
    """返回 (xy 误差[m], z 位移[m], 成功标志)，全部形状 [N]。"""
    u = env.unwrapped
    task_name = u.cfg_task.name
    fixed_cfg = u.cfg_task.fixed_asset_cfg
    held_pos, _ = factory_utils.get_held_base_pose(
        u.held_pos, u.held_quat, task_name, fixed_cfg, u.num_envs, u.device
    )
    target_pos, _ = factory_utils.get_target_held_base_pose(
        u.fixed_pos, u.fixed_quat, task_name, fixed_cfg, u.num_envs, u.device
    )
    xy = torch.linalg.vector_norm(held_pos[:, 0:2] - target_pos[:, 0:2], dim=-1)
    z = held_pos[:, 2] - target_pos[:, 2]
    ok = u._get_curr_successes(success_threshold=u.cfg_task.success_threshold, check_rot=False)
    return xy, z, ok


def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK_ID, cfg=env_cfg)
    u = env.unwrapped

    print("=" * 78)
    print(f"[INFO] 任务            : {TASK_ID}")
    print(f"[INFO] 并行环境数       : {u.num_envs}")
    print(f"[INFO] 物理步长 dt      : {u.cfg.sim.dt:.6f} s  ({1.0 / u.cfg.sim.dt:.1f} Hz)")
    print(f"[INFO] decimation       : {u.cfg.decimation}")
    print(f"[INFO] 控制频率         : {1.0 / (u.cfg.sim.dt * u.cfg.decimation):.2f} Hz")
    print(f"[INFO] 每回合步数       : {u.max_episode_length}  (episode_length_s={u.cfg.episode_length_s})")
    print(f"[INFO] 动作空间         : {env.action_space}")
    print(f"[INFO] 成功判据         : xy < 2.5 mm 且 z_disp < {u.cfg_task.success_threshold} * 孔深"
          f" = {u.cfg_task.success_threshold * u.cfg_task.fixed_asset_cfg.height * 1000:.2f} mm")
    print(f"[INFO] 孔 / 销直径      : {u.cfg_task.fixed_asset_cfg.diameter * 1000:.3f} mm /"
          f" {u.cfg_task.held_asset_cfg.diameter * 1000:.3f} mm"
          f"  => 单边间隙 {(u.cfg_task.fixed_asset_cfg.diameter - u.cfg_task.held_asset_cfg.diameter) / 2 * 1000:.3f} mm")
    print(f"[INFO] 位置动作上下界   : {u.cfg.ctrl.pos_action_bounds}")
    print(f"[INFO] 每环境随机化的接触力阈值范围 : {u.cfg_task.contact_penalty_threshold_range} N")
    print("=" * 78)

    obs, _ = env.reset()
    print(f"[INFO] policy 观测维度  : {tuple(obs['policy'].shape)}")
    print(f"[INFO] critic 状态维度  : {tuple(obs['critic'].shape)}")
    print(f"[INFO] 动作维度         : {u.cfg.action_space}")
    print("=" * 78)

    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"forge_inspect_{datetime.now():%Y%m%d_%H%M%S}.csv")
    fh = open(csv_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(
        ["step", "env_id", "force_x", "force_y", "force_z", "force_norm", "xy_err_mm", "z_disp_mm", "success"]
    )

    n_success = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    peak_force = torch.zeros(u.num_envs, device=u.device)

    with torch.inference_mode():
        for step in range(args_cli.steps):
            if args_cli.action == "zero":
                actions = torch.zeros(env.action_space.shape, device=u.device)
            elif args_cli.action == "random":
                actions = 0.1 * torch.randn(env.action_space.shape, device=u.device)
            else:  # down：z 方向恒定 -0.3（≈ -15 mm 目标，会被 5 cm 界裁剪）
                actions = torch.zeros(env.action_space.shape, device=u.device)
                actions[:, 2] = -0.3

            obs, rew, terminated, truncated, info = env.step(actions)

            force = u.force_sensor_smooth[:, 0:3]
            fnorm = torch.linalg.norm(force, dim=-1)
            peak_force = torch.maximum(peak_force, fnorm)
            xy, z, ok = _task_metrics(env)
            n_success |= ok

            for i in range(u.num_envs):
                writer.writerow(
                    [
                        step,
                        i,
                        f"{force[i, 0].item():.4f}",
                        f"{force[i, 1].item():.4f}",
                        f"{force[i, 2].item():.4f}",
                        f"{fnorm[i].item():.4f}",
                        f"{xy[i].item() * 1000:.4f}",
                        f"{z[i].item() * 1000:.4f}",
                        int(ok[i].item()),
                    ]
                )

            if step % 15 == 0:
                print(
                    f"step {step:4d} | rew {rew.mean().item():+.4f} | "
                    f"|F| mean {fnorm.mean().item():7.2f} N max {fnorm.max().item():7.2f} N | "
                    f"xy {xy.mean().item() * 1000:6.3f} mm | z {z.mean().item() * 1000:7.3f} mm | "
                    f"success {ok.float().mean().item():.2f}"
                )
            if bool(torch.all(terminated | truncated)):
                break

    fh.close()
    print("=" * 78)
    print(f"[RESULT] 任一时刻成功过的环境比例 : {n_success.float().mean().item():.2f}")
    print(f"[RESULT] 力峰值 (各环境)          : {[round(v, 2) for v in peak_force.tolist()]}")
    print(f"[RESULT] CSV                     : {csv_path}")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
