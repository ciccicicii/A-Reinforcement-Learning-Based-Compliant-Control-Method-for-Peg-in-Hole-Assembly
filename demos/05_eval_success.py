# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""统一的插接成功率评估（阶段二 / 阶段三共用）。

三种被测对象，用**同一个环境、同一批初始随机化、同一套指标**评估：

  --scripted                  解析柔顺基线（不加载网络）
  --checkpoint <path>         残差 PPO 的 checkpoint
  （都不给）                    纯零残差：等价于"残差策略输出恒 0"

指标：成功率 / 平均首次成功步（= 插接时间）/ 力峰值 / 末态 z 误差 / 末态 xy 误差。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab
    ./isaaclab.sh -p ../demos/05_eval_success.py --scripted --num_envs 32 --episodes 10 --headless
    ./isaaclab.sh -p ../demos/05_eval_success.py \
        --checkpoint ../outputs/rsl_rl/2026-11-23_10-00-00_residual/model_1450.pt \
        --num_envs 32 --episodes 10 --headless
"""

import argparse
import csv
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FORGE 插接成功率评估")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--checkpoint", type=str, default=None, help="rsl-rl checkpoint 路径；不给则零残差")
parser.add_argument("--scripted", action="store_true", default=False, help="用解析基线策略评估")
parser.add_argument("--out_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_ROOT)
import forge_ext  # noqa: E402,F401
from forge_ext.agents import ForgeResidualPPORunnerCfg  # noqa: E402
from forge_ext.base_policy import InsertBaseCfg, ScriptedInsertPolicy  # noqa: E402
from forge_ext.residual_env import ForgeResidualCfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

TASK_ID = "Isaac-Forge-PegInsert-Residual-v0"


def _metrics(u):
    held_pos, _ = factory_utils.get_held_base_pose(
        u.held_pos, u.held_quat, u.cfg_task.name, u.cfg_task.fixed_asset_cfg, u.num_envs, u.device
    )
    target_pos, _ = factory_utils.get_target_held_base_pose(
        u.fixed_pos, u.fixed_quat, u.cfg_task.name, u.cfg_task.fixed_asset_cfg, u.num_envs, u.device
    )
    xy = torch.linalg.vector_norm(held_pos[:, 0:2] - target_pos[:, 0:2], dim=-1)
    z = held_pos[:, 2] - target_pos[:, 2]
    ok = u._get_curr_successes(success_threshold=u.cfg_task.success_threshold, check_rot=False)
    return xy, z, ok


def main():
    device = args_cli.device or "cuda:0"

    env_cfg = ForgeResidualCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 1234
    env_cfg.sim.device = device

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    u = raw_env.unwrapped

    policy_fn = None
    policy = None
    if args_cli.checkpoint:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=1.0)
        agent_cfg = ForgeResidualPPORunnerCfg()
        agent_cfg.device = device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(args_cli.checkpoint)
        policy_fn = runner.get_inference_policy(device=device)
        u = env.unwrapped
    elif args_cli.scripted:
        policy = ScriptedInsertPolicy(InsertBaseCfg(), u.num_envs, u.device)

    out_dir = args_cli.out_dir or os.path.join(RL_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tag = "scripted" if args_cli.scripted else ("ckpt" if args_cli.checkpoint else "zero_residual")
    csv_path = os.path.join(out_dir, f"eval_{tag}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    fh = open(csv_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["episode", "env_id", "success", "first_success_step", "final_xy_mm", "final_z_mm", "peak_force_N"])

    raw_env.reset()
    if policy is not None:
        policy.reset()

    rates, times = [], []
    with torch.inference_mode():
        for ep in range(args_cli.episodes):
            success_any = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
            first_step = torch.full((u.num_envs,), -1, dtype=torch.long, device=u.device)
            peak = torch.zeros(u.num_envs, device=u.device)
            xy = z = torch.zeros(u.num_envs, device=u.device)

            if policy_fn is not None:
                obs = env.get_observations()
            for step in range(u.max_episode_length):
                if policy_fn is not None:
                    actions = policy_fn(obs)
                    obs, _, _, _ = env.step(actions)
                else:
                    fnorm = torch.linalg.norm(u.force_sensor_smooth[:, 0:3], dim=-1)
                    peak = torch.maximum(peak, fnorm)
                    actions = policy.act(fnorm) if policy is not None else torch.zeros(
                        (u.num_envs, u.cfg.action_space), device=u.device
                    )
                    raw_env.step(actions)

                fnorm = torch.linalg.norm(u.force_sensor_smooth[:, 0:3], dim=-1)
                peak = torch.maximum(peak, fnorm)
                xy, z, ok = _metrics(u)
                newly = ok & (~success_any)
                first_step[newly] = step
                success_any |= ok

            rate = success_any.float().mean().item()
            rates.append(rate)
            t = first_step[success_any].float().mean().item() if success_any.any() else float("nan")
            times.append(t)
            print(
                f"[ep {ep:02d}] 成功率 {rate:.3f} | 首次成功步 {t:.1f} | "
                f"xy {xy.mean().item() * 1000:.2f} mm | z {z.mean().item() * 1000:.2f} mm | "
                f"力峰值 {peak.mean().item():.2f} N"
            )
            for i in range(u.num_envs):
                writer.writerow(
                    [
                        ep,
                        i,
                        int(success_any[i].item()),
                        int(first_step[i].item()),
                        f"{xy[i].item() * 1000:.3f}",
                        f"{z[i].item() * 1000:.3f}",
                        f"{peak[i].item():.3f}",
                    ]
                )
            if policy is not None:
                policy.reset()

    fh.close()
    print("=" * 78)
    print(f"[RESULT] {tag:14s} 成功率 {sum(rates) / len(rates):.3f}  (n={len(rates)} 回合 × {u.num_envs} 环境)")
    valid = [t for t in times if t == t]
    print(f"[RESULT] 平均首次成功步 {sum(valid) / len(valid):.1f}" if valid else "[RESULT] 无成功回合")
    print(f"[RESULT] CSV: {csv_path}")
    print("=" * 78)
    raw_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
