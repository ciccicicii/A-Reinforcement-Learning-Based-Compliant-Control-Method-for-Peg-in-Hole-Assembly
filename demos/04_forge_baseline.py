"""FORGE PegInsert 基础控制基线（实验一：位置控制）。

控制策略：两段式手写控制器
    阶段 A（对齐）：把末端横向偏差压到 ``--xy_tol`` 以内，高度保持不变；
    阶段 B（下压）：横向对中后，直接命令恒定插入位姿 ``--depth``，直到成功或超力。

注意环境特性（会直接影响策略设计）：
    * 动作被 EMA 平滑，alpha 约 0.025–0.1，指令需要几十步才生效 → 用恒定目标而非增量；
    * 任务空间存在死区（最大 5 N），毫米级误差产生的力矩会被直接吃掉。

FORGE 的动作语义（见 forge_env.py:_apply_action）：
    action[0:3]  →  目标位置 = 孔位置 + action[0:3] * 0.05 m   （归一化绝对目标，非增量）
    action[3:5]  →  强制为 0（roll/pitch 被引擎清零）
    action[5]    →  yaw，重映射到关节限位区间
    action[6]    →  [−1,1] 的成功预测头，不是控制量

本脚本是后续柔顺控制与残差强化学习的对照基线，输出成功率、成功步数、最大接触力。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab
    ./isaaclab.sh -p /home/mi/graduation_design/RL/demos/04_forge_baseline.py --episodes 10

    # 无窗口，跑更多回合
    ./isaaclab.sh -p .../04_forge_baseline.py --episodes 50 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted positional baseline for FORGE peg insertion.")
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=400, help="Max control steps per episode.")
parser.add_argument("--xy_tol", type=float, default=0.002, help="Lateral tolerance to start descending [m].")
parser.add_argument("--depth", type=float, default=0.03, help="Constant insertion target depth below hole top [m].")
parser.add_argument("--force_limit", type=float, default=None, help="Abort episode if |F| exceeds this [N]. Defaults to force_threshold.")
parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
parser.add_argument("--debug_every", type=int, default=0, help="Print per-step debug info every N steps (0 = off).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os
from datetime import datetime

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

POS_ACTION_SCALE = 0.05  # cfg.ctrl.pos_action_bounds，动作 → 米


def make_action(rel_pos: torch.Tensor, prev_actions: torch.Tensor, target_rel: torch.Tensor) -> torch.Tensor:
    """把孔坐标系下的目标位姿转成 FORGE 的 7 维动作。"""
    action = torch.zeros(1, 7, device=rel_pos.device)
    action[:, 0:3] = torch.clamp(target_rel / POS_ACTION_SCALE, -1.0, 1.0)
    # roll/pitch 会被引擎清零，这里显式给 0 更清楚
    action[:, 3] = 0.0
    action[:, 4] = 0.0
    # 保持复位时的 yaw 目标，不做姿态调整
    action[:, 5] = prev_actions[5]
    # 第 7 维是成功预测头，不是控制量
    action[:, 6] = -1.0
    return action


def get_success(unwrapped, success_threshold: float) -> bool:
    """读取当前时刻的成功标志。"""
    successes = unwrapped._get_curr_successes(success_threshold=success_threshold, check_rot=False)
    return bool(successes[0])


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    success_threshold = env_cfg.task.success_threshold

    print(f"[INFO]: 任务 {args_cli.task}，回合数 {args_cli.episodes}")
    limit_desc = "按每回合 force_threshold" if args_cli.force_limit is None else f"{args_cli.force_limit:.2f} N"
    print(f"[INFO]: 成功阈值 {success_threshold}，力上限 {limit_desc}，横向容差 {args_cli.xy_tol*1000:.1f} mm")
    print(f"[INFO]: 阶段 A 对齐 xy → 阶段 B 恒定目标下压至 {args_cli.depth*1000:.0f} mm\n")

    rows = []

    for ep in range(args_cli.episodes):
        obs, _ = env.reset()
        policy_obs = obs["policy"][0]
        rel_pos = policy_obs[0:3].clone()
        prev_actions = policy_obs[17:24].clone()

        # 接触力阈值每回合重新随机，未指定时按本回合观测值取
        force_limit = (
            args_cli.force_limit if args_cli.force_limit is not None else float(policy_obs[16].item())
        )

        init_xy = torch.linalg.norm(rel_pos[0:2]).item()
        init_z = rel_pos[2].item()

        max_force = 0.0
        success = False
        aborted = False
        steps_used = 0
        phase = "align"

        for step in range(args_cli.max_steps):
            steps_used = step
            if get_success(unwrapped, success_threshold):
                success = True
                break

            force_norm = torch.linalg.norm(unwrapped.force_sensor_smooth[0, 0:3]).item()
            max_force = max(max_force, force_norm)
            if force_norm > force_limit:
                aborted = True
                break

            xy_err = torch.linalg.norm(rel_pos[0:2]).item()

            # 纯位置控制的定义：从第一帧起就命令完整的插入位姿，不做分阶段等待。
            # 环境对动作有 EMA 平滑（alpha 约 0.025–0.1），恒定目标会被自动缓入，
            # 不需要也不应该自己做增量下压。
            target_rel = torch.zeros_like(rel_pos)
            target_rel[2] = -args_cli.depth
            phase = "align" if xy_err > args_cli.xy_tol else "insert"

            action = make_action(rel_pos, prev_actions, target_rel)
            obs, rew, terminated, truncated, info = env.step(action)

            policy_obs = obs["policy"][0]
            rel_pos = policy_obs[0:3].clone()
            prev_actions = policy_obs[17:24].clone()

            if args_cli.debug_every and step % args_cli.debug_every == 0:
                print(
                    f"    [dbg {step:3d}] rel={[round(v, 4) for v in rel_pos.tolist()]} "
                    f"cmd_act={[round(v, 3) for v in action[0, 0:3].tolist()]} "
                    f"applied={[round(v, 3) for v in env.unwrapped.actions[0, 0:3].tolist()]} "
                    f"F={force_norm:.2f} term={bool(terminated[0])} trunc={bool(truncated[0])}"
                )

            if bool(terminated[0]) or bool(truncated[0]):
                # 回合被环境终止：结束时再判一次成功
                success = get_success(unwrapped, success_threshold)
                break

        status = "SUCCESS" if success else ("FORCE_ABORT" if aborted else "FAIL")
        rows.append(
            {
                "episode": ep,
                "status": status,
                "steps": steps_used,
                "init_xy_err": round(init_xy, 5),
                "init_z": round(init_z, 5),
                "final_xy_err": round(torch.linalg.norm(rel_pos[0:2]).item(), 5),
                "final_z": round(rel_pos[2].item(), 5),
                "max_force": round(max_force, 3),
                "force_limit": round(force_limit, 3),
            }
        )
        print(
            f"[EP {ep:02d}] {status:<11} steps={steps_used:3d} "
            f"init_xy={init_xy*1000:5.1f}mm  final_xy={rows[-1]['final_xy_err']*1000:5.1f}mm  "
            f"final_z={rows[-1]['final_z']*1000:6.1f}mm  max_F={max_force:5.2f}N"
        )

    # 汇总
    n = len(rows)
    n_success = sum(1 for r in rows if r["status"] == "SUCCESS")
    succ_rows = [r for r in rows if r["status"] == "SUCCESS"]
    print("\n[SUMMARY]")
    print(f"  成功率        : {n_success}/{n} = {100.0 * n_success / n:.1f}%")
    if succ_rows:
        print(f"  平均成功步数  : {sum(r['steps'] for r in succ_rows) / len(succ_rows):.1f}")
    print(f"  平均最大接触力: {sum(r['max_force'] for r in rows) / n:.2f} N")
    print(f"  最大接触力    : {max(r['max_force'] for r in rows):.2f} N")
    print(f"  力超限中止    : {sum(1 for r in rows if r['status'] == 'FORCE_ABORT')} 次")

    csv_path = args_cli.csv
    if csv_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", f"forge_baseline_{stamp}.csv"
        )
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO]: 已写入 {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
