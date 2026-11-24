"""FORGE PegInsert 柔顺控制基线（实验二：力反馈导纳）。

在实验一（纯位置目标）的基础上，只加一层力反馈修正，不改变底层控制器：

    1. 横向导纳：把接触力的横向分量转成目标位姿的横向偏移
       target_xy += k_force * F_xy * sign
       孔口倒角会把销钉往孔轴方向推，因此沿力的方向修正即可"顺坡入孔"。

    2. 轴向限力：接触力超过 backoff_force 时停止下压，超过更接近阈值时回退，
       避免像实验一那样一路顶到关节力矩饱和。

    3. 力信号做低通：obs 的 ft_force 每轴带 1 N 噪声，直接用会让目标抖动。

注意环境的两个特性：
    * 动作被 EMA 平滑（alpha 约 0.025–0.1），所以目标要恒定或缓变，不能做增量指令；
    * 任务空间死区最大 5 N，纯位置误差在毫米级会被吃掉 —— 这正是实验一失败的原因。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab
    ./isaaclab.sh -p /home/mi/graduation_design/RL/demos/05_forge_compliance.py --episodes 10
    ./isaaclab.sh -p .../05_forge_compliance.py --episodes 20 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Force-admittance baseline for FORGE peg insertion.")
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=400, help="Max control steps per episode.")
parser.add_argument("--depth", type=float, default=0.03, help="Constant insertion target depth below hole top [m].")
parser.add_argument("--k_force", type=float, default=0.0003, help="Lateral admittance gain [m/N].")
parser.add_argument("--max_corr", type=float, default=0.002, help="Max lateral correction per step [m].")
parser.add_argument("--max_offset", type=float, default=0.02, help="Max accumulated lateral offset [m].")
parser.add_argument("--force_deadband", type=float, default=1.5, help="Ignore |F_xy| below this [N].")
parser.add_argument("--filter_alpha", type=float, default=0.3, help="Low-pass factor for measured force.")
parser.add_argument("--backoff_ratio", type=float, default=0.6, help="Stop descending above ratio*force_limit.")
parser.add_argument("--sign", type=float, default=1.0, choices=[1.0, -1.0], help="Admittance direction sign.")
parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
parser.add_argument("--debug_every", type=int, default=0, help="Print debug info every N steps (0 = off).")
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

POS_ACTION_SCALE = 0.05


def make_action(target_rel: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
    action = torch.zeros(1, 7, device=target_rel.device)
    action[:, 0:3] = torch.clamp(target_rel / POS_ACTION_SCALE, -1.0, 1.0)
    action[:, 3] = 0.0
    action[:, 4] = 0.0
    action[:, 5] = prev_actions[5]
    action[:, 6] = -1.0
    return action


def get_success(unwrapped, success_threshold: float) -> bool:
    return bool(unwrapped._get_curr_successes(success_threshold=success_threshold, check_rot=False)[0])


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    success_threshold = env_cfg.task.success_threshold

    print(f"[INFO]: 任务 {args_cli.task}，回合数 {args_cli.episodes}")
    print(f"[INFO]: 导纳增益 {args_cli.k_force} m/N，力死区 {args_cli.force_deadband} N，方向 sign={args_cli.sign:.0f}\n")

    rows = []

    for ep in range(args_cli.episodes):
        obs, _ = env.reset()
        policy_obs = obs["policy"][0]
        rel_pos = policy_obs[0:3].clone()
        prev_actions = policy_obs[17:24].clone()
        force_limit = float(policy_obs[16].item())

        init_xy = torch.linalg.norm(rel_pos[0:2]).item()
        max_force = 0.0
        success = False
        aborted = False
        steps_used = 0

        f_xy_filtered = torch.zeros(2, device=rel_pos.device)
        lateral_offset = torch.zeros(2, device=rel_pos.device)

        for step in range(args_cli.max_steps):
            steps_used = step
            if get_success(unwrapped, success_threshold):
                success = True
                break

            force_vec = unwrapped.force_sensor_smooth[0, 0:3]
            force_norm = torch.linalg.norm(force_vec).item()
            max_force = max(max_force, force_norm)
            if force_norm > force_limit:
                aborted = True
                break

            # 低通滤波后的横向接触力
            f_xy_filtered = args_cli.filter_alpha * force_vec[0:2] + (1 - args_cli.filter_alpha) * f_xy_filtered
            f_xy_norm = torch.linalg.norm(f_xy_filtered).item()

            # (1) 横向导纳：沿接触力方向修正目标
            if f_xy_norm > args_cli.force_deadband:
                correction = args_cli.sign * args_cli.k_force * f_xy_filtered
                corr_norm = torch.linalg.norm(correction).item()
                if corr_norm > args_cli.max_corr:
                    correction = correction / corr_norm * args_cli.max_corr
                lateral_offset = lateral_offset + correction
                lateral_offset = torch.clamp(lateral_offset, -args_cli.max_offset, args_cli.max_offset)

            # (2) 轴向限力：力大时停止下压并略微回退
            if force_norm > args_cli.backoff_ratio * force_limit:
                target_z = rel_pos[2].item() + 0.004
            else:
                target_z = -args_cli.depth

            target_rel = torch.tensor(
                [lateral_offset[0].item(), lateral_offset[1].item(), target_z],
                device=rel_pos.device,
            )
            action = make_action(target_rel, prev_actions)
            obs, rew, terminated, truncated, info = env.step(action)

            policy_obs = obs["policy"][0]
            rel_pos = policy_obs[0:3].clone()
            prev_actions = policy_obs[17:24].clone()

            if args_cli.debug_every and step % args_cli.debug_every == 0:
                print(
                    f"    [dbg {step:3d}] rel={[round(v, 4) for v in rel_pos.tolist()]} "
                    f"F={[round(v, 2) for v in force_vec.tolist()]} |F|={force_norm:.2f} "
                    f"off={[round(v, 4) for v in lateral_offset.tolist()]}"
                )

            if bool(terminated[0]) or bool(truncated[0]):
                success = get_success(unwrapped, success_threshold)
                break

        status = "SUCCESS" if success else ("FORCE_ABORT" if aborted else "FAIL")
        rows.append(
            {
                "episode": ep,
                "status": status,
                "steps": steps_used,
                "init_xy_err": round(init_xy, 5),
                "final_xy_err": round(torch.linalg.norm(rel_pos[0:2]).item(), 5),
                "final_z": round(rel_pos[2].item(), 5),
                "max_force": round(max_force, 3),
                "force_limit": round(force_limit, 3),
                "final_offset": round(torch.linalg.norm(lateral_offset).item(), 5),
            }
        )
        print(
            f"[EP {ep:02d}] {status:<11} steps={steps_used:3d} "
            f"init_xy={init_xy*1000:5.1f}mm  final_xy={rows[-1]['final_xy_err']*1000:5.1f}mm  "
            f"final_z={rows[-1]['final_z']*1000:6.1f}mm  max_F={max_force:5.2f}N"
        )

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
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", f"forge_compliance_{stamp}.csv"
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
