"""阶段 A：安全高度横向对中试验（审查批准流程的第一步）。

依据：《docs/research/审查/2026-11-23_FORGE阶段二审查.md》批准的控制流程。

本脚本只回答一个问题：**在不降低高度的前提下，恒定横向目标 (0,0) 能否把
带噪末端相对孔位的横向误差压到门槛以下**。不接触、不插入、不加导纳。

两种模式：

``--mode align``（默认，审查门槛 1）
    保持复位高度，持续命令 target_xy=(0,0)。
    横向误差 < ``--xy_tol`` 连续 ``--hold_steps`` 步记为 CONVERGED。
    独立跑 ``--episodes 20`` 回合，保留逐步 CSV。

``--mode calibrate``（审查优先级 2）
    对中成功后，用独立状态变量 ``z_cmd`` 缓慢积分下降，
    记录首次接触时的 rel_z 与世界系 F_z 正负号 → 标定接触高度与轴向力方向。
    建议跑 3 回合。

记录字段（对齐审查门槛 3）：种子、回合、步、阶段、观测 xyz、目标 xyz、
F_x/F_y/F_z 世界系分量、力阈值、EMA 后实际动作（前 3 维）、横向误差、
对中保持计数、对中时刻、接触标志。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab
    ./isaaclab.sh -p /home/mi/graduation_design/RL/demos/07_phaseA_align.py \
        --mode align --episodes 20 --seed 42 --headless
    ./isaaclab.sh -p /home/mi/graduation_design/RL/demos/07_phaseA_align.py \
        --mode calibrate --episodes 3 --seed 42 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase-A safe-height lateral alignment trial for FORGE.")
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0", help="Task name.")
parser.add_argument("--mode", type=str, default="align", choices=["align", "calibrate"], help="Trial mode.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--episodes", type=int, default=20, help="Number of episodes.")
parser.add_argument("--max_steps", type=int, default=200, help="Max control steps per episode.")
parser.add_argument(
    "--keep_official_limit",
    action="store_true",
    default=False,
    help="使用官方 10 s 时限；默认延长到覆盖 --max_steps，避免窗口跨越环境自动复位。",
)
parser.add_argument("--xy_tol", type=float, default=0.002, help="Lateral tolerance for alignment [m].")
parser.add_argument("--hold_steps", type=int, default=10, help="Consecutive in-tolerance steps to count as converged.")
parser.add_argument("--contact_alert", type=float, default=1.0, help="|F| alert level during align mode [N].")
parser.add_argument("--descend_step", type=float, default=0.001, help="Calibrate mode: z_cmd decrement per step [m].")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--csv", type=str, default=None, help="CSV output path.")
parser.add_argument("--debug_every", type=int, default=0, help="Print debug every N steps (0 = off).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise SystemExit("[ERROR] 本脚本为单环境实现，请使用 --num_envs 1（或省略该参数）。")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import math
import os
from datetime import datetime

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

POS_ACTION_SCALE = 0.05
# 观测切片（FORGE 24 维 policy 观测，与 03_forge_inspect.py 的核查一致）
SL_REL = slice(0, 3)          # fingertip_pos_rel_fixed
SL_F_THRESH = slice(16, 17)   # force_threshold
SL_PREV_ACT = slice(17, 24)   # prev_actions


def make_action(target_rel: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
    """把孔坐标系下的目标位姿转成 FORGE 7 维动作（姿态沿用复位 yaw，预测头置 -1）。"""
    action = torch.zeros(1, 7, device=target_rel.device)
    action[:, 0:3] = torch.clamp(target_rel / POS_ACTION_SCALE, -1.0, 1.0)
    action[:, 3] = 0.0
    action[:, 4] = 0.0
    action[:, 5] = prev_actions[5]
    action[:, 6] = -1.0
    return action


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # 回合时限：环境默认 10 s（约 150 控制步）会自动复位。200 步的窗口会跨 2 个环境回合，
    # 把不同初态之间的跳变误当成"对中漂移"。默认延长到覆盖整个窗口。
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    if not args_cli.keep_official_limit:
        env_cfg.episode_length_s = args_cli.max_steps * control_dt + 1.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(
        f"[INFO]: 回合时限 {env_cfg.episode_length_s:.1f} s（控制 dt={control_dt*1000:.1f} ms）"
        + ("" if args_cli.keep_official_limit else "｜**已修改任务时限，非官方默认**")
    )
    unwrapped = env.unwrapped

    print(f"[INFO]: 模式 {args_cli.mode}，回合 {args_cli.episodes}，种子 {args_cli.seed}")
    print(f"[INFO]: 对中门槛 {args_cli.xy_tol*1000:.1f} mm，需连续 {args_cli.hold_steps} 步\n")

    rows = []
    summary = []

    for ep in range(args_cli.episodes):
        obs, _ = env.reset()
        policy_obs = obs["policy"][0]
        rel = policy_obs[SL_REL].clone()
        prev_actions = policy_obs[SL_PREV_ACT].clone()
        f_thresh = float(policy_obs[SL_F_THRESH].item())

        init_xy = torch.linalg.norm(rel[0:2]).item()
        hold_z = rel[2].item()          # 阶段 A：保持复位高度，禁止降低 z
        z_cmd = hold_z                  # calibrate 模式用的独立下降状态变量
        phase = "A"

        in_tol = 0
        converged_step = None
        contact_alerts = 0
        first_contact = None            # (step, rel_z, Fx, Fy, Fz)
        last_force = torch.zeros(3, device=rel.device)
        status = "TIMEOUT"

        for step in range(args_cli.max_steps):
            target = torch.tensor([0.0, 0.0, z_cmd], device=rel.device)
            action = make_action(target, prev_actions)
            obs, rew, terminated, truncated, info = env.step(action)

            policy_obs = obs["policy"][0]
            rel = policy_obs[SL_REL].clone()
            prev_actions = policy_obs[SL_PREV_ACT].clone()
            f_thresh = float(policy_obs[SL_F_THRESH].item())
            f_xyz = unwrapped.force_sensor_smooth[0, 0:3].detach().clone()
            last_force = f_xyz
            f_norm = torch.linalg.norm(f_xyz).item()
            applied = unwrapped.actions[0, 0:3].detach().clone()
            xy_err = torch.linalg.norm(rel[0:2]).item()

            if f_norm > args_cli.contact_alert:
                contact_alerts += 1
                if first_contact is None:
                    first_contact = (step, rel[2].item(), f_xyz[0].item(), f_xyz[1].item(), f_xyz[2].item())

            if xy_err < args_cli.xy_tol:
                in_tol += 1
                if in_tol >= args_cli.hold_steps and converged_step is None:
                    converged_step = step
            else:
                in_tol = 0

            rows.append(
                {
                    "seed": args_cli.seed,
                    "episode": ep,
                    "step": step,
                    "phase": phase,
                    "obs_x": round(rel[0].item(), 6),
                    "obs_y": round(rel[1].item(), 6),
                    "obs_z": round(rel[2].item(), 6),
                    "tgt_x": 0.0,
                    "tgt_y": 0.0,
                    "tgt_z": round(z_cmd, 6),
                    "F_x": round(f_xyz[0].item(), 4),
                    "F_y": round(f_xyz[1].item(), 4),
                    "F_z": round(f_xyz[2].item(), 4),
                    "force_threshold": round(f_thresh, 4),
                    "applied_x": round(applied[0].item(), 5),
                    "applied_y": round(applied[1].item(), 5),
                    "applied_z": round(applied[2].item(), 5),
                    "xy_err": round(xy_err, 6),
                    "in_tol_count": in_tol,
                    "converged": converged_step is not None,
                    "contact_alerts": contact_alerts,
                    "note": "",
                }
            )

            if args_cli.debug_every and step % args_cli.debug_every == 0:
                print(
                    f"    [dbg {step:3d}] xy={xy_err*1000:6.2f}mm z={rel[2].item()*1000:6.1f}mm "
                    f"F=({f_xyz[0].item():+.2f},{f_xyz[1].item():+.2f},{f_xyz[2].item():+.2f}) "
                    f"in_tol={in_tol}"
                )

            # calibrate 模式：对中后开始缓慢下降，寻找首次接触
            if args_cli.mode == "calibrate" and converged_step is not None and first_contact is None:
                phase = "B-calib"
                z_cmd = z_cmd - args_cli.descend_step
                rows[-1]["note"] = "descending"

            # calibrate 模式：记录到接触后 10 步即结束
            if args_cli.mode == "calibrate" and first_contact is not None and step >= first_contact[0] + 10:
                status = "CONTACT_FOUND"
                break

        if args_cli.mode == "align":
            status = "CONVERGED" if converged_step is not None else "TIMEOUT"
            if contact_alerts > 5:
                status += "+CONTACT"
            summary.append(
                {
                    "episode": ep,
                    "status": status,
                    "init_xy": round(init_xy, 5),
                    "converged_step": converged_step,
                    "final_xy": round(torch.linalg.norm(rel[0:2]).item(), 5),
                    "final_z": round(rel[2].item(), 5),
                    "contact_alerts": contact_alerts,
                }
            )
            print(
                f"[EP {ep:02d}] {status:<16} init_xy={init_xy*1000:5.1f}mm "
                f"converged@{converged_step}  final_xy={summary[-1]['final_xy']*1000:5.2f}mm "
                f"contact_alerts={contact_alerts}"
            )
        else:
            fc = first_contact or (None,) * 5
            summary.append(
                {
                    "episode": ep,
                    "status": status,
                    "converged_step": converged_step,
                    "contact_step": fc[0],
                    "contact_rel_z": round(fc[1], 5) if fc[0] is not None else None,
                    "contact_Fxyz": (round(fc[2], 3), round(fc[3], 3), round(fc[4], 3)) if fc[0] is not None else None,
                }
            )
            print(f"[EP {ep:02d}] {status}  converged@{converged_step}  contact@{summary[-1]['contact_step']}"
                  f"  rel_z={summary[-1]['contact_rel_z']}  F_xyz={summary[-1]['contact_Fxyz']}")

    # 汇总
    print("\n[SUMMARY]")
    if args_cli.mode == "align":
        n_conv = sum(1 for r in summary if r["status"] == "CONVERGED")
        conv_steps = [r["converged_step"] for r in summary if r["converged_step"] is not None]
        print(f"  对中收敛率   : {n_conv}/{len(summary)}")
        if conv_steps:
            print(f"  收敛步数     : min={min(conv_steps)} max={max(conv_steps)} mean={sum(conv_steps)/len(conv_steps):.1f}")
        n_contact = sum(1 for r in summary if "CONTACT" in r["status"])
        print(f"  保持高度下异常接触回合: {n_contact}")
    else:
        for r in summary:
            print(f"  EP{r['episode']}: converged@{r['converged_step']} contact@{r['contact_step']} "
                  f"rel_z={r['contact_rel_z']} F_xyz={r['contact_Fxyz']}")
        zs = [r["contact_rel_z"] for r in summary if r["contact_rel_z"] is not None]
        fzs = [r["contact_Fxyz"][2] for r in summary if r["contact_Fxyz"] is not None]
        if zs:
            print(f"  首次接触 rel_z 范围: {min(zs)*1000:.1f}–{max(zs)*1000:.1f} mm")
        if fzs:
            print(f"  首次接触 F_z 符号: {'全为正' if all(f > 0 for f in fzs) else '全为负' if all(f < 0 for f in fzs) else '不一致，需人工核查'}")

    csv_path = args_cli.csv
    if csv_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs",
            f"phaseA_{args_cli.mode}_{stamp}.csv",
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
