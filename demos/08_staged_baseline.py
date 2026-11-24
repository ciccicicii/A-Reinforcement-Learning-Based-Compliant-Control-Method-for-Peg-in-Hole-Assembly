"""三阶段解析基线：对中（A）→ 条件缓降（B）→ 近孔微调（C）。

依据：《docs/research/审查/2026-11-23_FORGE阶段二审查.md》批准的流程，与
《2026-11-23_FORGE阶段汇报复审.md》的修复要求，及 phaseA 标定数据
（初步接触区间 30.4–32.2 mm、F_z 正号、首次检出接触力 2.56–4.88 N @1mm/步）。

相对上一版的修复（复审阻塞项）：
    1. 阶段 C 导纳偏置改为**跨步状态** ``admittance_offset``，在构造动作**之前**
       用上一步滤波力更新并叠加到螺旋目标——保证真正下发（上一版修正被
       螺旋目标重新初始化覆盖，从未生效）；
    2. 阶段判定与状态更新全部移到 ``env.step()`` **之前**，CSV 记录的
       ``phase`` 即当步实际下发动作所属阶段；
    3. 记录基础螺旋目标、导纳偏置、最终下发目标三个量（base_x/y、adm_x/y、tgt_x/y/z）；
    4. 阶段 B 用带符号 ``F_z >= 阈值`` 识别向下压接触；负向 F_z 单独计数为异常；
    5. 入口显式拒绝 ``num_envs != 1``（动作/观测访问固定为单环境）；
    6. ``--c_sign`` 暴露导纳方向：复审要求先用小规模接触试验确认正增益
       （沿力方向修正）不会把销钉推离孔轴，符号未验证前不得宣称"螺旋+导纳"生效。

阶段设计
--------
A  对中：保持复位高度，恒定 target_xy=(0,0)；xy<--xy_tol 连续 --hold_steps 步 → B。
B  条件缓降：独立状态变量 z_cmd 积分下降（--descend_step）。下降许可（上一步数据判定）：
     1) xy < --descend_gate（对中漂移监视）；
     2) F_z < --warn_ratio × 力阈值（带符号）；
   F_z >= --backoff_ratio × 力阈值时 z_cmd 回退 --backoff_dist；负 F_z 记为异常。
C  近孔微调：已接触、xy<--c_xy_gate、连续 --c_trigger_steps 步未成功时启用——
   毫米级可扩张螺旋 + 导纳偏置（跨步状态）。

终止：SUCCESS（环境成功判据）/ FORCE_ABORT（|F|>力阈值，脚本安全规则）/ TIMEOUT。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab
    ./isaaclab.sh -p /home/mi/graduation_design/RL/demos/08_staged_baseline.py \
        --episodes 30 --seed 42 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Three-stage analytic baseline for FORGE peg insertion.")
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 (single-env implementation).")
parser.add_argument("--episodes", type=int, default=30, help="Number of episodes.")
parser.add_argument("--max_steps", type=int, default=600, help="Max control steps per episode.")
parser.add_argument("--neg_fz_alert", type=float, default=-0.5, help="Count F_z below this [N] as a physical anomaly (sensor noise is ~-0.16 N max).")
parser.add_argument("--c_enable", action="store_true", help="Enable Phase-C (default OFF: spiral-only C was 0/7 with 3 aborts in the 30-episode run).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
# 阶段 A
parser.add_argument("--xy_tol", type=float, default=0.002, help="Phase-A tolerance [m].")
parser.add_argument("--hold_steps", type=int, default=10, help="Consecutive in-tolerance steps to enter B.")
# 阶段 B
parser.add_argument("--descend_gate", type=float, default=0.003, help="Phase-B: pause descent if xy exceeds this [m].")
parser.add_argument("--descend_step", type=float, default=0.0005, help="Phase-B: z_cmd decrement per step [m].")
parser.add_argument("--z_min", type=float, default=-0.008, help="Phase-B: lowest z_cmd (rel to hole frame) [m].")
parser.add_argument("--warn_ratio", type=float, default=0.55, help="Phase-B: pause descent above ratio*force_threshold (signed F_z).")
parser.add_argument("--backoff_ratio", type=float, default=0.75, help="Phase-B: retreat above ratio*force_threshold.")
parser.add_argument("--backoff_dist", type=float, default=0.003, help="Phase-B: retreat distance [m].")
# 阶段 C
parser.add_argument("--c_trigger_steps", type=int, default=30, help="Phase-C: stuck steps after contact to trigger.")
parser.add_argument("--c_xy_gate", type=float, default=0.005, help="Phase-C: only engage if xy below this [m].")
parser.add_argument("--spiral_r0", type=float, default=0.001, help="Phase-C: initial spiral radius [m].")
parser.add_argument("--spiral_dr", type=float, default=0.00002, help="Phase-C: radius growth per step [m].")
parser.add_argument("--spiral_rmax", type=float, default=0.0025, help="Phase-C: max spiral radius [m].")
parser.add_argument("--spiral_omega_deg", type=float, default=15.0, help="Phase-C: angular step [deg].")
parser.add_argument("--c_force_deadband", type=float, default=0.8, help="Phase-C: admittance deadband [N].")
parser.add_argument("--c_k_force", type=float, default=0.0008, help="Phase-C: admittance gain [m/N].")
parser.add_argument("--c_max_corr", type=float, default=0.001, help="Phase-C: max admittance correction per step [m].")
parser.add_argument("--c_max_offset", type=float, default=0.005, help="Phase-C: max accumulated admittance offset (L2 norm) [m].")
parser.add_argument("--c_max_total", type=float, default=0.004, help="Phase-C: max final lateral target |base+admittance| (L2 norm) [m].")
parser.add_argument("--c_sign", type=float, default=1.0, choices=[1.0, -1.0], help="Phase-C admittance direction sign (UNVERIFIED — validate with a small contact trial first).")
parser.add_argument("--csv", type=str, default=None, help="CSV output path.")
parser.add_argument("--debug_every", type=int, default=0, help="Print debug every N steps (0 = off).")
# 注意：必须先注册 AppLauncher 参数（--headless/--device 等），再 parse
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
SL_REL = slice(0, 3)
SL_F_THRESH = slice(16, 17)
SL_PREV_ACT = slice(17, 24)


def make_action(target_rel: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
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
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    success_threshold = env_cfg.task.success_threshold

    # 版本快照（审查要求：CLI 参数 + 脚本哈希，写入 CSV 同目录 json）
    import hashlib
    import json
    import isaaclab
    script_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
    run_config = {
        "script_sha256_16": script_sha,
        "cli": {k: v for k, v in vars(args_cli).items()},
        "isaac_lab_version": getattr(isaaclab, "__version__", "unknown"),
    }
    print(f"[INFO]: 脚本 SHA-256[:16]={script_sha}  Isaac Lab={run_config['isaac_lab_version']}")

    print(f"[INFO]: 三阶段基线，回合 {args_cli.episodes}，种子 {args_cli.seed}，导纳方向 sign={args_cli.c_sign:+.0f}（未验证）")
    print(
        f"[INFO]: A判据 {args_cli.xy_tol*1000:.1f}mm×{args_cli.hold_steps}步 → "
        f"B缓降 {args_cli.descend_step*1000:.1f}mm/步(gate {args_cli.descend_gate*1000:.1f}mm, "
        f"预警 F_z>{args_cli.warn_ratio:.2f}×阈值, 回退 F_z>{args_cli.backoff_ratio:.2f}×阈值) → C微调\n"
    )

    rows = []
    summary = []

    for ep in range(args_cli.episodes):
        obs, _ = env.reset()
        policy_obs = obs["policy"][0]
        rel = policy_obs[SL_REL].clone()
        prev_actions = policy_obs[SL_PREV_ACT].clone()
        f_thresh = float(policy_obs[SL_F_THRESH].item())
        f_xyz = torch.zeros(3, device=rel.device)  # 上一步力（首步为零）

        init_xy = torch.linalg.norm(rel[0:2]).item()
        hold_z = rel[2].item()  # 复位高度（回退上限）
        z_cmd = hold_z
        phase = "A"  # 决定当步下发动作的阶段

        in_tol = 0
        b_step = None
        c_step = None
        first_contact = None
        stuck_count = 0
        spiral_r = args_cli.spiral_r0
        spiral_ang = 0.0
        f_xy_filtered = torch.zeros(2, device=rel.device)
        admittance_offset = torch.zeros(2, device=rel.device)
        neg_fz_events = 0
        descent_ok = False  # 本步是否满足下降许可（A 阶段恒为 False，B 内按上一步观测更新）
        max_force = 0.0
        success = False
        status = "TIMEOUT"

        for step in range(args_cli.max_steps):
            # ============ 1. 用上一步数据做阶段判定与状态更新 ============
            xy_err_prev = torch.linalg.norm(rel[0:2]).item()
            f_z_prev = f_xyz[2].item()
            f_norm_prev = torch.linalg.norm(f_xyz).item()

            if phase == "A":
                if xy_err_prev < args_cli.xy_tol:
                    in_tol += 1
                    if in_tol >= args_cli.hold_steps:
                        phase = "B"
                        b_step = step
                        stuck_count = 0
                else:
                    in_tol = 0
            elif phase == "B":
                # 记录本步下降许可条件（用上一步观测判定，与下文 z_cmd 更新同一依据）
                descent_ok = (
                    xy_err_prev <= args_cli.descend_gate
                    and f_z_prev < args_cli.warn_ratio * f_thresh
                    and z_cmd > args_cli.z_min
                )
                if f_norm_prev > 1.0:
                    if first_contact is None:
                        first_contact = step
                    if xy_err_prev < args_cli.c_xy_gate:
                        stuck_count += 1
                    else:
                        stuck_count = 0
                    if stuck_count >= args_cli.c_trigger_steps and args_cli.c_enable:
                        phase = "C"
                        c_step = step
                        spiral_r = args_cli.spiral_r0
                        spiral_ang = 0.0
                        admittance_offset.zero_()
                if f_z_prev < args_cli.neg_fz_alert:
                    neg_fz_events += 1  # 显著负向力（排除传感器微噪声）记为异常
                # 限力无条件生效（修复：此前 gate 外力失控导致 8 次 ABORT 中的多数）
                if f_z_prev >= args_cli.backoff_ratio * f_thresh:
                    z_cmd = min(z_cmd + args_cli.backoff_dist, hold_z)  # 回退不超过复位高度
                # 下降许可：对中保持 + 预警区间以下
                elif z_cmd > args_cli.z_min and xy_err_prev <= args_cli.descend_gate and f_z_prev < args_cli.warn_ratio * f_thresh:
                    # 硬下界：下降后不得越过 z_min（审查要求，此前可越界一个 descend_step）
                    z_cmd = max(args_cli.z_min, z_cmd - args_cli.descend_step)
                # 其余情况：预警区间内或 gate 外，暂停积分（不再下压）

            # ============ 2. 构造本步目标（含导纳偏置，真正下发） ============
            base_x = base_y = 0.0
            if phase == "C":
                spiral_ang += math.radians(args_cli.spiral_omega_deg)
                spiral_r = min(spiral_r + args_cli.spiral_dr, args_cli.spiral_rmax)
                base_x = spiral_r * math.cos(spiral_ang)
                base_y = spiral_r * math.sin(spiral_ang)
                # 导纳偏置：用上一步滤波力更新跨步状态（在构造动作前）
                f_xy_filtered = 0.3 * f_xyz[0:2] + 0.7 * f_xy_filtered
                if torch.linalg.norm(f_xy_filtered).item() > args_cli.c_force_deadband:
                    corr = args_cli.c_sign * args_cli.c_k_force * f_xy_filtered
                    cn = torch.linalg.norm(corr).item()
                    if cn > args_cli.c_max_corr:
                        corr = corr / cn * args_cli.c_max_corr
                    admittance_offset = admittance_offset + corr
                # L2 范数限幅（修复：逐分量 clamp 会使对角偏置范数达 √2×c_max_offset）
                off_norm = torch.linalg.norm(admittance_offset).item()
                if off_norm > args_cli.c_max_offset:
                    admittance_offset = admittance_offset / off_norm * args_cli.c_max_offset

            adm_x = admittance_offset[0].item()
            adm_y = admittance_offset[1].item()
            raw_x = base_x + adm_x
            raw_y = base_y + adm_y
            # 最终横向目标 L2 限幅（审查要求：单独限幅并记录限幅前值）
            pre_clip_x, pre_clip_y = raw_x, raw_y
            raw_norm = math.hypot(raw_x, raw_y)
            if raw_norm > args_cli.c_max_total:
                scale = args_cli.c_max_total / raw_norm
                raw_x *= scale
                raw_y *= scale
            tgt_x, tgt_y = raw_x, raw_y
            tgt_z = z_cmd

            # ============ 3. 记录并下发 ============
            action = make_action(
                torch.tensor([tgt_x, tgt_y, tgt_z], device=rel.device), prev_actions
            )
            obs, rew, terminated, truncated, info = env.step(action)

            policy_obs = obs["policy"][0]
            rel = policy_obs[SL_REL].clone()
            prev_actions = policy_obs[SL_PREV_ACT].clone()
            f_thresh = float(policy_obs[SL_F_THRESH].item())
            f_xyz = unwrapped.force_sensor_smooth[0, 0:3].detach().clone()
            f_norm = torch.linalg.norm(f_xyz).item()
            max_force = max(max_force, f_norm)
            applied = unwrapped.actions[0, 0:3].detach().clone()
            xy_err = torch.linalg.norm(rel[0:2]).item()

            note = ""
            if success := bool(unwrapped._get_curr_successes(success_threshold=success_threshold, check_rot=False)[0]):
                status, note = "SUCCESS", "success"
            elif f_norm > f_thresh:
                status, note = "FORCE_ABORT", "force_over_threshold"

            rows.append(
                {
                    "seed": args_cli.seed, "episode": ep, "step": step, "phase": phase,
                    "obs_x": round(rel[0].item(), 6), "obs_y": round(rel[1].item(), 6),
                    "obs_z": round(rel[2].item(), 6),
                    "base_x": round(base_x, 6), "base_y": round(base_y, 6),
                    "adm_x": round(adm_x, 6), "adm_y": round(adm_y, 6),
                    "pre_clip_x": round(pre_clip_x, 6), "pre_clip_y": round(pre_clip_y, 6),
                    "tgt_x": round(tgt_x, 6), "tgt_y": round(tgt_y, 6), "tgt_z": round(tgt_z, 6),
                    "b_descent_ok": int(descent_ok),
                    "F_x": round(f_xyz[0].item(), 4), "F_y": round(f_xyz[1].item(), 4),
                    "F_z": round(f_xyz[2].item(), 4), "force_threshold": round(f_thresh, 4),
                    "applied_x": round(applied[0].item(), 5), "applied_y": round(applied[1].item(), 5),
                    "applied_z": round(applied[2].item(), 5),
                    "xy_err": round(xy_err, 6), "z_cmd": round(z_cmd, 6),
                    "spiral_r": round(spiral_r, 6), "neg_fz_events": neg_fz_events,
                    "note": note,
                }
            )
            if note:
                break

            if args_cli.debug_every and step % args_cli.debug_every == 0:
                print(
                    f"    [dbg {step:3d}] {phase} xy={xy_err*1000:6.2f}mm z_cmd={z_cmd*1000:6.1f}mm "
                    f"F_z={f_xyz[2].item():+.2f} stuck={stuck_count} r={spiral_r*1000:.2f}mm "
                    f"adm=({adm_x*1000:+.2f},{adm_y*1000:+.2f})mm"
                )
        # for-else：未 break 即超时
        else:
            status = "TIMEOUT"

        summary.append(
            {
                "episode": ep,
                "status": status,
                "init_xy": round(init_xy, 5),
                "B@": b_step,
                "C@": c_step,
                "contact@": first_contact,
                "final_xy": round(torch.linalg.norm(rel[0:2]).item(), 5),
                "final_z_cmd": round(z_cmd, 5),
                "max_force": round(max_force, 3),
                "neg_fz_events": neg_fz_events,
            }
        )
        print(
            f"[EP {ep:02d}] {status:<11} init={init_xy*1000:5.1f}mm B@{b_step} C@{c_step} contact@{first_contact} "
            f"final_xy={summary[-1]['final_xy']*1000:5.2f}mm z_cmd={z_cmd*1000:6.1f}mm "
            f"max_F={max_force:5.2f}N negFz={neg_fz_events}"
        )

    # ---- 汇总 ----
    n = len(summary)
    n_ok = sum(1 for r in summary if r["status"] == "SUCCESS")
    print("\n[SUMMARY]")
    print(f"  成功率        : {n_ok}/{n} = {100.0*n_ok/n:.1f}%")
    ok_steps = [r["step"] for r in rows if r["note"] == "success"]
    if ok_steps:
        print(f"  成功步数      : min={min(ok_steps)} max={max(ok_steps)} mean={sum(ok_steps)/len(ok_steps):.1f}")
    print(f"  FORCE_ABORT   : {sum(1 for r in summary if r['status']=='FORCE_ABORT')}")
    print(f"  TIMEOUT       : {sum(1 for r in summary if r['status']=='TIMEOUT')}")
    reached_b = sum(1 for r in summary if r["B@"] is not None)
    reached_c = sum(1 for r in summary if r["C@"] is not None)
    contacted = sum(1 for r in summary if r["contact@"] is not None)
    print(f"  阶段分布      : 进入B {reached_b}/{n}；接触 {contacted}/{n}；进入C {reached_c}/{reached_b if reached_b else '-'}")
    print(f"  平均最大接触力: {sum(r['max_force'] for r in summary)/n:.2f} N")
    print(f"  负F_z事件总数 : {sum(r['neg_fz_events'] for r in summary)}")

    csv_path = args_cli.csv
    if csv_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs",
            f"staged_baseline_{stamp}.csv",
        )
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(csv_path.replace(".csv", "_config.json"), "w") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    print(f"[INFO]: 已写入 {csv_path}（配置快照 {os.path.basename(csv_path).replace('.csv','_config.json')}）")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
