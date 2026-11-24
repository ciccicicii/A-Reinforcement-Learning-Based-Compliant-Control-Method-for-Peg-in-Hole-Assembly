"""Franka 多航点运动 Demo（基础运控第 2 步）。

在 ``01_franka_hold.py`` 的基础上，依次到达一组末端位姿航点，形成一条
"抬升 → 横向移动 → 下降 → 返回" 的连续轨迹，并记录每个航点的收敛时间。

这一步是后续插入动作的基础：航点可以改成"预插入位姿 → 孔口上方 → 接触"，
控制结构不变。

用法::

    ./isaaclab.sh -p /home/mi/graduation_design/RL/demos/02_franka_waypoints.py --num_envs 1

    # 更换航点（每个航点 7 个数：x y z qw qx qy qz，用 ; 分隔多个航点）
    ./isaaclab.sh -p .../02_franka_waypoints.py \
        --waypoints "0.5 0 0.5 0 1 0 0;0.5 -0.2 0.45 0 1 0 0"

    # 无窗口
    ./isaaclab.sh -p .../02_franka_waypoints.py --num_envs 1 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

DEFAULT_WAYPOINTS = [
    [0.50, 0.00, 0.50, 0.0, 1.0, 0.0, 0.0],  # 起始/返回位姿
    [0.50, -0.15, 0.45, 0.0, 1.0, 0.0, 0.0],  # 横向移动到一侧
    [0.50, 0.15, 0.45, 0.0, 1.0, 0.0, 0.0],  # 横向移动到另一侧
    [0.50, 0.00, 0.38, 0.0, 1.0, 0.0, 0.0],  # 在中心位置下降
    [0.50, 0.00, 0.50, 0.0, 1.0, 0.0, 0.0],  # 返回起始位姿
]

parser = argparse.ArgumentParser(description="Franka multi-waypoint motion demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Keep at 1 on 8GB GPUs.")
parser.add_argument("--waypoints", type=str, default=None, help="Waypoints as 'x y z qw qx qy qz' groups separated by ';'.")
parser.add_argument("--max_steps", type=int, default=600, help="Max steps allowed per waypoint.")
parser.add_argument("--pos_tol", type=float, default=0.005, help="Position tolerance in meters.")
parser.add_argument("--rot_tol_deg", type=float, default=2.0, help="Orientation tolerance in degrees.")
parser.add_argument("--hold_steps", type=int, default=20, help="Consecutive in-tolerance steps required to advance.")
parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
parser.add_argument("--log_every", type=int, default=40, help="Print progress every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import math
import os
from datetime import datetime

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """桌面 + Franka 场景，与官方差分 IK 教程一致。"""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def quat_angle_error(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    """两个四元数之间的夹角（弧度），输入为 w x y z。"""
    dot = torch.abs(torch.sum(quat_a * quat_b, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def parse_waypoints(text: str | None) -> list[list[float]]:
    """把 'x y z qw qx qy qz;...' 解析成航点列表。"""
    if text is None:
        return DEFAULT_WAYPOINTS
    waypoints = []
    for chunk in text.split(";"):
        values = [float(v) for v in chunk.split()]
        if len(values) != 7:
            raise ValueError(f"每个航点需要 7 个数（x y z qw qx qy qz），收到 {len(values)} 个: {chunk!r}")
        waypoints.append(values)
    if not waypoints:
        raise ValueError("航点列表为空")
    return waypoints


def run_simulator(sim, scene, waypoints: torch.Tensor, csv_path: str):
    robot = scene["robot"]

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    sim_dt = sim.get_physics_dt()
    rot_tol = math.radians(args_cli.rot_tol_deg)

    print(f"[INFO]: 共 {waypoints.shape[0]} 个航点，容差 pos={args_cli.pos_tol} m / rot={args_cli.rot_tol_deg} deg")

    records = []
    summary = []
    global_step = 0

    for wp_idx in range(waypoints.shape[0]):
        target = waypoints[wp_idx]
        ik_commands[:] = target
        diff_ik_controller.reset()
        diff_ik_controller.set_command(ik_commands)
        print(f"[WAYPOINT {wp_idx}] 目标: {[round(v, 4) for v in target.tolist()]}")

        in_tol_count = 0
        converge_step = None
        last_pos_err = float("inf")
        last_rot_err = float("inf")

        for _ in range(args_cli.max_steps):
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            global_step += 1

            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            pos_err = torch.norm(ee_pos_b - ik_commands[:, 0:3], dim=-1)
            rot_err = quat_angle_error(ee_quat_b, ik_commands[:, 3:7])
            last_pos_err = pos_err[0].item()
            last_rot_err = rot_err[0].item()

            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

            records.append(
                {
                    "step": global_step,
                    "waypoint": wp_idx,
                    "t": round(global_step * sim_dt, 4),
                    "ee_x": round(ee_pos_b[0, 0].item(), 6),
                    "ee_y": round(ee_pos_b[0, 1].item(), 6),
                    "ee_z": round(ee_pos_b[0, 2].item(), 6),
                    "pos_err": round(last_pos_err, 6),
                    "rot_err_rad": round(last_rot_err, 6),
                }
            )

            if last_pos_err < args_cli.pos_tol and last_rot_err < rot_tol:
                in_tol_count += 1
                if in_tol_count >= args_cli.hold_steps and converge_step is None:
                    converge_step = global_step
            else:
                in_tol_count = 0

            if global_step % args_cli.log_every == 0:
                print(
                    f"[step {global_step:4d}] wp={wp_idx} pos_err={last_pos_err:.4f} m, "
                    f"rot_err={math.degrees(last_rot_err):.2f} deg"
                )

            if converge_step is not None:
                break

        status = "CONVERGED" if converge_step is not None else "TIMEOUT"
        summary.append(
            {
                "waypoint": wp_idx,
                "status": status,
                "steps": global_step if converge_step is None else converge_step,
                "final_pos_err": round(last_pos_err, 6),
                "final_rot_err_deg": round(math.degrees(last_rot_err), 2),
            }
        )
        print(
            f"[WAYPOINT {wp_idx}] {status}: 末误差 pos={last_pos_err:.4f} m / "
            f"rot={math.degrees(last_rot_err):.2f} deg"
        )

    print("\n[SUMMARY]")
    for row in summary:
        print(
            f"  wp {row['waypoint']}: {row['status']:<9} "
            f"pos_err={row['final_pos_err']:.4f} m  rot_err={row['final_rot_err_deg']:.2f} deg"
        )

    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"[INFO]: 已写入 {csv_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.6], [0.5, 0.0, 0.4])

    scene = InteractiveScene(TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5))
    sim.reset()
    print("[INFO]: Setup complete...")

    waypoints = torch.tensor(parse_waypoints(args_cli.waypoints), device=sim.device, dtype=torch.float32)

    csv_path = args_cli.csv
    if csv_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", f"franka_waypoints_{stamp}.csv"
        )

    run_simulator(sim, scene, waypoints, csv_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
