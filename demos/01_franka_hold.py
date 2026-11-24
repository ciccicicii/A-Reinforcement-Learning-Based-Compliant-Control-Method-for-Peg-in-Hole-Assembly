# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""Franka 末端位姿保持 Demo（基础运控第 1 步）。

与官方 ``run_diff_ik.py`` 的区别：
    * 目标位姿固定，不周期性重置、不切换目标；
    * 记录并输出末端跟踪误差，用于判断控制链是否真正收敛；
    * 结果写入 CSV，可作为毕设中的基础运控实验记录。

用法::

    ./isaaclab.sh -p ../demos/01_franka_hold.py --num_envs 1

    # 无窗口
    ./isaaclab.sh -p ../demos/01_franka_hold.py --num_envs 1 --headless

    # 自定义目标位姿（位置 + 四元数 w x y z）
    ./isaaclab.sh -p ../demos/01_franka_hold.py \
        --target 0.5 0.0 0.45 0.0 1.0 0.0 0.0 --steps 500
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka end-effector pose hold demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn. Keep at 1 on 8GB GPUs.")
parser.add_argument("--steps", type=int, default=400, help="Number of simulation steps to run.")
parser.add_argument(
    "--target",
    type=float,
    nargs=7,
    default=[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
    help="Target end-effector pose in the robot root frame: x y z qw qx qy qz.",
)
parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
parser.add_argument("--log_every", type=int, default=20, help="Print the tracking error every N steps.")
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
    """桌面 + Franka 的场景配置，与官方差分 IK 教程保持一致。"""

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
    """两个四元数之间的夹角（弧度），输入为 w x y z 顺序。"""
    dot = torch.abs(torch.sum(quat_a * quat_b, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, target: torch.Tensor, steps: int, csv_path: str):
    robot = scene["robot"]

    # 控制器：绝对位姿命令 + 阻尼最小二乘 IK
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # 目标位姿可视化标记
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # 初始化：写回默认关节状态
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    # 设定固定目标，之后不再改变
    ik_commands = target.repeat(scene.num_envs, 1).to(robot.device)
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands)

    sim_dt = sim.get_physics_dt()
    print(f"[INFO]: 目标位姿（root 系）: {ik_commands[0].tolist()}")
    print(f"[INFO]: 仿真步数: {steps}, dt: {sim_dt}, 环境数: {scene.num_envs}")

    records = []
    for step in range(steps):
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

        # 实际末端位姿与跟踪误差（使用 step 之后的新状态，避免一步滞后）
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        goal_pose_b = ik_commands
        pos_err = torch.norm(ee_pos_b - goal_pose_b[:, 0:3], dim=-1)
        rot_err = quat_angle_error(ee_quat_b, goal_pose_b[:, 3:7])

        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(goal_pose_b[:, 0:3] + scene.env_origins, goal_pose_b[:, 3:7])

        records.append(
            {
                "step": step,
                "t": round(step * sim_dt, 4),
                "ee_x": round(ee_pos_b[0, 0].item(), 6),
                "ee_y": round(ee_pos_b[0, 1].item(), 6),
                "ee_z": round(ee_pos_b[0, 2].item(), 6),
                "pos_err": round(pos_err[0].item(), 6),
                "rot_err_rad": round(rot_err[0].item(), 6),
            }
        )

        if step % args_cli.log_every == 0:
            print(
                f"[step {step:4d}] pos_err={pos_err[0].item():.4f} m, "
                f"rot_err={math.degrees(rot_err[0].item()):.2f} deg"
            )

    # 收敛结论用最后 10% 的步数判断，避免把启动瞬态算进去
    tail = records[len(records) * 9 // 10 :]
    final_pos_err = max(r["pos_err"] for r in tail)
    final_rot_err = math.degrees(max(r["rot_err_rad"] for r in tail))
    print("[RESULT]: 末段最大位置误差 %.4f m，最大姿态误差 %.2f deg" % (final_pos_err, final_rot_err))

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

    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    target = torch.tensor(args_cli.target, device=sim.device, dtype=torch.float32)

    csv_path = args_cli.csv
    if csv_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", f"franka_hold_{stamp}.csv")

    run_simulator(sim, scene, target, args_cli.steps, csv_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
