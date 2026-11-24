"""FORGE PegInsert 环境检查（进入 FORGE 的第一步）。

不做控制，只把环境的接口摊开看清楚：
    1. 动作空间到底几维、每一位是什么；
    2. 观测/状态向量的每一段对应哪个物理量；
    3. 复位后销钉与孔的相对位姿、接触力。
    4. 连续施加零动作时这些量如何变化。

这些信息决定后续"基础控制 + 残差"怎么接。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab
    ./isaaclab.sh -p /home/mi/graduation_design/RL/demos/03_forge_inspect.py --num_envs 1

    # 只看接口，不看画面
    ./isaaclab.sh -p /home/mi/graduation_design/RL/demos/03_forge_inspect.py --num_envs 1 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect the FORGE peg-insertion environment.")
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=5, help="Number of zero-action steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def build_slices(names: list[str], dims: dict) -> dict[str, slice]:
    """把观测名列表转成 name -> slice 的映射，用于按名字取值。"""
    out = {}
    start = 0
    for name in names:
        if name not in dims:
            raise KeyError(f"未知观测项 {name!r}，已知: {sorted(dims)}")
        out[name] = slice(start, start + dims[name])
        start += dims[name]
    return out


def print_slices(title: str, vector: torch.Tensor, slices: dict[str, slice]):
    print(f"\n--- {title}（共 {vector.shape[-1]} 维）---")
    for name, sl in slices.items():
        values = vector[sl].tolist()
        pretty = [round(v, 4) for v in values]
        print(f"  {name:<24} [{sl.start}:{sl.stop}] = {pretty}")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    print(f"\n[INFO]: 任务 {args_cli.task}")
    print(f"[INFO]: Gym action space  = {env.action_space}")
    print(f"[INFO]: Gym obs space     = {env.observation_space}")

    from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG

    # prev_actions 不在 factory 的维度表里，维度等于动作维度
    action_dim = int(env.action_space.shape[-1])
    obs_dims = dict(OBS_DIM_CFG)
    obs_dims["prev_actions"] = action_dim
    state_dims = dict(STATE_DIM_CFG)
    state_dims["prev_actions"] = action_dim

    obs_order = list(env_cfg.obs_order) + ["prev_actions"]
    state_order = list(env_cfg.state_order) + ["prev_actions"]
    obs_slices = build_slices(obs_order, obs_dims)
    state_slices = build_slices(state_order, state_dims)

    print(f"[INFO]: 动作维度 = {env.action_space.shape[-1]}")
    print("        FORGE 动作为 7 维：前 6 维是相对孔坐标系的任务空间位姿增量，")
    print("        第 7 维是 [−1,1] 的成功预测头，不是运动指令。")

    obs, _ = env.reset()
    print_slices("policy 观测（复位后）", obs["policy"][0], obs_slices)
    print_slices("critic 状态（复位后）", obs["critic"][0], state_slices)

    # 资产位姿：确认销钉与孔的初始关系
    try:
        fixed = unwrapped.scene["fixed_asset"]
        held = unwrapped.scene["held_asset"]
        print("\n[INFO]: fixed_asset（孔）位置 :", [round(v, 4) for v in fixed.data.root_pos_w[0].tolist()])
        print("[INFO]: held_asset（销）位置   :", [round(v, 4) for v in held.data.root_pos_w[0].tolist()])
    except Exception as exc:  # 资产名可能随版本变化，取不到不算失败
        print(f"[WARN]: 无法直接读取资产位姿: {exc}")

    # 零动作滚动，观察接触力与相对位姿的变化
    actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
    print(f"\n[INFO]: 连续施加零动作 {args_cli.steps} 步")
    for i in range(args_cli.steps):
        obs, rew, terminated, truncated, info = env.step(actions)
        rel_pos = obs["policy"][0][obs_slices["fingertip_pos_rel_fixed"]]
        force = obs["policy"][0][obs_slices["ft_force"]]
        force_norm = torch.linalg.norm(force).item()
        print(
            f"  step {i}: pos_rel_fixed={[round(v, 4) for v in rel_pos.tolist()]} "
            f"|F|={force_norm:.3f} N  reward={rew[0].item():.4f} "
            f"terminated={bool(terminated[0])} truncated={bool(truncated[0])}"
        )

    # 成功判据（若该版本提供）
    try:
        successes = unwrapped._get_curr_successes(success_threshold=env_cfg.task.success_threshold, check_rot=False)
        print(f"\n[INFO]: 当前成功标志 = {successes.tolist()}")
        print(f"[INFO]: success_threshold = {env_cfg.task.success_threshold}")
    except Exception as exc:
        print(f"[WARN]: 无法读取成功判据: {exc}")

    print("\n[INFO]: 检查完成。零动作不插入是预期结果。")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
