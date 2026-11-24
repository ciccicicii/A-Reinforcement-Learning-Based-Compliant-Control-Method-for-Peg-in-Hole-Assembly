# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""FORGE 官方 PPO 复现入口（rl_games，阶段二主实验）。

为什么不直接用 Isaac Lab 的 ``scripts/reinforcement_learning/rl_games/train.py``：

1. 那个脚本用 hydra 从 gym 注册表取 agent 配置，**不会 import 本仓库的 forge_ext**，
   于是 ``forge_ext`` 注册的 "带日志" 任务它看不见；
2. 官方 FORGE 任务把成功率写在 ``extras["successes"]``，而 rl_games 只记录
   ``extras["episode"]``，所以**官方训练命令的 TensorBoard 里看不到成功率**。

本脚本做两件事：用 ``forge_ext`` 注册的 ``Isaac-Forge-PegInsert-Logged-v0``
（与官方任务动力学完全等价，只多写一行日志），并原样复用官方的
``rl_games_ppo_cfg.yaml``（LSTM 1024×2 + 非对称 AC + central value）。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab

    # 冒烟：16 envs，20 epoch（确认能跑）
    ./isaaclab.sh -p ../demos/06_train_forge_rl_games.py --num_envs 16 --max_epochs 20 --headless

    # 正式：64 envs，500 epoch
    ./isaaclab.sh -p ../demos/06_train_forge_rl_games.py --num_envs 64 --max_epochs 500 --headless

    # 断点续训
    ./isaaclab.sh -p ../demos/06_train_forge_rl_games.py --num_envs 64 --max_epochs 500 \
        --checkpoint logs/rl_games/Forge/<时间戳>/nn/Forge.pth

日志：``IsaacLab/logs/rl_games/Forge/<时间戳>/``，TensorBoard::

    tensorboard --logdir logs/rl_games
    # 看 episode/successes、episode/logs_rew_contact_penalty、episode/early_term_recall/*
"""

import argparse
import math
import os
import sys

import yaml

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FORGE PegInsert PPO（rl_games）复现")
parser.add_argument("--num_envs", type=int, default=64, help="并行环境数（8GB 建议 32-96）")
parser.add_argument("--max_epochs", type=int, default=500, help="训练 epoch 数（沿用官方 yaml 的超参）")
parser.add_argument("--seed", type=int, default=0, help="随机种子（对比实验固定）")
parser.add_argument("--run_name", type=str, default=None, help="日志目录名（默认时间戳）")
parser.add_argument("--checkpoint", type=str, default=None, help="续训 checkpoint（.pth）")
parser.add_argument("--agent_yaml", type=str, default=None, help="agent 配置 yaml，默认官方 FORGE 那份")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Forge-PegInsert-Logged-v0",
    help="任务 id；改成 Isaac-Forge-PegInsert-Direct-v0 可跑官方原版（但 TB 里没有成功率）",
)
parser.add_argument(
    "--ablate",
    choices=["none", "no_fp", "no_force"],
    default="none",
    help="消融：no_fp=去掉过力惩罚；no_force=策略不看力（等于 FORGE(No Force)）",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下必须在 AppLauncher 之后 import。"""

import gymnasium as gym
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401

RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_ROOT)
import forge_ext  # noqa: E402,F401  注册 Logged / Residual 任务

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402


def _default_agent_yaml() -> str:
    import isaaclab_tasks.direct.forge.agents as forge_agents

    return os.path.join(os.path.dirname(forge_agents.__file__), "rl_games_ppo_cfg.yaml")


def main():
    device = args_cli.device or "cuda:0"

    # ---------------- 环境 ----------------
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    # ---------------- 消融 ----------------
    if args_cli.ablate == "no_fp":
        # 去掉"超过力阈值"的惩罚项（论文 Table 里的 FORGE (No FP)）
        env_cfg.task.contact_penalty_scale = 0.0
        print("[INFO] 消融: no_fp —— contact_penalty_scale = 0")
    elif args_cli.ablate == "no_force":
        # 策略不看力（但仍保留特权状态里的真值力，供 critic 用）
        env_cfg.obs_order = [k for k in env_cfg.obs_order if k != "ft_force"]
        n_obs = sum(
            {
                "fingertip_pos_rel_fixed": 3,
                "fingertip_quat": 4,
                "ee_linvel": 3,
                "ee_angvel": 3,
                "force_threshold": 1,
            }[k]
            for k in env_cfg.obs_order
        ) + 7  # prev_actions
        env_cfg.observation_space = n_obs
        print(f"[INFO] 消融: no_force —— 观测维度降为 {n_obs}")

    # ---------------- agent 配置 ----------------
    agent_yaml = args_cli.agent_yaml or _default_agent_yaml()
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)
    print(f"[INFO] agent 配置: {agent_yaml}")

    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["device"] = device
    agent_cfg["params"]["config"]["device_name"] = device
    agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_epochs
    env_cfg.seed = args_cli.seed

    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO] 加载 checkpoint: {resume_path}")

    # ---------------- 日志目录 ----------------
    config_name = agent_cfg["params"]["config"]["name"]
    log_root = os.path.abspath(os.path.join("logs", "rl_games", config_name))
    log_dir = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["params"]["config"]["train_dir"] = log_root
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    env_cfg.log_dir = os.path.join(log_root, log_dir)
    os.makedirs(env_cfg.log_dir, exist_ok=True)
    dump_yaml(os.path.join(env_cfg.log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(env_cfg.log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"[INFO] 日志目录: {env_cfg.log_dir}")

    # ---------------- 建环境 ----------------
    env = gym.make(args_cli.task, cfg=env_cfg)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # ---------------- 注册进 rl_games ----------------
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run(
        {"train": True, "play": False, "sigma": None, "checkpoint": resume_path if args_cli.checkpoint else None}
        if args_cli.checkpoint
        else {"train": True, "play": False, "sigma": None}
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
