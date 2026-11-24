# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""残差 PPO 训练入口（阶段三）。

用自己的入口脚本而不是 Isaac Lab 的 ``scripts/reinforcement_learning/rsl_rl/train.py``，
是因为后者通过 hydra 从 gym 注册表里取 agent 配置，而我们的 ``forge_ext`` 是仓库外的包，
必须在 AppLauncher 之后显式 import；自己写入口最省事、也最容易调试。

用法::

    cd /home/mi/graduation_design/RL/IsaacLab
    # 小规模冒烟（约 10 分钟，确认能跑通）
    ./isaaclab.sh -p ../demos/04_train_residual.py --num_envs 16 --max_iterations 20 --headless

    # 正式训练（8GB 显存建议 64 环境，约 8-12 小时）
    ./isaaclab.sh -p ../demos/04_train_residual.py --num_envs 64 --max_iterations 1500 --headless

    # 断点续训
    ./isaaclab.sh -p ../demos/04_train_residual.py --resume --load_run 2026-11-23_10-00-00

日志：``RL/outputs/rsl_rl/<时间戳>_<run_name>/``，TensorBoard 指向 ``RL/outputs/rsl_rl``。
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FORGE 残差 PPO 训练")
parser.add_argument("--num_envs", type=int, default=64, help="并行环境数（8GB 显存建议 32-96）")
parser.add_argument("--max_iterations", type=int, default=1000, help="PPO 迭代次数")
parser.add_argument("--seed", type=int, default=0, help="随机种子（对比实验需固定）")
parser.add_argument("--run_name", type=str, default="residual", help="日志目录后缀")
parser.add_argument("--resume", action="store_true", default=False, help="断点续训")
parser.add_argument("--load_run", type=str, default=".*", help="续训时匹配的运行目录（正则）")
parser.add_argument("--checkpoint", type=str, default=None, help="续训时加载的 checkpoint 文件名（正则）")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_ext  # noqa: E402,F401  —— 注册 Isaac-Forge-PegInsert-Residual-v0
from forge_ext.agents import ForgeResidualPPORunnerCfg  # noqa: E402
from forge_ext.residual_env import ForgeResidualCfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

TASK_ID = "Isaac-Forge-PegInsert-Residual-v0"
RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    device = args_cli.device or "cuda:0"

    env_cfg = ForgeResidualCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = device

    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    agent_cfg = ForgeResidualPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = device
    agent_cfg.max_iterations = args_cli.max_iterations

    log_root = os.path.join(RL_ROOT, "outputs", "rsl_rl")
    log_dir = os.path.join(log_root, f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args_cli.run_name}")
    print(f"[INFO] 日志目录: {log_dir}")
    print(f"[INFO] num_envs={args_cli.num_envs}  max_iterations={args_cli.max_iterations}  seed={args_cli.seed}")

    if args_cli.resume:
        resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint or "model_.*.pt")
        agent_cfg.resume = True
        agent_cfg.load_run = args_cli.load_run
        print(f"[INFO] 从 {resume_path} 续训")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.resume:
        runner.load(resume_path)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
