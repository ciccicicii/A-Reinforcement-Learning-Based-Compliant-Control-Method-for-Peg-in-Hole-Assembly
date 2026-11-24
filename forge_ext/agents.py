# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""残差 PPO 的训练配置（rsl-rl 3.x）。

超参参考 Isaac Lab 的 Franka 操作任务默认值，并按 FORGE 的稀疏接触特性做了调整：
  * 每轮采样步数 32（15 Hz 控制、150 步/回合，约 1/5 个回合）；
  * gamma 0.99：插接是 100+ 步的序列决策，折扣不能太短；
  * 网络 [256, 128, 64]：二维动作不需要 LSTM，MLP 足够且显存友好。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ForgeResidualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "forge_residual_peginsert"
    # 非对称 actor-critic：actor 只看可观测（带噪）量，critic 看特权状态
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
