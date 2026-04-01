from dataclasses import dataclass, field

from ms_lab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@dataclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: RslRlPpoActorCriticCfg = field(default_factory=lambda: RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        activation="elu",
    ))

    num_steps_per_env: int = 16
    max_iterations: int = 300
    save_interval: int = 50
    experiment_name: str = "cartpole"
    logger: str = "tensorboard"
    obs_groups: dict = field(default_factory=lambda: {"policy": ["policy"], "critic": ["policy"]})
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=lambda: RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ))
    multi_gpu: dict = field(default_factory=lambda: {
        "enabled": False,
        "global_rank": 0,
        "world_size": 1,
    })
