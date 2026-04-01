import gymnasium as gym

gym.register(
    id="ms_lab-Cartpole-Mozi",
    entry_point="ms_lab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:CartpoleClassicEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:CartpolePPORunnerCfg",
    },
)
