import torch
from ms_lab.envs import ManagerBasedRlEnv
from ms_lab.managers.scene_entity_config import SceneEntityCfg

def joint_pos_limit(
    env: ManagerBasedRlEnv,
    lower: float,
    upper: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the joint position is outside the given limits."""
    asset = env._backend.get_robot(asset_cfg.name)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.any((joint_pos < lower) | (joint_pos > upper), dim=1)
