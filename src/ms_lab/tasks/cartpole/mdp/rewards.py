from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from ms_lab.managers.scene_entity_config import SceneEntityCfg


if TYPE_CHECKING:
    from ms_lab.envs.manager_based_rl_env import ManagerBasedRlEnv

def joint_pos_target_l2(
    env: ManagerBasedRlEnv, target: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    #\"\"\"Penalize joint position deviation from target.\"\"\"
    asset = env._backend.get_robot(asset_cfg.name)
    return torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - target).sum(dim=1)

def joint_vel_l1(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint velocities using L1-kernel (absolute value)."""
    asset = env._backend.get_robot(asset_cfg.name)
    return torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]).sum(dim=1)

def joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint velocities using L2-kernel (squared)."""
    asset = env._backend.get_robot(asset_cfg.name)
    return torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]).sum(dim=1)

def pole_pos_limit(
    env: ManagerBasedRlEnv, limit: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    #\"\"\"Terminate when pole angle is too large.\"\"\"
    asset = env._backend.get_robot(asset_cfg.name)
    return torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]).sum(dim=1) > limit

def cart_pos_limit(
    env: ManagerBasedRlEnv, limit: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    #\"\"\"Terminate when cart position is too far.\"\"\"
    asset = env._backend.get_robot(asset_cfg.name)
    return torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]).sum(dim=1) > limit
