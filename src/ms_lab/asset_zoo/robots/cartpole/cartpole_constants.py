"""Cartpole robot constants."""

from pathlib import Path

from ms_lab import ms_lab_SRC_PATH
from ms_lab.entity import EntityArticulationInfoCfg, EntityCfg
from ms_lab.utils.spec_config import ActuatorCfg, CollisionCfg

##
# USD asset path.
##

CARTPOLE_USD: Path = (
    ms_lab_SRC_PATH / "asset_zoo" / "robots" / "cartpole" /"cartpole.usd"
)
assert CARTPOLE_USD.exists(), f"Cartpole USD file not found at {CARTPOLE_USD}"


##
# Actuator config.
##

# Cart actuator: Direct force control with damping (与IsaacLab配置一致)
# 参考: isaaclab_assets/robots/cartpole.py - cart_actuator
CARTPOLE_CART_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=["slider_to_cart"],
    effort_limit=400.0,  # 与IsaacLab一致: effort_limit_sim=400.0
    stiffness=0.0,       # 与IsaacLab一致: stiffness=0.0
    damping=10.0,        # 与IsaacLab一致: damping=10.0
    armature=0.0,
)

# Pole actuator: Passive joint with no damping (与IsaacLab配置一致)
# 参考: isaaclab_assets/robots/cartpole.py - pole_actuator
CARTPOLE_POLE_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=["cart_to_pole"],
    effort_limit=400.0,  # 与IsaacLab一致: effort_limit_sim=400.0
    stiffness=0.0,       # 与IsaacLab一致: stiffness=0.0
    damping=0.0,         # 与IsaacLab一致: damping=0.0 (自由摆动)
    armature=0.0,
)
##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 2.0),  # Cart starts at origin
    joint_pos={
        "slider_to_cart": 0.0,  # Cart at center
        "cart_to_pole": 0.0,    # Pole upright
    },
    joint_vel={
        "slider_to_cart": 0.0,
        "cart_to_pole": 0.0,
    },
)

##
# Collision config.
##

# Simple collision for cartpole - enable all collisions
# CARTPOLE_COLLISION = CollisionCfg(
#     geom_names_expr=[".*"],
#     contype=1,
#     conaffinity=1,
#     condim=3,
#     friction=(0.5,),
# )

##
# Final config.
##

CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(CARTPOLE_CART_ACTUATOR_CFG, CARTPOLE_POLE_ACTUATOR_CFG),
    # soft_joint_pos_limit_factor=0.95,
)

CARTPOLE_ROBOT_CFG = EntityCfg(
    init_state=INIT_STATE,
    # collisions=(CARTPOLE_COLLISION,),
    asset_file=str(CARTPOLE_USD),  # Use USD file
    articulation=CARTPOLE_ARTICULATION,
)
