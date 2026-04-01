"""Cartpole Task Configuration."""

import math
from dataclasses import dataclass, field

from ms_lab.envs import ManagerBasedRlEnvCfg
from ms_lab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg, EventTermCfg, term
from ms_lab.managers.scene_entity_config import SceneEntityCfg
from ms_lab.scene import SceneCfg
from ms_lab.terrains.terrain_importer import TerrainImporterCfg
from ms_lab.viewer import ViewerConfig
from ms_lab.tasks.cartpole import mdp
import ms_lab.envs.mdp as mdp_std

##
# Scene.
##

SCENE_CFG = SceneCfg(
    num_envs=4096,
    env_spacing=4.0,
    terrain=TerrainImporterCfg(env_spacing=4.0),
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="cart",  
    distance=10.0,
    elevation=10.0,
    azimuth=0.0,
)

@dataclass
class ActionsCfg:
    """Action specifications for the environment."""
    # Use JointEffortsActionCfg for direct torque control
    joint_effort: mdp.JointEffortsActionCfg = term(
        mdp.JointEffortsActionCfg,
        asset_name="robot",
        actuator_names=["slider_to_cart"],
        scale=100.0,  # Scale factor for the action
    )

@dataclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @dataclass
    class PolicyCfg(ObservationGroupCfg):
        # Joint positions
        joint_pos: ObservationTermCfg = field(default_factory=lambda: ObservationTermCfg(
            func=mdp_std.observations.joint_pos_rel,
        ))
        # Joint velocities
        joint_vel: ObservationTermCfg = field(default_factory=lambda: ObservationTermCfg(
            func=mdp_std.observations.joint_vel_rel,
        ))

    policy: PolicyCfg = field(default_factory=PolicyCfg)


@dataclass
class RewardsCfg:
    """Reward terms for the environment."""
    
    # (1) Constant running reward
    alive: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(func=mdp_std.rewards.is_alive, weight=1.0))
    
    # (2) Failure penalty
    terminating: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(func=mdp_std.rewards.is_terminated, weight=-2.0))
    
    # (3) Primary task: keep pole upright
    pole_pos: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"target": 0.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])}
    ))

    # (4) Shaping tasks: lower cart velocity
    cart_vel: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])}
    ))
    
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel: RewardTermCfg = field(default_factory=lambda: RewardTermCfg(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])}
    ))


@dataclass
class EventCfg:
    """Event specifications for the environment."""

    # Reset cart joint with a uniform random offset from default (default=0).
    # position_range: cart position offset in metres [-1.0, +1.0]
    # velocity_range: cart velocity offset in m/s    [-0.5, +0.5]
    reset_cart_position: EventTermCfg = term(
        EventTermCfg,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
        },
    )

    # Reset pole joint with a uniform random offset from default (default=0).
    # position_range: pole angle offset in radians  [-0.25*pi, +0.25*pi]  (~±45°)
    # velocity_range: pole angular velocity offset  [-0.25*pi, +0.25*pi]  rad/s
    reset_pole_position: EventTermCfg = term(
        EventTermCfg,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
        },
    )


@dataclass
class TerminationsCfg:
    """Termination terms for the environment."""
    
    # (1) Time out - 必须设置time_out=True，否则PPO不会对超时状态做value bootstrap
    time_out: TerminationTermCfg = field(default_factory=lambda: TerminationTermCfg(func=mdp_std.terminations.time_out, time_out=True))
    
    # (2) Cart out of bounds
    cart_out_of_bounds: TerminationTermCfg = field(default_factory=lambda: TerminationTermCfg(
        func=mdp.cart_pos_limit,
        params={
            "limit": 3.0, 
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])
        }
    ))
    
    

@dataclass
class CartpoleEnvCfg(ManagerBasedRlEnvCfg):
    """Configuration for the Cartpole environment."""
    
    # Required parameters from parent classes
    decimation: int = 2  # Control frequency decimation
    episode_length_s: float = 5.0
    
    # Scene settings
    scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
    viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)

    # Basic Settings
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    events: EventCfg = field(default_factory=EventCfg)

    def __post_init__(self):
        """Post initialization."""
        # 注意: mozi中必须设置 sim.mujoco.timestep，而不是 sim.dt（后者无效）
        self.sim.mujoco.timestep = 1.0 / 120.0
