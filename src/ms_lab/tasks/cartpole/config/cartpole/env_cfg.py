from dataclasses import dataclass

from ms_lab.tasks.cartpole.cartpole_env_cfg import CartpoleEnvCfg
from ms_lab.asset_zoo.robots.cartpole.cartpole_constants import CARTPOLE_ROBOT_CFG

@dataclass
class CartpoleClassicEnvCfg(CartpoleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Add Robot
        self.scene.entities = {"robot": CARTPOLE_ROBOT_CFG}
