"""
PhysX物理仿真后端核心模块
==========================
该模块实现了基于PhysX引擎的机器人仿真后端，提供场景管理、实体控制、
物理计算、域随机化、渲染等核心功能，适配Mozisim/IsaacLab仿真框架。

核心功能:
1. 多环境并行物理仿真
2. 机器人/物体实体管理
3. 地形生成与场景构建
4. 物理参数域随机化（质量、摩擦系数等）
5. 多模式渲染（可视化窗口/图像数组/纯物理仿真）
"""
import numpy as np
import torch

# 导入ms_lab的PhysX仿真核心模块
from ms_lab.ms_physx.sim import Simulation  # PhysX仿真器核心类：负责物理计算、步进、状态更新
from ms_lab.ms_physx.scene import Scene  # 仿真场景管理类：负责地形、实体、环境的创建与管理
from ms_lab.ms_physx.entity import Entity  # 实体（机器人/物体）管理类：封装机器人关节、物理参数、状态

# 导入Mozisim仿真应用框架
from mozisim.core.simulation_app import SimulationApp  # 仿真应用入口类：管理仿真生命周期、渲染窗口
from mozisim.utils.mozi_utils import MoziUtils  # Mozisim通用工具类：提供物理-USD同步、数据转换等工具
from mozisim.utils.stage_utils import save_stage  # 场景保存工具（USD格式）：将仿真场景保存为USD文件，用于离线查看

# 导入IsaacLab数学采样工具
from ms_lab.third_party.isaaclab.isaaclab.utils.math import (
    sample_log_uniform,  # 对数均匀分布采样函数：适合数值范围跨度大的参数（如质量、摩擦系数）
    sample_uniform,  # 均匀分布采样函数：适合数值范围跨度小的参数
)

# ==============================
# 全局配置 - 域随机化字段维度定义
# ==============================
# 定义支持域随机化的物理参数及其维度
# - body_mass: 机器人本体质量，标量（维度1）
# - foot_friction: 机器人脚部摩擦系数，向量（维度3，对应x/y/z轴）
FIELD_DIM = {
    "body_mass": 1,
    "foot_friction": 3,
}


def _sample_distribution(
        distribution: str,
        lower: torch.Tensor,
        upper: torch.Tensor,
        shape: tuple,
        device: str,
) -> torch.Tensor:
    """
    从指定概率分布中采样生成随机张量（域随机化核心工具函数）

    参数:
        distribution: str，分布类型
            - "uniform": 均匀分布（等概率采样）
            - "log_uniform": 对数均匀分布（小数值采样概率更高）
            - "gaussian": 高斯分布（需补充导入sample_gaussian函数）
        lower: torch.Tensor，采样下界（与输出同维度，必须≤upper）
        upper: torch.Tensor，采样上界（与输出同维度，必须≥lower）
        shape: tuple，输出张量的形状（例如：[num_envs, 1] 表示每个环境1个采样值）
        device: str，计算设备（"cpu"/"cuda"，需与仿真器设备一致）

    返回:
        torch.Tensor，指定分布的随机采样结果，形状=shape，设备=device，dtype=float32

    异常:
        ValueError: 传入不支持的分布类型时抛出
        NotImplementedError: 高斯分布未实现时抛出
    """
    # 统一转换为float32张量，确保数值类型一致
    lower = lower.to(dtype=torch.float32, device=device)
    upper = upper.to(dtype=torch.float32, device=device)

    if distribution == "uniform":
        # 均匀分布采样：lower ≤ x ≤ upper
        return sample_uniform(lower, upper, shape, device=device)
    elif distribution == "log_uniform":
        # 对数均匀分布采样：log(lower) ≤ log(x) ≤ log(upper) → x ∈ [lower, upper]
        return sample_log_uniform(lower, upper, shape, device=device)
    elif distribution == "gaussian":
        # 高斯分布采样：均值=(lower+upper)/2，标准差=(upper-lower)/6（99.7%数据落在[lower,upper]）
        # 注意：需补充导入from ms_lab.third_party.isaaclab.isaaclab.utils.math import sample_gaussian
        raise NotImplementedError("高斯分布采样函数sample_gaussian尚未导入，请补充导入后使用")
        # return sample_gaussian(lower, upper, shape, device=device)
    else:
        # 抛出不支持的分布类型异常，提示合法选项
        raise ValueError(f"不支持的分布类型: {distribution}，仅支持uniform/log_uniform/gaussian")


class PhysxBackend:
    """
    PhysX物理仿真后端核心类

    负责完整的仿真生命周期管理：
    1. 仿真应用初始化（渲染模式配置）
    2. 场景构建（地形、机器人、物体）
    3. 物理计算（步进、重置、状态更新）
    4. 域随机化（物理参数随机调整）
    5. 渲染输出（可视化窗口/图像数组）
    6. 资源释放（仿真应用关闭）

    核心属性:
        device: str，计算设备（cpu/cuda）
        is_headless: bool，是否无头模式（无可视化窗口）
        is_render: bool，是否启用渲染（输出图像数据）
        scene: Scene，仿真场景对象（只读属性）
        sim: Simulation，PhysX仿真器对象（只读属性）
        num_envs: int，并行仿真环境数量（只读属性）
        entities: dict，机器人实体字典 {实体名: Entity对象}
        objects: dict，场景物体字典 {物体名: 物体对象}
        terrain: 地形对象，场景地形管理
    """

    def __init__(self, render_mode: str, device: str):
        """
        初始化PhysX仿真后端

        参数:
            render_mode: str，渲染模式（决定仿真的可视化方式）
                - "human": 人类可视化模式（显示窗口，支持鼠标/键盘交互）
                - "rgb_array": 图像数组模式（无头，输出RGB图像数据用于算法处理）
                - 其他值: 纯物理仿真（无头，无渲染，计算效率最高）
            device: str，计算设备（"cpu"/"cuda"，建议使用cuda加速物理计算）

        初始化流程:
            1. 配置基础参数（设备、渲染模式）
            2. 初始化Mozisim工具类
            3. 创建实体/物体存储字典
            4. 初始化仿真应用（SimulationApp）
            5. 延迟初始化场景/仿真器（通过_init_scene/_init_sim）
        """
        # 基础配置：保存计算设备（后续所有张量都使用该设备）
        self.device = device
        # 初始化Mozisim工具类：提供物理状态与USD可视化同步等功能
        self.utils = MoziUtils()

        # 实体/物体存储字典：按名称索引，方便快速访问
        self.entities = {}  # 机器人实体字典 {实体名: Entity对象}
        self.objects = {}  # 场景物体字典 {物体名: 物体对象}

        # 渲染模式配置：根据输入模式设置无头/渲染标志
        if render_mode == "human":
            self.is_headless = False  # 非无头模式（显示可视化窗口）
            self.is_render = True  # 启用渲染（窗口渲染）
        elif render_mode == "rgb_array":
            self.is_headless = True  # 无头模式（无窗口）
            self.is_render = True  # 启用渲染（输出图像数组）
        else:
            self.is_headless = True  # 无头模式（无窗口）
            self.is_render = False  # 禁用渲染（纯物理仿真，速度最快）

        # 初始化仿真应用（Mozisim核心入口）：管理仿真生命周期和渲染上下文
        self.app = SimulationApp(
            config={
                "headless": self.is_headless,  # 是否无头模式（控制窗口显示）
                "is_output_texture_buff": self.is_render  # 是否输出纹理缓冲区（用于获取图像数据）
            }
        )

        # 场景相关初始化（延迟到_init_scene/_init_sim调用时初始化）
        self.terrain = None  # 地形对象：存储地形生成的高度图、环境原点等
        self._scene = None  # 私有场景对象（通过@property暴露，防止外部修改）
        self._sim = None  # 私有仿真器对象（通过@property暴露，防止外部修改）

    # ==============================
    # 只读属性访问器（安全访问私有变量）
    # ==============================
    @property
    def scene(self) -> Scene:
        """获取场景对象（只读）

        安全访问私有_scene变量，确保场景已初始化，避免空指针异常
        """
        if self._scene is None:
            raise RuntimeError("场景尚未初始化，请先调用_init_scene方法")
        return self._scene

    @property
    def sim(self) -> Simulation:
        """获取仿真器对象（只读）

        安全访问私有_sim变量，确保仿真器已初始化，避免空指针异常
        """
        if self._sim is None:
            raise RuntimeError("仿真器尚未初始化，请先调用_init_sim方法")
        return self._sim

    @property
    def num_envs(self) -> int:
        """获取并行仿真环境数量（只读）

        从场景对象中获取环境数量，确保与场景配置一致
        """
        return self.scene.num_envs

    # ==============================
    # 初始化相关方法（内部调用，完成场景/仿真器构建）
    # ==============================
    def init_scene(self, cfg):
        """
        初始化仿真场景（内部方法，需在_init_sim前调用）

        完成场景构建的核心步骤：
        1. 配置地形环境数量（与总环境数同步）
        2. 创建场景对象（加载配置中的地形、实体、物体）
        3. 添加地形到场景（生成高度图、环境原点）
        4. 加载机器人实体配置并添加到场景（生成USD路径）
        5. 初始化环境原点坐标（默认全零，地形生成后覆盖）

        参数:
            cfg: 配置对象（如Hydra配置），需包含以下字段：
                - num_envs: 并行环境数量（int）
                - terrain: 地形配置（包含类型、尺寸、生成参数）
                - entities: 机器人实体配置字典（{实体名: 实体配置}）
                - objects: 场景物体配置字典（可选，{物体名: 物体配置}）

        输出:
            初始化self._scene、self.terrain、self.ent_prim_dict、self.ent_cfg_dict等属性
        """
        # 同步地形环境数量与总环境数：确保地形生成的环境数与仿真器一致
        cfg.terrain.num_envs = cfg.num_envs

        # 创建场景对象：传入配置和设备，初始化场景上下文
        self._scene = Scene(cfg, device=self.device)

        # 保存配置引用：后续初始化仿真器时使用
        self.ent_cfgs = cfg.entities  # 机器人实体配置
        self.object_cfgs = cfg.objects  # 场景物体配置

        # 添加地形到场景：生成地形网格、高度图、环境原点
        self.scene.add_terrain()
        self.terrain = self.scene.terrain  # 保存地形对象，方便后续访问

        # 实体USD路径/配置存储（用于后续仿真器初始化）
        self.ent_prim_dict = {}  # {实体名: 实体USD路径列表}：每个实体对应多个环境的USD路径
        self.ent_cfg_dict = {}  # {实体名: 实体配置}：保存实体的原始配置

        # 遍历添加所有机器人实体到场景
        for ent_name, ent_cfg in self.ent_cfgs.items():
            # 添加机器人到场景：生成实体的USD路径（每个环境一个路径）
            ent_prim_paths = self.scene.add_robot(ent_name, ent_cfg)
            self.ent_prim_dict[ent_name] = ent_prim_paths
            self.ent_cfg_dict[ent_name] = ent_cfg

        # 初始化默认环境原点：所有环境初始位置为(0,0,0)，地形生成后会覆盖
        self._default_env_origins = torch.zeros(
            (cfg.num_envs, 3), device=self.device, dtype=torch.float32
        )

    def init_sim(self, cfg):
        """
        初始化PhysX仿真器（内部方法，需在_init_scene后调用）

        完成仿真器构建的核心步骤：
        1. 创建PhysX仿真器对象（加载PhysX引擎配置）
        2. 初始化机器人实体并添加到仿真器（绑定USD路径和配置）
        3. 加载场景物体并添加到仿真器（同步环境数量）
        4. 初始化域随机化字段维度（修正拼写错误，确保与全局配置一致）

        参数:
            cfg: 仿真器配置对象，需包含PhysX引擎相关配置：
                - dt: 仿真步长（秒）
                - gravity: 重力加速度（m/s²）
                - solver_iterations: 求解器迭代次数
                - gpu: 是否使用GPU加速

        输出:
            初始化self._sim、self.entities、self.objects等属性
        """
        # 创建PhysX仿真器核心对象：传入配置、设备、渲染标志
        self._sim = Simulation(
            cfg=cfg,
            device=self.device,
            is_render=self.is_render,
        )

        # 为每个机器人实体创建Entity对象并添加到仿真器
        for ent_name, ent_prim_paths in self.ent_prim_dict.items():
            ent_cfg = self.ent_cfg_dict[ent_name]
            # 创建实体对象：绑定配置、USD路径、设备
            ent = Entity(ent_cfg, ent_prim_paths, device=self.device)
            self.entities[ent_name] = ent  # 保存到实体字典

        # 添加场景物体（如果有配置）
        if self.object_cfgs is not None:
            for object_name, object_cfg in self.object_cfgs.items():
                object_cfg.num_envs = self.num_envs  # 同步环境数量
                # 添加物体到场景并保存到物体字典
                obj = self.scene.add_object(object_name, object_cfg)
                self.objects[object_name] = obj

        # 域随机化字段维度配置（修正原代码拼写错误：body_friction -> foot_friction）
        self.field_dim = {
            "body_mass": 1,  # 机器人本体质量：标量（每个实体1个值）
            "foot_friction": 3,  # 机器人脚部摩擦系数：向量（x/y/z轴各1个值）
        }

        # 调试用：保存场景到USD文件（如需启用请取消注释）
        # save_stage("./rough_scene_0107_3072.usda")
        # exit()

    # ==============================
    # 核心功能方法（对外提供的仿真控制接口）
    # ==============================
    def expand_model_fields(self, domain_randomization_fields: list):
        """
        扩展仿真模型的域随机化字段（预注册需要随机化的参数）

        作用：告诉仿真器需要跟踪哪些物理参数，为后续随机化做准备
        调用时机：初始化仿真器后，首次随机化前

        参数:
            domain_randomization_fields: list[str]，需要随机化的字段名列表
                例如: ["body_mass", "foot_friction"]
        """
        self._sim.expand_model_fields(domain_randomization_fields)

    def create_graph(self):
        """创建PhysX GPU计算图（加速物理仿真计算）

        作用：将物理计算流程编译为GPU计算图，避免重复编译，提升仿真速度
        调用时机：初始化仿真器后，首次仿真步前（仅需调用一次）
        """
        self._sim.create_graph()

    def get_robot(self, asset_name: str) -> Entity:
        """
        获取指定名称的机器人实体对象

        参数:
            asset_name: str，机器人实体名称（需与配置中的名称一致）

        返回:
            Entity，对应的机器人实体对象（可访问关节状态、物理参数等）

        异常:
            KeyError: 实体名称不存在时抛出，提示已加载的实体列表
        """
        if asset_name not in self.entities:
            raise KeyError(
                f"机器人实体 {asset_name} 不存在，已加载的实体: {list(self.entities.keys())}"
            )
        return self.entities[asset_name]

    def get_all_robots(self) -> list:
        """
        获取所有已加载的机器人实体对象列表

        返回:
            list[Entity]，所有机器人实体对象（按添加顺序排列）
        """
        return list(self.entities.values())

    def get_terrain(self):
        """
        获取地形对象

        返回:
            场景地形对象（Terrain类实例），可访问高度图、环境原点、地形类型等
        """
        return self.terrain

    def get_env_origins(self) -> torch.Tensor:
        """
        获取所有仿真环境的原点坐标

        返回:
            torch.Tensor，形状[num_envs, 3]，dtype=float32，存储每个环境的(x,y,z)原点坐标
            - 优先返回地形生成的环境原点（非零）
            - 地形未初始化时返回默认原点（全零）
        """
        # 优先返回地形定义的环境原点（地形生成后会自动计算）
        if self.terrain is not None and self.terrain.env_origins is not None:
            return self.terrain.env_origins
        # 否则返回默认原点（全零）
        return self._default_env_origins

    def write_data_to_sim(self):
        """
        将数据写入仿真器（预留接口）
        用于将外部数据（如机器人关节指令、力控制指令）同步到PhysX仿真器

        典型使用场景：
            在每个仿真步前，将关节目标角度/速度写入仿真器，控制机器人运动
        """
        pass
        # 示例实现（如需启用请取消注释）
        # self._scene.write_data_to_sim()

    def forward(self):
        """执行仿真前向计算（更新物理状态）

        作用：计算当前时间步的物理状态（位置、速度、力、关节角度等）
        调用时机：每个仿真步前（step()前）
        """
        self._sim.forward()

    def step(self):
        """执行仿真步（PhysX引擎单次步进）

        作用：推进物理时间，执行一次完整的物理计算
        调用时机：forward()后，update()前
        """
        self._sim.step()

    def update(self, dt: float):
        """
        更新仿真状态（指定时间步长）

        参数:
            dt: float，时间步长（秒），例如0.01表示10ms（需与仿真器配置的dt一致）

        作用：同步仿真时间，更新所有实体的物理状态
        调用时机：step()后
        """
        self._sim.update(dt)

    def reset(self, env_ids: torch.Tensor or list):
        """
        重置指定环境的仿真状态

        参数:
            env_ids: torch.Tensor/list，需要重置的环境ID列表
                例如: torch.tensor([0, 2, 5], device="cuda") 或 [0, 1, 2]

        作用：将指定环境的机器人/物体恢复到初始状态（位置、速度、物理参数）
        典型使用场景：机器人摔倒后重置环境
        """
        self._scene.reset(env_ids)

    def get_model_field(self, field: str) -> torch.Tensor:
        """
        获取仿真模型的指定物理字段值

        参数:
            field: str，字段名（如"body_mass", "foot_friction"）

        返回:
            torch.Tensor，该字段的当前值（形状[num_envs, 字段维度]）

        异常:
            AttributeError: 字段不存在时抛出，提示合法字段
        """
        if not hasattr(self._sim.model, field):
            raise AttributeError(
                f"仿真模型不存在字段 {field}，合法字段: {dir(self._sim.model)}"
            )
        return getattr(self._sim.model, field)

    def random_values(self, ranges: list, distribution: str) -> torch.Tensor:
        """
        生成指定分布的随机值（示例实现，建议根据实际需求修改）

        参数:
            ranges: list[float/float]，随机值范围 [最小值, 最大值]
            distribution: str，分布类型（"uniform"/"log_uniform"/"gaussian"）

        返回:
            torch.Tensor，形状[2,2]的随机张量（示例形状，需根据实际需求调整）

        注意：
            该方法为示例实现，实际使用时应根据random_field的需求调整形状
            建议替换为调用_sample_distribution函数
        """
        # 注意：该方法为示例实现，实际使用时应根据random_field的需求调整形状
        return torch.randn([2, 2], dtype=torch.float32, device=self.device)

    def random_field(
            self,
            env_ids: torch.Tensor or None,
            field: str,
            ranges: list,
            distribution: str,
            operation: str,
            asset_cfg,
            axes=None
    ):
        """
        执行域随机化：随机调整指定环境中机器人的物理参数

        核心逻辑：
            1. 确定需要随机化的环境ID
            2. 找到目标机器人的身体部位索引
            3. 从指定分布采样随机值
            4. 将随机值应用到机器人的指定物理字段

        参数:
            env_ids: torch.Tensor/None，需要随机化的环境ID
                - None: 所有环境
                - 张量/列表: 指定环境ID（如torch.tensor([0,2,5])）
            field: str，要随机化的物理字段名（如"body_mass", "foot_friction"）
            ranges: list[torch.Tensor/torch.Tensor]，随机值范围 [下界, 上界]
                - 示例: [torch.tensor(1.0), torch.tensor(5.0)]（body_mass范围1-5kg）
            distribution: str，采样分布类型（"uniform"/"log_uniform"/"gaussian"）
            operation: str，操作类型（如"assign"赋值/"scale"缩放）
                - "assign": 将采样值直接赋值给字段
                - "scale": 将采样值作为缩放因子，乘以原字段值
            asset_cfg: 资产配置对象，需包含：
                - name: 机器人实体名称
                - body_names: 需要随机化的身体部位名称列表（如["foot_left", "foot_right"]）
            axes: 轴配置（预留参数，当前未使用，用于指定需要随机化的轴）

        异常:
            ValueError: 未找到匹配的身体部位时抛出
            KeyError: 机器人实体不存在时抛出（由get_robot方法抛出）
        """
        # 1. 处理环境ID（默认所有环境）
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int)

        # 2. 获取目标机器人实体
        ent = self.get_robot(asset_cfg.name)

        # 3. 找到需要随机化的身体部位索引
        idx_list = []
        for idx, body_name in enumerate(ent.body_names):
            if body_name in asset_cfg.body_names:
                idx_list.append(idx)
        # 检查是否找到匹配的身体部位
        if not idx_list:
            raise ValueError(
                f"机器人 {asset_cfg.name} 未找到匹配的身体部位，"
                f"配置的部位: {asset_cfg.body_names}, 实体的部位: {ent.body_names}"
            )

        # 4. 定义采样形状：[环境数, 身体部位数, 字段维度]
        shape = [self.num_envs, len(idx_list), self.field_dim[field]]

        # 5. 从指定分布采样随机值
        values = _sample_distribution(
            distribution=distribution,
            lower=torch.tensor(ranges[0], device=self.device, dtype=torch.float32),
            upper=torch.tensor(ranges[1], device=self.device, dtype=torch.float32),
            shape=tuple(shape),
            device=self.device
        )

        # 6. 将随机值应用到机器人实体的指定字段
        ent.set_field(env_ids, values, field, operation, idx_list)

    def get_stiffness(self, ctrl_ids=None) -> torch.Tensor:
        """
        获取机器人关节的刚度值

        参数:
            ctrl_ids: 控制ID（预留参数，当前未使用，用于指定需要获取的关节）

        返回:
            torch.Tensor，机器人关节的刚度值张量（形状[num_envs, num_joints]）

        注意：
            此处硬编码"robot"为机器人名称，建议根据实际情况调整为参数
        """
        # 注意：此处硬编码"robot"为机器人名称，建议根据实际情况调整
        return self.get_robot("robot").get_stiffness()

    def get_damping(self, ctrl_ids=None) -> torch.Tensor:
        """
        获取机器人关节的阻尼值（修正原代码dampling拼写错误）

        参数:
            ctrl_ids: 控制ID（预留参数，当前未使用，用于指定需要获取的关节）

        返回:
            torch.Tensor，机器人关节的阻尼值张量（形状[num_envs, num_joints]）

        注意：
            此处硬编码"robot"为机器人名称，建议根据实际情况调整为参数
        """
        # 注意：此处硬编码"robot"为机器人名称，建议根据实际情况调整
        return self.get_robot("robot").get_damping()

    def render(self) -> np.ndarray:
        """
        渲染场景并获取RGB图像数据

        返回:
            np.ndarray，形状[720, 1280, 3]，dtype=uint8，RGB格式的图像数据
            - 720: 图像高度（像素）
            - 1280: 图像宽度（像素）
            - 3: RGB通道

        渲染流程:
            1. 将物理状态同步到USD（确保可视化与物理状态一致）
            2. 获取视口图像数据（RGBA格式）
            3. 重塑形状并提取RGB通道（丢弃Alpha通道）
        """
        # 将物理状态同步到USD（用于可视化）
        self.utils.physics_to_usd()
        # 获取视口图像数据（RGBA格式，形状[720*1280*4]）
        img = self.app.get_viewport_image_data()
        # 重塑形状并提取RGB通道（丢弃Alpha通道）
        img = img.reshape([720, 1280, 4])[:, :, :3]
        return img

    def close(self):
        """关闭仿真应用，释放所有资源（必须调用）

        作用：
            1. 关闭渲染窗口
            2. 释放PhysX引擎资源
            3. 清理GPU/CPU内存

        注意：
            必须在仿真结束时调用，否则会导致资源泄漏、程序卡死
        """
        self.app.close()
