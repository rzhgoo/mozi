# 导入场景、仿真、配置相关的核心模块
from ms_lab.scene import Scene
from ms_lab.sim.sim import Simulation
from ms_lab.managers.scene_entity_config import SceneEntityCfg
# 导入类型注解相关模块
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union
# 导入实体相关模块
from ms_lab.entity import Entity, EntityIndexing
import torch
# 导入数据类装饰器
from dataclasses import dataclass
# 导入IsaacLab的数学工具函数
from ms_lab.third_party.isaaclab.isaaclab.utils.math import (
    quat_apply_inverse,  # 四元数逆应用
    sample_log_uniform,  # 对数均匀分布采样
    sample_uniform,  # 均匀分布采样
)

# 默认的资产配置，指定机器人类型的场景实体配置
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass
class FieldSpec:
    """
    字段规格类：定义如何处理特定字段的规范
    用于描述 Mujoco 模型中不同类型字段（如关节、刚体、几何形状等）的随机化规则
    """
    # 实体类型：自由度、关节、刚体、几何、站点、执行器
    entity_type: Literal["dof", "joint", "body", "geom", "site", "actuator"]
    # 是否使用地址（q_adr, v_adr）：True表示需要地址来定位字段
    use_address: bool = False
    # 默认轴列表：如果用户未指定轴时使用的默认轴
    default_axes: Optional[List[int]] = None
    # 有效轴列表：限制该字段可随机化的轴范围
    valid_axes: Optional[List[int]] = None


# 预定义的字段规格映射表
# 定义了Mujoco模型中各种可随机化字段的处理规则
FIELD_SPECS = {
    # 自由度相关字段 - 使用地址定位
    "dof_armature": FieldSpec("dof", use_address=True),  # 自由度电枢
    "dof_frictionloss": FieldSpec("dof", use_address=True),  # 自由度摩擦损耗
    "dof_damping": FieldSpec("dof", use_address=True),  # 自由度阻尼
    # 关节相关字段 - 直接使用ID
    "jnt_range": FieldSpec("joint"),  # 关节范围
    "jnt_stiffness": FieldSpec("joint"),  # 关节刚度
    # 刚体相关字段 - 直接使用ID
    "body_mass": FieldSpec("body"),  # 刚体质量
    "body_ipos": FieldSpec("body", default_axes=[0, 1, 2]),  # 刚体初始位置（xyz轴）
    "body_iquat": FieldSpec("body", default_axes=[0, 1, 2, 3]),  # 刚体初始四元数（wxyz）
    "body_inertia": FieldSpec("body"),  # 刚体惯性
    "body_pos": FieldSpec("body", default_axes=[0, 1, 2]),  # 刚体位置
    "body_quat": FieldSpec("body", default_axes=[0, 1, 2, 3]),  # 刚体四元数
    # 几何形状相关字段 - 直接使用ID
    "geom_friction": FieldSpec("geom", default_axes=[0], valid_axes=[0, 1, 2]),  # 几何摩擦（0:切向,1:滚动,2:扭转）
    "geom_pos": FieldSpec("geom", default_axes=[0, 1, 2]),  # 几何位置
    "geom_quat": FieldSpec("geom", default_axes=[0, 1, 2, 3]),  # 几何四元数
    "geom_rgba": FieldSpec("geom", default_axes=[0, 1, 2, 3]),  # 几何颜色（rgba）
    # 站点相关字段 - 直接使用ID
    "site_pos": FieldSpec("site", default_axes=[0, 1, 2]),  # 站点位置
    "site_quat": FieldSpec("site", default_axes=[0, 1, 2, 3]),  # 站点四元数
    # 特殊情况 - 使用地址
    "qpos0": FieldSpec("joint", use_address=True),  # 关节初始位置
}


def _get_entity_indices(
        indexing: EntityIndexing, asset_cfg, spec: FieldSpec
) -> torch.Tensor:
    """
    获取实体索引：根据实体类型和配置获取对应的索引张量

    参数：
        indexing: 实体索引对象，包含各类实体的ID和地址映射
        asset_cfg: 资产配置对象
        spec: 字段规格对象

    返回：
        对应实体类型的索引张量
    """
    match spec.entity_type:
        case "dof":
            # 自由度：返回关节速度地址
            return indexing.joint_v_adr[asset_cfg.joint_ids]
        case "joint" if spec.use_address:
            # 关节（需要地址）：返回关节位置地址
            return indexing.joint_q_adr[asset_cfg.joint_ids]
        case "joint":
            # 关节（不需要地址）：直接返回关节ID
            return indexing.joint_ids[asset_cfg.joint_ids]
        case "body":
            # 刚体：返回刚体ID
            return indexing.body_ids[asset_cfg.body_ids]
        case "geom":
            # 几何形状：返回几何ID
            return indexing.geom_ids[asset_cfg.geom_ids]
        case "site":
            # 站点：返回站点ID
            return indexing.site_ids[asset_cfg.site_ids]
        case "actuator":
            # 执行器：返回控制ID（需确保控制ID不为空）
            assert indexing.ctrl_ids is not None
            return indexing.ctrl_ids[asset_cfg.actuator_ids]
        case _:
            # 未知实体类型抛出异常
            raise ValueError(f"Unknown entity type: {spec.entity_type}")


def _determine_target_axes(
        model_field,
        spec: FieldSpec,
        axes: Optional[List[int]],
        ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]],
) -> List[int]:
    """
    确定目标轴：计算需要随机化的轴列表

    参数：
        model_field: 模型字段张量
        spec: 字段规格对象
        axes: 用户指定的轴列表（可选）
        ranges: 随机化范围（元组或字典）

    返回：
        目标轴列表
    """
    # 计算字段维度：减去环境维度（第一个维度）
    field_ndim = len(model_field.shape) - 1

    if axes is not None:
        # 1. 用户显式指定了轴
        target_axes = axes
    elif isinstance(ranges, dict):
        # 2. 通过字典键指定轴
        target_axes = list(ranges.keys())
    elif spec.default_axes is not None:
        # 3. 使用字段规格的默认轴
        target_axes = spec.default_axes
    else:
        # 4. 随机化所有轴
        if field_ndim > 1:
            target_axes = list(range(model_field.shape[-1]))  # 多维字段：最后一维的所有轴
        else:
            target_axes = [0]  # 标量字段：仅第0轴

    # 验证轴的有效性
    if spec.valid_axes is not None:
        invalid_axes = set(target_axes) - set(spec.valid_axes)
        if invalid_axes:
            raise ValueError(
                f"Invalid axes {invalid_axes} for field. Valid axes: {spec.valid_axes}"
            )

    return target_axes


def _prepare_axis_ranges(
        ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]],
        target_axes: List[int],
        field: str,
) -> Dict[int, Tuple[float, float]]:
    """
    准备轴范围：将输入的范围转换为统一的字典格式

    参数：
        ranges: 随机化范围（元组或字典）
        target_axes: 目标轴列表
        field: 字段名称

    返回：
        轴到范围的字典映射
    """
    if isinstance(ranges, tuple):
        # 1. 元组格式：所有轴使用相同的范围
        return {axis: ranges for axis in target_axes}
    elif isinstance(ranges, dict):
        # 2. 字典格式：验证所有目标轴都有对应的范围
        missing_axes = set(target_axes) - set(ranges.keys())
        if missing_axes:
            raise ValueError(
                f"Missing ranges for axes {missing_axes} in field '{field}'. "
                f"Required axes: {target_axes}"
            )
        return {axis: ranges[axis] for axis in target_axes}
    else:
        # 不支持的格式
        raise TypeError(f"ranges must be tuple or dict, got {type(ranges)}")


def _generate_random_values(
        distribution: str,
        axis_ranges: Dict[int, Tuple[float, float]],
        indexed_data: torch.Tensor,
        target_axes: List[int],
        device,
) -> torch.Tensor:
    """
    生成随机值：为指定轴生成符合分布的随机值

    参数：
        distribution: 分布类型（uniform/log_uniform/gaussian）
        axis_ranges: 轴范围字典
        indexed_data: 索引数据张量
        target_axes: 目标轴列表
        device: 计算设备（cpu/cuda）

    返回：
        包含随机值的张量
    """
    # 克隆原始数据以避免修改原张量
    result = indexed_data.clone()

    for axis in target_axes:
        # 获取当前轴的上下限
        lower, upper = axis_ranges[axis]

        # 转换为张量并移至指定设备
        lower_bound = torch.tensor(lower).to(device)
        upper_bound = torch.tensor([upper], device=device)

        # 确定随机值的形状
        if len(indexed_data.shape) > 2:
            # 多维字段：保持前n-1维，最后一维为1
            shape = (*indexed_data.shape[:-1], 1)
        else:
            # 标量字段：使用原形状
            shape = indexed_data.shape

        # 根据分布类型采样随机值
        random_vals = _sample_distribution(
            distribution, lower_bound, upper_bound, shape, device
        )

        # 将随机值赋值到对应轴
        if len(indexed_data.shape) > 2:
            result[..., axis] = random_vals.squeeze(-1)  # 移除最后一维
        else:
            result = random_vals

    return result


def _apply_operation(
        model_field,
        env_grid,
        entity_grid,
        indexed_data,
        random_values,
        operation,
):
    """
    应用操作：将随机值以指定方式应用到模型字段

    参数：
        model_field: 模型字段张量
        env_grid: 环境网格索引
        entity_grid: 实体网格索引
        indexed_data: 原始索引数据
        random_values: 生成的随机值
        operation: 操作类型（add/scale/abs）
    """
    if operation == "add":
        # 加法操作：原始值 + 随机值
        model_field[env_grid, entity_grid] = indexed_data + random_values
    elif operation == "scale":
        # 缩放操作：原始值 * 随机值
        model_field[env_grid, entity_grid] = indexed_data * random_values
    elif operation == "abs":
        # 替换操作：直接使用随机值替换原始值
        model_field[env_grid, entity_grid] = random_values
    else:
        # 未知操作类型
        raise ValueError(f"Unknown operation: {operation}")


def _sample_distribution(
        distribution: str,
        lower: torch.Tensor,
        upper: torch.Tensor,
        shape: tuple,
        device: str,
) -> torch.Tensor:
    """
    分布采样：从指定分布中采样数据

    参数：
        distribution: 分布类型
        lower: 下限张量
        upper: 上限张量
        shape: 输出形状
        device: 计算设备

    返回：
        采样得到的张量
    """
    if distribution == "uniform":
        # 均匀分布采样
        return sample_uniform(lower, upper, shape, device=device)
    elif distribution == "log_uniform":
        # 对数均匀分布采样
        return sample_log_uniform(lower, upper, shape, device=device)
    elif distribution == "gaussian":
        # 高斯分布采样（注：原代码未实现sample_gaussian函数，可能需要补充）
        return sample_gaussian(lower, upper, shape, device=device)
    else:
        # 未知分布类型
        raise ValueError(f"Unknown distribution: {distribution}")


class MujocoBackend:
    """
    Mujoco后端类：封装Mujoco仿真的核心操作
    提供场景初始化、仿真控制、模型字段随机化等功能
    """

    def __init__(self, device):
        """
        初始化函数

        参数：
            device: 计算设备（cpu/cuda）
        """
        self.device = device

    @property
    def scene(self):
        """场景属性：获取当前场景对象"""
        return self._scene

    @property
    def sim(self):
        """仿真属性：获取当前仿真对象"""
        return self._sim

    @property
    def num_envs(self):
        """环境数量属性：获取环境总数"""
        return self.scene.num_envs

    def init_scene(self, cfg):
        """
        初始化场景

        参数：
            cfg: 场景配置对象
        """
        self._scene = Scene(cfg, device=self.device)

    def init_sim(self, cfg):
        """
        初始化仿真

        参数：
            cfg: 仿真配置对象
        """
        # 编辑场景规格
        cfg.mujoco.edit_spec(self._scene.spec)

        # 创建仿真对象
        self._sim = Simulation(
            num_envs=self._scene.num_envs,
            cfg=cfg,
            model=self._scene.compile(),  # 编译场景为Mujoco模型
            device=self.device,
        )
        # 初始化场景
        self._scene.initialize(
            mj_model=self._sim.mj_model,
            model=self._sim.model,
            data=self._sim.data,
        )

    def expand_model_fields(self, domain_randomization_fields):
        """
        扩展模型字段：为域随机化扩展模型字段

        参数：
            domain_randomization_fields: 需要扩展的字段列表
        """
        self._sim.expand_model_fields(domain_randomization_fields)

    def create_graph(self):
        """创建计算图：初始化仿真的计算图"""
        self._sim.create_graph()

    def get_robot(self, asset_name):
        """
        获取机器人对象

        参数：
            asset_name: 资产名称

        返回：
            对应的机器人实体对象
        """
        return self._scene[asset_name]

    def random_field(self,
                     env_ids,
                     field,
                     ranges,
                     distribution,
                     operation,
                     asset_cfg,
                     axes):
        """
        字段随机化：对指定字段进行域随机化

        参数：
            env_ids: 环境ID列表（None表示所有环境）
            field: 要随机化的字段名称
            ranges: 随机化范围
            distribution: 分布类型
            operation: 操作类型（add/scale/abs）
            asset_cfg: 资产配置
            axes: 要随机化的轴列表
        """
        # 获取字段规格
        spec = FIELD_SPECS[field]
        # 使用默认配置（如果未指定）
        asset_cfg = asset_cfg or _DEFAULT_ASSET_CFG
        # 获取机器人资产
        asset = self.get_robot(asset_cfg.name)

        # 处理环境ID
        if env_ids is None:
            # 未指定环境ID：使用所有环境
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int)
        else:
            # 转换为指定设备的整型张量
            env_ids = env_ids.to(self.device, dtype=torch.int)

        # 获取模型字段
        model_field = self.get_model_field(field)

        # 获取实体索引
        entity_indices = _get_entity_indices(asset.indexing, asset_cfg, spec)

        # 确定目标轴
        target_axes = _determine_target_axes(model_field, spec, axes, ranges)

        # 准备轴范围
        axis_ranges = _prepare_axis_ranges(ranges, target_axes, field)

        # 创建环境-实体网格索引
        env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")

        # 获取索引数据（注意：entity_grid需要转CPU以匹配模型字段的设备）
        indexed_data = model_field[env_grid, entity_grid.cpu()]

        # 生成随机值
        random_values = _generate_random_values(
            distribution, axis_ranges, indexed_data, target_axes, self.device
        )

        # 应用随机化操作
        _apply_operation(
            model_field, env_grid, entity_grid, indexed_data, random_values, operation
        )

    def get_all_robots(self):
        """
        获取所有机器人（注：原代码拼写错误 _cene → _scene）

        返回：
            所有机器人实体的迭代器
        """
        return self._scene.entities.values()

    def get_terrain(self):
        """获取地形对象"""
        return self._scene.terrain

    def get_env_origins(self):
        """获取环境原点坐标"""
        return self._scene.env_origins

    def write_data_to_sim(self):
        """将数据写入仿真"""
        self._scene.write_data_to_sim()

    def forward(self):
        """前向计算：执行仿真的前向动力学"""
        self._sim.forward()

    def step(self):
        """步进：执行仿真的一步"""
        self._sim.step()

    def update(self, dt):
        """
        更新场景

        参数：
            dt: 时间步长
        """
        self._scene.update(dt)

    def reset(self, env_ids):
        """
        重置环境

        参数：
            env_ids: 要重置的环境ID列表
        """
        self._scene.reset(env_ids)

    def get_model_field(self, field):
        """
        获取模型字段

        参数：
            field: 字段名称

        返回：
            对应的模型字段张量
        """
        return getattr(self._sim.model, field)

    def get_stiffness(self, ctrl_ids):
        """
        获取执行器刚度

        参数：
            ctrl_ids: 控制ID列表

        返回：
            刚度值张量
        """
        return self._sim.mj_model.actuator_gainprm[ctrl_ids, 0]

    def get_damping(self, ctrl_ids):
        """
        获取执行器阻尼

        参数：
            ctrl_ids: 控制ID列表

        返回：
            阻尼值张量
        """
        return self._sim.mj_model.actuator_biasprm[ctrl_ids, 2]
