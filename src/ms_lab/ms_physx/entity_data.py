# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Union, Sequence

# 注：如果以下导入报错，请确保对应模块已安装/路径正确
from ms_lab.entity import EntityIndexing
from ms_lab.third_party.isaaclab.isaaclab.utils.math import combine_frame_transforms
from mozisim.physx_engine.articulations import BatchArticulation


# ============================== 工具函数 ==============================


# ============================== 核心类型转换工具函数 ==============================
def convert_to_python_type(data, max_length: int = None) -> Union[int, list[int]]:
    """
    将Tensor/ndarray/slice类型的索引统一转换为原生Python整数/列表，
    避免C++底层接口不支持Torch/Numpy/Slice类型而报错。

    Args:
        data: 输入索引，可以是Tensor、ndarray、int、list、slice、None
        max_length: 当data为None或slice时，用于生成完整索引范围的最大长度

    Returns:
        纯Python原生类型的索引：int 或 list[int]

    Raises:
        ValueError: 当data为None或slice但未指定max_length时
        TypeError: 当传入不支持的索引类型时
    """
    if data is None:
        # None表示选取所有索引，返回 0 ~ max_length-1 的完整列表
        if max_length is None:
            raise ValueError("当data为None时，必须指定max_length参数")
        return list(range(max_length))

    elif isinstance(data, slice):
        # 将切片对象转换为具体整数索引列表
        if max_length is None:
            raise ValueError("当data为slice时，必须指定max_length参数")
        start = data.start if data.start is not None else 0
        stop = data.stop if data.stop is not None else max_length
        step = data.step if data.step is not None else 1
        return list(range(start, stop, step))

    elif isinstance(data, torch.Tensor):
        # Tensor标量 → int；Tensor数组 → Python列表
        if data.numel() == 1:
            return int(data.item())
        else:
            return data.cpu().tolist()

    elif isinstance(data, np.ndarray):
        # Numpy标量 → int；数组 → Python列表
        if data.size == 1:
            return int(data.item())
        else:
            return data.tolist()

    elif isinstance(data, int):
        # 直接返回整数
        return data

    elif isinstance(data, list):
        # 确保列表内所有元素均为int类型，防止隐式类型问题
        return [int(x) for x in data]

    else:
        # 不支持的索引类型直接抛出异常，便于调试
        raise TypeError(f"不支持的索引类型: {type(data)}")


# ============================== 核心数学转换函数 ==============================
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    四元数旋转向量：v' = q * v * q⁻¹ 的纯向量计算实现
    用于将局部坐标系向量旋转到世界坐标系。

    Args:
        quat: 四元数张量，格式 (w, x, y, z)，shape: [..., 4]
        vec: 待旋转的三维向量张量，shape: [..., 3]

    Returns:
        旋转后的三维向量张量，shape: [..., 3]
    """
    quat_w = quat[..., 0:1]
    quat_xyz = quat[..., 1:4]
    cross = torch.cross(quat_xyz, vec, dim=-1)
    return 2 * (quat_w * cross + torch.cross(quat_xyz, cross, dim=-1)) + vec


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    四元数逆旋转：使用共轭四元数将世界坐标系向量转回局部坐标系。

    Args:
        quat: 原始四元数张量，格式 (w, x, y, z)，shape: [..., 4]
        vec: 待逆旋转的三维向量张量，shape: [..., 3]

    Returns:
        逆旋转后的三维向量张量，shape: [..., 3]
    """
    # 构造共轭四元数（实部不变，虚部取反）
    quat_conj = torch.cat([quat[..., 0:1], -quat[..., 1:4]], dim=-1)
    return quat_apply(quat_conj, vec)


def compute_velocity_from_cvel(pos: torch.Tensor, subtree_com: torch.Tensor, cvel: torch.Tensor) -> torch.Tensor:
    """
    从质心速度（cvel）计算世界坐标系下的实际速度
    用于处理多体动力学中质心偏移带来的速度转换。

    Args:
        pos: 刚体位置张量，shape: [..., 3]
        subtree_com: 子树质心位置张量，shape: [..., 3]
        cvel: 质心处速度张量，[角速度, 线速度]，shape: [..., 6]

    Returns:
        世界坐标系下的刚体实际速度张量，shape: [..., 6]
    """
    lin_vel_c = cvel[..., 3:6]   # 质心线速度
    ang_vel_c = cvel[..., 0:3]   # 质心角速度
    offset = subtree_com - pos   # 质心相对于刚体位置的偏移
    # 考虑旋转带来的线速度补偿
    lin_vel_w = lin_vel_c - torch.cross(ang_vel_c, offset, dim=-1)
    return torch.cat([lin_vel_w, ang_vel_c], dim=-1)


# ============================== 核心数据类 ==============================
@dataclass
class EntityData:
    """
    机器人/实体状态数据封装类
    统一管理仿真中机器人的姿态、速度、关节、传感器、控制指令等数据
    内置懒加载缓存机制，减少重复底层调用，提升仿真循环速度。
    """
    # 索引与仿真接口
    indexing: EntityIndexing                # 实体索引管理（关节、刚体、传感器地址映射）
    batch_artic: BatchArticulation          # 批量多体物理引擎接口
    device: str                             # 运行设备（cuda/cpu）

    # 默认状态参数
    default_root_state: torch.Tensor        # 默认根节点状态（位置+姿态+线速度+角速度）
    default_joint_pos: torch.Tensor         # 默认关节角度
    default_joint_vel: torch.Tensor          # 默认关节速度
    default_joint_stiffness: torch.Tensor   # 默认关节刚度
    default_joint_damping: torch.Tensor     # 默认关节阻尼

    # 关节限位
    default_joint_pos_limits: torch.Tensor  # 默认关节硬限位
    joint_pos_limits: torch.Tensor          # 实际关节硬限位
    soft_joint_pos_limits: torch.Tensor     # 关节软限位（用于安全约束）

    # 环境与坐标系参数
    gravity_vec_w: torch.Tensor             # 世界坐标系重力向量
    forward_vec_b: torch.Tensor              # 机体坐标系前向向量（通常为x轴）

    # 机器人结构属性
    is_fixed_base: bool                      # 是否为固定基机器人
    is_articulated: bool                     # 是否为多体铰接结构
    is_actuated: bool                        # 是否带执行器（可控制）

    # 维度常量定义
    POS_DIM = 3                             # 位置维度
    QUAT_DIM = 4                            # 四元数维度
    LIN_VEL_DIM = 3                         # 线速度维度
    ANG_VEL_DIM = 3                         # 角速度维度
    ROOT_POSE_DIM = POS_DIM + QUAT_DIM      # 根节点姿态维度
    ROOT_VEL_DIM = LIN_VEL_DIM + ANG_VEL_DIM# 根节点速度维度
    ROOT_STATE_DIM = ROOT_POSE_DIM + ROOT_VEL_DIM  # 根节点完整状态维度

    def __post_init__(self):
        """初始化后自动计算环境、刚体、关节数量。"""
        self.num_envs = self.default_root_state.shape[0]       # 并行环境数量
        self.num_bodies = len(self.indexing.body_ids)          # 刚体数量
        self.num_joints = self.batch_artic.num_joint           # 总关节数
        self.num_actuators = self.batch_artic.num_dof          # 可驱动关节数

        # ========== 懒加载缓存优化 ==========
        self._cache = {}                      # 缓存数据存储
        self._cache_timestamps = {}            # 缓存项更新时间戳
        self._step_counter = 0                 # 当前仿真步数，用于判断缓存是否过期

    def _get_cached_item(self, key, loader_func):
        """
        统一懒加载缓存入口
        仅当缓存不存在或已过期时，才调用loader_func重新获取。

        Args:
            key: 缓存键名
            loader_func: 真实数据加载函数（调用底层仿真接口）

        Returns:
            缓存中或新加载的数据
        """
        # 缓存不存在或已过期 → 重新加载
        if key not in self._cache or self._cache_timestamps.get(key, -1) < self._step_counter:
            data = loader_func()
            self._cache[key] = data
            self._cache_timestamps[key] = self._step_counter
        return self._cache[key]

    def invalidate_cache(self):
        """
        使当前所有缓存失效
        优化方式：仅递增步数计数器，不直接清空字典，减少内存操作。
        """
        self._step_counter += 1

    def _resolve_env_ids(self, env_ids: Optional[Union[torch.Tensor, slice, int, list]]) -> Union[int, list[int]]:
        """
        解析环境ID，统一转为Python原生int/list，避免slice/张量导致C++接口报错。

        Args:
            env_ids: 环境ID，可以是Tensor、ndarray、int、list、slice、None

        Returns:
            解析后的环境ID，int 或 list[int]
        """
        return convert_to_python_type(env_ids, max_length=self.num_envs)

    def _resolve_joint_ids(self, joint_ids: Optional[Union[torch.Tensor, slice, int, list]]) -> Union[int, list[int]]:
        """
        解析关节ID，统一转为Python原生类型。

        Args:
            joint_ids: 关节ID，可以是Tensor、ndarray、int、list、slice、None

        Returns:
            解析后的关节ID，int 或 list[int]
        """
        return convert_to_python_type(joint_ids, max_length=self.num_actuators)

    def _resolve_body_ids(self, body_ids: Optional[Union[torch.Tensor, slice, int, list]]) -> Union[int, list[int]]:
        """
        解析刚体ID，统一转为Python原生类型。

        Args:
            body_ids: 刚体ID，可以是Tensor、ndarray、int、list、slice、None

        Returns:
            解析后的刚体ID，int 或 list[int]
        """
        return convert_to_python_type(body_ids, max_length=self.num_bodies)

    # ============================== 属性（懒加载优化） ==============================
    @property
    def heading_w(self) -> torch.Tensor:
        """
        世界坐标系下机器人朝向角（yaw/heading），由前向向量计算得到。

        Returns:
            朝向角张量，shape: [num_envs]
        """
        forward_w = quat_apply(self.root_link_quat_w, self.forward_vec_b)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def qacc(self) -> torch.Tensor:
        """
        广义加速度（根节点6维 + 关节维），懒加载。

        Returns:
            广义加速度张量，shape: [num_envs, 6 + num_joints]
        """
        return self._get_cached_item(
            "qacc",
            lambda: torch.as_tensor(self.batch_artic.get_qacc(), dtype=torch.float32, device=self.device)
        )

    @property
    def joint_acc(self) -> torch.Tensor:
        """
        关节角加速度（从广义加速度中截取关节部分）。

        Returns:
            关节角加速度张量，shape: [num_envs, num_joints]
        """
        return self.qacc[:, 6:]

    @property
    def ctrl(self) -> torch.Tensor:
        """
        关节位置控制目标（PD目标位置）。

        Returns:
            关节位置控制目标张量，shape: [num_envs, num_actuators]
        """
        return self._get_cached_item(
            "dof_position_targets",
            lambda: torch.as_tensor(self.batch_artic.get_dof_position_targets(), dtype=torch.float32, device=self.device)
        )

    @property
    def actuator_force(self) -> torch.Tensor:
        """
        执行器输出力矩估算
        基于PD控制公式：τ = kp*(target - pos) - kd*vel。

        Returns:
            执行器输出力矩张量，shape: [num_envs, num_actuators]
        """
        params = self._get_cached_item(
            "dof_drive_params",
            lambda: torch.as_tensor(self.batch_artic.get_dof_drive_params().copy(), dtype=torch.float32,
                                 device=self.device)
        )
        stiffness = params[:, :, 0]
        damping = params[:, :, 1]
        dof_pos = self.joint_pos
        dof_vel = self.joint_vel
        ctrl = self.ctrl
        return stiffness * (ctrl - dof_pos) - damping * dof_vel

    @property
    def contact_sensor_data(self) -> dict[str, torch.Tensor]:
        """
        接触传感器数据
        返回每个刚体是否与地面接触，格式 {body_name: [env_num, 2]}。

        Returns:
            接触传感器数据字典，键为刚体名，值为接触状态张量
        """
        contact_sensor_datas = self._get_cached_item(
            "contact_sensor_datas",
            lambda: self.batch_artic.get_contact_sensor_datas()
        )
        body_names = self.batch_artic.rigid_body_names
        sensordata = {b: torch.zeros([self.num_envs, 2], dtype=torch.float32, device=self.device) for b in body_names}

        for i in range(self.num_envs):
            robot_datas = contact_sensor_datas[i]
            for idx, rd in enumerate(robot_datas):
                if not rd:
                    continue
                for data in rd:
                    if data.forces.size != 0 and "GroundPlane" in data.actor2:
                        sensordata[body_names[idx]][i, 0] = 1
        return sensordata

    @property
    def subtree_com(self) -> torch.Tensor:
        """
        子树质心位置（根节点）。

        Returns:
            子树质心位置张量，shape: [num_envs, 3]
        """
        link_c_mass_pose = self._get_cached_item(
            "link_c_mass_pose",
            lambda: self.batch_artic.get_link_c_mass_pose(rigid_body_indices=[0])
        )
        return torch.tensor(link_c_mass_pose[:, :, :3], dtype=torch.float32, device=self.device)

    @property
    def cvel(self) -> torch.Tensor:
        """
        质心速度 [角速度, 线速度]。

        Returns:
            质心速度张量，shape: [num_envs, 6]
        """
        rigid_body_velocity = self._get_cached_item(
            "rigid_body_velocity",
            lambda: self.batch_artic.get_rigid_body_velocity(rigid_body_indices=[0])
        )
        return torch.tensor(rigid_body_velocity, dtype=torch.float32, device=self.device)

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """
        世界坐标系下根节点姿态
        内部做了四元数顺序调整：(x,y,z,w) → (w,x,y,z)。

        Returns:
            根节点姿态张量，shape: [num_envs, 7]
        """
        root_pose = self._get_cached_item(
            "root_pose",
            lambda: torch.as_tensor(self.batch_artic.get_root_pose().copy(), dtype=torch.float32, device=self.device)
        )
        return self._get_cached_item(
            "root_link_pose_w",
            lambda: self._compute_root_link_pose_w(root_pose)
        )

    def _compute_root_link_pose_w(self, root_pose):
        """
        内部工具：调整根节点姿态的四元数顺序。

        Args:
            root_pose: 原始根节点姿态张量，shape: [num_envs, 7]

        Returns:
            调整四元数顺序后的根节点姿态张量，shape: [num_envs, 7]
        """
        new_pose = root_pose.clone().to(self.device)
        new_pose[:, 3:7] = new_pose[:, [6, 3, 4, 5]]
        return new_pose

    @property
    def root_link_pos_w(self):
        """
        世界坐标系根节点位置 (x,y,z)。

        Returns:
            根节点位置张量，shape: [num_envs, 3]
        """
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self):
        """
        世界坐标系根节点姿态四元数 (w,x,y,z)。

        Returns:
            根节点四元数张量，shape: [num_envs, 4]
        """
        return self.root_link_pose_w[:, 3:]

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """
        世界坐标系根节点速度 [线速度, 角速度]。

        Returns:
            根节点速度张量，shape: [num_envs, 6]
        """
        return self._get_cached_item(
            "root_link_vel_w",
            lambda: torch.as_tensor(self.batch_artic.get_root_velocity(), dtype=torch.float32, device=self.device)
        )

    @property
    def root_link_lin_vel_w(self):
        """
        世界坐标系根节点线速度。

        Returns:
            根节点线速度张量，shape: [num_envs, 3]
        """
        return self.root_link_vel_w[..., :3]

    @property
    def root_link_ang_vel_w(self):
        """
        世界坐标系根节点角速度。

        Returns:
            根节点角速度张量，shape: [num_envs, 3]
        """
        return self.root_link_vel_w[..., 3:]

    @property
    def root_link_lin_vel_b(self):
        """
        局部坐标系根节点线速度。

        Returns:
            局部坐标系根节点线速度张量，shape: [num_envs, 3]
        """
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self):
        """
        局部坐标系根节点角速度。

        Returns:
            局部坐标系根节点角速度张量，shape: [num_envs, 3]
        """
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_forward_vector_w(self):
        """
        世界坐标系下机器人前向向量。

        Returns:
            前向向量张量，shape: [num_envs, 3]
        """
        return quat_apply(self.root_link_quat_w, self.forward_vec_b)

    @property
    def projected_gravity_b(self):
        """
        局部坐标系下的重力投影向量。

        Returns:
            重力投影向量张量，shape: [num_envs, 3]
        """
        return quat_apply_inverse(self.root_link_quat_w, self.gravity_vec_w)

    @property
    def joint_pos(self):
        """
        关节角度位置。

        Returns:
            关节角度位置张量，shape: [num_envs, num_joints]
        """
        return self._get_cached_item(
            "dof_positions",
            lambda: torch.as_tensor(self.batch_artic.get_dof_positions(), dtype=torch.float32, device=self.device)
        )

    @property
    def joint_vel(self):
        """
        关节角速度。

        Returns:
            关节角速度张量，shape: [num_envs, num_joints]
        """
        return self._get_cached_item(
            "dof_velocities",
            lambda: torch.as_tensor(self.batch_artic.get_dof_velocities(), dtype=torch.float32, device=self.device)
        )

    def site_pose_w(self, site_name: str, offset: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        获取指定site（刚体/末端执行器）在世界坐标系下的姿态
        支持叠加偏移量（offset），用于计算末端执行器点位。

        Args:
            site_name: 刚体名称
            offset: 位置+姿态偏移 [x,y,z,qw,qx,qy,qz]，shape: [7]

        Returns:
            世界坐标系下最终姿态张量，shape: [7]
        """
        rigid_body_names = self.batch_artic.rigid_body_names
        site_id = next((i for i, n in enumerate(rigid_body_names) if n == site_name), -1)
        if self.is_fixed_base and site_id >= 0:
            site_id -= 1

        rigid_body_pose = self._get_cached_item(
            "rigid_body_pose",
            lambda: self.batch_artic.get_rigid_body_pose()
        )
        site_pose = rigid_body_pose[:, site_id]
        site_pos = site_pose[:3]
        site_quat = np.array([site_pose[6], *site_pose[3:6]])

        ee_offset_pos = offset[:3]
        ee_offset_quat = offset[3:]
        if isinstance(ee_offset_pos, np.ndarray):
            ee_offset_pos = torch.tensor(ee_offset_pos, device=self.device)
        if isinstance(ee_offset_quat, np.ndarray):
            ee_offset_quat = torch.tensor(ee_offset_quat, device=self.device)

        site_pos_t = torch.tensor(site_pos, device=self.device)
        site_quat_t = torch.tensor(site_quat, device=self.device)
        ee_pos, ee_quat = combine_frame_transforms(site_pos_t, site_quat_t, ee_offset_pos, ee_offset_quat)
        return torch.cat([ee_pos, ee_quat], dim=0)

    # ============================== 写入接口（优化拷贝） ==============================
    def write_root_pose(self, pose: torch.Tensor, env_ids=None):
        """
        设置机器人根节点姿态（仅浮动基）
        自动转换四元数顺序并写入底层仿真。

        Args:
            pose: 根节点姿态张量，shape: [num_envs, 7]
            env_ids: 环境ID，可选，默认所有环境

        Raises:
            ValueError: 当机器人为固定基时
        """
        if self.is_fixed_base:
            raise ValueError("Fixed base cannot write root pose")
        env_ids = self._resolve_env_ids(env_ids)

        new_pose = torch.empty_like(pose, device=self.device)
        new_pose[:, :3] = pose[:, :3]
        new_pose[:, 6] = pose[:, 3]
        new_pose[:, 3:6] = pose[:, 4:]

        # 仅在必要时拷贝到CPU，减少数据传输开销
        new_pose_np = new_pose.detach().cpu().numpy() if new_pose.device.type != "cpu" else new_pose.numpy()

        self.batch_artic.set_root_pose(new_pose_np, env_ids)
        self.invalidate_cache()

    def write_root_velocity(self, vel: torch.Tensor, env_ids=None):
        """
        设置根节点速度（仅浮动基）。

        Args:
            vel: 根节点速度张量，shape: [num_envs, 6]
            env_ids: 环境ID，可选，默认所有环境

        Raises:
            ValueError: 当机器人为固定基时
        """
        if self.is_fixed_base:
            raise ValueError("Fixed base cannot write root velocity")
        env_ids = self._resolve_env_ids(env_ids)

        vel_np = vel.detach().cpu().numpy() if vel.device.type != "cpu" else vel.numpy()
        self.batch_artic.set_root_velocity(vel_np, env_ids)
        self.invalidate_cache()

    def write_joint_state(self, position, velocity, joint_ids=None, env_ids=None):
        """
        一次性写入关节位置+速度。

        Args:
            position: 关节位置张量，shape: [num_envs, num_joints]
            velocity: 关节速度张量，shape: [num_envs, num_joints]
            joint_ids: 关节ID，可选，默认所有关节
            env_ids: 环境ID，可选，默认所有环境
        """
        self.write_joint_position(position, joint_ids, env_ids)
        self.write_joint_velocity(velocity, joint_ids, env_ids)

    def write_joint_position(self, position, joint_ids=None, env_ids=None):
        """
        设置关节角度位置。

        Args:
            position: 关节位置张量，shape: [num_envs, num_joints]
            joint_ids: 关节ID，可选，默认所有关节
            env_ids: 环境ID，可选，默认所有环境

        Raises:
            ValueError: 当机器人非铰接结构时
            RuntimeError: 当设置关节位置失败时
        """
        if not self.is_articulated:
            raise ValueError("Not articulated")
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)

        pos_np = position.detach().cpu().numpy() if position.device.type != "cpu" else position.numpy()

        try:
            self.batch_artic.set_dof_positions(pos_np, env_ids, joint_ids)
        except Exception as e:
            raise RuntimeError(
                f"设置关节位置失败 - "
                f"position类型: {type(pos_np)}, shape: {pos_np.shape}, "
                f"env_ids类型: {type(env_ids)}, 值: {env_ids}, "
                f"joint_ids类型: {type(joint_ids)}, 值: {joint_ids}"
            ) from e

        self.invalidate_cache()

    def write_joint_velocity(self, velocity, joint_ids=None, env_ids=None):
        """
        设置关节速度。

        Args:
            velocity: 关节速度张量，shape: [num_envs, num_joints]
            joint_ids: 关节ID，可选，默认所有关节
            env_ids: 环境ID，可选，默认所有环境

        Raises:
            ValueError: 当机器人非铰接结构时
        """
        if not self.is_articulated:
            raise ValueError("Not articulated")
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)

        vel_np = velocity.detach().cpu().numpy() if velocity.device.type != "cpu" else velocity.numpy()
        self.batch_artic.set_dof_velocities(vel_np, env_ids, joint_ids)
        self.invalidate_cache()

    def write_joint_efforts(self, efforts, joint_ids=None, env_ids=None):
        """
        直接设置关节力矩/力控。

        Args:
            efforts: 关节力矩张量，shape: [num_envs, num_joints]
            joint_ids: 关节ID，可选，默认所有关节
            env_ids: 环境ID，可选，默认所有环境

        Raises:
            ValueError: 当机器人非铰接结构时
        """
        if not self.is_articulated:
            raise ValueError("Not articulated")
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)

        efforts_np = efforts.detach().cpu().numpy() if efforts.device.type != "cpu" else efforts.numpy()
        self.batch_artic.set_dof_efforts(efforts_np, env_ids, joint_ids)
        self.invalidate_cache()

    def write_external_wrench(self, force=None, torque=None, body_ids=None, env_ids=None):
        """
        对外施加外力/外力矩。

        Args:
            force: 外力张量，shape: [num_envs, num_bodies, 3]，可选，默认0
            torque: 外力矩张量，shape: [num_envs, num_bodies, 3]，可选，默认0
            body_ids: 刚体ID，可选，默认所有刚体
            env_ids: 环境ID，可选，默认所有环境
        """
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)

        force = force if force is not None else torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        torque = torque if torque is not None else torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)

        data = torch.cat([force, torque], dim=-1)
        data_np = data.detach().cpu().numpy() if data.device.type != "cpu" else data.numpy()

        self.batch_artic.set_link_force_torque(data_np, env_ids, body_ids)
        self.invalidate_cache()

    def write_ctrl(self, ctrl, ctrl_ids=None, env_ids=None):
        """
        设置关节位置控制目标（PD target）。

        Args:
            ctrl: 关节位置控制目标张量，shape: [num_envs, num_actuators]
            ctrl_ids: 控制目标ID，可选，默认所有执行器
            env_ids: 环境ID，可选，默认所有环境

        Raises:
            ValueError: 当机器人非驱动结构时
            RuntimeError: 当设置关节位置目标失败时
        """
        if not self.is_actuated:
            raise ValueError("Not actuated")
        env_ids = self._resolve_env_ids(env_ids)
        ctrl_ids = self._resolve_joint_ids(ctrl_ids)

        ctrl_np = ctrl.detach().cpu().numpy() if ctrl.device.type != "cpu" else ctrl.numpy()

        try:
            self.batch_artic.set_dof_position_targets(ctrl_np, env_ids, ctrl_ids)
        except Exception as e:
            raise RuntimeError(
                f"设置关节位置目标失败 - "
                f"ctrl类型: {type(ctrl_np)}, shape: {ctrl_np.shape}, "
                f"env_ids类型: {type(env_ids)}, 值: {env_ids}, "
                f"ctrl_ids类型: {type(ctrl_ids)}, 值: {ctrl_ids}"
            ) from e
        self.invalidate_cache()

    def write_velocity(self, ctrl, ctrl_ids=None, env_ids=None):
        """
        设置关节速度控制目标。

        Args:
            ctrl: 关节速度控制目标张量，shape: [num_envs, num_actuators]
            ctrl_ids: 控制目标ID，可选，默认所有执行器
            env_ids: 环境ID，可选，默认所有环境

        Raises:
            ValueError: 当机器人非驱动结构时
        """
        if not self.is_actuated:
            raise ValueError("Not actuated")
        env_ids = self._resolve_env_ids(env_ids)
        ctrl_ids = self._resolve_joint_ids(ctrl_ids)

        ctrl_np = ctrl.detach().cpu().numpy() if ctrl.device.type != "cpu" else ctrl.numpy()
        self.batch_artic.set_dof_velocity_targets(ctrl_np, env_ids, ctrl_ids)
        self.invalidate_cache()

    def set_rigid_body_mass(self, mass, env_ids, body_ids):
        """
        动态设置刚体质量。

        Args:
            mass: 刚体质量张量，shape: [num_envs]
            env_ids: 环境ID
            body_ids: 刚体ID
        """
        mass = mass.unsqueeze(1)
        mass_np = mass.detach().cpu().numpy() if mass.device.type != "cpu" else mass.numpy()

        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)

        self.batch_artic.set_rigid_body_mass(mass_np, env_ids, body_ids)
        self.invalidate_cache()

    def set_body_link_friction_params(self, params, env_ids, body_ids):
        """
        设置刚体静摩擦/动摩擦参数。

        Args:
            params: 摩擦参数张量，[静摩擦, 动摩擦]，shape: [num_envs, num_bodies, 2]
            env_ids: 环境ID
            body_ids: 刚体ID
        """
        sf = params[:, :, 0]
        df = params[:, :, 1]

        sf_np = sf.detach().cpu().numpy() if sf.device.type != "cpu" else sf.numpy()
        df_np = df.detach().cpu().numpy() if df.device.type != "cpu" else df.numpy()

        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)

        self.batch_artic.set_link_static_friction(sf_np, env_ids, body_ids)
        self.batch_artic.set_link_dynamic_friction(df_np, env_ids, body_ids)
        self.invalidate_cache()

    def get_rigid_body_mass(self):
        """
        获取刚体质量。

        Returns:
            刚体质量张量，shape: [num_envs, num_bodies]
        """
        return self._get_cached_item(
            "rigid_body_mass",
            lambda: torch.as_tensor(self.batch_artic.get_rigid_body_mass(), dtype=torch.float32, device=self.device)
        )

    def get_body_link_friction_params(self):
        """
        获取刚体摩擦参数。

        Returns:
            摩擦参数张量，[静摩擦, 动摩擦]，shape: [num_envs, num_bodies, 2]
        """
        return self._get_cached_item(
            "link_friction_params",
            lambda: torch.as_tensor(self.batch_artic.get_link_friction_params(), dtype=torch.float32, device=self.device)
        )


# ============================== 测试 ==============================
if __name__ == "__main__":
    device = "cpu"
    num_envs = 4
    num_bodies = 5
    num_joints = 4

    # 模拟EntityIndexing（实际使用替换为真实实例）
    class MockEntityIndexing:
        def __init__(self, body_ids, geom_ids, site_ids, ctrl_ids, joint_ids, mocap_id,
                     joint_q_adr, joint_v_adr, free_joint_q_adr, free_joint_v_adr, sensor_adr):
            self.body_ids = body_ids
            self.geom_ids = geom_ids
            self.site_ids = site_ids
            self.ctrl_ids = ctrl_ids
            self.joint_ids = joint_ids
            self.mocap_id = mocap_id
            self.joint_q_adr = joint_q_adr
            self.joint_v_adr = joint_v_adr
            self.free_joint_q_adr = free_joint_q_adr
            self.free_joint_v_adr = free_joint_v_adr
            self.sensor_adr = sensor_adr


    indexing = MockEntityIndexing(
        body_ids=torch.arange(num_bodies, device=device),
        geom_ids=torch.arange(3, device=device),
        site_ids=torch.arange(2, device=device),
        ctrl_ids=torch.arange(4, device=device),
        joint_ids=torch.arange(num_joints, device=device),
        mocap_id=None,
        joint_q_adr=torch.arange(7, 7 + num_joints, device=device),
        joint_v_adr=torch.arange(6, 6 + num_joints, device=device),
        free_joint_q_adr=torch.arange(7, device=device),
        free_joint_v_adr=torch.arange(6, device=device),
        sensor_adr={"force": torch.arange(3, device=device)}
    )

    # 模拟BatchArticulation（实际使用替换为真实接口）
    class MockBatchArticulation:
        def __init__(self):
            self.num_joint = num_joints
            self.num_dof = num_joints
            self.rigid_body_names = [f"body_{i}" for i in range(num_bodies)]

        def get_root_pose(self): return np.zeros((num_envs, 7))
        def get_dof_positions(self): return np.zeros((num_envs, num_joints))
        def get_dof_velocities(self): return np.zeros((num_envs, num_joints))
        def get_rigid_body_pose(self): return np.zeros((num_envs, num_bodies, 7))
        def get_rigid_body_velocity(self, rigid_body_indices): return np.zeros((num_envs, num_bodies, 6))
        def get_link_c_mass_pose(self, rigid_body_indices): return np.zeros((num_envs, num_bodies, 7))
        def get_contact_sensor_datas(self): return [[] for _ in range(num_envs)]
        def get_dof_drive_params(self): return np.zeros((num_envs, num_joints, 2))
        def get_qacc(self): return np.zeros((num_envs, 6 + num_joints))
        def get_dof_position_targets(self): return np.zeros((num_envs, num_joints))
        def get_rigid_body_mass(self): return np.zeros((num_envs, num_bodies))
        def get_link_friction_params(self): return np.zeros((num_envs, num_bodies, 2))

        def set_root_pose(self, pose, env_ids): pass
        def set_root_velocity(self, vel, env_ids): pass
        def set_dof_positions(self, pos, env_ids, joint_ids): pass
        def set_dof_velocities(self, vel, env_ids, joint_ids): pass
        def set_dof_efforts(self, efforts, env_ids, joint_ids): pass
        def set_link_force_torque(self, data, env_ids, body_ids): pass
        def set_dof_position_targets(self, ctrl, env_ids, ctrl_ids): pass
        def set_dof_velocity_targets(self, ctrl, env_ids, ctrl_ids): pass
        def set_rigid_body_mass(self, mass, env_ids, body_ids): pass
        def set_link_static_friction(self, sf, env_ids, body_ids): pass
        def set_link_dynamic_friction(self, df, env_ids, body_ids): pass


    batch_artic = MockBatchArticulation()

    # 初始化默认状态
    default_root_state = torch.zeros(num_envs, 13, device=device)
    default_joint_pos = torch.zeros(num_envs, num_joints, device=device)
    default_joint_vel = torch.zeros(num_envs, num_joints, device=device)
    default_joint_stiffness = torch.zeros(num_envs, num_joints, device=device)
    default_joint_damping = torch.zeros(num_envs, num_joints, device=device)
    default_joint_pos_limits = torch.zeros(num_joints, 2, device=device)
    joint_pos_limits = torch.zeros(num_joints, 2, device=device)
    soft_joint_pos_limits = torch.zeros(num_joints, 2, device=device)
    gravity_vec_w = torch.tensor([0.0, 0.0, -9.81], device=device)
    forward_vec_b = torch.tensor([1.0, 0.0, 0.0], device=device)

    # 创建实例
    entity_data = EntityData(
        indexing=indexing,
        batch_artic=batch_artic,
        device=device,
        default_root_state=default_root_state,
        default_joint_pos=default_joint_pos,
        default_joint_vel=default_joint_vel,
        default_joint_stiffness=default_joint_stiffness,
        default_joint_damping=default_joint_damping,
        default_joint_pos_limits=default_joint_pos_limits,
        joint_pos_limits=joint_pos_limits,
        soft_joint_pos_limits=soft_joint_pos_limits,
        gravity_vec_w=gravity_vec_w,
        forward_vec_b=forward_vec_b,
        is_fixed_base=False,
        is_articulated=True,
        is_actuated=True
    )

    print("✅ EntityData 优化版初始化完成！")
    print(f"  - 环境数量: {entity_data.num_envs}")
    print(f"  - 刚体数量: {entity_data.num_bodies}")
    print(f"  - 关节数量: {entity_data.num_joints}")
    print(f"  - 执行器数量: {entity_data.num_actuators}")
