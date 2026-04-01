# 导入数据类相关装饰器和字段函数
from dataclasses import dataclass, field
# 导入mozisim仿真库的动态立方体类
from mozisim.core.api.objects import DynamicCuboid
# 导入为prim添加关节根API的工具函数
from mozisim.utils.articulation_utils import add_articulation_root_api_to_prim
# 导入物理材质类
from mozisim.core.api.materials import PhysicsMaterial
# 导入PyTorch库（用于张量计算）
import torch
# 导入numpy库（用于数值计算）
import numpy as np
from typing import Optional, Union, Sequence


# ============================== 工具函数 ==============================
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
        if max_length is None:
            raise ValueError("当data为None时，必须指定max_length参数")
        return list(range(max_length))
    elif isinstance(data, slice):
        if max_length is None:
            raise ValueError("当data为slice时，必须指定max_length参数")
        start = data.start if data.start is not None else 0
        stop = data.stop if data.stop is not None else max_length
        step = data.step if data.step is not None else 1
        return list(range(start, stop, step))
    elif isinstance(data, torch.Tensor):
        if data.numel() == 1:
            return int(data.item())
        else:
            return data.cpu().tolist()
    elif isinstance(data, np.ndarray):
        if data.size == 1:
            return int(data.item())
        else:
            return data.tolist()
    elif isinstance(data, int):
        return data
    elif isinstance(data, list):
        return [int(x) for x in data]
    else:
        raise TypeError(f"不支持的索引类型: {type(data)}")


# ========== 全局配置 ==========
OBJECT_DICT = {
    "cube": DynamicCuboid,
}


# ========== 配置数据类 ==========
@dataclass(kw_only=True)
class ObjectCfg:
    object_type: str = "cube"
    num_envs: int = 1
    size: list | None = None
    mass: float | None = None


# ========== 核心物体类 ==========
class Object:
    """
    仿真环境中的物体管理类（带懒加载缓存优化）
    负责创建、初始化、管理仿真物体的状态（位姿、速度等）
    """

    def __init__(self, object_name, object_cfg, device):
        """
        初始化物体实例
        Args:
            object_name (str): 物体名称（用于生成唯一的prim路径）
            object_cfg (ObjectCfg): 物体配置实例
            device (torch.device): 计算设备（cpu/cuda），用于张量存储
        """
        # 保存配置
        self.cfg = object_cfg
        self.num_envs = self.cfg.num_envs
        self.device = device

        # 校验物体类型是否合法
        if self.cfg.object_type not in OBJECT_DICT.keys():
            raise ValueError(f"不支持的物体类型: {self.cfg.object_type}，支持类型: {list(OBJECT_DICT.keys())}")

        # 初始化物体默认根状态
        self.default_root_state = torch.tensor(
            [0.0, 0.0, self.cfg.size[2] / 2, 1.0, 0.0, 0.0, 0.0],
            device=self.device
        )

        # 创建物理材质
        physics_material = PhysicsMaterial(
            prim_path="/Materials/DefaultGroundMaterial",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0,
        )

        # 存储每个环境中物体的prim路径
        self._object_prim_paths = []

        # 为每个环境创建物体实例
        for i in range(self.num_envs):
            object_prim_path = f"/World/{object_name}_{i}"
            OBJECT_DICT[self.cfg.object_type](
                prim_path=object_prim_path,
                name=f"{object_name}_{i}",
                size=self.cfg.size,
                mass=self.cfg.mass,
                physics_material=physics_material,
                color=np.array([0.0, 0.0, 0.0]),
            )
            add_articulation_root_api_to_prim(object_prim_path)
            self._object_prim_paths.append(object_prim_path)

    @property
    def object_prim_paths(self):
        """只读属性：获取所有环境的物体prim路径列表"""
        return self._object_prim_paths

    def init_object(self):
        """初始化物体的批量控制接口（带缓存）"""
        from mozisim.core.api.articulation import BatchArticulation
        self.batch_object = BatchArticulation(self.object_prim_paths)
        # 初始化带缓存的物体数据管理类
        self.init_data()

    def init_data(self):
        """初始化带懒加载缓存的物体数据管理实例"""
        self._data = ObjectData(
            batch_object=self.batch_object,
            default_root_state=self.default_root_state,
            num_envs=self.num_envs,
            device=self.device
        )

    @property
    def data(self):
        """只读属性：获取物体数据管理实例"""
        return self._data

    def write_root_link_pose_to_sim(self, pose, env_ids):
        """将位姿数据写入仿真环境（自动处理缓存失效）"""
        self.data.write_root_link_pose(pose, env_ids)

    def write_root_link_velocity_to_sim(self, velocity, env_ids):
        """将速度数据写入仿真环境（自动处理缓存失效）"""
        self.data.write_root_link_velocity(velocity, env_ids)


# ========== 物体数据管理类（带懒加载缓存） ==========
class ObjectData:
    """
    物体状态数据管理类（带懒加载缓存优化）
    负责读取/写入仿真物体的位姿、速度等状态，避免重复调用底层接口
    """

    def __init__(self, batch_object, default_root_state, num_envs, device):
        """
        初始化数据管理实例
        Args:
            batch_object (BatchArticulation): 批量关节控制实例
            default_root_state (torch.Tensor): 默认根状态
            num_envs (int): 环境数量
            device (torch.device): 计算设备
        """
        self.batch_object = batch_object
        self.default_root_state = default_root_state
        self.num_envs = num_envs
        self.device = device

        # ========== 懒加载缓存核心配置 ==========
        self._cache = {}  # 缓存数据存储
        self._cache_timestamps = {}  # 缓存项更新时间戳
        self._step_counter = 0  # 当前仿真步数（用于判断缓存是否过期）

    def _get_cached_item(self, key, loader_func):
        """
        统一懒加载缓存入口
        仅当缓存不存在或已过期时，才调用loader_func重新获取数据
        Args:
            key: 缓存键名
            loader_func: 真实数据加载函数（调用底层仿真接口）
        Returns:
            缓存中或新加载的数据
        """
        if key not in self._cache or self._cache_timestamps.get(key, -1) < self._step_counter:
            data = loader_func()
            self._cache[key] = data
            self._cache_timestamps[key] = self._step_counter
        return self._cache[key]

    def invalidate_cache(self):
        """使所有缓存失效（仅递增步数计数器，优化内存操作）"""
        self._step_counter += 1

    def _resolve_env_ids(self, env_ids: Optional[Union[torch.Tensor, slice, int, list]]) -> Union[int, list[int]]:
        """解析环境ID为Python原生类型，避免底层接口报错"""
        return convert_to_python_type(env_ids, max_length=self.num_envs)

    # ============================== 带缓存的属性 ==============================
    @property
    def root_link_pose_w(self):
        """
        只读属性：获取世界坐标系下的根关节位姿（带缓存）
        返回格式：torch.Tensor(N,7)，[x,y,z,w,x,y,z]
        """
        return self._get_cached_item(
            "root_link_pose_w",
            self._load_root_link_pose_w  # 真实加载函数
        )

    def _load_root_link_pose_w(self):
        """真实加载根关节位姿（仅缓存失效时调用）"""
        root_link_pose = self.batch_object.get_root_pose()
        root_link_pose_wxyz = np.zeros_like(root_link_pose)
        # 位置信息（x,y,z）
        root_link_pose_wxyz[:, :3] = root_link_pose[:, :3]
        # 四元数转换：[x,y,z,w] → [w,x,y,z]
        root_link_pose_wxyz[:, 3] = root_link_pose[:, 6]
        root_link_pose_wxyz[:, 4:] = root_link_pose[:, 3:6]
        # 转换为torch张量
        return torch.tensor(
            root_link_pose_wxyz,
            dtype=torch.float32,
            device=self.device
        )

    @property
    def root_link_pos_w(self):
        """只读属性：获取世界坐标系下的根关节位置（带缓存）"""
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self):
        """只读属性：获取世界坐标系下的根关节四元数（带缓存）"""
        return self.root_link_pose_w[:, 3:]

    @property
    def root_link_velocity_w(self):
        """
        只读属性：获取世界坐标系下的根关节速度（带缓存）
        返回格式：torch.Tensor(N,6)，[线速度x,y,z,角速度x,y,z]
        """
        return self._get_cached_item(
            "root_link_velocity_w",
            self._load_root_link_velocity_w
        )

    def _load_root_link_velocity_w(self):
        """真实加载根关节速度（仅缓存失效时调用）"""
        root_link_velocity = self.batch_object.get_root_velocity()
        return torch.tensor(
            root_link_velocity,
            dtype=torch.float32,
            device=self.device
        )

    @property
    def root_lin_vel_w(self):
        """只读属性：获取世界坐标系下的根关节线速度（带缓存）"""
        return self.root_link_velocity_w[:, :3]

    @property
    def root_ang_vel_w(self):
        """只读属性：获取世界坐标系下的根关节角速度（带缓存）"""
        return self.root_link_velocity_w[:, 3:]

    # ============================== 写入接口（自动失效缓存） ==============================
    def write_root_link_pose(self, pose, env_ids):
        """
        将位姿写入仿真环境（自动使缓存失效）
        Args:
            pose (torch.Tensor): 要设置的位姿（N,7）
            env_ids (torch.Tensor): 要更新的环境ID列表
        """
        # 解析环境ID为Python原生类型
        resolved_env_ids = self._resolve_env_ids(env_ids)
        # 写入仿真环境
        self.batch_object.set_root_pose(
            pose.cpu().numpy(),
            resolved_env_ids
        )
        # 使缓存失效（下次读取会重新加载）
        self.invalidate_cache()

    def write_root_link_velocity(self, velocity, env_ids):
        """
        将速度写入仿真环境（自动使缓存失效）
        Args:
            velocity (torch.Tensor): 要设置的速度（N,6）
            env_ids (torch.Tensor): 要更新的环境ID列表
        """
        resolved_env_ids = self._resolve_env_ids(env_ids)
        self.batch_object.set_root_velocity(
            velocity.cpu().numpy(),
            resolved_env_ids
        )
        # 使缓存失效
        self.invalidate_cache()


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 测试初始化
    device = torch.device("cpu")
    cfg = ObjectCfg(
        object_type="cube",
        num_envs=4,
        size=[0.1, 0.1, 0.1],
        mass=1.0
    )

    # 创建物体实例
    obj = Object("test_cube", cfg, device)
    # 初始化批量接口（模拟）
    # obj.init_object()  # 实际使用时需确保mozisim库可用

    print("✅ 带懒加载缓存的Object类初始化完成！")
    print(f"  - 环境数量: {obj.num_envs}")
    print(f"  - 物体路径数量: {len(obj.object_prim_paths)}")
