import torch
import trimesh
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable


@dataclass
class HeightFieldConfig:
    """高度场配置参数"""
    size: Tuple[float, float] = (10.0, 10.0)  # 地形尺寸 (x, y)，单位：米
    horizontal_scale: float = 0.1  # x/y 方向的离散化精度，单位：米/像素
    vertical_scale: float = 0.1  # z 方向的缩放比例，单位：米/单位高度
    border_width: float = 0.2  # 边界宽度，用于避免边缘 artifacts
    slope_threshold: Optional[float] = None  # 坡度阈值，超过则修正为垂直面
    device: str = "cpu"  # 计算设备（cpu/cuda）
    dtype: torch.dtype = torch.float32  # 数据类型


def convert_height_field_to_mesh(
        height_field: torch.Tensor,
        horizontal_scale: float,
        vertical_scale: float,
        slope_threshold: Optional[float] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将高度场 Tensor 转换为三角形网格的顶点和三角形索引

    Args:
        height_field: 2D 高度场 Tensor (num_rows, num_cols)
        horizontal_scale: x/y 方向的离散化精度（米/像素）
        vertical_scale: z 方向的缩放比例（米/单位高度）
        slope_threshold: 坡度阈值（弧度），超过则修正垂直面，None 表示不修正
        device: 计算设备
        dtype: 数据类型

    Returns:
        vertices: 顶点 Tensor (num_vertices, 3)，每个元素为 (x, y, z) 坐标
        triangles: 三角形索引 Tensor (num_triangles, 3)，每个元素为顶点索引
    """
    num_rows, num_cols = height_field.shape
    height_field = height_field.to(device=device, dtype=dtype)

    # 创建网格坐标（使用 torch.meshgrid，注意索引顺序）
    y = torch.linspace(0, (num_cols - 1) * horizontal_scale, num_cols, device=device, dtype=dtype)
    x = torch.linspace(0, (num_rows - 1) * horizontal_scale, num_rows, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="xy")  # (num_rows, num_cols)
    hf = height_field.clone()

    # 修正超过坡度阈值的垂直面
    if slope_threshold is not None:
        # 根据缩放比例调整坡度阈值
        slope_threshold_scaled = slope_threshold * horizontal_scale / vertical_scale

        # 计算顶点移动量
        move_x = torch.zeros((num_rows, num_cols), device=device, dtype=torch.int32)
        move_y = torch.zeros((num_rows, num_cols), device=device, dtype=torch.int32)
        move_corners = torch.zeros((num_rows, num_cols), device=device, dtype=torch.int32)

        # 沿 x 轴方向修正（上下相邻像素）
        move_x[:-1, :] += (hf[1:, :] - hf[:-1, :]) > slope_threshold_scaled
        move_x[1:, :] -= (hf[:-1, :] - hf[1:, :]) > slope_threshold_scaled

        # 沿 y 轴方向修正（左右相邻像素）
        move_y[:, :-1] += (hf[:, 1:] - hf[:, :-1]) > slope_threshold_scaled
        move_y[:, 1:] -= (hf[:, :-1] - hf[:, 1:]) > slope_threshold_scaled

        # 沿对角线方向修正（对角相邻像素）
        move_corners[:-1, :-1] += (hf[1:, 1:] - hf[:-1, :-1]) > slope_threshold_scaled
        move_corners[1:, 1:] -= (hf[:-1, :-1] - hf[1:, 1:]) > slope_threshold_scaled

        # 应用顶点移动（转换为 float 类型进行计算）
        move_x_float = move_x.to(dtype=dtype)
        move_y_float = move_y.to(dtype=dtype)
        move_corners_float = move_corners.to(dtype=dtype)

        xx += (move_x_float + move_corners_float * (move_x == 0).to(dtype=dtype)) * horizontal_scale
        yy += (move_y_float + move_corners_float * (move_y == 0).to(dtype=dtype)) * horizontal_scale

    # 生成顶点 (num_rows * num_cols, 3)
    vertices = torch.zeros((num_rows * num_cols, 3), device=device, dtype=dtype)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale

    # 生成三角形索引
    num_triangles = 2 * (num_rows - 1) * (num_cols - 1)
    triangles = torch.zeros((num_triangles, 3), device=device, dtype=torch.int32)

    # 使用向量化操作生成三角形索引（比循环更高效）
    i = torch.arange(num_rows - 1, device=device)
    j = torch.arange(num_cols - 1, device=device)
    ii, jj = torch.meshgrid(i, j, indexing="xy")  # (num_rows-1, num_cols-1)

    # 四个顶点索引
    idx0 = ii * num_cols + jj  # (i, j)
    idx1 = ii * num_cols + (jj + 1)  # (i, j+1)
    idx2 = (ii + 1) * num_cols + jj  # (i+1, j)
    idx3 = (ii + 1) * num_cols + (jj + 1)  # (i+1, j+1)

    # 展开为一维数组
    idx0_flat = idx0.flatten()
    idx1_flat = idx1.flatten()
    idx2_flat = idx2.flatten()
    idx3_flat = idx3.flatten()

    # 填充三角形索引
    triangles[::2, 0] = idx0_flat
    triangles[::2, 1] = idx3_flat
    triangles[::2, 2] = idx1_flat



    triangles[1::2, 0] = idx0_flat
    triangles[1::2, 1] = idx2_flat
    triangles[1::2, 2] = idx3_flat

    return vertices, triangles


def generate_mesh_from_height_field(

        difficulty: float = 0.5,
        cfg: Optional[HeightFieldConfig] = None
) -> Tuple[trimesh.Trimesh, torch.Tensor]:
    """
    从高度场函数生成 trimesh 网格（全程使用 Tensor 计算）

    Args:
        height_field_func: 高度场生成函数，输入 (difficulty, cfg)，返回 2D Tensor 高度场
        difficulty: 难度参数（用于高度场函数），范围 [0, 1]
        cfg: 高度场配置，默认使用默认配置

    Returns:
        mesh: trimesh 网格对象
        origin: 地形原点坐标 Tensor (x, y, z)，位于地形中心底部
    """


    # 使用默认配置如果未提供
    if cfg is None:
        cfg = HeightFieldConfig()


    # 验证边界宽度
    if cfg.border_width > 0 and cfg.border_width < cfg.horizontal_scale:
        raise ValueError(
            f"边界宽度 ({cfg.border_width}) 必须大于等于水平缩放比例 ({cfg.horizontal_scale})"
        )

    # 计算像素尺寸（包含边界）
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    border_pixels = int(cfg.border_width / cfg.horizontal_scale)

    # 初始化高度场数组（包含边界）
    heights = torch.zeros((width_pixels+2*border_pixels, length_pixels+2*border_pixels),
                          device=cfg.device, dtype=torch.int16)


    # 计算实际生成区域的尺寸（扣除边界）
    sub_terrain_size = (
        (width_pixels - 2 * border_pixels) * cfg.horizontal_scale,
        (length_pixels - 2 * border_pixels) * cfg.horizontal_scale
    )

    # 临时修改配置用于生成核心区域



    # 生成核心区域高度场（函数返回 Tensor）
    core_heights = cfg.func(difficulty)
    core_heights = core_heights.to(device=cfg.device, dtype=torch.int16)



    # 将核心区域高度场放入包含边界的数组中
    heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = core_heights


    # 转换为网格（返回 Tensor 顶点和三角形）
    vertices, triangles = convert_height_field_to_mesh(
        heights,
        horizontal_scale=cfg.horizontal_scale,
        vertical_scale=cfg.vertical_scale,
        slope_threshold=cfg.slope_threshold,
        device=cfg.device,
        dtype=cfg.dtype
    )

    # 转换为 numpy 数组用于 trimesh（trimesh 不直接支持 Tensor）
    vertices_np = vertices.cpu().numpy()
    triangles_np = triangles.cpu().numpy().astype(np.uint32)

    # 创建 trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=triangles_np)

    # 计算地形原点（中心底部）
    center_x = cfg.size[0] / 2.0
    center_y = cfg.size[1] / 2.0

    # 计算中心区域的最大高度（作为原点 z 坐标）
    x1 = int((center_x - 1.0) / cfg.horizontal_scale)
    x2 = int((center_x + 1.0) / cfg.horizontal_scale)
    y1 = int((center_y - 1.0) / cfg.horizontal_scale)
    y2 = int((center_y + 1.0) / cfg.horizontal_scale)

    # 确保索引在有效范围内
    x1 = max(0, x1)
    x2 = min(width_pixels, x2)
    y1 = max(0, y1)
    y2 = min(length_pixels, y2)

    # 计算中心区域最大高度
    center_heights = heights[x1:x2, y1:y2]

    origin_z = center_heights.max() * cfg.vertical_scale


    # 原点 Tensor
    origin = torch.tensor([center_x, center_y, origin_z],
                          device=cfg.device, dtype=cfg.dtype)

    return mesh, origin


# ------------------------------
# 示例：如何使用（全程 Tensor 计算）
# ------------------------------
if __name__ == "__main__":
    # 1. 定义一个基于 Tensor 的高度场生成函数（正弦波浪地形）
    def sine_wave_terrain(difficulty: float, cfg: HeightFieldConfig) -> torch.Tensor:
        """生成正弦波浪高度场（返回 Tensor）"""
        # 计算核心区域的像素尺寸
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1

        # 创建网格（Tensor 版本）
        x = torch.linspace(0, cfg.size[0], width_pixels,
                           device=cfg.device, dtype=cfg.dtype)
        y = torch.linspace(0, cfg.size[1], length_pixels,
                           device=cfg.device, dtype=cfg.dtype)
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        # 生成正弦波浪（难度控制波浪幅度）
        amplitude = 0.5 * difficulty  # 最大高度 0.5 米
        frequency = 2.0  # 每米的波浪数
        heights = amplitude * torch.sin(2 * torch.pi * frequency * xx / cfg.size[0]) + \
                  amplitude * torch.cos(2 * torch.pi * frequency * yy / cfg.size[1])

        # 转换为 int16 类型（高度场存储为整数以节省空间）
        return (heights / cfg.vertical_scale).to(dtype=torch.int16)


    # 2. 配置参数（支持 CUDA 加速）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HeightFieldConfig(
        size=(20.0, 20.0),  # 地形尺寸 20x20 米
        horizontal_scale=0.05,  # 5 厘米/像素的精度
        vertical_scale=0.05,  # 5 厘米/单位高度
        border_width=0.1,  # 10 厘米边界
        slope_threshold=torch.tan(torch.tensor(np.deg2rad(60), dtype=torch.float32)),  # 60度坡度阈值
        device=device,
        dtype=torch.float32
    )

    # 3. 生成网格（全程使用 Tensor 计算）
    mesh, origin = generate_mesh_from_height_field(
        height_field_func=sine_wave_terrain,
        difficulty=0.8,  # 高难度 = 更大的波浪
        cfg=config
    )

    # 4. 输出信息并可视化
    print(f"计算设备：{device}")
    print(f"生成的网格：{len(mesh.vertices)} 个顶点，{len(mesh.faces)} 个三角形")
    print(f"地形原点：{origin.cpu().numpy()}")

    # 显示网格（需要安装 pycollada）
    mesh.show()

    # 5. 保存网格（可选）
    # mesh.export("terrain_tensor.obj")
