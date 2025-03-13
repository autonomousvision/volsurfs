import numpy as np
import torch


def get_planar_2d_points(height, width, min=-1, max=1, device="cuda"):
    # create 2d grid to evaluate sdfs
    points_2d = np.meshgrid(np.linspace(min, max, height), np.linspace(min, max, width))
    points_2d = np.stack(points_2d, axis=-1)
    points_2d = points_2d.reshape(-1, 2)
    points_2d = torch.tensor(points_2d, dtype=torch.float32, device=device)
    return points_2d


def get_planar_3d_points(
    height, width, axis="xy", plane=0, min=-1, max=1, device="cuda"
):
    points_2d = get_planar_2d_points(height, width, min=min, max=max, device=device)
    additional_axis = torch.ones(points_2d.shape[0], 1, device=device) * plane
    points_3d = torch.cat([points_2d, additional_axis], dim=-1)
    if axis == "xy" or axis == "yx":
        return points_3d
    if axis == "xz" or axis == "zx":
        return points_3d[:, [0, 2, 1]]
    if axis == "yz" or axis == "zy":
        return points_3d[:, [1, 2, 0]]
    return None


def get_spatial_3d_points(height, width, depth, device="cuda"):
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    z = np.linspace(-1, 1, depth)
    X, Y, Z = np.meshgrid(x, y, z)
    points_3d = np.stack([X, Y, Z], axis=-1)
    points_3d = points_3d.reshape(-1, 3)
    points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
    return points_3d
