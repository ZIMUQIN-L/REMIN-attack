from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
import numpy as np
from sklearn.preprocessing import RobustScaler


def orthogonal_align_to_grid(coords, grid_shape):
    x, y = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]))
    grid = np.column_stack([x.ravel(), y.ravel()])[:len(coords)]

    grid_centered = grid - np.mean(grid, axis=0)
    coords_centered = coords - np.mean(coords, axis=0)

    R, scale = orthogonal_procrustes(coords_centered, grid_centered)
    aligned = coords_centered.dot(R) * scale + np.mean(grid, axis=0)

    return aligned


def scale_coords(coords, x_range, y_range):
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])

    # 线性缩放
    scaled_x = (coords[:, 0] - x_min) / (x_max - x_min) * (x_range[1] - x_range[0]) + x_range[0]
    scaled_y = (coords[:, 1] - y_min) / (y_max - y_min) * (y_range[1] - y_range[0]) + y_range[0]
    return np.column_stack([scaled_x, scaled_y])


def original_align_to_grid(coords, grid_shape):
    n = len(coords)
    grid_size = int(np.ceil(np.sqrt(n)))
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    grid = np.column_stack([x.ravel(), y.ravel()])[:n]

    _, aligned, _ = procrustes(grid, coords)
    scaled = robust_scale_coords(aligned, [1, grid_shape[0] - 1], [1, grid_shape[1] - 1])
    return scaled


def robust_scale_coords(coords, x_range, y_range):
    scaler = RobustScaler(
        quantile_range=(25, 75),  # 使用25%-75%分位数范围
        with_centering=True,
        with_scaling=True
    )
    scaled = scaler.fit_transform(coords)

    # 映射到目标范围
    scaled[:, 0] = (scaled[:, 0] - scaled[:, 0].min()) / (scaled[:, 0].max() - scaled[:, 0].min()) * (
                x_range[1] - x_range[0]) + x_range[0]
    scaled[:, 1] = (scaled[:, 1] - scaled[:, 1].min()) / (scaled[:, 1].max() - scaled[:, 1].min()) * (
                y_range[1] - y_range[0]) + y_range[0]

    return scaled