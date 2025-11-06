import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import KDTree


def visualize_3d_results(original_coords, tsne_coords, title="3D Visualization"):
    """
    可视化3D数据，包括原始坐标和t-SNE结果，支持对重建结果进行交互式旋转
    :param original_coords: 原始3D坐标（列表或numpy数组）
    :param tsne_coords: t-SNE降维后的3D坐标（列表或numpy数组）
    :param title: 图表标题
    """
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8

    # 确保输入是numpy数组
    original_coords = np.array(original_coords)
    tsne_coords = np.array(tsne_coords)

    fig = plt.figure(figsize=(15, 7))

    # 创建两个子图，一个用于原始坐标，一个用于t-SNE结果
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # 计算点的顺序（从左到右）
    order = np.argsort(original_coords[:, 0])
    colors = order / len(order)  # 归一化到[0,1]区间

    # 创建颜色映射
    cmap = plt.cm.viridis

    # 绘制原始坐标（固定视角）
    scatter1 = ax1.scatter(original_coords[:, 0],
                           original_coords[:, 1],
                           original_coords[:, 2],
                           c=colors,
                           cmap=cmap,
                           s=20)
    ax1.set_title('Original Coordinates', fontsize=12)
    ax1.view_init(elev=20, azim=45)  # 固定视角

    # 绘制t-SNE结果（可交互旋转）
    scatter2 = ax2.scatter(tsne_coords[:, 0],
                           tsne_coords[:, 1],
                           tsne_coords[:, 2],
                           c=colors,  # 使用相同的颜色映射
                           cmap=cmap,
                           s=20)
    ax2.set_title('Reconstruction Result', fontsize=12)

    # 添加颜色条
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Point Index')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Point Index')
    cbar1.ax.set_ylabel('Point Index', fontsize=10)
    cbar2.ax.set_ylabel('Point Index', fontsize=10)

    # 设置坐标轴标签和其他属性
    for ax in [ax1, ax2]:
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.grid(True)

        # 设置背景色为白色
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # 设置网格线为灰色
        ax.grid(True, linestyle='--', alpha=0.3)

    # 添加三个旋转控制滑块（只控制重建结果）
    ax_x = plt.axes([0.2, 0.02, 0.6, 0.02])  # X轴旋转
    ax_y = plt.axes([0.2, 0.05, 0.6, 0.02])  # Y轴旋转
    ax_z = plt.axes([0.2, 0.08, 0.6, 0.02])  # Z轴旋转

    x_slider = plt.Slider(ax_x, 'X-Rotation', 0, 360, valinit=0)
    y_slider = plt.Slider(ax_y, 'Y-Rotation', 0, 360, valinit=0)
    z_slider = plt.Slider(ax_z, 'Z-Rotation', 0, 360, valinit=0)

    # 添加缩放滑块
    ax_scale = plt.axes([0.2, 0.11, 0.6, 0.02])
    scale_slider = plt.Slider(ax_scale, 'Scale', 0.1, 2.0, valinit=1.0)

    # 添加重置按钮
    ax_reset = plt.axes([0.45, 0.15, 0.1, 0.04])
    reset_button = plt.Button(ax_reset, 'Reset View')

    def update(val):
        # 获取当前角度和缩放值
        x_rot = np.radians(x_slider.val)
        y_rot = np.radians(y_slider.val)
        z_rot = np.radians(z_slider.val)
        scale = scale_slider.val

        # 计算旋转矩阵
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(x_rot), -np.sin(x_rot)],
                       [0, np.sin(x_rot), np.cos(x_rot)]])

        Ry = np.array([[np.cos(y_rot), 0, np.sin(y_rot)],
                       [0, 1, 0],
                       [-np.sin(y_rot), 0, np.cos(y_rot)]])

        Rz = np.array([[np.cos(z_rot), -np.sin(z_rot), 0],
                       [np.sin(z_rot), np.cos(z_rot), 0],
                       [0, 0, 1]])

        # 组合旋转矩阵（注意顺序：先X，后Y，最后Z）
        R = Rz @ Ry @ Rx

        # 计算中心点
        center = np.mean(tsne_coords, axis=0)

        # 应用变换
        rotated_coords = (tsne_coords - center) @ R.T * scale + center

        # 更新散点图
        scatter2.set_offsets(rotated_coords[:, :2])  # 更新xy坐标
        scatter2._offsets3d = (rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2])  # 更新3D坐标

        fig.canvas.draw_idle()

    def reset(event):
        x_slider.set_val(0)
        y_slider.set_val(0)
        z_slider.set_val(0)
        scale_slider.set_val(1.0)
        update(None)

    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)
    scale_slider.on_changed(update)
    reset_button.on_clicked(reset)

    # 添加说明文本
    ax2.text2D(0.02, 0.98, 'Controls:\nX/Y/Z-Rotation: Rotate around axes\nScale: Resize\nReset: Default view',
               transform=ax2.transAxes,
               fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(title, fontsize=14, y=0.95)
    plt.subplots_adjust(bottom=0.25)  # 为滑块留出更多空间
    plt.show()


def procrustes_with_scale(A, B):
    """
    Procrustes 对齐（带缩放）
    Args:
        A: 源点集 (n, m)
        B: 目标点集 (n, m)
    Returns:
        A_aligned: 对齐后的A
        scale: 缩放因子
        R: 旋转矩阵
    """
    # 1. 中心化（如果尚未中心化）
    A_centered = A - np.mean(A, axis=0)
    B_centered = B - np.mean(B, axis=0)

    # 2. 计算缩放因子（最小二乘优化）
    scale = np.trace(B_centered.T @ A_centered) / np.trace(A_centered.T @ A_centered)

    # 3. 计算旋转矩阵（SVD，不包含缩放）
    R, _ = orthogonal_procrustes(A_centered, B_centered)

    # 4. 应用变换：缩放 -> 旋转 -> 平移
    A_aligned = scale * A_centered @ R + np.mean(B, axis=0)

    return A_aligned, scale, R


class CorrespondMetrics:
    def __init__(self, map_to_original, point_to_index, size):
        """
        初始化评估器
        参数:
            - map_to_original: 从点映射到原始坐标的字典
            - point_to_index: 从点映射到索引的字典
        """
        # 从map_to_original和point_to_index中提取原始坐标
        # 首先创建一个临时字典，将索引映射到坐标
        self.size = size
        index_to_coords = {}
        for point, coords in map_to_original.items():
            index = point_to_index[point]
            index_to_coords[index] = coords

        self.original_coords = np.array([index_to_coords[i] for i in range(len(index_to_coords))])

    def scale_coords_dim(self, coords, size):
        """
        将坐标线性缩放到 [0, size[dim]] 区间，支持任意维度。

        Args:
            coords (np.ndarray): 输入坐标，形状为 (n_samples, n_dims)
            size (Tuple[float]): 每维目标区间的最大值（最小值固定为 0）

        Returns:
            np.ndarray: 缩放后的坐标，形状同 coords
        """
        coords = np.asarray(coords)
        if coords.ndim != 2:
            raise ValueError("coords 必须是二维数组")

        n_samples, n_dims = coords.shape

        if len(size) != n_dims:
            raise ValueError(f"size 的维度数量({len(size)})应与 coords 的维度({n_dims})一致")

        if not all(isinstance(s, (int, float)) and s > 0 for s in size):
            raise ValueError("size 中所有元素必须是正数")

        scaled = np.empty_like(coords, dtype=np.float64)

        for dim in range(n_dims):
            c_min, c_max = coords[:, dim].min(), coords[:, dim].max()
            target_max = size[dim]

            if np.isclose(c_max, c_min):
                # 如果该维是常数，统一映射到中间值
                scaled[:, dim] = target_max / 2
            else:
                scaled[:, dim] = (coords[:, dim] - c_min) / (c_max - c_min) * target_max

        return scaled

    def robust_scale_coords(self, coords, size):
        """
        使用鲁棒缩放将坐标归一化到目标范围（支持多维）

        Args:
            coords: 输入坐标数组，形状必须为 (n_samples, n_dims)
            size: 目标尺寸元组 (N0, N1,...,Nx)，长度必须匹配 n_dims

        Returns:
            缩放后的坐标数组，形状与输入相同
        """
        # 参数检查
        assert len(coords.shape) == 2, "coords must be 2D array (n_samples, n_dims)"
        assert coords.shape[1] == len(size), "size dimensions must match coords dimensions"

        # 初始化RobustScaler
        scaler = RobustScaler(
            quantile_range=(25, 75),
            with_centering=True,
            with_scaling=True
        )

        # 鲁棒缩放
        scaled = scaler.fit_transform(coords)

        # 对每个维度分别映射到目标范围
        for dim in range(scaled.shape[1]):
            dim_min = scaled[:, dim].min()
            dim_max = scaled[:, dim].max()
            size_min, size_max = 0, size[dim] - 1  # 默认映射到 [0, size[dim]]

            # 线性映射到目标范围
            scaled[:, dim] = (scaled[:, dim] - dim_min) / (dim_max - dim_min) * \
                             (size_max - size_min) + size_min

        return scaled

    def visualize_alignment(self, reconstructed_coords, reconstructed_aligned):
        """
        可视化原始坐标、重建坐标和对齐后的坐标
        参数:
            - reconstructed_coords: (n, 2) 重建坐标（对齐前）
            - reconstructed_aligned: (n, 2) 对齐后的坐标
        """
        # 创建图形
        plt.figure(figsize=(15, 5))

        # 绘制原始坐标
        plt.subplot(131)
        plt.scatter(self.original_coords[:, 0], self.original_coords[:, 1], c='blue', label='Original')
        plt.title('Original Coordinates')
        plt.legend()

        # 绘制重建坐标（对齐前）
        plt.subplot(132)
        plt.scatter(reconstructed_coords[:, 0], reconstructed_coords[:, 1], c='red', label='Reconstructed')
        plt.title('Reconstructed Coordinates (Before Alignment)')
        plt.legend()

        # 绘制对齐后的坐标
        plt.subplot(133)
        plt.scatter(self.original_coords[:, 0], self.original_coords[:, 1], c='blue', label='Original', alpha=0.3)
        plt.scatter(reconstructed_aligned[:, 0], reconstructed_aligned[:, 1], c='orange', label='Reconstructed',
                    alpha=0.8)
        plt.title('After Procrustes Alignment')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_alignment_3d(self, reconstructed_coords, reconstructed_aligned):
        """
        可视化原始坐标、重建坐标和对齐后的坐标（3D版本）

        参数:
            - reconstructed_coords: (n, 3) 重建坐标（对齐前）
            - reconstructed_aligned: (n, 3) 对齐后的坐标
        """
        # 创建 3D 图形
        fig = plt.figure(figsize=(18, 6))

        # 原始坐标（3D）
        ax1 = fig.add_subplot(131, projection='3d')
        # ax1 = fig.add_subplot(131)
        ax1.scatter(
            self.original_coords[:, 0],
            self.original_coords[:, 1],
            self.original_coords[:, 2],
            c='blue', label='Original'
        )
        ax1.set_title('Original Coordinates (3D)')
        ax1.legend()

        # 重建坐标（对齐前，3D）
        ax2 = fig.add_subplot(132, projection='3d')
        # ax2 = fig.add_subplot(132)
        ax2.scatter(
            reconstructed_coords[:, 0],
            reconstructed_coords[:, 1],
            reconstructed_coords[:, 2],
            c='red', label='Reconstructed'
        )
        ax2.set_title('Reconstructed (Before Alignment)')
        ax2.legend()

        # 对齐后的坐标（3D，叠加显示）
        ax3 = fig.add_subplot(133, projection='3d')
        # ax3 = fig.add_subplot(133)
        ax3.scatter(
            self.original_coords[:, 0],
            self.original_coords[:, 1],
            self.original_coords[:, 2],
            c='blue', label='Original', alpha=0.3
        )
        ax3.scatter(
            reconstructed_aligned[:, 0],
            reconstructed_aligned[:, 1],
            reconstructed_aligned[:, 2],
            c='orange', label='Reconstructed', alpha=0.8
        )
        ax3.set_title('After Procrustes Alignment (3D)')
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def get_aligned_coords(self, reconstructed_coords):
        original_center = np.mean(self.original_coords, axis=0)
        reconstructed_center = np.mean(reconstructed_coords, axis=0)

        # 将两个点集都平移到原点
        original_centered = self.original_coords - original_center
        reconstructed_centered = reconstructed_coords - reconstructed_center

        # # 使用Procrustes分析只进行旋转对齐
        _, reconstructed_aligned, _ = procrustes(original_centered, reconstructed_centered)

        reconstructed_aligned = self.robust_scale_coords(reconstructed_aligned, self.size)
        return reconstructed_aligned

    def snap_to_integer(self, coords):
        """将坐标对齐到最近的整数"""
        # 计算整体偏移量（使坐标分布更均匀）
        offsets = coords - np.round(coords)
        median_offset = np.median(offsets, axis=0)

        return np.round(coords - median_offset)

    def evaluate(self, reconstructed_coords):
        """
        评估重建效果
        参数:
            - reconstructed_coords: (n, 2) 重建坐标
        返回:
            - metrics: 包含各项指标的字典
        """
        # 计算原始坐标和重建坐标的中心
        original_center = np.mean(self.original_coords, axis=0)
        reconstructed_center = np.mean(reconstructed_coords, axis=0)

        # 将两个点集都平移到原点
        original_centered = self.original_coords - original_center
        reconstructed_centered = reconstructed_coords - reconstructed_center

        # # 使用Procrustes分析只进行旋转对齐
        # print(original_centered.shape, reconstructed_centered.shape)
        _, reconstructed_aligned, _ = procrustes(original_centered, reconstructed_centered)

        reconstructed_aligned = self.robust_scale_coords(reconstructed_aligned, self.size)
        # reconstructed_aligned = self.snap_to_integer(reconstructed_aligned)

        # 可视化对齐结果
        # self.visualize_alignment_3d(reconstructed_coords, reconstructed_aligned)

        # 计算各项指标
        metrics = {
            'mse': self._mse_nearest_neighbor(reconstructed_aligned),
            # 'exact_match': self.tolerant_match_rate(reconstructed_aligned),
            # 'neighbor_accuracy': self._calc_neighbor_accuracy(reconstructed_aligned),
            'chamfer_distance': self.chamfer_distance(reconstructed_aligned),
        }

        return metrics

    def chamfer_distance(self, reconstructed):
        tree_orig = KDTree(self.original_coords)
        tree_recon = KDTree(reconstructed)
        dist_orig, _ = tree_orig.query(reconstructed)
        dist_recon, _ = tree_recon.query(self.original_coords)
        return (dist_orig.mean() + dist_recon.mean()) / 2

    def tolerant_match_rate(self, reconstructed, radius=2):
        """
        计算松弛匹配率：重建点是否在原始点的邻域内
        Args:
            reconstructed: (n, 2) 重建坐标
            radius: 匹配阈值
        Returns:
            match_rate: 匹配比例
        """
        if len(self.original_coords) == 0 or len(reconstructed) == 0:
            return 0.0

        # 直接计算对应点之间的距离
        distances = np.square(np.linalg.norm(self.original_coords - reconstructed, axis=1))
        matched = np.sum(distances <= radius)

        return matched / len(reconstructed)

    def _mse_with_correspondence(self, reconstructed, reconstructed_to_original):
        """使用点对应关系计算MSE"""
        # 使用点对应关系计算MSE
        mse = 0
        for i in range(len(self.original_coords)):
            reconstructed_idx = reconstructed_to_original[i]
            mse += np.sum((self.original_coords[i] - reconstructed[reconstructed_idx]) ** 2)
        return mse / len(self.original_coords)

    def _exact_match_with_correspondence(self, reconstructed, reconstructed_to_original):
        """使用点对应关系计算完全匹配率"""
        # 使用点对应关系计算完全匹配率
        exact_matches = 0
        for i in range(len(self.original_coords)):
            reconstructed_idx = reconstructed_to_original[i]
            if np.all(np.round(reconstructed[reconstructed_idx]) == self.original_coords[i]):
                exact_matches += 1
        return exact_matches / len(self.original_coords)

    def _visualize_neighbors(self, reconstructed, reconstructed_to_original, indices_original, indices_reconstructed,
                             k=5):
        """可视化原始点和重建点的最近邻关系"""
        plt.figure(figsize=(12, 6))

        # 绘制原始点
        plt.subplot(121)
        plt.scatter(self.original_coords[:, 0], self.original_coords[:, 1], c='blue', label='Original')
        # 绘制原始点的最近邻关系
        for i in range(len(self.original_coords)):
            neighbors = indices_original[i]
            for j in neighbors:
                plt.plot([self.original_coords[i, 0], self.original_coords[j, 0]],
                         [self.original_coords[i, 1], self.original_coords[j, 1]],
                         'b-', alpha=0.2)
        plt.title('Original Points and Neighbors')
        plt.legend()

        # 绘制重建点
        plt.subplot(122)
        plt.scatter(reconstructed[:, 0], reconstructed[:, 1], c='red', label='Reconstructed')
        # 绘制重建点的最近邻关系
        for i in range(len(reconstructed)):
            neighbors = indices_reconstructed[i]
            for j in neighbors:
                plt.plot([reconstructed[i, 0], reconstructed[j, 0]],
                         [reconstructed[i, 1], reconstructed[j, 1]],
                         'r-', alpha=0.2)
        plt.title('Reconstructed Points and Neighbors')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def _calc_neighbor_accuracy_nearest(self, reconstructed, reconstructed_to_original, k=5):
        """使用最近邻匹配计算最近邻准确率"""
        # 确保输入数据有效
        if len(reconstructed) == 0 or len(self.original_coords) == 0:
            return 0.0

        # 计算最近邻
        nn_original = NearestNeighbors(n_neighbors=min(k, len(self.original_coords))).fit(self.original_coords)
        indices_original = nn_original.kneighbors(return_distance=False)

        nn_reconstructed = NearestNeighbors(n_neighbors=min(k, len(reconstructed))).fit(reconstructed)
        indices_reconstructed = nn_reconstructed.kneighbors(return_distance=False)

        # 可视化最近邻关系
        # self._visualize_neighbors(reconstructed, reconstructed_to_original, indices_original, indices_reconstructed, k)

        # 确保k值不超过点集大小
        k = min(k, len(self.original_coords), len(reconstructed))

        overlap = 0
        for i in range(len(self.original_coords)):
            # 获取原始点i的最近邻
            original_neighbors = indices_original[i]
            # 获取对应的重建点
            reconstructed_idx = reconstructed_to_original[i]
            # 获取重建点的最近邻
            reconstructed_neighbors = indices_reconstructed[reconstructed_idx]

            # 将重建点的最近邻映射回原始点的索引
            mapped_reconstructed_neighbors = [reconstructed_to_original[j] for j in reconstructed_neighbors]

            # 计算重叠
            overlap += len(np.intersect1d(original_neighbors, mapped_reconstructed_neighbors))

        return overlap / (k * len(self.original_coords))

    def _mse_hungarian(self, reconstructed):
        """
        使用匈牙利算法计算两个点集之间的 MSE。

        参数:
        - A: (n, d) 形状的 NumPy 数组，表示 n 个 d 维点
        - B: (n, d) 形状的 NumPy 数组，表示 n 个 d 维点

        返回:
        - mse_value: 计算得到的 MSE 值
        """
        cost_matrix = cdist(self.original_coords, reconstructed) ** 2  # 计算欧几里得距离的平方
        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # 匈牙利算法求最优匹配
        return np.mean(cost_matrix[row_ind, col_ind])  # 计算 MSE

    def _mse_nearest_neighbor(self, reconstructed):
        """
        """
        kd_tree = cKDTree(reconstructed)  # 在 B 上构建 KD-Tree
        distances, _ = kd_tree.query(self.original_coords)  # 查询 A 中每个点到 B 的最近邻距离
        return np.mean(distances ** 2)  # 计算 MSE

    def _calc_mse(self, reconstructed):
        """均方误差"""
        return np.mean(np.sum((self.original_coords - reconstructed) ** 2, axis=1))

    def _calc_exact_match(self, reconstructed):
        """完全匹配率"""
        rounded = np.round(reconstructed)
        return np.mean(np.all(rounded == self.original_coords, axis=1))

    def _calc_exact_match_nearest(self, reconstructed):
        """使用最近邻匹配计算完全匹配率"""
        kd_tree = cKDTree(self.original_coords)  # 在 original_coords 上构建 KD-Tree
        distances, _ = kd_tree.query(reconstructed)  # 查询 reconstructed 每个点的最近邻
        return np.mean(distances == 0)  # 统计完全匹配的比例

    def _calc_neighbor_accuracy(self, reconstructed, k=5):
        """最近邻准确率"""
        nn_original = NearestNeighbors(n_neighbors=k).fit(self.original_coords)
        indices_original = nn_original.kneighbors(return_distance=False)

        nn_reconstructed = NearestNeighbors(n_neighbors=k).fit(reconstructed)
        indices_reconstructed = nn_reconstructed.kneighbors(return_distance=False)
        overlap = 0
        for i in range(len(self.original_coords)):
            if i < len(indices_reconstructed):  # 确保索引在范围内
                overlap += len(np.intersect1d(indices_original[i], indices_reconstructed[i]))
        return overlap / (k * len(self.original_coords))
