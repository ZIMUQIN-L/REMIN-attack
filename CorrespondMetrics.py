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


def procrustes_with_scale(A, B):
    """
    Procrustes alignment(with scaling)
    Args:
        A: original data points (n, m)
        B: target data points (n, m)
    Returns:
        A_aligned: A after alignment
        scale: scaling factor
        R: rotate matrix
    """
    # 1. centralized (if not centralized)
    A_centered = A - np.mean(A, axis=0)
    B_centered = B - np.mean(B, axis=0)

    # 2. compute the scaling factor (Least squares optimization)
    scale = np.trace(B_centered.T @ A_centered) / np.trace(A_centered.T @ A_centered)

    # 3. compute rotate matrix（SVD, without scaling）
    R, _ = orthogonal_procrustes(A_centered, B_centered)

    # 4. apply the transformation: scaling -> rotate -> translation
    A_aligned = scale * A_centered @ R + np.mean(B, axis=0)

    return A_aligned, scale, R


class CorrespondMetrics:
    def __init__(self, map_to_original, point_to_index, N0, N1):
        """
        Initialize evaluator
        Parameters:
            - map_to_original: Dictionary mapping points to original coordinates
            - point_to_index: Dictionary mapping points to indices
        """
        # Extract original coordinates from map_to_original and point_to_index
        # First create a temporary dictionary mapping indices to coordinates
        self.N0 = N0
        self.N1 = N1
        index_to_coords = {}
        for point, coords in map_to_original.items():
            index = point_to_index[point]
            index_to_coords[index] = coords
        
        self.original_coords = np.array([index_to_coords[i] for i in range(len(index_to_coords))])

    def scale_coords(self, coords, x_range, y_range):
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])

        # linear scaling
        scaled_x = (coords[:, 0] - x_min) / (x_max - x_min) * (x_range[1] - x_range[0]) + x_range[0]
        scaled_y = (coords[:, 1] - y_min) / (y_max - y_min) * (y_range[1] - y_range[0]) + y_range[0]
        return np.column_stack([scaled_x, scaled_y])

    def robust_scale_coords(self, coords, x_range, y_range):
        scaler = RobustScaler(
            quantile_range=(25, 75),
            with_centering=True,
            with_scaling=True
        )
        scaled = scaler.fit_transform(coords)

        # Map to target range
        scaled[:, 0] = (scaled[:, 0] - scaled[:, 0].min()) / (scaled[:, 0].max() - scaled[:, 0].min()) * (
                x_range[1] - x_range[0]) + x_range[0]
        scaled[:, 1] = (scaled[:, 1] - scaled[:, 1].min()) / (scaled[:, 1].max() - scaled[:, 1].min()) * (
                y_range[1] - y_range[0]) + y_range[0]

        return scaled

    def visualize_alignment(self, reconstructed_coords, reconstructed_aligned):
        plt.rcParams.update({'font.size': 3})  # set font size
        point_size = 2  # size point size

        plt.figure(figsize=(12, 5), dpi=300)

        # original data points
        plt.subplot(131)
        plt.scatter(self.original_coords[:, 0], self.original_coords[:, 1], c='blue', label='Original', s=point_size)
        plt.title('Original Coordinates')
        plt.legend()

        # reconstruction data points (before alignment)
        plt.subplot(132)
        plt.scatter(reconstructed_coords[:, 0], reconstructed_coords[:, 1], c='red', label='Reconstructed',
                    s=point_size)
        plt.title('Reconstructed Coordinates (Before Alignment)')
        plt.legend()

        # reconstruction data points (after alignment)
        plt.subplot(133)
        plt.scatter(self.original_coords[:, 0], self.original_coords[:, 1], c='blue', label='Original', alpha=0.3,
                    s=point_size)
        plt.scatter(reconstructed_aligned[:, 0], reconstructed_aligned[:, 1], c='orange', label='Reconstructed',
                    alpha=0.8, s=point_size)
        plt.title('After Procrustes Alignment')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def snap_to_integer(self, coords):
        """Align coordinates to the nearest integer"""
        # Calculate overall offset (to distribute coordinates more evenly)
        offsets = coords - np.round(coords)
        median_offset = np.median(offsets, axis=0)

        return np.round(coords - median_offset)

    def get_aligned_coords(self, reconstructed_coords):
        original_center = np.mean(self.original_coords, axis=0)
        reconstructed_center = np.mean(reconstructed_coords, axis=0)

        # Translate both sets of points to the origin
        original_centered = self.original_coords - original_center
        reconstructed_centered = reconstructed_coords - reconstructed_center

        # Use Procrustes analysis for rotational alignment only
        _, reconstructed_aligned, _ = procrustes(original_centered, reconstructed_centered)

        reconstructed_aligned = self.robust_scale_coords(reconstructed_aligned, [1, self.N0 - 1], [1, self.N1 - 1])
        return reconstructed_aligned

    def evaluate(self, reconstructed_coords, align=True):
        """
        Evaluate Reconstruction Effect
        Parameters:
        - reconstructed_coords: (n, 2) Reconstructed coordinates
        Returns:
        - metrics: Dictionary containing various metrics
        """
        if align:
        # Calculate the center of the original coordinates and reconstructed coordinates
            original_center = np.mean(self.original_coords, axis=0)
            reconstructed_center = np.mean(reconstructed_coords, axis=0)

            # Translate both sets of points to the origin
            original_centered = self.original_coords - original_center
            reconstructed_centered = reconstructed_coords - reconstructed_center


            # Use Procrustes analysis for rotational alignment only
            _, reconstructed_aligned, _ = procrustes(original_centered, reconstructed_centered)

            reconstructed_aligned = self.robust_scale_coords(reconstructed_aligned, [1, self.N0 - 1], [1, self.N1 - 1])
            # reconstructed_aligned = self.snap_to_integer(reconstructed_aligned)

            # visualization
            self.visualize_alignment(reconstructed_coords, reconstructed_aligned)
        else:
            reconstructed_aligned = reconstructed_coords
            reconstructed_aligned = self.robust_scale_coords(reconstructed_aligned, [0, self.N0], [0, self.N1])
            self.visualize_alignment(self.original_coords, reconstructed_aligned)

        # metrics computation
        metrics = {
            'mse': self._calc_mse(reconstructed_aligned),
            'exact_match': self.tolerant_match_rate(reconstructed_aligned),
            'neighbor_accuracy': self._calc_neighbor_accuracy(reconstructed_aligned),
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
        Calculate relaxation matching rate: Whether reconstructed points fall within the neighborhood of original points
        Args:
            reconstructed: (n, 2) Reconstructed coordinates
            radius: Matching threshold
        Returns:
            match_rate: Matching ratio
        """
        if len(self.original_coords) == 0 or len(reconstructed) == 0:
            return 0.0

        # Directly calculate the distance between corresponding points
        distances = np.square(np.linalg.norm(self.original_coords - reconstructed, axis=1))
        matched = np.sum(distances <= radius)

        return matched / len(reconstructed)

    def _mse_with_correspondence(self, reconstructed, reconstructed_to_original):
        """Calculate MSE Using Point Correspondence"""
        mse = 0
        for i in range(len(self.original_coords)):
            reconstructed_idx = reconstructed_to_original[i]
            mse += np.sum((self.original_coords[i] - reconstructed[reconstructed_idx]) ** 2)
        return mse / len(self.original_coords)

    def _exact_match_with_correspondence(self, reconstructed, reconstructed_to_original):
        # Calculate the exact match rate using the point correspondence relationship
        exact_matches = 0
        for i in range(len(self.original_coords)):
            reconstructed_idx = reconstructed_to_original[i]
            if np.all(np.round(reconstructed[reconstructed_idx]) == self.original_coords[i]):
                exact_matches += 1
        return exact_matches / len(self.original_coords)

    def _visualize_neighbors(self, reconstructed, reconstructed_to_original, indices_original, indices_reconstructed,
                             k=5):
        """Visualizing the Nearest Neighbor Relationships Between Original Points and Reconstructed Points"""
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.scatter(self.original_coords[:, 0], self.original_coords[:, 1], c='blue', label='Original')
        for i in range(len(self.original_coords)):
            neighbors = indices_original[i]
            for j in neighbors:
                plt.plot([self.original_coords[i, 0], self.original_coords[j, 0]],
                         [self.original_coords[i, 1], self.original_coords[j, 1]],
                         'b-', alpha=0.2)
        plt.title('Original Points and Neighbors')
        plt.legend()

        plt.subplot(122)
        plt.scatter(reconstructed[:, 0], reconstructed[:, 1], c='red', label='Reconstructed')
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
        """Calculate the nearest neighbor accuracy using nearest neighbor matching."""
        if len(reconstructed) == 0 or len(self.original_coords) == 0:
            return 0.0

        # compute nearest neighbor
        nn_original = NearestNeighbors(n_neighbors=min(k, len(self.original_coords))).fit(self.original_coords)
        indices_original = nn_original.kneighbors(return_distance=False)

        nn_reconstructed = NearestNeighbors(n_neighbors=min(k, len(reconstructed))).fit(reconstructed)
        indices_reconstructed = nn_reconstructed.kneighbors(return_distance=False)

        # make sure k is smaller than the dataset size
        k = min(k, len(self.original_coords), len(reconstructed))

        overlap = 0
        for i in range(len(self.original_coords)):
            # get the nearest neighbor for i
            original_neighbors = indices_original[i]
            # get the corresponding reconstruction point
            reconstructed_idx = reconstructed_to_original[i]
            reconstructed_neighbors = indices_reconstructed[reconstructed_idx]

            # Map the nearest neighbor of the reconstruction point back to the index of the original point.
            mapped_reconstructed_neighbors = [reconstructed_to_original[j] for j in reconstructed_neighbors]

            # compute overlap
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