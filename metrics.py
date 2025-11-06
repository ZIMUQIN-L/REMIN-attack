import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


class ReconstructionMetrics:
    def __init__(self, original_coords):
        """
        初始化评估器
        参数:
            - original_coords: (n, 2) 原始坐标
        """
        self.original_coords = original_coords

    def evaluate(self, reconstructed_coords):
        """
        评估重建效果
        参数:
            - reconstructed_coords: (n, 2) 重建坐标
        返回:
            - metrics: 包含各项指标的字典
        """
        # 找到重建点和原始点之间的对应关系
        kd_tree = cKDTree(reconstructed_coords)
        _, reconstructed_to_original = kd_tree.query(self.original_coords)
        
        # 计算各项指标
        metrics = {
            'mse_nn': self._mse_with_correspondence(reconstructed_coords, reconstructed_to_original),
            'exact_match_nn': self._exact_match_with_correspondence(reconstructed_coords, reconstructed_to_original),
            'neighbor_accuracy': self._calc_neighbor_accuracy(reconstructed_coords),  # 使用原来的方法，只关注分布
            # 'nn_na': self._calc_neighbor_accuracy_nearest(reconstructed_coords, reconstructed_to_original),
        }

        return metrics

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

    def _visualize_neighbors(self, reconstructed, reconstructed_to_original, indices_original, indices_reconstructed, k=5):
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

        # 可视化最近邻关系
        # plt.figure(figsize=(12, 6))
        
        # # 绘制原始点
        # plt.subplot(121)
        # plt.scatter(self.original_coords[:, 0], self.original_coords[:, 1], c='blue', label='Original')
        # # 绘制原始点的最近邻关系
        # for i in range(len(self.original_coords)):
        #     neighbors = indices_original[i]
        #     for j in neighbors:
        #         plt.plot([self.original_coords[i, 0], self.original_coords[j, 0]], 
        #                 [self.original_coords[i, 1], self.original_coords[j, 1]], 
        #                 'b-', alpha=0.2)
        # plt.title('Original Points and Neighbors')
        # plt.legend()
        
        # # 绘制重建点
        # plt.subplot(122)
        # plt.scatter(reconstructed[:, 0], reconstructed[:, 1], c='red', label='Reconstructed')
        # # 绘制重建点的最近邻关系
        # for i in range(len(reconstructed)):
        #     neighbors = indices_reconstructed[i]
        #     for j in neighbors:
        #         plt.plot([reconstructed[i, 0], reconstructed[j, 0]], 
        #                 [reconstructed[i, 1], reconstructed[j, 1]], 
        #                 'r-', alpha=0.2)
        # plt.title('Reconstructed Points and Neighbors')
        # plt.legend()
        
        # plt.tight_layout()
        # plt.show()

        overlap = 0
        for i in range(len(self.original_coords)):
            if i < len(indices_reconstructed):  # 确保索引在范围内
                overlap += len(np.intersect1d(indices_original[i], indices_reconstructed[i]))
        return overlap / (k * len(self.original_coords))

    def _calc_distance_correlation(self, reconstructed):
        """距离秩相关系数"""
        original_dist = pdist(self.original_coords)
        reconstructed_dist = pdist(reconstructed)
        return spearmanr(original_dist, reconstructed_dist).correlation

    def _calc_grid_alignment(self, coords):
        """网格对齐误差"""
        snapped = np.round(coords)
        return np.mean(np.linalg.norm(coords - snapped, axis=1))