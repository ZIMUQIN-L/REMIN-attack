import numpy as np
from sklearn.manifold import TSNE, MDS, Isomap
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import networkx as nx
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors, KernelDensity
from DensityAwareTsne import DensityAwareTSNE
from AggregateTsne import AggregateTSNE


def topology_preserving_tsne(distance_matrix: np.ndarray, W, D, dim: int = 2, alpha: float = 0.1,
                             **kwargs) -> np.ndarray:

    # L = laplacian(W, normed=True)
    #
    # # 计算 Laplacian 特征向量（使用 subset_by_index 替代 eigvals）
    # _, eig_vectors = eigh(L, subset_by_index=[1, dim])  # 取前 dim 个非零特征向量
    #
    # # 先用 Laplacian Embedding 作为初始值
    # init_embedding = eig_vectors

    # t-SNE 降维（初始化为 Laplacian Embedding）
    model = TSNE(n_components=dim, metric='precomputed', square_distances=True, init=W, **kwargs)
    distance_matrix[distance_matrix == np.inf] = 1e12  # 避免 inf
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))


def tsne_with_pca(distance_matrix: np.ndarray, dim: int = 2, pca_dim: int = 50, **kwargs) -> np.ndarray:
    """PCA + t-SNE 降维方法
    1. 先用 PCA 降维到 pca_dim 维度，减少高维噪声
    2. 再用 t-SNE 降维到最终的 dim 维度
    """
    # 1️⃣ 处理无穷值（避免计算错误）
    distance_matrix[distance_matrix == np.inf] = 1e12
    distance_matrix = np.nan_to_num(distance_matrix, nan=np.inf)

    # 2️⃣ 先用 PCA 进行降维（仅当 pca_dim < 原始维度时才降维）
    if distance_matrix.shape[1] > pca_dim:
        print(f"Applying PCA reduction to {pca_dim} dimensions before t-SNE...")
        pca = PCA(n_components=pca_dim)
        reduced_matrix = pca.fit_transform(distance_matrix)
    else:
        reduced_matrix = distance_matrix

    # 3️⃣ t-SNE 进一步降维
    print(f"Applying t-SNE reduction to {dim} dimensions...")
    model = TSNE(
        n_components=dim,
        metric='euclidean',  # PCA 处理后变成坐标点，不再是距离矩阵
        **kwargs  # 允许外部传入 perplexity 等参数
    )

    return model.fit_transform(reduced_matrix)


def tsne(distance_matrix: np.ndarray, dim: int = 2, **kwargs) -> np.ndarray:

    """t-SNE降维（可调perplexity等参数）"""
    model = TSNE(
        metric='precomputed',
        square_distances=True,
        early_exaggeration=12.0,
        **kwargs  # 接收perplexity等参数
    )
    distance_matrix[distance_matrix == np.inf] = 1000000000000000
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))

# from openTSNE import TSNE
#
# def tsne(distance_matrix: np.ndarray, dim: int = 2, **kwargs) -> np.ndarray:
#     distance_matrix = np.nan_to_num(distance_matrix, nan=np.inf)
#     distance_matrix[distance_matrix == np.inf] = 1e12
#     model = TSNE(
#         # n_components=dim,
#         metric="precomputed",
#         n_jobs=-1,
#         negative_gradient_method="barnes_hut",
#         **kwargs
#     )
#     return model.fit(distance_matrix)



def mds(distance_matrix: np.ndarray, dim: int = 2, **kwargs) -> np.ndarray:
    """MDS降维"""
    model = MDS(
        n_components=dim,
        dissimilarity='precomputed',
        **kwargs  # 接收epslon等参数
    )
    distance_matrix[distance_matrix == np.inf] = 1000000000000000
    return model.fit_transform(distance_matrix)


def isomap(distance_matrix: np.ndarray, dim: int = 2, **kwargs) -> np.ndarray:
    """Isomap降维"""
    model = Isomap(
        n_components=dim,
        metric='precomputed',
        **kwargs  # 接收n_neighbors等参数
    )
    distance_matrix[distance_matrix == np.inf] = 1000000000000000
    return model.fit_transform(distance_matrix)


def joint_optimization(freq_table, grid_size, init_dim=2):
    N0, N1 = grid_size
    """联合优化：推断虚拟坐标 + 降维"""
    points = list(set([p for pair in freq_table.keys() for p in pair]))
    n_points = len(points)

    # 初始化虚拟坐标（高维空间）
    np.random.seed(42)
    init_coords = np.random.rand(n_points, init_dim) * np.array(grid_size)

    def loss(coords_flat):
        coords = coords_flat.reshape(n_points, init_dim)
        loss_val = 0

        # 1. 共现频率匹配损失
        for (a_idx, b_idx), co_occur in freq_table.items():
            # 计算假设的重叠面积
            x_a, y_a = coords[points.index(a_idx)]
            x_b, y_b = coords[points.index(b_idx)]
            overlap = max(0, min(x_a, x_b)) * max(0, min(y_a, y_b))

            # 预测的共现概率与观测值的差异
            pred = overlap / (x_a * y_a * x_b * y_b + 1e-6)
            loss_val += (pred - co_occur) ** 2

        # 2. 低维空间正则化
        if init_dim == 2:
            # 鼓励坐标在网格范围内
            loss_val += 0.1 * np.sum(np.clip(coords, 0, grid_size) - coords) ** 2

        return loss_val

    # 优化求解
    result = minimize(loss, init_coords.flatten(), method='L-BFGS-B')
    optimized_coords = result.x.reshape(n_points, init_dim)

    return optimized_coords


class DensityPreservingTSNE(TSNE):
    def __init__(self, densities=None, alpha=0.5, lambda_=0.2, **kwargs):
        self.densities = densities
        self.alpha = alpha  # 密度权重系数
        self.lambda_ = lambda_  # 密度正则强度
        super().__init__(**kwargs)

    def _fit(self, X):
        # 计算原始密度（如果未提供）
        print("Custom _fit() is being called!")
        if self.densities is None:
            kde = KernelDensity(bandwidth=0.3).fit(X)
            self.densities = np.exp(kde.score_samples(X))
            self.densities = (self.densities - self.densities.min()) / \
                             (self.densities.max() - self.densities.min() + 1e-8)

        # 运行标准t-SNE流程
        embedding = super()._fit(X)

        # 添加密度正则项
        if self.lambda_ > 0:
            self._apply_density_constraint(embedding)

        return embedding

    def _apply_density_constraint(self, embedding):
        # 计算嵌入空间密度
        kde = KernelDensity(bandwidth=0.3, kernel='cosine').fit(embedding)
        reco_densities = np.exp(kde.score_samples(embedding))

        # 密度匹配梯度
        grad = 2 * self.lambda_ * (reco_densities - self.densities)[:, None] * \
               (embedding - np.mean(embedding, axis=0))

        # 更新嵌入结果
        embedding -= 0.01 * grad  # 学习率可调


def density_preserving_tsne(distance_matrix: np.ndarray, pos, densities, dim: int = 2, alpha: float = 0.1,
                             **kwargs) -> np.ndarray:

    model = DensityPreservingTSNE(
        densities=densities,
        alpha=0.5,
        lambda_=0.2,
        init=pos,  # 保持您的拓扑初始化
        metric='precomputed',
        **kwargs
    )
    distance_matrix[distance_matrix == np.inf] = 1e12  # 避免 inf
    distance_matrix[distance_matrix == np.nan] = 1e12  # 避免 nan
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))


def density_aware_tsne(distance_matrix: np.ndarray, pos, densities, dim: int = 2, alpha: float = 0.1,
                             **kwargs) -> np.ndarray:

    model = DensityAwareTSNE(
        densities=densities,
        alpha=0.5,
        lambda_=0.2,
        init=pos,  # 保持您的拓扑初始化
        metric='precomputed',
        **kwargs
    )
    distance_matrix[distance_matrix == np.inf] = 1e12  # 避免 inf
    distance_matrix[distance_matrix == np.nan] = 1e12  # 避免 nan
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))


def aggregate_tsne(distance_matrix: np.ndarray, true_mean=None, true_var=None, dim: int = 2, alpha: float = 0.1,
                             **kwargs) -> np.ndarray:

    model = AggregateTSNE(
        true_mean=true_mean,
        true_var=true_var,
        alpha=0.5,
        lambda_=0.2,
        # init=pos,  # 保持您的拓扑初始化
        metric='precomputed',
        **kwargs
    )
    distance_matrix[distance_matrix == np.inf] = 1e12  # 避免 inf
    distance_matrix[distance_matrix == np.nan] = 1e12  # 避免 nan
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))
