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
    # t-SNE dimensionality reduction (initialized with Laplacian Embedding)
    model = TSNE(n_components=dim, metric='precomputed', square_distances=True, init=W, **kwargs)
    distance_matrix[distance_matrix == np.inf] = 1e12  # 避免 inf
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))


def tsne_with_pca(distance_matrix: np.ndarray, dim: int = 2, pca_dim: int = 50, **kwargs) -> np.ndarray:
    """PCA + t-SNE dimensionality reduction method
    1. First use PCA to reduce to pca_dim dimensions, reducing high-dimensional noise
    2. Then use t-SNE to reduce to final dim dimensions
    """
    # Handle infinite values (avoid calculation errors)
    distance_matrix[distance_matrix == np.inf] = 1e12
    distance_matrix = np.nan_to_num(distance_matrix, nan=np.inf)

    # First use PCA for dimensionality reduction (only reduce if pca_dim < original dimension)
    if distance_matrix.shape[1] > pca_dim:
        print(f"Applying PCA reduction to {pca_dim} dimensions before t-SNE...")
        pca = PCA(n_components=pca_dim)
        reduced_matrix = pca.fit_transform(distance_matrix)
    else:
        reduced_matrix = distance_matrix

    # Further reduction with t-SNE
    print(f"Applying t-SNE reduction to {dim} dimensions...")
    model = TSNE(
        n_components=dim,
        metric='euclidean',  # After PCA processing, becomes coordinate points, no longer distance matrix
        **kwargs  # Allow external parameters like perplexity
    )

    return model.fit_transform(reduced_matrix)


def tsne(distance_matrix: np.ndarray, dim: int = 2, **kwargs) -> np.ndarray:

    """t-SNE dimensionality reduction (with adjustable perplexity and other parameters)"""
    model = TSNE(
        metric='precomputed',
        square_distances=True,
        early_exaggeration=12.0,
        **kwargs  # Receive parameters like perplexity
    )
    distance_matrix[distance_matrix == np.inf] = 1000000000000000
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))


def mds(distance_matrix: np.ndarray, dim: int = 2, **kwargs) -> np.ndarray:
    """MDS dimensionality reduction"""
    model = MDS(
        n_components=dim,
        dissimilarity='precomputed',
        **kwargs  # Receive parameters like epsilon
    )
    distance_matrix[distance_matrix == np.inf] = 1000000000000000
    return model.fit_transform(distance_matrix)


def isomap(distance_matrix: np.ndarray, dim: int = 2, **kwargs) -> np.ndarray:
    """Isomap dimensionality reduction"""
    model = Isomap(
        n_components=dim,
        metric='precomputed',
        **kwargs  # Receive parameters like n_neighbors
    )
    distance_matrix[distance_matrix == np.inf] = 1000000000000000
    return model.fit_transform(distance_matrix)


def joint_optimization(freq_table, grid_size, init_dim=2):
    N0, N1 = grid_size
    """Joint optimization: infer virtual coordinates + dimensionality reduction"""
    points = list(set([p for pair in freq_table.keys() for p in pair]))
    n_points = len(points)

    # Initialize virtual coordinates (high-dimensional space)
    np.random.seed(42)
    init_coords = np.random.rand(n_points, init_dim) * np.array(grid_size)

    def loss(coords_flat):
        coords = coords_flat.reshape(n_points, init_dim)
        loss_val = 0

        # 1. Co-occurrence frequency matching loss
        for (a_idx, b_idx), co_occur in freq_table.items():
            # Calculate hypothetical overlap area
            x_a, y_a = coords[points.index(a_idx)]
            x_b, y_b = coords[points.index(b_idx)]
            overlap = max(0, min(x_a, x_b)) * max(0, min(y_a, y_b))

            # Difference between predicted co-occurrence probability and observed value
            pred = overlap / (x_a * y_a * x_b * y_b + 1e-6)
            loss_val += (pred - co_occur) ** 2

        # 2. Low-dimensional space regularization
        if init_dim == 2:
            # Encourage coordinates to be within grid range
            loss_val += 0.1 * np.sum(np.clip(coords, 0, grid_size) - coords) ** 2

        return loss_val

    # Optimization solution
    result = minimize(loss, init_coords.flatten(), method='L-BFGS-B')
    optimized_coords = result.x.reshape(n_points, init_dim)

    return optimized_coords


class DensityPreservingTSNE(TSNE):
    def __init__(self, densities=None, alpha=0.5, lambda_=0.2, **kwargs):
        self.densities = densities
        self.alpha = alpha  # Density weight coefficient
        self.lambda_ = lambda_  # Density regularization strength
        super().__init__(**kwargs)

    def _fit(self, X):
        # Calculate original density (if not provided)
        print("Custom _fit() is being called!")
        if self.densities is None:
            kde = KernelDensity(bandwidth=0.3).fit(X)
            self.densities = np.exp(kde.score_samples(X))
            self.densities = (self.densities - self.densities.min()) / \
                             (self.densities.max() - self.densities.min() + 1e-8)

        # Run standard t-SNE process
        embedding = super()._fit(X)

        # Add density regularization term
        if self.lambda_ > 0:
            self._apply_density_constraint(embedding)

        return embedding

    def _apply_density_constraint(self, embedding):
        # Calculate embedding space density
        kde = KernelDensity(bandwidth=0.3, kernel='cosine').fit(embedding)
        reco_densities = np.exp(kde.score_samples(embedding))

        # Density matching gradient
        grad = 2 * self.lambda_ * (reco_densities - self.densities)[:, None] * \
               (embedding - np.mean(embedding, axis=0))

        # Update embedding results
        embedding -= 0.01 * grad  # Adjustable learning rate


def density_preserving_tsne(distance_matrix: np.ndarray, pos, densities, dim: int = 2, alpha: float = 0.1,
                             **kwargs) -> np.ndarray:

    model = DensityPreservingTSNE(
        densities=densities,
        alpha=0.5,
        lambda_=0.2,
        init=pos,  # Maintain your topological initialization
        metric='precomputed',
        **kwargs
    )
    distance_matrix[distance_matrix == np.inf] = 1e12  # Avoid inf
    distance_matrix[distance_matrix == np.nan] = 1e12  # Avoid nan
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))


def density_aware_tsne(distance_matrix: np.ndarray, pos, densities, dim: int = 2, alpha: float = 0.1,
                             **kwargs) -> np.ndarray:

    model = DensityAwareTSNE(
        densities=densities,
        alpha=0.5,
        lambda_=0.2,
        init=pos,  # Maintain your topological initialization
        metric='precomputed',
        **kwargs
    )
    distance_matrix[distance_matrix == np.inf] = 1e12  # Avoid inf
    distance_matrix[distance_matrix == np.nan] = 1e12  # Avoid nan
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))


def aggregate_tsne(distance_matrix: np.ndarray, true_mean=None, true_var=None, dim: int = 2, alpha: float = 0.1,
                             **kwargs) -> np.ndarray:

    model = AggregateTSNE(
        true_mean=true_mean,
        true_var=true_var,
        alpha=0.5,
        lambda_=0.2,
        metric='precomputed',
        **kwargs
    )
    distance_matrix[distance_matrix == np.inf] = 1e12  # Avoid inf
    distance_matrix[distance_matrix == np.nan] = 1e12  # Avoid nan
    return model.fit_transform(np.nan_to_num(distance_matrix, nan=np.inf))
