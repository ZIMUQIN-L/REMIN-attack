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
    Visualize 3D data, including original coordinates and t-SNE results, with interactive rotation for reconstruction results
    :param original_coords: original 3D coordinates (list or numpy array)
    :param tsne_coords: 3D coordinates after t-SNE dimensionality reduction (list or numpy array)
    :param title: chart title
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8

    original_coords = np.array(original_coords)
    tsne_coords = np.array(tsne_coords)

    fig = plt.figure(figsize=(15, 7))

    # Create two subplots, one for original coordinates, one for t-SNE results
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Calculate point order (left to right)
    order = np.argsort(original_coords[:, 0])
    colors = order / len(order)  # Normalize to [0,1] range

    cmap = plt.cm.viridis

    scatter1 = ax1.scatter(original_coords[:, 0],
                           original_coords[:, 1],
                           original_coords[:, 2],
                           c=colors,
                           cmap=cmap,
                           s=20)
    ax1.set_title('Original Coordinates', fontsize=12)
    ax1.view_init(elev=20, azim=45)  # Fixed viewpoint

    # Plot t-SNE results (interactive rotation)
    scatter2 = ax2.scatter(tsne_coords[:, 0],
                           tsne_coords[:, 1],
                           tsne_coords[:, 2],
                           c=colors,
                           cmap=cmap,
                           s=20)
    ax2.set_title('Reconstruction Result', fontsize=12)

    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Point Index')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Point Index')
    cbar1.ax.set_ylabel('Point Index', fontsize=10)
    cbar2.ax.set_ylabel('Point Index', fontsize=10)

    for ax in [ax1, ax2]:
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.grid(True)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.grid(True, linestyle='--', alpha=0.3)

    ax_x = plt.axes([0.2, 0.02, 0.6, 0.02])  # X-axis rotation
    ax_y = plt.axes([0.2, 0.05, 0.6, 0.02])  # Y-axis rotation
    ax_z = plt.axes([0.2, 0.08, 0.6, 0.02])  # Z-axis rotation

    x_slider = plt.Slider(ax_x, 'X-Rotation', 0, 360, valinit=0)
    y_slider = plt.Slider(ax_y, 'Y-Rotation', 0, 360, valinit=0)
    z_slider = plt.Slider(ax_z, 'Z-Rotation', 0, 360, valinit=0)

    # Add scale slider
    ax_scale = plt.axes([0.2, 0.11, 0.6, 0.02])
    scale_slider = plt.Slider(ax_scale, 'Scale', 0.1, 2.0, valinit=1.0)

    # Add reset button
    ax_reset = plt.axes([0.45, 0.15, 0.1, 0.04])
    reset_button = plt.Button(ax_reset, 'Reset View')

    def update(val):
        # Get current angles and scale value
        x_rot = np.radians(x_slider.val)
        y_rot = np.radians(y_slider.val)
        z_rot = np.radians(z_slider.val)
        scale = scale_slider.val

        # Calculate rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(x_rot), -np.sin(x_rot)],
                       [0, np.sin(x_rot), np.cos(x_rot)]])

        Ry = np.array([[np.cos(y_rot), 0, np.sin(y_rot)],
                       [0, 1, 0],
                       [-np.sin(y_rot), 0, np.cos(y_rot)]])

        Rz = np.array([[np.cos(z_rot), -np.sin(z_rot), 0],
                       [np.sin(z_rot), np.cos(z_rot), 0],
                       [0, 0, 1]])

        # Combine rotation matrices (order: X first, then Y, finally Z)
        R = Rz @ Ry @ Rx

        # Calculate center point
        center = np.mean(tsne_coords, axis=0)

        # Apply transformation
        rotated_coords = (tsne_coords - center) @ R.T * scale + center

        scatter2.set_offsets(rotated_coords[:, :2])
        scatter2._offsets3d = (rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2])

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

    ax2.text2D(0.02, 0.98, 'Controls:\nX/Y/Z-Rotation: Rotate around axes\nScale: Resize\nReset: Default view',
               transform=ax2.transAxes,
               fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(title, fontsize=14, y=0.95)
    plt.subplots_adjust(bottom=0.25)
    plt.show()


def procrustes_with_scale(A, B):
    """
    Procrustes alignment (with scaling)
    Args:
        A: source point set (n, m)
        B: target point set (n, m)
    Returns:
        A_aligned: aligned A
        scale: scaling factor
        R: rotation matrix
    """
    # 1. Center (if not already centered)
    A_centered = A - np.mean(A, axis=0)
    B_centered = B - np.mean(B, axis=0)

    # 2. Calculate scaling factor (least squares optimization)
    scale = np.trace(B_centered.T @ A_centered) / np.trace(A_centered.T @ A_centered)

    # 3. Calculate rotation matrix (SVD, without scaling)
    R, _ = orthogonal_procrustes(A_centered, B_centered)

    # 4. Apply transformation: scale -> rotate -> translate
    A_aligned = scale * A_centered @ R + np.mean(B, axis=0)

    return A_aligned, scale, R


class CorrespondMetrics:
    def __init__(self, map_to_original, point_to_index, size):
        """
        Initialize evaluator
        Parameters:
            - map_to_original: dictionary mapping points to original coordinates
            - point_to_index: dictionary mapping points to indices
        """
        # Extract original coordinates from map_to_original and point_to_index
        # First create a temporary dictionary mapping indices to coordinates
        self.size = size
        index_to_coords = {}
        for point, coords in map_to_original.items():
            index = point_to_index[point]
            index_to_coords[index] = coords

        self.original_coords = np.array([index_to_coords[i] for i in range(len(index_to_coords))])

    def scale_coords_dim(self, coords, size):
        """
        Linearly scale coordinates to [0, size[dim]] range, supporting any dimension.

        Args:
            coords (np.ndarray): input coordinates, shape (n_samples, n_dims)
            size (Tuple[float]): maximum value of target range for each dimension (minimum fixed at 0)

        Returns:
            np.ndarray: scaled coordinates, same shape as coords
        """
        coords = np.asarray(coords)
        if coords.ndim != 2:
            raise ValueError("coords must be 2D array")

        n_samples, n_dims = coords.shape

        if len(size) != n_dims:
            raise ValueError(f"size dimensions ({len(size)}) should match coords dimensions ({n_dims})")

        if not all(isinstance(s, (int, float)) and s > 0 for s in size):
            raise ValueError("all elements in size must be positive numbers")

        scaled = np.empty_like(coords, dtype=np.float64)

        for dim in range(n_dims):
            c_min, c_max = coords[:, dim].min(), coords[:, dim].max()
            target_max = size[dim]

            if np.isclose(c_max, c_min):
                # If dimension is constant, uniformly map to middle value
                scaled[:, dim] = target_max / 2
            else:
                scaled[:, dim] = (coords[:, dim] - c_min) / (c_max - c_min) * target_max

        return scaled

    def robust_scale_coords(self, coords, size):
        """
        Normalize coordinates to target range using robust scaling (supports multiple dimensions)

        Args:
            coords: input coordinate array, must be shape (n_samples, n_dims)
            size: target size tuple (N0, N1,...,Nx), length must match n_dims

        Returns:
            scaled coordinate array, same shape as input
        """
        # Parameter checking
        assert len(coords.shape) == 2, "coords must be 2D array (n_samples, n_dims)"
        assert coords.shape[1] == len(size), "size dimensions must match coords dimensions"

        # Initialize RobustScaler
        scaler = RobustScaler(
            quantile_range=(25, 75),
            with_centering=True,
            with_scaling=True
        )

        # Robust scaling
        scaled = scaler.fit_transform(coords)

        # Map each dimension to target range separately
        for dim in range(scaled.shape[1]):
            dim_min = scaled[:, dim].min()
            dim_max = scaled[:, dim].max()
            size_min, size_max = 0, size[dim] - 1  # Default map to [0, size[dim]]

            # Linear mapping to target range
            scaled[:, dim] = (scaled[:, dim] - dim_min) / (dim_max - dim_min) * \
                             (size_max - size_min) + size_min

        return scaled

    def visualize_alignment(self, reconstructed_coords, reconstructed_aligned):
        """
        Visualize original coordinates, reconstructed coordinates, and aligned coordinates
        Parameters:
            - reconstructed_coords: (n, 2) reconstructed coordinates (before alignment)
            - reconstructed_aligned: (n, 2) aligned coordinates
        """
        # Create figure
        plt.figure(figsize=(15, 5))

        # Plot original coordinates
        plt.subplot(131)
        plt.scatter(self.original_coords[:, 0], self.original_coords[:, 1], c='blue', label='Original')
        plt.title('Original Coordinates')
        plt.legend()

        # Plot reconstructed coordinates (before alignment)
        plt.subplot(132)
        plt.scatter(reconstructed_coords[:, 0], reconstructed_coords[:, 1], c='red', label='Reconstructed')
        plt.title('Reconstructed Coordinates (Before Alignment)')
        plt.legend()

        # Plot aligned coordinates
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
        Visualize original coordinates, reconstructed coordinates, and aligned coordinates (3D version)

        Parameters:
            - reconstructed_coords: (n, 3) reconstructed coordinates (before alignment)
            - reconstructed_aligned: (n, 3) aligned coordinates
        """
        # Create 3D figure
        fig = plt.figure(figsize=(18, 6))

        # Original coordinates (3D)
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(
            self.original_coords[:, 0],
            self.original_coords[:, 1],
            self.original_coords[:, 2],
            c='blue', label='Original'
        )
        ax1.set_title('Original Coordinates (3D)')
        ax1.legend()

        # Reconstructed coordinates (before alignment, 3D)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(
            reconstructed_coords[:, 0],
            reconstructed_coords[:, 1],
            reconstructed_coords[:, 2],
            c='red', label='Reconstructed'
        )
        ax2.set_title('Reconstructed (Before Alignment)')
        ax2.legend()

        # Aligned coordinates (3D, overlay display)
        ax3 = fig.add_subplot(133, projection='3d')
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

        # Translate both point sets to origin
        original_centered = self.original_coords - original_center
        reconstructed_centered = reconstructed_coords - reconstructed_center

        # Use Procrustes analysis for rotation alignment only
        _, reconstructed_aligned, _ = procrustes(original_centered, reconstructed_centered)

        reconstructed_aligned = self.robust_scale_coords(reconstructed_aligned, self.size)
        return reconstructed_aligned

    def snap_to_integer(self, coords):
        """Snap coordinates to nearest integer"""
        # Calculate overall offset (to make coordinate distribution more uniform)
        offsets = coords - np.round(coords)
        median_offset = np.median(offsets, axis=0)

        return np.round(coords - median_offset)

    def evaluate(self, reconstructed_coords):
        """
        Evaluate reconstruction quality
        Parameters:
            - reconstructed_coords: (n, 2) reconstructed coordinates
        Returns:
            - metrics: dictionary containing various metrics
        """
        # Calculate centers of original and reconstructed coordinates
        original_center = np.mean(self.original_coords, axis=0)
        reconstructed_center = np.mean(reconstructed_coords, axis=0)

        # Translate both point sets to origin
        original_centered = self.original_coords - original_center
        reconstructed_centered = reconstructed_coords - reconstructed_center

        # Use Procrustes analysis for rotation alignment only
        _, reconstructed_aligned, _ = procrustes(original_centered, reconstructed_centered)

        reconstructed_aligned = self.robust_scale_coords(reconstructed_aligned, self.size)
        # reconstructed_aligned = self.snap_to_integer(reconstructed_aligned)

        # Calculate various metrics
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
        Calculate relaxed match rate: whether reconstructed points are within neighborhood of original points
        Args:
            reconstructed: (n, 2) reconstructed coordinates
            radius: matching threshold
        Returns:
            match_rate: match proportion
        """
        if len(self.original_coords) == 0 or len(reconstructed) == 0:
            return 0.0

        # Directly calculate distances between corresponding points
        distances = np.square(np.linalg.norm(self.original_coords - reconstructed, axis=1))
        matched = np.sum(distances <= radius)

        return matched / len(reconstructed)

    def _mse_with_correspondence(self, reconstructed, reconstructed_to_original):
        """Calculate MSE using point correspondence"""
        mse = 0
        for i in range(len(self.original_coords)):
            reconstructed_idx = reconstructed_to_original[i]
            mse += np.sum((self.original_coords[i] - reconstructed[reconstructed_idx]) ** 2)
        return mse / len(self.original_coords)

    def _exact_match_with_correspondence(self, reconstructed, reconstructed_to_original):
        """Calculate exact match rate using point correspondence"""
        exact_matches = 0
        for i in range(len(self.original_coords)):
            reconstructed_idx = reconstructed_to_original[i]
            if np.all(np.round(reconstructed[reconstructed_idx]) == self.original_coords[i]):
                exact_matches += 1
        return exact_matches / len(self.original_coords)

    def _visualize_neighbors(self, reconstructed, reconstructed_to_original, indices_original, indices_reconstructed,
                             k=5):
        """Visualize nearest neighbor relationships between original and reconstructed points"""
        plt.figure(figsize=(12, 6))

        # Plot original points
        plt.subplot(121)
        plt.scatter(self.original_coords[:, 0], self.original_coords[:, 1], c='blue', label='Original')
        # Plot nearest neighbor relationships for original points
        for i in range(len(self.original_coords)):
            neighbors = indices_original[i]
            for j in neighbors:
                plt.plot([self.original_coords[i, 0], self.original_coords[j, 0]],
                         [self.original_coords[i, 1], self.original_coords[j, 1]],
                         'b-', alpha=0.2)
        plt.title('Original Points and Neighbors')
        plt.legend()

        # Plot reconstructed points
        plt.subplot(122)
        plt.scatter(reconstructed[:, 0], reconstructed[:, 1], c='red', label='Reconstructed')
        # Plot nearest neighbor relationships for reconstructed points
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
        """Calculate nearest neighbor accuracy using nearest neighbor matching"""
        # Ensure input data is valid
        if len(reconstructed) == 0 or len(self.original_coords) == 0:
            return 0.0

        # Calculate nearest neighbors
        nn_original = NearestNeighbors(n_neighbors=min(k, len(self.original_coords))).fit(self.original_coords)
        indices_original = nn_original.kneighbors(return_distance=False)

        nn_reconstructed = NearestNeighbors(n_neighbors=min(k, len(reconstructed))).fit(reconstructed)
        indices_reconstructed = nn_reconstructed.kneighbors(return_distance=False)

        # Ensure k value does not exceed point set size
        k = min(k, len(self.original_coords), len(reconstructed))

        overlap = 0
        for i in range(len(self.original_coords)):
            # Get nearest neighbors of original point i
            original_neighbors = indices_original[i]
            # Get corresponding reconstructed point
            reconstructed_idx = reconstructed_to_original[i]
            # Get nearest neighbors of reconstructed point
            reconstructed_neighbors = indices_reconstructed[reconstructed_idx]

            # Map reconstructed neighbors back to original point indices
            mapped_reconstructed_neighbors = [reconstructed_to_original[j] for j in reconstructed_neighbors]

            # Calculate overlap
            overlap += len(np.intersect1d(original_neighbors, mapped_reconstructed_neighbors))

        return overlap / (k * len(self.original_coords))

    def _mse_hungarian(self, reconstructed):
        """
        Calculate MSE between two point sets using Hungarian algorithm.

        Parameters:
        - A: (n, d) NumPy array representing n d-dimensional points
        - B: (n, d) NumPy array representing n d-dimensional points

        Returns:
        - mse_value: calculated MSE value
        """
        cost_matrix = cdist(self.original_coords, reconstructed) ** 2  # Calculate squared Euclidean distance
        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Hungarian algorithm for optimal matching
        return np.mean(cost_matrix[row_ind, col_ind])  # Calculate MSE

    def _mse_nearest_neighbor(self, reconstructed):
        """
        Calculate MSE using nearest neighbor matching
        """
        kd_tree = cKDTree(reconstructed)  # Build KD-Tree on B
        distances, _ = kd_tree.query(self.original_coords)  # Query nearest neighbor distance for each point in A
        return np.mean(distances ** 2)  # Calculate MSE

    def _calc_mse(self, reconstructed):
        """Mean squared error"""
        return np.mean(np.sum((self.original_coords - reconstructed) ** 2, axis=1))

    def _calc_exact_match(self, reconstructed):
        """Exact match rate"""
        rounded = np.round(reconstructed)
        return np.mean(np.all(rounded == self.original_coords, axis=1))

    def _calc_exact_match_nearest(self, reconstructed):
        """Calculate exact match rate using nearest neighbor matching"""
        kd_tree = cKDTree(self.original_coords)
        distances, _ = kd_tree.query(reconstructed)
        return np.mean(distances == 0)

    def _calc_neighbor_accuracy(self, reconstructed, k=5):
        """Nearest neighbor accuracy"""
        nn_original = NearestNeighbors(n_neighbors=k).fit(self.original_coords)
        indices_original = nn_original.kneighbors(return_distance=False)

        nn_reconstructed = NearestNeighbors(n_neighbors=k).fit(reconstructed)
        indices_reconstructed = nn_reconstructed.kneighbors(return_distance=False)
        overlap = 0
        for i in range(len(self.original_coords)):
            if i < len(indices_reconstructed):  # Ensure index is within range
                overlap += len(np.intersect1d(indices_original[i], indices_reconstructed[i]))
        return overlap / (k * len(self.original_coords))
