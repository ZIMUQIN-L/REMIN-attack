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
        Initialize evaluator
        Parameters:
            - original_coords: (n, 2) original coordinates
        """
        self.original_coords = original_coords

    def evaluate(self, reconstructed_coords):
        """
        Evaluate reconstruction quality
        Parameters:
            - reconstructed_coords: (n, 2) reconstructed coordinates
        Returns:
            - metrics: dictionary containing various metrics
        """
        # Find correspondence between reconstructed points and original points
        kd_tree = cKDTree(reconstructed_coords)
        _, reconstructed_to_original = kd_tree.query(self.original_coords)
        
        # Calculate various metrics
        metrics = {
            'mse_nn': self._mse_with_correspondence(reconstructed_coords, reconstructed_to_original),
            'exact_match_nn': self._exact_match_with_correspondence(reconstructed_coords, reconstructed_to_original),
            'neighbor_accuracy': self._calc_neighbor_accuracy(reconstructed_coords),  # Use original method, focus on distribution only
            # 'nn_na': self._calc_neighbor_accuracy_nearest(reconstructed_coords, reconstructed_to_original),
        }

        return metrics

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

    def _visualize_neighbors(self, reconstructed, reconstructed_to_original, indices_original, indices_reconstructed, k=5):
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
        kd_tree = cKDTree(self.original_coords)  # Build KD-Tree on original_coords
        distances, _ = kd_tree.query(reconstructed)  # Query nearest neighbor for each reconstructed point
        return np.mean(distances == 0)  # Count exact match proportion

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

    def _calc_distance_correlation(self, reconstructed):
        """Distance rank correlation coefficient"""
        original_dist = pdist(self.original_coords)
        reconstructed_dist = pdist(reconstructed)
        return spearmanr(original_dist, reconstructed_dist).correlation

    def _calc_grid_alignment(self, coords):
        """Grid alignment error"""
        snapped = np.round(coords)
        return np.mean(np.linalg.norm(coords - snapped, axis=1))