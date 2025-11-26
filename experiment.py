import itertools
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from distance import *
from dimension import *
from align import *
from snapping import *
from metrics import *
from topology import *
from sklearn.neighbors import KernelDensity
import range_attack

class ExperimentRunner:
    def __init__(self, points, map_to_original, responses):
        # Ensure points and map_to_original have consistent key types
        self.points = points
        self.responses = responses
        self.dist_matrix = None
        self.map_to_original = map_to_original
        self.point_to_index = {p: i for i, p in enumerate(self.points)}
        self.original_result = None
        self.aligned_result = None
        self.snapped_result = None
        self.W = None
        self.D = None
        self.metrics = ReconstructionMetrics(
            np.array([self.map_to_original[p] for p in self.points])
        )
        self.full_pos = None
        self.densities = None
        self.true_mean = None
        self.true_var = None

    def set_true_aggregate(self, mean, var):
        self.true_mean = mean
        self.true_var = var

    def evaluate_reconstruction(self, reconstructed_coords, method_name=''):
        """Evaluate reconstruction quality"""
        # print(reconstructed_coords)
        return self.metrics.evaluate(reconstructed_coords)

    def distance_matrix_from_table(self, distance_table):
        size = len(self.points)
        distance_matrix = np.full((size, size), np.inf)

        for (id1, id2), distance in distance_table.items():
            # Ensure id1 and id2 types are consistent with points
            i, j = self.point_to_index[id1], self.point_to_index[id2]
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # Symmetric

        np.fill_diagonal(distance_matrix, 0)
        self.dist_matrix = distance_matrix
        return distance_matrix


    def get_geometric_density(self, pos, bandwidth=2.0):
        """
        Spatial density estimation based on Kamada-Kawai layout coordinates
        :param pos: (n_points, 2) topological initialization coordinates
        :param bandwidth: bandwidth selection strategy
        :return: Normalized density array (n_points,)
        """

        # Normalize coordinates
        pos_norm = (pos - pos.mean(0)) / pos.std(0)

        # 2D kernel density estimation
        kde = KernelDensity(bandwidth=bandwidth, kernel='cosine')
        kde.fit(pos_norm)

        # Calculate log density and normalize to [0,1]
        log_dens = kde.score_samples(pos_norm)
        densities = np.exp(log_dens)
        return (densities - densities.min()) / (densities.max() - densities.min() + 1e-8)

    def plot_density_heatmap(self, pos, densities):
        plt.scatter(pos[:, 0], pos[:, 1], c=densities, cmap='viridis', s=50)
        plt.colorbar(label='Density')
        plt.show()

    def build_graph_from_responses(self):
        """
        Generate graph adjacency matrix from response set and construct networkx Graph
        :param response_set: List[List[int]] - returned point sets for each query
        :param num_points: int - number of database points
        :return: networkx Graph
        """
        G = nx.Graph()

        # Iterate through all size=2 responses to construct edges
        response_set = augment_responses(self.responses)
        for (i, j) in response_set:
            G.add_edge(i, j)
        pos = nx.kamada_kawai_layout(G, dim=2)

        # use point_to_index to correctly map positions
        pos_array = np.zeros((len(self.points), 2))
        for point_id, coord in pos.items():
            if point_id in self.point_to_index:
                pos_array[self.point_to_index[point_id]] = coord

        # Density estimation based on geometric positions
        densities = self.get_geometric_density(pos_array)

        self.plot_graph_layout(pos)

        full_pos = np.zeros((len(self.points), 2))  # Default fill with (0,0)
        for point_id, coord in pos.items():
            full_pos[self.point_to_index[point_id]] = coord  # Only fill connected points

        self.plot_density_heatmap(full_pos, densities)
        self.full_pos = full_pos
        self.densities = densities

        return full_pos, densities


    def classic_run(self, responses, distance_config, reduction_config):
        """Execute experiment and return results"""
        # Calculate frequency table
        freq_table = self.build_frequency_table(responses)
        total_responses = self.count_total_responses(responses)


        # Execute all combinations
        results = {}
        for (dist_name, dist_params), (red_name, red_params) in itertools.product(distance_config, reduction_config):
            # Calculate distance
            if dist_name == 'jaccard':
                dist_table = jaccard(freq_table, total_responses)
            elif dist_name == 'log':
                dist_table = logarithmic(freq_table)
            elif dist_name == 'reciprocal':
                dist_table = reciprocal(freq_table)
            elif dist_name == 'gaussian':
                dist_table = gaussian_kernel(freq_table, **dist_params)
            elif dist_name == 'euclidean':
                dist_table = cooccurrence_euclidean(freq_table, total_responses)
            else:
                dist_table = origin_reciprocal(freq_table)
            # Build matrix
            dist_matrix = self.distance_matrix_from_table(dist_table)

            # Dimensionality reduction
            if red_name == 'tsne':
                coords = tsne(dist_matrix, **red_params)
            elif red_name == 'topo_tsne':
                if self.full_pos is None or self.densities is None:
                    pos, densities = self.build_graph_from_responses()
                else:
                    pos, densities = self.full_pos, self.densities
                coords = topology_preserving_tsne(dist_matrix, pos, self.D, **red_params)
            elif red_name == 'density_tsne':
                if self.full_pos is None or self.densities is None:
                    pos, densities = self.build_graph_from_responses()
                else:
                    pos, densities = self.full_pos, self.densities
                coords = density_preserving_tsne(dist_matrix, pos, densities, **red_params)
            elif red_name == 'aware_tsne':
                if self.full_pos is None or self.densities is None:
                    pos, densities = self.build_graph_from_responses()
                else:
                    pos, densities = self.full_pos, self.densities
                coords = density_aware_tsne(dist_matrix, pos, densities, **red_params)
            elif red_name == 'aggregate_tsne':
                coords = aggregate_tsne(dist_matrix, self.true_mean, self.true_var, **red_params)
            elif red_name == 'mds':
                coords = mds(dist_matrix, **red_params)
            elif red_name == 'isomap':
                coords = isomap(dist_matrix, **red_params)
            else:
                return

            # Store results
            key = f"{dist_name} + {dist_params} + {red_name} + {red_params}"
            # key = f"{dist_name} distance with {red_name}"
            # key = f"{red_params}"
            results[key] = coords
            self.original_result = results
        return results

    def align_coords(self, grid_size):
        new_results = {}
        for key, coords in self.original_result.items():
            aligned_coords = original_align_to_grid(coords, grid_size)
            new_results[key] = aligned_coords
        self.aligned_result = new_results
        return new_results

    def snap_coords(self, grid_size):
        new_results = {}
        for key, coords in self.aligned_result.items():
            snapped_coords = simulated_annealing_snap(coords, grid_size)
            new_results[key] = snapped_coords
        return new_results


    def plot_results(self, results):
        """Visualize all results"""
        n = len(results)
        n_cols = 3
        n_rows = int(np.ceil(n / n_cols))

        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        for idx, (title, coords) in enumerate(results.items(), 1):
            plt.subplot(n_rows, n_cols, idx)
            plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
            plt.title(title)
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    def build_frequency_table(self, responses):
        frequency_table = {}
        for response in responses:
            for id1, id2 in itertools.combinations(response, 2):
                if id1 > id2:  # Ensure id1 < id2 to avoid duplicates
                    id1, id2 = id2, id1
                frequency_table[(id1, id2)] = frequency_table.get((id1, id2), 0) + 1
        return frequency_table

    def count_total_responses(self, responses):
        """Count total occurrences for each point (fix type issues)"""
        frequency_table = {}
        for response in responses:
            for id in response:
                frequency_table[id] = frequency_table.get(id, 0) + 1
        return frequency_table

    def plot_graph_layout(self, pos):
        X = []
        Y = []

        for i in pos:
            point = pos[i]
            X.append(point[0])
            Y.append(point[1])

        fig = plt.figure()

        plt.scatter(X, Y, s=5)
        plt.gca().set_aspect('equal')
        plt.show()