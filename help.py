import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calculate_total_responses(responses, points):
    """Calculate total occurrences for each point (for Jaccard)"""
    total_counts = {point: 0 for point in points}
    for response in responses:
        for point in response:
            total_counts[point] += 1
    return total_counts


def build_frequency_table(responses):
    """Generate co-occurrence frequency table (keep as is)"""
    frequency_table = {}
    for response in responses:
        for id1, id2 in itertools.combinations(response, 2):
            if id1 > id2:
                id1, id2 = id2, id1
            frequency_table[(id1, id2)] = frequency_table.get((id1, id2), 0) + 1
    return frequency_table


def scale_and_reduce_points(points, N0, N1, method="grid"):
    """
    Scale database to specified size and reduce number of points
    Parameters:
        - points: [(x, y), ...] original point list
        - N0: target scaling width (x direction)
        - N1: target scaling height (y direction)
        - method: "grid" uses grid filtering, "kmeans" uses clustering
    Returns:
        - new_points: processed new point list
    """
    points = np.array(points)  # Convert to NumPy array
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Normalization: scale to [0,1] range
    norm_points = (points - [min_x, min_y]) / (max_x - min_x)

    # Rescale: adjust to [0, N0-1] x [0, N1-1] range
    scaled_points = np.round(norm_points * [N0 - 2, N1 - 2] + 1).astype(int)

    if method == "grid":
        # Use set deduplication, preserve basic shape but reduce point count
        unique_points = list(set(map(tuple, scaled_points)))
    elif method == "kmeans":
        num_clusters = int(len(points) * 0.5)  # reduce points by 50%
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_points)
        unique_points = kmeans.cluster_centers_.astype(int).tolist()
    else:
        raise ValueError("method must be 'grid' or 'kmeans'")

    return unique_points


def plot_metrics(evaluation):
    """Visualize evaluation metrics"""
    df = pd.DataFrame(evaluation).T
    df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
    plt.title('Reconstruction Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_metrics_normalized(evaluation):
    """Single chart visualization of normalized evaluation metrics"""
    df_normalized = pd.DataFrame(evaluation).T

    # Plot bar chart
    df_normalized.plot(kind='bar', figsize=(12, 6), colormap='viridis')
    plt.title('Normalized Reconstruction Metrics')
    plt.ylabel('Normalized Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot2D(pos):
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