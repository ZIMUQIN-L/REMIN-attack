import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calculate_total_responses(responses, points):
    """计算每个点的总出现次数（用于Jaccard）"""
    total_counts = {point: 0 for point in points}
    for response in responses:
        for point in response:
            total_counts[point] += 1
    return total_counts


def build_frequency_table(responses):
    """生成共现频率表（保持原样）"""
    frequency_table = {}
    for response in responses:
        for id1, id2 in itertools.combinations(response, 2):
            if id1 > id2:
                id1, id2 = id2, id1
            frequency_table[(id1, id2)] = frequency_table.get((id1, id2), 0) + 1
    return frequency_table


def scale_and_reduce_points(points, N0, N1, method="grid"):
    """
    将数据库缩放到指定大小，并减少点的数量
    参数:
        - points: [(x, y), ...] 原始点列表
        - N0: 目标缩放尺寸的宽度 (x 方向)
        - N1: 目标缩放尺寸的高度 (y 方向)
        - method: "grid" 使用网格过滤, "kmeans" 使用聚类
    返回:
        - new_points: 处理后的新点列表
    """
    points = np.array(points)  # 转换为 NumPy 数组
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # 1️⃣ 归一化: 缩放到 [0,1] 区间
    norm_points = (points - [min_x, min_y]) / (max_x - min_x)

    # 2️⃣ 重新缩放: 调整到 [0, N0-1] x [0, N1-1] 区间
    scaled_points = np.round(norm_points * [N0 - 2, N1 - 2] + 1).astype(int)

    if method == "grid":
        # 3️⃣ 使用集合去重，保留基本形状但减少点数
        unique_points = list(set(map(tuple, scaled_points)))
    elif method == "kmeans":
        num_clusters = int(len(points) * 0.5)  # 例如减少 50% 的点
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_points)
        unique_points = kmeans.cluster_centers_.astype(int).tolist()
    else:
        raise ValueError("method must be 'grid' or 'kmeans'")

    return unique_points


def plot_metrics(evaluation):
    """可视化评估指标"""
    df = pd.DataFrame(evaluation).T
    df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
    plt.title('Reconstruction Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_metrics_normalized(evaluation):
    """归一化后单图可视化评估指标"""
    df_normalized = pd.DataFrame(evaluation).T

    # 归一化处理（按列归一化到 [0,1] 区间）
    # df_normalized = (df) / (df.max())

    # 绘制柱状图
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