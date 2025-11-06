import random
import numpy
import tqdm
from itertools import product
import itertools
import math
from typing import List, Set, Tuple, Dict
import statistics


def scale_points(points, N0, N1):
    maxN0 = 0
    maxN1 = 0

    new_points = []

    for i, j in points:
        maxN0 = max(i, maxN0)
        maxN1 = max(j, maxN1)

    for i, j in points:
        new_i = int(max(1, i * N0 / maxN0))
        new_j = int(max(1, j * N1 / maxN1))
        new_points.append((new_i, new_j))
    return new_points


def make_database_from_points(points):
    """
    给所有点一个标识符
    give each points an identifier
    :param points:
    :return:
    """
    newN0 = 0
    newN1 = 0
    new_points = []

    map_to_original = {}
    for point in points:
        i, j = point
        search_token = random.randrange(10000000)
        while search_token in map_to_original:
            search_token = random.randrange(10000000)
        map_to_original[search_token] = (i, j)
        new_points.append(search_token)
        newN0 = max(i, newN0)
        newN1 = max(j, newN1)
    return new_points, map_to_original, newN0 + 1, newN1 + 1


def make_database_from_points_3D(points):
    newN0 = 0
    newN1 = 0
    newN2 = 0
    new_points = []

    map_to_original = {}
    for point in points:
        i, j, k = point
        search_token = random.randrange(10000000)
        while search_token in map_to_original:
            search_token = random.randrange(10000000)
        map_to_original[search_token] = (i, j, k)
        new_points.append(search_token)
        newN0 = max(i, newN0)
        newN1 = max(j, newN1)
        newN2 = max(k, newN2)
    return new_points, map_to_original, newN0 + 1, newN1 + 1, newN2 + 1


def get_class_database(N0, N1, max_points, plaintext=False):
    map_to_original = {}
    points = []

    # 定义左上角和右下角的范围
    top_left_range = (0, N0 // 2, 0, N1 // 2)  # 左上角范围
    bottom_right_range = (N0 // 2, N0, N1 // 2, N1)  # 右下角范围

    # 生成左上角的点
    for i in range(top_left_range[0], top_left_range[1]):
        for j in range(top_left_range[2], top_left_range[3]):
            if random.randrange(100) < 30:
                continue
            repeats = int(1 + (max_points - 1) * random.random())
            for num in range(repeats):
                if plaintext:
                    search_token = (i, j)
                else:
                    search_token = random.randrange(10000000)
                map_to_original[search_token] = (i, j)
                points.append(search_token)

    # 生成右下角的点
    for i in range(bottom_right_range[0], bottom_right_range[1]):
        for j in range(bottom_right_range[2], bottom_right_range[3]):
            if random.randrange(100) < 30:
                continue
            repeats = int(1 + (max_points - 1) * random.random())
            for num in range(repeats):
                if plaintext:
                    search_token = (i, j)
                else:
                    search_token = random.randrange(10000000)
                map_to_original[search_token] = (i, j)
                points.append(search_token)

    return points, map_to_original


def get_random_database_with_outlier(N0, N1, max_points, plaintext=False):
    map_to_original = {}
    points = []

    # 生成主要点集（集中在某个区域，比如中间）
    main_range = (N0 // 4, 3 * N0 // 4, N1 // 4, 3 * N1 // 4)  # 主要点集的范围

    for i in range(main_range[0], main_range[1]):
        for j in range(main_range[2], main_range[3]):
            if random.randrange(100) < 20:  # 30% 的概率跳过
                continue
            repeats = int(1 + (max_points - 1) * random.random())
            for num in range(repeats):
                if plaintext:
                    search_token = (i, j)
                else:
                    search_token = random.randrange(10000000)
                map_to_original[search_token] = (i, j)
                points.append(search_token)

    # 添加一个离群点（远离主要点集）
    outlier_i = random.choice([0, N0 - 1])  # 离群点在左上角或右下角
    outlier_j = random.choice([0, N1 - 1])
    if plaintext:
        outlier_token = (outlier_i, outlier_j)
    else:
        outlier_token = random.randrange(10000000)
    map_to_original[outlier_token] = (outlier_i, outlier_j)
    points.append(outlier_token)

    if plaintext:
        outlier_token = (outlier_i-1, outlier_j)
    else:
        outlier_token = random.randrange(10000000)
    map_to_original[outlier_token] = (outlier_i-1, outlier_j)
    points.append(outlier_token)
    return points, map_to_original


def get_random_database(N0, N1, max_points, sparsity=0, plaintext=False):
    map_to_original = {}
    points = []
    for i in range(1, N0):
        for j in range(1, N1):
            if random.randrange(100) < sparsity:
                continue
            repeats = int(1 + (max_points - 1) * random.random())
            for num in range(repeats):
                if plaintext:
                    search_token = (i, j)
                else:
                    search_token = random.randrange(10000000)
                map_to_original[search_token] = (i, j)
                points.append(search_token)

    return points, map_to_original


def get_random_database_3D(N0, N1, N2, max_points):
    map_to_original = {}
    points = []
    for i in range(1, N0):
        for j in range(1, N1):
            for z in range(1, N2):
                repeats = int(1 + (max_points - 1) * random.random())
                for num in range(repeats):
                    search_token = random.randrange(10000000)
                    map_to_original[search_token] = (i, j, z)
                    points.append(search_token)

    return points, map_to_original


def get_random_database_nd(N, dim, max_points):
    """
    生成一个任意维度的随机数据库

    参数:
        N: 每个维度的长度（所有维度相同）
        dim: 维度数量
        max_points: 每个坐标点最多生成的搜索令牌数

    返回:
        points: 搜索令牌列表
        map_to_original: 从搜索令牌到原始坐标的字典
    """
    map_to_original = {}
    points = []

    # 生成各维度的范围 (从1到N-1)
    ranges = [range(1, N) for _ in range(dim)]

    # 遍历所有可能的坐标组合
    for coords in product(*ranges):
        # 为每个坐标点生成随机数量的搜索令牌
        if random.randrange(100) < 40:
            continue
        repeats = int(1 + (max_points - 1) * random.random())
        for _ in range(repeats):
            search_token = random.randrange(10000000)
            map_to_original[search_token] = coords
            points.append(search_token)

    return points, map_to_original


def get_responses_no_vals_nd(points, coord_map, Ns, dim):
    """
    Ns: list[int]，每个维度的最大值（不含）
    dim: int，维度数量
    返回：所有合法的 (min_0, max_0, ..., min_{d-1}, max_{d-1}) 区间组合
    """

    # 构造每个维度的 (min_i, max_i) 组合
    all_ranges = []
    for i in range(dim):
        ranges_i = [(min_i, max_i) for min_i in range(1, Ns) for max_i in range(min_i, Ns)]
        all_ranges.append(ranges_i)

    # 笛卡尔积组合每个维度的区间
    resps = []
    for combo in tqdm.tqdm(itertools.product(*all_ranges)):
        # combo 是 ((min0, max0), (min1, max1), ...)
        flat_combo = []
        for pair in combo:
            flat_combo.extend(pair)
        resps.append(tuple(flat_combo))

    return resps


def get_actual_query_resps_after_sampling_nd(resps, points, map_points_to_coordinates):
    """
    计算每个查询范围实际匹配的点（多维版本）

    参数:
        resps: 查询范围列表，格式 [(min0, max0, min1, max1, ...), ...]
        points: 所有搜索令牌列表
        map_points_to_coordinates: 令牌到坐标的映射

    返回:
        actual: 每个查询范围对应的点集合列表 [set(), set(), ...]
        unique_rs: 所有满足至少一个查询的点的坐标（去重）
    """
    actual = []
    unique_rs = set()
    seen_ranges = set()

    # 遍历所有查询范围
    for bounds in tqdm.tqdm(resps):
        # 检查范围是否已处理过（去重）
        current_range = tuple(bounds)
        if current_range in seen_ranges:
            continue
        seen_ranges.add(current_range)

        # 解析 min_i 和 max_i
        dim = len(bounds) // 2
        min_bounds = bounds[::2]  # (min0, min1, ..., min_k)
        max_bounds = bounds[1::2]  # (max0, max1, ..., max_k)

        # 收集当前范围内的点
        matched_points = set()
        for p in points:
            coords = map_points_to_coordinates[p]
            in_range = True
            for i in range(dim):
                if not (min_bounds[i] <= coords[i] <= max_bounds[i]):
                    in_range = False
                    break
            if in_range:
                matched_points.add(p)
                unique_rs.add(coords)  # 记录坐标（自动去重）

        actual.append(matched_points)

    return actual, unique_rs


def get_responses(points, map_points_to_coordinates, N0, N1):
    # 循环获取所有的response set
    resps = []
    for min0 in tqdm.tqdm(range(1, N0)):
        for min1 in range(1, N1):
            for max0 in range(min0, N0):
                for max1 in range(min1, N1):
                    r = []
                    for p in points:
                        if map_points_to_coordinates[p][0] <= max0 and map_points_to_coordinates[p][0] >= min0 and \
                                map_points_to_coordinates[p][1] <= max1 and map_points_to_coordinates[p][1] >= min1:
                            r.append(p)
                    resps.append(set(r))

    return resps


def get_responses_no_vals(points, map_points_to_coordinates, N0, N1):
    resps = []
    for min0 in tqdm.tqdm(range(1, N0)):
        for min1 in range(1, N1):
            for max0 in range(min0, N0):
                for max1 in range(min1, N1):
                    # r = []
                    # for p in points:
                    #     if map_points_to_coordinates[p][0] <= max0 and map_points_to_coordinates[p][0] >= min0 and map_points_to_coordinates[p][1] <= max1 and map_points_to_coordinates[p][1] >= min1:
                    #         r.append(p)
                    resps.append((min0, max0, min1, max1))

    return resps


def get_actual_resps_after_sampling(resps, points, map_points_to_coordinates):
    actual = []

    unique_rs = set()

    for min0, max0, min1, max1 in tqdm.tqdm(resps):
        r = []
        for p in points:
            if map_points_to_coordinates[p][0] <= max0 and map_points_to_coordinates[p][0] >= min0 and \
                    map_points_to_coordinates[p][1] <= max1 and map_points_to_coordinates[p][1] >= min1:
                r.append(p)
                unique_rs.add(map_points_to_coordinates[p])
        actual.append(set(r))
    return actual, unique_rs


def get_actual_query_resps_after_sampling(resps, points, map_points_to_coordinates):
    actual = []
    unique_rs = set()

    seen_ranges = set()

    for min0, max0, min1, max1 in tqdm.tqdm(resps):
        current_range = (min0, max0, min1, max1)

        if current_range in seen_ranges:
            continue
        seen_ranges.add(current_range)

        r = []
        for p in points:
            if map_points_to_coordinates[p][0] <= max0 and map_points_to_coordinates[p][0] >= min0 and \
                    map_points_to_coordinates[p][1] <= max1 and map_points_to_coordinates[p][1] >= min1:
                r.append(p)
                unique_rs.add(map_points_to_coordinates[p])
        actual.append(set(r))  # Add sorted tuple of points to preserve duplicates

    return actual, unique_rs


def get_responses_3D(points, d, N0, N1, N2):
    resps = []
    i = 0
    for min0 in tqdm.tqdm(range(1, N0)):
        for min1 in range(1, N1):
            for min2 in range(1, N2):
                for max0 in range(min0, N0):
                    for max1 in range(min1, N1):
                        for max2 in range(min2, N2):
                            r = []
                            for p in points:
                                if d[p][0] <= max0 and d[p][0] >= min0 and d[p][1] <= max1 and d[p][1] >= min1 and d[p][
                                    2] <= max2 and d[p][2] >= min2:
                                    r.append(p)
                            resps.append(set(r))
                            i += 1
    return resps


def get_responses_no_vals_3D(points, d, N0, N1, N2):
    resps = []
    for min0 in tqdm.tqdm(range(1, N0)):
        for min1 in range(1, N1):
            for min2 in range(1, N2):
                for max0 in range(min0, N0):
                    for max1 in range(min1, N1):
                        for max2 in range(min2, N2):
                            # r = []
                            # for p in points:
                            #    if d[p][0] <= max0 and d[p][0] >= min0 and d[p][1] <= max1 and d[p][1] >= min1 and d[p][2] <= max2 and d[p][2] >= min2:
                            #        r.append(p)
                            resps.append((min0, max0, min1, max1, min2, max2))
    return resps


def get_actual_query_resps_after_sampling_3D(resps, points, map_points_to_coordinates):
    actual = []

    unique_rs = set()
    seen_ranges = set()

    for min0, max0, min1, max1, min2, max2 in tqdm.tqdm(resps):
        current_range = (min0, max0, min1, max1, min2, max2)

        if current_range in seen_ranges:
            continue
        seen_ranges.add(current_range)
        r = []
        for p in points:
            if map_points_to_coordinates[p][0] <= max0 and map_points_to_coordinates[p][0] >= min0 and \
                    map_points_to_coordinates[p][1] <= max1 and map_points_to_coordinates[p][1] >= min1 and \
                    map_points_to_coordinates[p][2] <= max2 and map_points_to_coordinates[p][2] >= min2:
                r.append(p)
                unique_rs.add(map_points_to_coordinates[p])
        actual.append(set(r))
    return actual, unique_rs


def get_actual_resps_after_sampling_3D(resps, points, map_points_to_coordinates):
    actual = []

    unique_rs = set()

    for min0, max0, min1, max1, min2, max2 in tqdm.tqdm(resps):
        r = []
        for p in points:
            if map_points_to_coordinates[p][0] <= max0 and map_points_to_coordinates[p][0] >= min0 and \
                    map_points_to_coordinates[p][1] <= max1 and map_points_to_coordinates[p][1] >= min1 and \
                    map_points_to_coordinates[p][2] <= max2 and map_points_to_coordinates[p][2] >= min2:
                r.append(p)
                unique_rs.add(map_points_to_coordinates[p])
        actual.append(set(r))
    return actual, unique_rs


def sample_gaussian(resps, needed):
    to_return = []
    while len(to_return) < needed:
        index = int(random.gauss(len(resps) / 2, len(resps) / 5))
        if index >= 0 and index < len(resps) - 1:
            to_return.append(resps[index])
    return to_return


def sample_beta(resps, needed):
    to_return = []
    for i in range(needed):
        index = int(random.betavariate(2, 1) * (len(resps) - 1))
        index = min(index, len(resps) - 1)
        index = max(index, 0)
        to_return.append(resps[index])
    return to_return


def sample_uniform(resps, needed):
    return random.sample(resps, needed)

'''
about countermeasure
'''

def generate_bogus_pool(
    map_to_original: Dict,
    M: int
) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """
    生成一个 bogus 池：
    - 坐标从真实分布中采样
    - fake_token 命名为 'fake_<k>'
    返回:
      bogus_points : List[str]  —— fake token 列表
      bogus_map    : Dict[str, (x,y)] —— fake token -> 坐标
    """
    real_coords = list(map_to_original.values())
    bogus_points, bogus_map = [], {}
    for k in range(M):
        coord = random.choice(real_coords)      # 从真实分布采样
        fake_token = k + 10000000               # 唯一 ID
        bogus_points.append(fake_token)
        bogus_map[fake_token] = coord
    return bogus_points, bogus_map


def pad_results_to_power_of_x(
    results: List[Set],
    query_rects: List[Tuple[float,float,float,float]],
    x: int,
    bogus_points: List[str],
    bogus_map: Dict[str, Tuple[float,float]]
) -> Tuple[List[Set], Dict[str, Tuple[float,float]], Dict]:
    """
    对每个查询结果集合做 padding:
    - target size = 最近的 x 的幂 (如果 len(res)==0 则不 pad)
    - 只从 bogus_map 中坐标落在该 query range 的假点中选（优先）
    - 允许同一个 fake token 在不同 query 之间重复使用（跨 query 可复用）
    返回:
      padded_results, used_bogus_map, metrics_dict
    metrics_dict 包含 avg_true_size, avg_padded_size, bandwidth_ratio, total_fake_inserted, index_size_overhead, per_query_stats
    """
    padded_results: List[Set] = []
    used_bogus: Dict[str, Tuple[float,float]] = {}

    true_sizes = []
    padded_sizes = []
    fake_insert_count = 0

    for res, rect in zip(results, query_rects):
        # treat res as a set (if it's list convert to set)
        if res is None:
            res = set()
        res_set = set(res)

        # record true size for non-empty queries (you may choose to include empty queries if desired)
        true_sizes.append(len(res_set))

        if len(res_set) == 0:
            padded_results.append(set())
            padded_sizes.append(0)
            continue

        # 计算目标大小：最近的 x 的幂
        # defensive: if len(res_set) == 1, math.log ok
        target_size = x ** math.ceil(math.log(max(1, len(res_set)), x))
        padded = set(res_set)

        xmin, xmax, ymin, ymax = rect

        # collect candidate fake ids inside rect
        candidates = [
            fid for fid, coord in bogus_map.items()
            if xmin <= coord[0] <= xmax and ymin <= coord[1] <= ymax
        ]

        # sample to fill up to target_size
        # avoid duplicates within the same query by removing chosen from local candidates
        while len(padded) < target_size:
            if candidates:
                chosen = random.choice(candidates)
                candidates.remove(chosen)  # avoid duplicate within same query
            else:
                chosen = random.choice(bogus_points)
            # if chosen already in padded (possible if chosen from bogus_points but already added), skip
            if chosen in padded:
                # if full pool exhausted and duplicates happen, try continue to pick different; to avoid infinite loop:
                # if all bogus_points are already used and still need more, break (rare)
                if len(padded) >= target_size:
                    break
                # try again
                continue
            padded.add(chosen)
            used_bogus[chosen] = bogus_map[chosen]
            fake_insert_count += 1

        padded_results.append(padded)
        padded_sizes.append(len(padded))

    # metrics
    true_sizes_arr = numpy.array(true_sizes, dtype=float)
    padded_sizes_arr = numpy.array(padded_sizes, dtype=float)

    # Avoid division by zero: if there are zero non-empty queries, set metrics to zeros/NaN
    nonzero_mask = true_sizes_arr > 0
    if nonzero_mask.sum() == 0:
        avg_true_size = 0.0
        avg_padded_size = float(numpy.mean(padded_sizes_arr)) if padded_sizes_arr.size>0 else 0.0
    else:
        avg_true_size = float(true_sizes_arr[nonzero_mask].mean())
        avg_padded_size = float(padded_sizes_arr[nonzero_mask].mean())

    bandwidth_ratio = (avg_padded_size / avg_true_size) if avg_true_size > 0 else float('inf')

    total_true_items = int(true_sizes_arr.sum())
    total_padded_items = int(padded_sizes_arr.sum())
    total_fake_inserted = int(fake_insert_count)

    index_size_overhead = (total_padded_items - total_true_items) / total_true_items if total_true_items > 0 else float('inf')

    per_query_stats = {
        "true_mean": float(numpy.mean(true_sizes_arr)) if true_sizes_arr.size>0 else 0.0,
        "true_std": float(numpy.std(true_sizes_arr)) if true_sizes_arr.size>0 else 0.0,
        "padded_mean": float(numpy.mean(padded_sizes_arr)) if padded_sizes_arr.size>0 else 0.0,
        "padded_std": float(numpy.std(padded_sizes_arr)) if padded_sizes_arr.size>0 else 0.0,
        "num_queries": int(len(results))
    }

    metrics = {
        "avg_true_size": avg_true_size,
        "avg_padded_size": avg_padded_size,
        "bandwidth_ratio": bandwidth_ratio,
        "total_true_items": total_true_items,
        "total_padded_items": total_padded_items,
        "total_fake_inserted": total_fake_inserted,
        "index_size_overhead": index_size_overhead,
        "per_query_stats": per_query_stats
    }

    # print summary
    print("Padding summary:")
    print(f"  queries: {len(results)}")
    print(f"  avg true size (non-empty): {avg_true_size:.3f}")
    print(f"  avg padded size (non-empty): {avg_padded_size:.3f}")
    print(f"  bandwidth ratio (avg_padded/avg_true): {bandwidth_ratio:.3f}")
    print(f"  total true items (sum over queries): {total_true_items}")
    print(f"  total padded items (sum over queries): {total_padded_items}")
    print(f"  total fake inserted (counting duplicates across queries): {total_fake_inserted}")
    print(f"  index size overhead (relative): {index_size_overhead:.3f}")

    return padded_results, used_bogus


'''
padding to mutiple of x
'''


def pad_results_to_multiple_of_x(
    results: List[Set],
    query_rects: List[Tuple[float,float,float,float]],
    x: int,
    bogus_points: List[str],
    bogus_map: Dict[str, Tuple[float,float]]
) -> Tuple[List[Set], Dict[str, Tuple[float,float]], Dict]:
    """
    对每个查询结果集合做 padding:
    - target size = ceil(len(res)/x) * x (即 pad 到 x 的倍数)
    - 只从 bogus_map 中坐标落在该 query range 的假点中选（优先）
    - 允许同一个 fake token 在不同 query 之间重复使用
    返回:
      padded_results, used_bogus_map, metrics_dict
    """
    padded_results: List[Set] = []
    used_bogus: Dict[str, Tuple[float,float]] = {}

    true_sizes = []
    padded_sizes = []
    fake_insert_count = 0

    for res, rect in zip(results, query_rects):
        res_set = set(res) if res is not None else set()

        true_sizes.append(len(res_set))

        if len(res_set) == 0:
            padded_results.append(set())
            padded_sizes.append(0)
            continue

        # 目标大小：向上取整到 x 的倍数
        target_size = math.ceil(len(res_set) / x) * x
        padded = set(res_set)

        xmin, xmax, ymin, ymax = rect

        candidates = [
            fid for fid, coord in bogus_map.items()
            if xmin <= coord[0] <= xmax and ymin <= coord[1] <= ymax
        ]

        while len(padded) < target_size:
            if candidates:
                chosen = random.choice(candidates)
                candidates.remove(chosen)
            else:
                chosen = random.choice(bogus_points)
            if chosen in padded:
                continue
            padded.add(chosen)
            used_bogus[chosen] = bogus_map[chosen]
            fake_insert_count += 1

        padded_results.append(padded)
        padded_sizes.append(len(padded))

    # metrics
    true_sizes_arr = numpy.array(true_sizes, dtype=float)
    padded_sizes_arr = numpy.array(padded_sizes, dtype=float)

    nonzero_mask = true_sizes_arr > 0
    if nonzero_mask.sum() == 0:
        avg_true_size = 0.0
        avg_padded_size = float(padded_sizes_arr.mean()) if padded_sizes_arr.size > 0 else 0.0
    else:
        avg_true_size = float(true_sizes_arr[nonzero_mask].mean())
        avg_padded_size = float(padded_sizes_arr[nonzero_mask].mean())

    bandwidth_ratio = (avg_padded_size / avg_true_size) if avg_true_size > 0 else float("inf")

    total_true_items = int(true_sizes_arr.sum())
    total_padded_items = int(padded_sizes_arr.sum())
    total_fake_inserted = int(fake_insert_count)

    index_size_overhead = (
        (total_padded_items - total_true_items) / total_true_items if total_true_items > 0 else float("inf")
    )

    per_query_stats = {
        "true_mean": float(true_sizes_arr.mean()) if true_sizes_arr.size > 0 else 0.0,
        "true_std": float(true_sizes_arr.std()) if true_sizes_arr.size > 0 else 0.0,
        "padded_mean": float(padded_sizes_arr.mean()) if padded_sizes_arr.size > 0 else 0.0,
        "padded_std": float(padded_sizes_arr.std()) if padded_sizes_arr.size > 0 else 0.0,
        "num_queries": int(len(results)),
    }

    metrics = {
        "avg_true_size": avg_true_size,
        "avg_padded_size": avg_padded_size,
        "bandwidth_ratio": bandwidth_ratio,
        "total_true_items": total_true_items,
        "total_padded_items": total_padded_items,
        "total_fake_inserted": total_fake_inserted,
        "index_size_overhead": index_size_overhead,
        "per_query_stats": per_query_stats,
    }

    print("Padding summary (multiple-of-x):")
    print(f"  queries: {len(results)}")
    print(f"  avg true size (non-empty): {avg_true_size:.3f}")
    print(f"  avg padded size (non-empty): {avg_padded_size:.3f}")
    print(f"  bandwidth ratio (avg_padded/avg_true): {bandwidth_ratio:.3f}")
    print(f"  total true items: {total_true_items}")
    print(f"  total padded items: {total_padded_items}")
    print(f"  total fake inserted: {total_fake_inserted}")
    print(f"  index size overhead: {index_size_overhead:.3f}")

    return padded_results, used_bogus


'''
response hiding version:
using canonical ranges to cover each query, and return union of responses
'''


def generate_canonical_ranges(N0, N1):
    """
    生成 2D canonical ranges (power-of-2 尺寸的矩形)
    """
    canonical_ranges = []
    lengths0 = [2**k for k in range(int(math.ceil(math.log2(N0)))+1)]
    lengths1 = [2**k for k in range(int(math.ceil(math.log2(N1)))+1)]

    for l0 in lengths0:
        for l1 in lengths1:
            for i in range(0, N0, l0):
                for j in range(0, N1, l1):
                    canonical_ranges.append(
                        (i, min(i+l0-1, N0-1), j, min(j+l1-1, N1-1))
                    )
    return canonical_ranges


def precompute_canonical_responses(canonical_ranges, points, map_points_to_coordinates):
    """
    预计算每个 canonical range 覆盖的点集
    """
    precomputed = {}
    for rect in canonical_ranges:
        min0, max0, min1, max1 = rect
        res = []
        for p in points:
            x, y = map_points_to_coordinates[p]
            if min0 <= x <= max0 and min1 <= y <= max1:
                res.append(p)
        precomputed[rect] = set(res)
    return precomputed


def find_covering_canonical_range(canonical_ranges, query):
    """
    找到最小的 canonical range 覆盖 query
    """
    qmin0, qmax0, qmin1, qmax1 = query
    candidates = []
    for rect in canonical_ranges:
        min0, max0, min1, max1 = rect
        if (min0 <= qmin0 and max0 >= qmax0 and
            min1 <= qmin1 and max1 >= qmax1):
            candidates.append(rect)

    def area(r): return (r[1]-r[0]+1) * (r[3]-r[2]+1)
    return min(candidates, key=area)


def response_hiding_with_bandwidth(
    queries, canonical_ranges, precomputed_responses, map_points_to_coordinates
):
    """
    批量处理一组 queries：
    - 对每个 query 找到最小的覆盖 canonical range
    - 返回 superset (服务器返回) + true set (客户端过滤)
    - 计算带宽开销统计
    """
    supersets = []
    true_sets = []
    ratios = []
    true_sizes = []
    superset_sizes = []

    seen_ranges = set()

    for q in queries:
        if q in seen_ranges:
            continue
        seen_ranges.add(q)

        # 服务器端：找到覆盖 canonical range
        covering = find_covering_canonical_range(canonical_ranges, q)
        superset = precomputed_responses[covering]
        supersets.append(superset)

        # 客户端：过滤
        true_resp, _ = get_actual_query_resps_after_sampling(
            [q], list(superset), map_points_to_coordinates
        )
        true_set = true_resp[0]
        true_sets.append(true_set)

        # 带宽统计
        true_size = len(true_set)
        sup_size = len(superset)
        true_sizes.append(true_size)
        superset_sizes.append(sup_size)

        if true_size > 0:
            ratios.append(sup_size / true_size)
        else:
            # 空查询时，ratio 定义为 1
            ratios.append(1.0)

    stats = {
        "avg_true_size": sum(true_sizes)/len(true_sizes),
        "avg_superset_size": sum(superset_sizes)/len(superset_sizes),
        "avg_ratio": sum(ratios)/len(ratios),
        "std_ratio": statistics.pstdev(ratios),
    }
    print("Response hiding stats:", stats)

    return supersets