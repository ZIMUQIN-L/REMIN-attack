import math
from typing import Dict, Tuple
import numpy as np


def reciprocal(freq_table):
    """1/sqrt(freq+1)"""
    distance_table = {pair: 1 / math.sqrt(freq + 1) for pair, freq in freq_table.items()}
    return distance_table


def origin_reciprocal(freq_table):
    """1/(freq + 1)"""
    distance_table = {pair: 1 / (freq + 1) for pair, freq in freq_table.items()}
    return distance_table


def logarithmic(freq_table: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], float]:
    """1/log(freq + e)"""
    distance_table = {pair: 1 / math.log(freq + math.e) for pair, freq in freq_table.items()}
    return distance_table


def jaccard(freq_table: Dict[Tuple[str, str], int], total_responses: Dict[str, int]) -> Dict[Tuple[str, str], float]:
    """Jaccard距离"""
    distance = {}
    for (a, b), freq in freq_table.items():
        jaccard_sim = freq / (total_responses[a] + total_responses[b] - freq)
        distance[(a, b)] = 1 - jaccard_sim
    return distance


def gaussian_kernel(freq_table, sigma=1.0):
    """Gaussian Kernel"""
    distance_table = {pair: np.exp(- (freq ** 2) / (2 * sigma ** 2)) for pair, freq in freq_table.items()}
    return distance_table


def cooccurrence_euclidean(freq_table, total_responses):
    distance = {}
    for (a, b), freq in freq_table.items():
        da = total_responses[a] - freq
        db = total_responses[b] - freq
        dist = math.sqrt(da**2 + db**2)
        distance[(a, b)] = dist
    return distance
