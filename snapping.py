import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools
import math
import random

'''
Simulated annealing and snapping
'''

def snap_to_integer(coords):
    """Snap points to nearby integers after removing a global median offset."""
    offsets = coords - np.round(coords)
    median_offset = np.median(offsets, axis=0)

    return np.round(coords - median_offset)


def snap_to_integer_hungary(coords, grid_shape):
    """Assign points to integer grid cells using the Hungarian algorithm (min total distance)."""
    grid = np.array(list(itertools.product(range(grid_shape[0]), range(grid_shape[1]))))
    cost_matrix = np.linalg.norm(coords[:, None] - grid[None], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return grid[col_ind]


def probabilistic_snap(coords, grid_shape):
    """Stochastically round each coordinate up or down based on its fractional part."""
    fractional = coords - np.floor(coords)
    prob_x = fractional[:, 0]
    prob_y = fractional[:, 1]

    # decide direction of snapping based on probability
    snapped_x = np.where(
        np.random.rand(len(coords)) < prob_x,
        np.ceil(coords[:, 0]),
        np.floor(coords[:, 0])
    )
    snapped_y = np.where(
        np.random.rand(len(coords)) < prob_y,
        np.ceil(coords[:, 1]),
        np.floor(coords[:, 1])
    )

    snapped = np.column_stack([snapped_x, snapped_y])
    return np.clip(snapped, [0, 0], [grid_shape[0] - 1, grid_shape[1] - 1])


def simulated_annealing_snap(coords, grid_shape):
    """Resolve grid assignment conflicts via simulated annealing to reduce overlaps and distance."""
    current_solution = np.round(coords).astype(int)
    current_energy = _calculate_energy(current_solution, coords)

    T = 1.0
    T_min = 1e-3
    alpha = 0.9

    while T > T_min:
        new_solution = _perturb(current_solution.copy(), grid_shape)
        new_energy = _calculate_energy(new_solution, coords)

        if new_energy < current_energy or \
                random.random() < math.exp(-(new_energy - current_energy) / T):
            current_solution = new_solution
            current_energy = new_energy

        T *= alpha

    return current_solution


def _calculate_energy(solution, original):
    """Energy = total displacement distance + overlap penalty (quadratic)."""
    offset_energy = np.sum(np.linalg.norm(solution - original, axis=1))

    _, counts = np.unique(solution, axis=0, return_counts=True)
    conflict_penalty = np.sum(counts[counts > 1] ** 2) * 100

    return offset_energy + conflict_penalty


def _perturb(solution, grid_shape):
    """Randomly move one point to a neighboring valid grid cell."""
    idx = random.randint(0, len(solution) - 1)
    old = solution[idx]

    candidates = [
        (old[0] + dx, old[1] + dy)
        for dx in [-1, 0, 1] for dy in [-1, 0, 1]
        if 0 <= old[0] + dx < grid_shape[0] and 0 <= old[1] + dy < grid_shape[1]
    ]
    solution[idx] = random.choice(candidates)
    return solution