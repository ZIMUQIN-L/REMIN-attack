import numpy as np
import matplotlib.pyplot as plt
import math

import random


def get_rho(points, N0, N1):
    rho_dict = {}
    r = 0
    for p in points:
        rho_dict[(p)] = p[0] * p[1] * (N0 - p[0]) * (N1 - p[1])
        for q in points:
            if p != q:
                min_0 = min(p[0], q[0])
                max_0 = max(p[0], q[0])
                min_1 = min(p[1], q[1])
                max_1 = max(p[1], q[1])
                rho_dict[(p, q)] = min_0 * min_1 * (N0 - max_0) * (N1 - max_1)

    return rho_dict


def all_points(N0, N1):
    points = []
    for i in range(1, N0):
        for j in range(1, N1):
            points.append((i, j))
    return points


def random_points(N0, N1, l):
    points = []

    if l > (N0 - 2) * (N1 - 2):
        l = (N0 - 2) * (N1 - 2)

    while len(points) <= l:
        x = random.randint(1, N0 - 2)
        y = random.randint(1, N1 - 2)

        if (x, y) not in points:
            points.append((x, y))

    return points



def rho(N0, N1, P, Q, K):
    min0 = min([P[0], Q[0], K[0]])
    min1 = min([P[1], Q[1], K[1]])
    max0 = max([P[0], Q[0], K[0]])
    max1 = max([P[1], Q[1], K[1]])
    return (min0 * min1 * (N0 + 1 - max0) * (N1 + 1 - max1))


def plot_linear(points, new_points, new_points2, title, show, N0, N1):
    X = []
    Y = []
    for i in new_points:
        X.append(i[0])
        Y.append(i[1])
    X2 = []
    Y2 = []
    for i in new_points2:
        X2.append(i[0])
        Y2.append(i[1])
    X3 = []
    Y3 = []
    for i in points:
        X3.append(i[0])
        Y3.append(i[1])
    plt.ylim(0, N1 + 1)
    plt.xlim(0, N0 + 1)
    plt.plot(X2, Y2, "o")
    plt.plot(X, Y, "*")
    plt.plot(X3, Y3, "+")
    plt.savefig(title)

    if show:
        plt.show()
    plt.clf()


def get_random_database(N0, N1, max_points, plaintext=False):
    map_to_original = {}
    points = []
    ori_points = []
    for i in range(1, N0):
        for j in range(1, N1):
            if random.randrange(100) < 40:
                continue
            repeats = int(1 + (max_points - 1) * random.random())
            for num in range(repeats):
                if plaintext:
                    search_token = (i, j)
                else:
                    search_token = random.randrange(10000000)
                map_to_original[search_token] = (i, j)
                points.append(search_token)
                ori_points.append((i, j))

    return points, map_to_original, ori_points


def get_random_database_with_outlier(N0, N1, max_points, plaintext=False):
    map_to_original = {}
    points = []
    ori_points = []

    # Generate main point set (concentrated in a certain area, e.g., center)
    main_range = (N0 // 4, 3 * N0 // 4, N1 // 4, 3 * N1 // 4)  # Range of main point set

    for i in range(main_range[0], main_range[1]):
        for j in range(main_range[2], main_range[3]):
            if random.randrange(100) < 60:  # x% probability to skip
                continue
            repeats = int(1 + (max_points - 1) * random.random())
            for num in range(repeats):
                if plaintext:
                    search_token = (i, j)
                else:
                    search_token = random.randrange(10000000)
                map_to_original[search_token] = (i, j)
                points.append(search_token)
                ori_points.append((i, j))

    # Add an outlier point (far from main point set)
    outlier_i = random.choice([0, N0 - 1])  # Outlier at top-left or bottom-right corner
    outlier_j = random.choice([0, N1 - 1])
    if plaintext:
        outlier_token = (outlier_i, outlier_j)
    else:
        outlier_token = random.randrange(10000000)
    map_to_original[outlier_token] = (outlier_i, outlier_j)
    points.append(outlier_token)
    ori_points.append((outlier_i, outlier_j))

    if plaintext:
        outlier_token = (outlier_i-1, outlier_j)
    else:
        outlier_token = random.randrange(10000000)
    map_to_original[outlier_token] = (outlier_i-1, outlier_j)
    points.append(outlier_token)
    return points, map_to_original, ori_points