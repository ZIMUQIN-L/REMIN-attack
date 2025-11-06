from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import _utils
from scipy.spatial.distance import pdist
import numpy as np
from scipy import linalg
from sklearn.utils import check_random_state
from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform
from time import time
from sklearn.neighbors import KernelDensity

MACHINE_EPSILON = np.finfo(np.double).eps

'''
note: 0.0001是一个还不错的参数？
'''

def _density_aware_loss(params, P, degrees_of_freedom, n_samples,
                       n_components, densities, lambda_=0.00045,
                       bandwidth=2.0, skip_num_points=0, compute_error=True, iter=0):
    """
    带密度约束的t-SNE损失函数

    参数：
        params: 展平的嵌入坐标 (n_samples * n_components,)
        P: 高维相似度矩阵
        densities: 预先计算的密度数组 (n_samples,)
        lambda_: 密度正则强度
        bandwidth: 核密度估计带宽
    """
    X_embedded = params.reshape(n_samples, n_components)

    # 1. 计算原始KL散度和梯度
    kl_divergence, grad = _loss_function(
        params, P, degrees_of_freedom, n_samples, n_components,
        compute_error=compute_error, skip_num_points=0
    )

    # 2. 计算密度约束项
    if lambda_ > 0 and iter >= 150 and iter % 10 == 0:
        # 核密度估计
        # X_embedded_norm = (X_embedded - X_embedded.mean(0)) / X_embedded.std(0)
        # kde = KernelDensity(bandwidth=bandwidth).fit(X_embedded_norm)
        #
        # log_dens = kde.score_samples(X_embedded_norm)
        # computed_densities = np.exp(log_dens)
        # computed_densities = (computed_densities - computed_densities.min()) / (computed_densities.max() - computed_densities.min() + 1e-8)

        '''
        方法1：密度梯度计算（循环）
        '''
        # # 计算密度梯度
        # grad_dens = np.zeros_like(X_embedded)
        # dist_matrix = squareform(pdist(X_embedded))
        # kernel_vals = np.exp(-0.5 * (dist_matrix / bandwidth) ** 2)
        #
        # for i in range(n_samples):
        #     # 核密度梯度分量
        #     grad_kernel = (X_embedded[i] - X_embedded) * kernel_vals[i][:, None]
        #     grad_dens[i] = 2 * (densities[i] - computed_densities[i]) * \
        #                    np.sum(grad_kernel, axis=0) / (bandwidth ** 2)

        ''' 
        方法2：密度梯度计算（向量化）'''
        # # 密度梯度计算（向量化）
        # diff = X_embedded[:, np.newaxis, :] - X_embedded[np.newaxis, :, :]  # (n,n,2)
        # dist_sq = np.sum(diff ** 2, axis=-1)  # (n,n)
        # kernel_vals = np.exp(-0.5 * dist_sq / (bandwidth ** 2))  # (n,n)
        #
        # # 计算梯度系数
        # coeff = 2 * (densities - computed_densities) / (bandwidth ** 2)  # (n,)
        #
        # # 向量化梯度计算（等效于原循环）
        # grad_dens = np.einsum('i,ij,ijk->ik', coeff, kernel_vals, diff)  # (n,2)

        '''
        方法3：spearman'''
        # X_norm = (X_embedded - X_embedded.mean(0)) / (X_embedded.std(0) + 1e-8)
        #
        # # 核密度估计（相同参数）
        # kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        # kde.fit(X_norm)
        # log_dens = kde.score_samples(X_norm)
        # lowdim_density = np.exp(log_dens)
        #
        # # 同款归一化方式
        # lowdim_density = (lowdim_density - lowdim_density.min()) / \
        #                  (lowdim_density.max() - lowdim_density.min() + 1e-8)
        # highdim_density = (densities - densities.min()) / \
        #                   (densities.max() - densities.min() + 1e-8)
        #
        # # 计算Spearman排名损失
        # highdim_rank = np.argsort(highdim_density)
        # lowdim_rank = np.argsort(lowdim_density)
        #
        # # 安全的Spearman计算
        # with np.errstate(invalid='ignore'):
        #     rank_corr = np.corrcoef(highdim_rank, lowdim_rank)[0, 1]
        # rank_loss = -np.nan_to_num(rank_corr, nan=0.0)  # 目标是最小化负相关
        #
        # # 梯度计算：推动点向同排名邻居移动
        # grad_dens = np.zeros_like(X_embedded)
        # dist_matrix = np.sqrt(np.sum((X_norm[:, None] - X_norm) ** 2, axis=-1))
        # np.fill_diagonal(dist_matrix, np.inf)  # 忽略自身
        #
        # for i in range(n_samples):
        #     # 找到排名最接近的k个点（非空间最近邻）
        #     rank_diff = np.abs(lowdim_rank - highdim_rank[i])
        #     closest = np.argpartition(rank_diff, min(30, n_samples - 1))[:min(30, n_samples - 1)]
        #
        #     for j in closest:
        #         dx = X_embedded[j] - X_embedded[i]
        #         step = (highdim_rank[i] - lowdim_rank[i]) * dx
        #         grad_dens[i] += step / (np.linalg.norm(step) + 1e-12)  # 稳定化

        '''
        方法4：spearman'''
        X_norm = (X_embedded - X_embedded.mean(0)) / (X_embedded.std(0) + 1e-8)

        # 核密度估计（相同参数）
        kde = KernelDensity(bandwidth=bandwidth, kernel='cosine')
        kde.fit(X_norm)
        log_dens = kde.score_samples(X_norm)
        lowdim_density = np.exp(log_dens)

        # 同款归一化方式
        lowdim_density = (lowdim_density - lowdim_density.min()) / \
                         (lowdim_density.max() - lowdim_density.min() + 1e-8)
        highdim_density = (densities - densities.min()) / \
                          (densities.max() - densities.min() + 1e-8)

        # 计算Spearman排名损失
        highdim_rank = np.argsort(highdim_density)
        lowdim_rank = np.argsort(lowdim_density)

        # 安全的Spearman计算
        with np.errstate(invalid='ignore'):
            rank_corr = np.corrcoef(highdim_rank, lowdim_rank)[0, 1]
        rank_loss = -np.nan_to_num(rank_corr, nan=0.0)  # 目标是最小化负相关

        # 向量化梯度计算
        n_samples = X_embedded.shape[0]
        k_neighbors = min(30, n_samples - 1)

        # 计算所有点的排名差异矩阵
        rank_diff = np.abs(lowdim_rank[:, None] - highdim_rank[None, :])

        # 找到每个点的k个排名最接近的点
        closest_indices = np.argpartition(rank_diff, k_neighbors, axis=1)[:, :k_neighbors]

        # 准备所有点的位置差异
        X_expanded = X_embedded[:, None, :]  # shape: (n_samples, 1, dim)
        X_broadcast = X_embedded[None, :, :]  # shape: (1, n_samples, dim)
        dX = X_broadcast - X_expanded  # shape: (n_samples, n_samples, dim)

        # 使用高级索引获取相关差异
        selected_dX = dX[np.arange(n_samples)[:, None], closest_indices]  # shape: (n_samples, k_neighbors, dim)

        # 计算排名差异权重
        rank_diffs = (highdim_rank[None, :] - lowdim_rank[:, None])[np.arange(n_samples)[:, None], closest_indices]

        # 计算梯度步长
        steps = rank_diffs[:, :, None] * selected_dX  # shape: (n_samples, k_neighbors, dim)
        norm_steps = np.linalg.norm(steps, axis=2, keepdims=True) + 1e-12
        normalized_steps = steps / norm_steps

        # 聚合梯度
        grad_dens = np.sum(normalized_steps, axis=1)

        # 合并梯度（注意逆标准化）
        std = X_embedded.std(0) + 1e-8
        grad_dens = grad_dens / std

        # 合并梯度
        grad += lambda_ * grad_dens.ravel()

        # 计算总损失
        if compute_error:
            # kl_divergence += lambda_ * np.sum((densities - computed_densities) ** 2)
            kl_divergence += lambda_ * rank_loss

    return kl_divergence, grad


def _loss_function(params, P, degrees_of_freedom, n_samples, n_components, density=None,
                   skip_num_points=0, compute_error=True):
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(
            P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def _joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose)
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P


def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs, iter=i)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i


class DensityAwareTSNE(TSNE):
    def __init__(self, densities=None, alpha=0.5, lambda_=0.2, init=None, **kwargs):
        self.densities = densities
        self.alpha = alpha  # 密度权重系数
        self.lambda_ = lambda_  # 密度正则强度
        super().__init__(**kwargs)
        self.init = init

    def _fit(self, X, skip_num_points=0):
        random_state = check_random_state(self.random_state)
        n_samples = X.shape[0]
        neighbors_nn = None

        distances = X
        distances **= 2
        P = _joint_probabilities(distances, self.perplexity, self.verbose)
        assert np.all(np.isfinite(P)), "All probabilities should be finite"
        assert np.all(P >= 0), "All probabilities should be non-negative"
        assert np.all(P <= 1), ("All probabilities should be less "
                                "or then equal to one")

        print("Computing pairwise distances...", isinstance(self.init, np.ndarray))
        if isinstance(self.init, np.ndarray):
            print("Using precomputed init")
            X_embedded = self.init
        else:
            X_embedded = 1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)

        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(P, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points)

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded,
              neighbors=None, skip_num_points=0):
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points, densities=self.densities,),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }

        # obj_func = _loss_function
        obj_func = _density_aware_loss

        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                      **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                          **opt_args)
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    def fit_transform(self, X, y=None):
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

        # print("Custom _fit() is being called!")
        # if self.densities is None:
        #     kde = KernelDensity(bandwidth=0.3).fit(X)
        #     self.densities = np.exp(kde.score_samples(X))
        #     self.densities = (self.densities - self.densities.min()) / \
        #                      (self.densities.max() - self.densities.min() + 1e-8)
        #
        # # 运行标准t-SNE流程
        # embedding = super()._fit(X)
        #
        # # 添加密度正则项
        # if self.lambda_ > 0:
        #     self._apply_density_constraint(embedding)
        #
        # return embedding

    def _apply_density_constraint(self, embedding):
        # 计算嵌入空间密度
        kde = KernelDensity(bandwidth=0.3).fit(embedding)
        reco_densities = np.exp(kde.score_samples(embedding))

        # 密度匹配梯度
        grad = 2 * self.lambda_ * (reco_densities - self.densities)[:, None] * \
               (embedding - np.mean(embedding, axis=0))

        # 更新嵌入结果
        embedding -= 0.01 * grad  # 学习率可调
