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


def _density_aware_loss(params, P, degrees_of_freedom, n_samples,
                       n_components, densities, lambda_=0.00045,
                       bandwidth=2.0, skip_num_points=0, compute_error=True, iter=0):
    """
    t-SNE loss function with density constraints

    Parameters:
        params: Flattened embedding coordinates (n_samples * n_components,)
        P: High-dimensional similarity matrix
        densities: Precomputed density array (n_samples,)
        lambda_: Density regularization strength
        bandwidth: Kernel density estimation bandwidth
    """
    X_embedded = params.reshape(n_samples, n_components)

    # 1. Calculate original KL divergence and gradient
    kl_divergence, grad = _loss_function(
        params, P, degrees_of_freedom, n_samples, n_components,
        compute_error=compute_error, skip_num_points=0
    )

    # 2. Calculate density constraint term
    if lambda_ > 0 and iter >= 150 and iter % 10 == 0:

        '''
        Method 4: Spearman '''
        X_norm = (X_embedded - X_embedded.mean(0)) / (X_embedded.std(0) + 1e-8)

        # Kernel density estimation (same parameters)
        kde = KernelDensity(bandwidth=bandwidth, kernel='cosine')
        kde.fit(X_norm)
        log_dens = kde.score_samples(X_norm)
        lowdim_density = np.exp(log_dens)

        # Same normalization method
        lowdim_density = (lowdim_density - lowdim_density.min()) / \
                         (lowdim_density.max() - lowdim_density.min() + 1e-8)
        highdim_density = (densities - densities.min()) / \
                          (densities.max() - densities.min() + 1e-8)

        # Calculate Spearman rank loss
        highdim_rank = np.argsort(highdim_density)
        lowdim_rank = np.argsort(lowdim_density)

        # Safe Spearman calculation
        with np.errstate(invalid='ignore'):
            rank_corr = np.corrcoef(highdim_rank, lowdim_rank)[0, 1]
        rank_loss = -np.nan_to_num(rank_corr, nan=0.0)  # Goal is to minimize negative correlation

        # Vectorized gradient calculation
        n_samples = X_embedded.shape[0]
        k_neighbors = min(30, n_samples - 1)

        # Calculate rank difference matrix for all points
        rank_diff = np.abs(lowdim_rank[:, None] - highdim_rank[None, :])

        # Find k nearest points by rank for each point
        closest_indices = np.argpartition(rank_diff, k_neighbors, axis=1)[:, :k_neighbors]

        # Prepare position differences for all points
        X_expanded = X_embedded[:, None, :]  # shape: (n_samples, 1, dim)
        X_broadcast = X_embedded[None, :, :]  # shape: (1, n_samples, dim)
        dX = X_broadcast - X_expanded  # shape: (n_samples, n_samples, dim)

        # Get relevant differences using advanced indexing
        selected_dX = dX[np.arange(n_samples)[:, None], closest_indices]  # shape: (n_samples, k_neighbors, dim)

        # Calculate rank difference weights
        rank_diffs = (highdim_rank[None, :] - lowdim_rank[:, None])[np.arange(n_samples)[:, None], closest_indices]

        # Calculate gradient steps
        steps = rank_diffs[:, :, None] * selected_dX  # shape: (n_samples, k_neighbors, dim)
        norm_steps = np.linalg.norm(steps, axis=2, keepdims=True) + 1e-12
        normalized_steps = steps / norm_steps

        # Aggregate gradients
        grad_dens = np.sum(normalized_steps, axis=1)

        # Merge gradients (note inverse normalization)
        std = X_embedded.std(0) + 1e-8
        grad_dens = grad_dens / std

        # Combine gradients
        grad += lambda_ * grad_dens.ravel()

        # Calculate total loss
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
        self.alpha = alpha  # Density Weighting Coefficient
        self.lambda_ = lambda_  # Density Regular Strength
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

    def _apply_density_constraint(self, embedding):
        # Calculate Embedding Space Density
        kde = KernelDensity(bandwidth=0.3).fit(embedding)
        reco_densities = np.exp(kde.score_samples(embedding))

        # Density-Matching Gradient
        grad = 2 * self.lambda_ * (reco_densities - self.densities)[:, None] * \
               (embedding - np.mean(embedding, axis=0))

        # Update embedded results
        embedding -= 0.01 * grad  # adjust learning rate
