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
t-SNE with a light density regularization (mean/variance matching)
'''

def _density_aware_loss(params, P, degrees_of_freedom, n_samples,
                       n_components, true_mean, true_var, lambda_=0.025,
                       bandwidth=2.0, skip_num_points=0, compute_error=True, iter=0):
    """
    t-SNE loss with a simple density constraint.

    Parameters
    ----------
    params : ndarray, shape (n_samples * n_components,)
        Flattened embedding coordinates.
    P : ndarray
        High-dimensional joint probabilities.
    degrees_of_freedom : float
        Degrees of freedom of the Student-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Embedding dimensionality.
    true_mean : float
        Target mean of the embedding (used for regularization).
    true_var : float
        Target variance of the embedding (used for regularization).
    lambda_ : float, default=0.025
        Strength of the density regularization.
    bandwidth : float, default=2.0
        Reserved for KDE-based constraints (not used here).
    skip_num_points : int
        Number of points to skip in gradient computation.
    compute_error : bool
        Whether to compute and return the KL divergence.
    iter : int
        Current iteration index (used to occasionally apply the constraint).
    """
    X_embedded = params.reshape(n_samples, n_components)
    N = n_samples * n_components

    # 1) Original t-SNE KL divergence and gradient
    kl_divergence, grad = _loss_function(
        params, P, degrees_of_freedom, n_samples, n_components,
        compute_error=compute_error, skip_num_points=0
    )

    # 2) Density constraint term (mean/variance matching)
    if lambda_ > 0 and iter % 2 == 0:
        embedded_mean = X_embedded.mean()
        embedded_var = X_embedded.var()

        # Add mean/variance supervision terms
        mean_diff = embedded_mean - true_mean
        var_diff = embedded_var - true_var
        mean_loss = mean_diff ** 2
        var_loss = var_diff ** 2

        if compute_error:
            kl_divergence = kl_divergence + lambda_ * var_loss

        # 1) Gradient of mean loss (identical for all parameters)
        d_mean_loss_dx = (2.0 / N) * mean_diff

        # 2) Gradient of variance loss
        flat_embedded = X_embedded.flatten()
        d_var_loss_dx = (4.0 / N) * var_diff * (flat_embedded - embedded_mean)

        # print("d_mean_loss_dx", lambda_ * d_mean_loss_dx.mean())
        # print("d_var_loss_dx", lambda_ * d_var_loss_dx.mean())
        # print("grad", grad.mean())

        # Optionally include mean term as well
        # grad += lambda_ * d_mean_loss_dx
        grad += lambda_ * d_var_loss_dx

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


class AggregateTSNE(TSNE):
    def __init__(self, true_mean=None, true_var=None, alpha=0.5, lambda_=0.2, init=None, **kwargs):
        self.true_mean = true_mean  # target mean
        self.true_var = true_var  # target variance
        self.alpha = alpha  # density weight coefficient
        self.lambda_ = lambda_  # density regularization strength
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
            "kwargs": dict(skip_num_points=skip_num_points, true_mean=self.true_mean, true_var=self.true_var,),
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
        # Estimate density in the embedding space
        kde = KernelDensity(bandwidth=0.3).fit(embedding)
        reco_densities = np.exp(kde.score_samples(embedding))

        # Density matching gradient
        grad = 2 * self.lambda_ * (reco_densities - self.densities)[:, None] * \
               (embedding - np.mean(embedding, axis=0))

        # Update embedding (tunable learning rate)
        embedding -= 0.01 * grad
