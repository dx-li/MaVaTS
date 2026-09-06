"""Matrix factor estimators with a common orthonormal loading convention.

Observations have shape ``(time, rows, columns)``. Loading matrices describe
subspaces: their signs and rotations are not identifiable. Compare projectors
or reconstructed signals, not individual loading entries.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import as_series, finite_scalar, positive_int


def _factor_ranks(ranks, shape):
    if ranks is None:
        return (None,) * len(shape)
    try:
        ranks = tuple(ranks)
    except TypeError as exc:
        raise ValueError("ranks must be a sequence, one per spatial mode") from exc
    if len(ranks) != len(shape):
        raise ValueError("ranks must have one entry per spatial mode")
    result = tuple(
        None if rank is None else positive_int(rank, "rank") for rank in ranks
    )
    if any(rank is not None and rank > dim for rank, dim in zip(result, shape)):
        raise ValueError("ranks cannot exceed spatial dimensions")
    return result


def _canonical_signs(v):
    v = v.copy()
    idx = np.argmax(np.abs(v), axis=0)
    v *= np.where(v[idx, np.arange(v.shape[1])] < 0, -1.0, 1.0)
    return v


def eigenvalue_ratio(eigenvalues, max_rank=None):
    """Estimate rank from the largest adjacent gap in descending eigenvalues.

    Accepts eigenvalues in either order. Searches up to half the dimension by
    default, as in Wang, Liu and Chen (2019). A relative numerical floor prevents
    dividing by roundoff at an exact rank deficiency. A zero spectrum returns
    rank one; this heuristic does not test for absence of factors. Specify ranks
    explicitly for small dimensions or weak signals.

    References
    ----------
    Wang, Liu and Chen (2019), Factor Models for Matrix-Valued
    High-Dimensional Time Series, Section 3.
    https://doi.org/10.1016/j.jeconom.2018.09.013
    The numerical floor and zero-spectrum fallback are implementation
    conventions, not a calibrated test of the absence of factors.
    """
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("eigenvalues must be a nonempty finite vector")
    values = np.sort(np.maximum(values, 0))[::-1]
    if values.size == 1:
        return 1
    upper = (
        max(1, values.size // 2)
        if max_rank is None
        else positive_int(max_rank, "max_rank")
    )
    if upper >= values.size:
        raise ValueError("max_rank must be smaller than the spectrum length")
    if values[0] == 0:
        return 1
    values = values / values[0]
    floor = np.finfo(float).eps * values.size
    ratios = (values[1 : upper + 1] + floor) / (values[:upper] + floor)
    return int(np.argmin(ratios)) + 1


def _eigenspace(moment, rank):
    values, vectors = np.linalg.eigh((moment + moment.T) * 0.5)
    values = np.maximum(values[::-1], 0)
    if rank is None:
        rank = eigenvalue_ratio(values)
    return _canonical_signs(vectors[:, ::-1][:, :rank]), values


def _mode_product(X, matrix, mode):
    """Multiply a spatial mode, with mode zero denoting the first after time."""
    return np.moveaxis(np.tensordot(X, matrix.T, axes=(mode + 1, 0)), -1, mode + 1)


def _project(X, loadings, inverse=False, skip=None):
    for mode, loading in enumerate(loadings):
        if mode != skip:
            X = _mode_product(X, loading if inverse else loading.T, mode)
    return X


@dataclass
class FactorResult:
    """Fitted matrix or Tucker tensor factor decomposition.

    ``loadings[k].T @ loadings[k]`` is identity. ``signal`` includes ``mean``
    when centering was requested; ``factors`` describe deviations from that
    mean. ``eigenvalues`` are descending spectra of moments computed after
    dividing data by ``data_scale`` (and after any alpha mean adjustment;
    alpha-PCA additionally divides its moment by ``max(1, 1 + alpha)``).
    This scaling avoids overflow and does not change the estimated subspaces.
    ``converged`` describes the projection iteration, not statistical accuracy.
    """

    factors: np.ndarray
    signal: np.ndarray
    loadings: tuple
    residuals: np.ndarray
    mean: np.ndarray
    eigenvalues: tuple
    method: str
    data_scale: float = 1.0
    n_iter: int = 0
    converged: bool = True

    @property
    def ranks(self):
        return tuple(loading.shape[1] for loading in self.loadings)

    def transform(self, X):
        """Project new observations using the fitted loadings and training mean.

        References
        ----------
        Chen, Yang and Zhang (2022), Factor Models for High-Dimensional Tensor
        Time Series. https://doi.org/10.1080/01621459.2021.1912757
        This is fixed-loading Tucker projection, not a new forecasting fit.
        """
        X = as_series(X, min_samples=1, ndim=len(self.loadings) + 1)
        if X.shape[1:] != self.mean.shape:
            raise ValueError("observation dimensions must match the fitted model")
        return _project(X - self.mean, self.loadings)

    def inverse_transform(self, factors):
        """Reconstruct observations from core factors, adding the training mean.

        References
        ----------
        Chen, Yang and Zhang (2022), Factor Models for High-Dimensional Tensor
        Time Series. https://doi.org/10.1080/01621459.2021.1912757
        Restoring the optional training mean is an implementation convention.
        """
        factors = as_series(factors, min_samples=1, ndim=len(self.loadings) + 1)
        if factors.shape[1:] != self.ranks:
            raise ValueError("factor dimensions must match fitted ranks")
        return _project(factors, self.loadings, inverse=True) + self.mean


def _result(X, loadings, mean, eigenvalues, method, scale, n_iter=0, converged=True):
    factors = _project(X - mean, loadings)
    signal = _project(factors, loadings, inverse=True) + mean
    return FactorResult(
        factors,
        signal,
        tuple(loadings),
        X - signal,
        mean,
        tuple(eigenvalues),
        method,
        scale,
        n_iter,
        converged,
    )


def _prepare(X, center, ndim=3):
    X = as_series(X, ndim=ndim)
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError("center must be boolean")
    # Average normalized observations so even a large finite mean stays finite.
    scale = max(float(np.max(np.abs(X))), np.finfo(float).tiny)
    normalized = X / scale
    mean_scaled = normalized.mean(axis=0) if center else np.zeros(X.shape[1:])
    mean = mean_scaled * scale
    return X, normalized - mean_scaled, mean, scale


def _matrix_moments(X):
    denominator = X.shape[0] * X.shape[1] * X.shape[2]
    return (
        np.einsum("tij,tkj->ik", X, X, optimize=True) / denominator,
        np.einsum("tij,tik->jk", X, X, optimize=True) / denominator,
    )


def _space_distance(first, second):
    return max(
        np.linalg.norm(a @ a.T - b @ b.T, ord="fro") for a, b in zip(first, second)
    )


def fit_alpha_pca(X, ranks=None, *, alpha=0.0, center=False):
    """Fit alpha-PCA of Chen and Fan (2023; published online 2021).

    The moment combines temporal covariance and ``(1 + alpha)`` times the
    outer product of the mean. ``alpha=-1`` uses covariance only; ``alpha=0``
    uses uncentered second moments. ``center=True`` first removes a fixed mean
    and restores it on reconstruction, making alpha immaterial.

    Parameters
    ----------
    X : array_like, shape (T, p, q)
        Finite, real observations.
    ranks : pair of int or None, optional
        Loading ranks. A missing rank uses an adjacent eigenvalue ratio.
    alpha : float, default 0
        First-moment weight adjustment, finite and at least -1.
    center : bool, default False
        Estimate and subtract an unrestricted temporal mean.

    References
    ----------
    Chen and Fan (2023; online 2021), Statistical Inference for
    High-Dimensional Matrix-Variate Factor Models.
    https://doi.org/10.1080/01621459.2021.1970569
    """
    alpha = finite_scalar(alpha, "alpha", minimum=-1)
    X, work, mean, scale = _prepare(X, center)
    ranks = _factor_ranks(ranks, X.shape[1:])
    # A convex rescaling avoids overflow even for alpha near float64's maximum.
    weight = np.sqrt(float(alpha) + 1)
    divisor = max(1.0, weight)
    work_mean = work.mean(axis=0)
    adjusted = (work - work_mean) / divisor + (weight / divisor) * work_mean
    pairs = [
        _eigenspace(moment, rank)
        for moment, rank in zip(_matrix_moments(adjusted), ranks)
    ]
    return _result(
        X,
        [pair[0] for pair in pairs],
        mean,
        [pair[1] for pair in pairs],
        "alpha-pca",
        scale,
    )


def fit_projected_pca(X, ranks=None, *, max_iter=1, tol=1e-8, center=False):
    """Fit projected matrix PCA of Yu, He, Kong and Zhang (2022).

    The default implements Algorithm 1: alpha=0 initialization followed by
    one simultaneous update of both spaces using projections onto the opposite
    initial loading space. ``max_iter>1`` recursively repeats those simultaneous
    updates. Automatic ranks are selected once from the initial moments; this
    is not the paper's iterative rank selection algorithm.

    References
    ----------
    Yu, He, Kong and Zhang (2022), Projected Estimation for Large-Dimensional
    Matrix Factor Models, Algorithm 1. https://doi.org/10.1016/j.jeconom.2021.04.001
    Author manuscript: https://arxiv.org/abs/2003.10285
    """
    max_iter = positive_int(max_iter, "max_iter")
    tol = finite_scalar(tol, "tol", minimum=0)
    if tol == 0:
        raise ValueError("tol must be finite and positive")
    X, work, mean, scale = _prepare(X, center)
    ranks = _factor_ranks(ranks, X.shape[1:])
    pairs = [
        _eigenspace(moment, rank) for moment, rank in zip(_matrix_moments(work), ranks)
    ]
    loadings = [pair[0] for pair in pairs]
    ranks = [loading.shape[1] for loading in loadings]
    converged = max_iter == 1
    for iteration in range(1, max_iter + 1):
        previous = loadings
        projected_rows = work @ previous[1]
        projected_cols = work.transpose(0, 2, 1) @ previous[0]
        denominator = np.prod(X.shape)
        moments = [
            np.einsum("tij,tkj->ik", Z, Z, optimize=True) / denominator
            for Z in (projected_rows, projected_cols)
        ]
        pairs = [_eigenspace(moment, rank) for moment, rank in zip(moments, ranks)]
        loadings = [pair[0] for pair in pairs]
        if _space_distance(loadings, previous) < tol:
            converged = True
            break
    return _result(
        X,
        loadings,
        mean,
        [pair[1] for pair in pairs],
        "projected-pca",
        scale,
        iteration,
        converged,
    )


def _lags(lags, n_samples):
    if isinstance(lags, (int, np.integer)) and not isinstance(lags, (bool, np.bool_)):
        result = tuple(range(1, positive_int(lags, "lags") + 1))
    else:
        try:
            result = tuple(positive_int(lag, "lag") for lag in lags)
        except TypeError as exc:
            raise ValueError(
                "lags must be a positive integer or sequence of positive integers"
            ) from exc
    if not result or len(set(result)) != len(result) or max(result) >= n_samples:
        raise ValueError(
            "lags must be nonempty, distinct, and smaller than the sample count"
        )
    return result


def _lagged_moment(unfolded, lags, method):
    """Accumulate exact TOPUP/TIPUP Gram matrices without a full outer tensor."""
    _, dimension, complement = unfolded.shape
    moment = np.zeros((dimension, dimension))
    for lag in lags:
        past, future = unfolded[:-lag], unfolded[lag:]
        n = len(past)
        if method == "tipup":
            covariance = np.einsum("tai,tbi->ab", past, future, optimize=True) / n
            moment += covariance @ covariance.T
        elif n < dimension * complement:
            # Dual identity: W_ab = sum_st <future_s,future_t>
            #                         sum_i past_sai past_tbi / n**2.
            # This costs O(T**2 * prod(dims)) instead of O(T * prod(dims)**2)
            # when the number of spatial entries exceeds the sample count.
            past_flat = past.reshape(n, -1)
            future_flat = future.reshape(n, -1)
            block = max(1, min(n, (4 * 1024**2) // max(n, dimension * complement)))
            for start in range(0, n, block):
                gram = future_flat[start : start + block] @ future_flat.T
                weighted = (gram @ past_flat).reshape(-1, dimension, complement)
                moment += (
                    np.einsum(
                        "tai,tbi->ab",
                        past[start : start + block],
                        weighted,
                        optimize=True,
                    )
                    / n**2
                )
        else:
            # Slice columns of the mode-k outer-product unfolding; summing their
            # Gram matrices is exact. Bound the temporary to about 32 MiB.
            block = max(
                1, min(complement, (4 * 1024**2) // (dimension**2 * complement))
            )
            for start in range(0, complement, block):
                covariance = (
                    np.einsum(
                        "tai,tbj->aibj",
                        past[:, :, start : start + block],
                        future,
                        optimize=True,
                    ).reshape(dimension, -1)
                    / n
                )
                moment += covariance @ covariance.T
    return moment


def fit_lagged_factor(X, ranks=None, *, lags=1, center=False):
    """Fit the lagged cross-covariance matrix factor model of Wang et al. (2019).

    Uses all pairs of columns (and rows), summing squared lagged cross-moments.
    This is matrix TOPUP, not the sum-before-squaring TIPUP approximation. An
    integer ``lags=h`` uses 1 through h; a sequence selects individual lags.
    White measurement noise and serially informative factors are required for
    identification. With ``center=False`` the model assumes zero mean.

    References
    ----------
    Wang, Liu and Chen (2019), Factor Models for Matrix-Valued
    High-Dimensional Time Series, Section 3.
    https://doi.org/10.1016/j.jeconom.2018.09.013
    """
    X, work, mean, scale = _prepare(X, center)
    ranks = _factor_ranks(ranks, X.shape[1:])
    lags = _lags(lags, len(X))
    moments = [
        _lagged_moment(work, lags, "topup"),
        _lagged_moment(work.transpose(0, 2, 1), lags, "topup"),
    ]
    pairs = [_eigenspace(moment, rank) for moment, rank in zip(moments, ranks)]
    return _result(
        X,
        [pair[0] for pair in pairs],
        mean,
        [pair[1] for pair in pairs],
        "lagged-factor",
        scale,
    )
