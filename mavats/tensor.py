"""Dynamic Tucker tensor factor models (time is always the first axis)."""

import numpy as np

from ._validation import finite_scalar, positive_int
from .factors import (
    _eigenspace,
    _factor_ranks,
    _lagged_moment,
    _lags,
    _prepare,
    _project,
    _result,
    _space_distance,
)


def _unfold_series(X, mode):
    return np.moveaxis(X, mode + 1, 1).reshape(X.shape[0], X.shape[mode + 1], -1)


def fit_tensor_factor(
    X,
    ranks=None,
    *,
    method="tipup",
    lags=1,
    iterative=False,
    max_iter=100,
    tol=1e-8,
    center=False,
):
    """Fit TOPUP, TIPUP, iTOPUP or iTIPUP to matrix or higher-order tensor series.

    Parameters
    ----------
    X : array_like, shape (T, d1, ..., dK)
        Finite real observations, with at least two spatial modes.
    ranks : sequence of int or None, optional
        One rank per spatial mode. Missing ranks use the initial eigenvalue
        ratio heuristic and stay fixed during iteration.
    method : {'topup', 'tipup'}, default 'tipup'
        TOPUP uses all lagged fiber cross-products; TIPUP contracts matching
        fibers before squaring and is cheaper but can suffer cancellation.
    lags : int or sequence of int, default 1
        Integer h includes lags 1,...,h. A sequence selects specific lags.
    iterative : bool, default False
        Project the other modes onto the current loading estimates and update
        modes in sequence (Gauss-Seidel) until projectors converge.
    max_iter : int, default 100
        Maximum projection sweeps after initialization.
    tol : float, default 1e-8
        Maximum Frobenius change of a loading projector for convergence.
    center : bool, default False
        Remove and retain the training mean for future transformations.

    Returns
    -------
    FactorResult
        Orthonormal loadings, core factors, signal, residuals and diagnostics.

    Notes
    -----
    Assumes serially white measurement error and informative factor lag
    moments. Nonzero means require centering or a justified model for the mean.
    TOPUP bounds its intermediate allocation using exact column blocks.

    References
    ----------
    Chen, Yang and Zhang (2022), Factor Models for High-Dimensional Tensor
    Time Series. https://arxiv.org/abs/1905.07530
    Han, Chen, Yang and Zhang, Tensor Factor Model Estimation by Iterative
    Projection. Algorithm and implementation described in tensorTS, eq. (14):
    https://yuefenghan.github.io/papers/R_software_paper_tensorTS.pdf
    """
    method = str(method).lower()
    if method not in {"topup", "tipup"}:
        raise ValueError("method must be 'topup' or 'tipup'")
    if not isinstance(iterative, (bool, np.bool_)):
        raise ValueError("iterative must be boolean")
    max_iter = positive_int(max_iter, "max_iter")
    tol = finite_scalar(tol, "tol", minimum=0)
    if tol == 0:
        raise ValueError("tol must be finite and positive")
    ndim = np.ndim(X)
    if ndim < 3:
        raise ValueError("X needs a time axis and at least two spatial modes")
    X, work, mean, scale = _prepare(X, center, ndim=ndim)
    ranks = _factor_ranks(ranks, X.shape[1:])
    lags = _lags(lags, len(X))
    pairs = [
        _eigenspace(_lagged_moment(_unfold_series(work, mode), lags, method), rank)
        for mode, rank in enumerate(ranks)
    ]
    loadings = [pair[0] for pair in pairs]
    spectra = [pair[1] for pair in pairs]
    ranks = [loading.shape[1] for loading in loadings]
    converged = not iterative
    iteration = 0
    if iterative:
        for iteration in range(1, max_iter + 1):
            previous = list(loadings)
            for mode, rank in enumerate(ranks):
                projected = _project(work, loadings, skip=mode)
                moment = _lagged_moment(_unfold_series(projected, mode), lags, method)
                loadings[mode], spectra[mode] = _eigenspace(moment, rank)
            if _space_distance(loadings, previous) < tol:
                converged = True
                break
    return _result(
        X,
        loadings,
        mean,
        spectra,
        ("i" if iterative else "") + method,
        scale,
        iteration,
        converged,
    )
