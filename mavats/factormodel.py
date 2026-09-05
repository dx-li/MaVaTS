from typing import Tuple, Union

import numpy as np

from mavats._validation import as_series, positive_int
from mavats.factors import (
    _eigenspace,
    _factor_ranks,
    _lagged_moment,
    _lags,
    eigenvalue_ratio,
    fit_lagged_factor,
)


def estimate_factor_model(
    X: np.ndarray, h0: int, k1: Union[int, None] = None, k2: Union[int, None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Estimates the high-dimensional matrix factor model in Wang, Liu, Chen 2019 (https://doi.org/10.1016/j.jeconom.2018.09.013)
    where $X_t = R F_t C^T + E_t = Q_1 Z_t Q_2^T + E_t$

    Parameters
    ----------
    X : (T, p1, p2) ndarray
        The observed data.
    h0 : int
        Lag parameter.
    k1 : int, optional
        Rank of the front loading matrix. If None, the rank is estimated.
    k2 : int, optional
        Rank of the back loading matrix. If None, the rank is estimated.

    Returns
    -------
    Z : (T, k1, k2) ndarray
        The estimated transformed factor matrix.
    S : (T, p1, p2) ndarray
        The estimated dynamic signal $S_t = Q_1 Z_t Q_2^T$.
    Q1 : (p1, k1) ndarray
        The estimated front loading matrix.
    Q2 : (p2, k2) ndarray
        The estimated back loading matrix.

    """
    result = fit_lagged_factor(X, (k1, k2), lags=h0)
    return result.factors, result.signal, *result.loadings


def _compute_omega_hat(X: np.ndarray, h: int) -> np.ndarray:
    X = as_series(X)
    h = positive_int(h, "h")
    T, p1, p2 = X.shape
    if h >= T:
        raise ValueError("h must be smaller than the sample count")
    omega_hat = np.zeros((p2, p2, p1, p1))
    X_t = X[: T - h]
    X_th = X[h:T]
    for i in range(p2):
        for j in range(p2):
            omega_hat[i, j] = np.einsum("tk,tl->kl", X_t[:, :, i], X_th[:, :, j])
    return omega_hat / (T - h)


def _compute_M(X: np.ndarray, h0: int) -> np.ndarray:
    X = as_series(X)
    return _lagged_moment(X, _lags(h0, len(X)), "topup")


def _compute_Q(M: np.ndarray, k: Union[int, None]) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if (
        M.ndim != 2
        or M.shape[0] != M.shape[1]
        or not M.size
        or not np.isfinite(M).all()
    ):
        raise ValueError("M must be a finite nonempty square matrix")
    k = _factor_ranks((k,), (len(M),))[0]
    return _eigenspace(M, k)[0]


def _estimate_k(w: np.ndarray) -> int:
    return eigenvalue_ratio(w)


def _ensure_positive_eigenvecs(v: np.ndarray) -> np.ndarray:
    v_summed = np.sum(v, axis=0)
    lt_zero = v_summed < 0
    v[:, lt_zero] *= -1
    return v
