from typing import Tuple, Union

import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh


def estimate_factor_model(
    X: np.ndarray, h0: int, k1: Union[int, None] = None, k2: Union[int, None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Estimates the high-dimensional matrix factor model in Chen, Yang, Zhang 2020 (https://doi.org/10.1080/01621459.2021.1912757)
    where $X_t = R F_t C^T + E_t$

    Parameters
    ----------
    X : (T, p1, p2) ndarray
        The data matrix.
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
        The estimated dynamic signal.
    Q1 : (p1, k1) ndarray
        The estimated front loading matrix.
    Q2 : (p2, k2) ndarray
        The estimated back loading matrix.

    """
    M1 = _compute_M(X, h0)
    Q1 = _compute_Q(M1, k1)
    M2 = _compute_M(X.transpose(0, 2, 1), h0)
    Q2 = _compute_Q(M2, k2)
    Z = Q1.T @ X @ Q2
    S = Q1 @ Z @ Q2.T
    return Z, S, Q1, Q2


def _compute_omega_hat(X: np.ndarray, h: int) -> np.ndarray:
    T, p1, p2 = X.shape
    omega_hat = np.zeros((p2, p2, p1, p1))
    X_t = X[: T - h]
    X_th = X[h:T]
    for i in range(p2):
        for j in range(p2):
            omega_hat[i, j] = np.einsum("tk,tl->kl", X_t[:, :, i], X_th[:, :, j])
    return omega_hat / (T - h)


def _compute_M(X: np.ndarray, h0: int) -> np.ndarray:
    T, p1, p2 = X.shape
    M = np.zeros((p1, p1))
    for h in range(1, h0 + 1):
        omega_hat = _compute_omega_hat(X, h)
        for i in range(p2):
            for j in range(p2):
                M += omega_hat[i, j] @ omega_hat[i, j].T
    return M


def _compute_Q(M: np.ndarray, k: Union[int, None]) -> np.ndarray:
    if k is None:
        w, v = eigh(M)
        k = _estimate_k(w)
        w, v = w[-k:], v[:, -k:]
    else:
        w, v = eigh(M, subset_by_index=(M.shape[0] - k, M.shape[0] - 1))
    v = _ensure_positive_eigenvecs(v)
    return v


def _estimate_k(w: np.ndarray) -> int:
    w = np.flipud(w)[: len(w) // 2 + 1]
    ratios = w[1:] / w[:-1]
    return int(np.argmin(ratios)) + 1


def _ensure_positive_eigenvecs(v: np.ndarray) -> np.ndarray:
    v_summed = np.sum(v, axis=0)
    lt_zero = v_summed < 0
    v[:, lt_zero] *= -1
    return v


if __name__ == "__main__":

    k1, k2 = 3, 4
    p1, p2 = 3, 12
    R = np.eye(p1)
    T = 10
    X = np.zeros((T, p1, p2))
    C = np.random.randn(p2, k2) * 3
    for i in range(T):
        X[i] = R @ np.random.randn(k1, k2) @ C.T + np.random.randn(p1, p2)
    Z, S, Q1, Q2 = estimate_factor_model(X, 3, k1, k2)
