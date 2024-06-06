import math
from typing import Tuple, Union

import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh

from mavats.factormodel import _estimate_k


def estimate_alpha_PCA(
    Y: np.ndarray, alpha: Union[float, int], k: Union[int, None], r: Union[int, None]
):
    r"""
    Estimates the $\alpha$-PCA model in Chen and Fan 2021 (https://doi.org/10.1080/01621459.2021.1970569)
    where $Y_t = R F_t C^T + E_t$

    Parameters
    ----------
    Y : (T, p, q) ndarray
        The data matrix.
    alpha : float
        The hyperparameter $\alpha$ that aims to balance information between first and second moments.
    k : int, optional
        Rank of the front loading matrix. If None, the rank is estimated.
    r : int, optional
        Rank of the back loading matrix. If None, the rank is estimated.

    Returns
    -------
    F : (T, k, r) ndarray
        The estimated factor matrix.
    S : (T, p, q) ndarray
        The estimated signal $S_t = R F_t C^T$.
    R : (p, k) ndarray
        The estimated front loading matrix.
    C : (q, r) ndarray
        The estimated back loading matrix.

    """
    if not (alpha >= -1):
        raise ValueError("alpha must be greater than or equal to -1")
    T, p, q = Y.shape
    pqT = p * q * T
    pq = p * q
    alpha_tilde = math.sqrt(alpha + 1) - 1
    Y_bar = np.mean(Y, axis=0)
    Y_tilde = Y + alpha_tilde * Y_bar
    M_R = np.einsum("tij,tkj->ik", Y_tilde, Y_tilde) / pqT
    M_C = np.einsum("tji,tjk->ik", Y_tilde, Y_tilde) / pqT
    R = _compute_loading(M_R, k)
    R *= math.sqrt(p)
    C = _compute_loading(M_C, r)
    C *= math.sqrt(q)
    F = R.T @ Y @ C / pq
    S = R @ F @ C.T / pq
    return F, S, R, C


def _compute_loading(M: np.ndarray, k: Union[int, None]) -> np.ndarray:
    if k is None:
        w, v = eigh(M)
        k = _estimate_k(w)
        w, v = w[-k:], v[:, -k:]
    else:
        w, v = eigh(M, subset_by_index=(M.shape[0] - k, M.shape[0] - 1))
    return v


def estimate_cov_R():
    r"""
    To be implemented
    """
    ...


def estimate_cov_C():
    r"""
    To be implemented
    """
    ...
