import math
from typing import Tuple, Union
import warnings

import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh

from mavats.factormodel import _estimate_k


def estimate_alpha_PCA(
    Y: np.ndarray, alpha: Union[float, int], k: Union[int, None] = None, r: Union[int, None] = None
):
    r"""
    Estimates the $\alpha$-PCA model in Chen and Fan 2021 (https://doi.org/10.1080/01621459.2021.1970569)
    where $Y_t = R F_t C^T + E_t$

    Parameters
    ----------
    Y : (T, p, q) ndarray
        The observed data.
    alpha : float
        The hyperparameter $\alpha \geq -1$ that aims to balance information between first and second moments.
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
    Y_tilde = _compute_Y_tilde(Y, alpha)
    M_R = np.einsum("tij,tkj->ik", Y_tilde, Y_tilde) / pqT
    M_C = np.einsum("tji,tjk->ik", Y_tilde, Y_tilde) / pqT
    R = _compute_loading(M_R, k)
    R *= math.sqrt(p)
    C = _compute_loading(M_C, r)
    C *= math.sqrt(q)
    F = R.T @ Y @ C / pq
    S = R @ F @ C.T / pq
    return F, S, R, C

def _compute_Y_tilde(Y, alpha):
    alpha_tilde = math.sqrt(alpha + 1) - 1
    Y_bar = np.mean(Y, axis=0)
    Y_tilde = Y + alpha_tilde * Y_bar
    return Y_tilde


def _compute_loading(M: np.ndarray, k: Union[int, None]) -> np.ndarray:
    if k is None:
        w, v = eigh(M)
        k = _estimate_k(w)
        w, v = w[-k:], v[:, -k:]
    else:
        w, v = eigh(M, subset_by_index=(M.shape[0] - k, M.shape[0] - 1))
    return v


def estimate_cov_Ri(Y: np.ndarray, F: np.ndarray, R: np.ndarray, C: np.ndarray, i:int, alpha: Union[float, int], m: int) -> np.ndarray:
    r"""
    Estimates the covariance matrix of the $i$-th row (0 indexed) of $\hat{R}$.

    Parameters
    ----------
    Y : (T, p, q) ndarray
        The observed data.
    F : (T, k, r) ndarray
        The estimated factors.
    R : (p, k) ndarray
        The estimated front loading matrix.
    C : (q, r) ndarray
        The estimated back loading matrix.
    i : int
        The row index.
    alpha : float
        The hyperparameter $\alpha \geq -1$. Should be the same as the one used in `estimate_alpha_PCA`.
    m : int
        A tuning parameter. Theoretically, it satisfies $m \rightarrow \infty$ and $m/(qT)^{\frac{1}{4}} \rightarrow 0$.

    Returns
    -------
    Sig_Ri : (k, k) ndarray
        The estimated covariance matrix of the $i$-th row of $\hat{R}$.
    """
    if not (alpha >= -1):
        raise ValueError("alpha must be greater than or equal to -1")
    T, p, q = Y.shape
    if m >= T:
        warnings.warn("m is greater than T, unexpected results may occur.")
    _, k, r = F.shape
    E = Y - R @ F @ C.T
    Y_tilde = _compute_Y_tilde(Y, alpha)
    YYT = (Y_tilde @ Y_tilde.transpose(0, 2, 1)).sum(axis=0) / (p*q*T)
    V, _ = eigh(YYT, subset_by_index=(p-k, p-1))
    V_inv = 1 / V
    V_inv = np.diag(V_inv)
    F_bar = np.mean(F, axis=0)
    middle = _compute_D_Rnui(F, F_bar, C, E, alpha, 0, i, q)
    for nu in range(1, m+1):
        D_Rnui = _compute_D_Rnui(F, F_bar, C, E, alpha, nu, i, q)
        middle += (1 - nu / (m+1)) * (D_Rnui + D_Rnui.T)
    Sig_Ri = V_inv @ middle @ V_inv
    return Sig_Ri

def _compute_D_Rnui(F: np.ndarray, F_bar: np.ndarray, C: np.ndarray, E: np.ndarray, alpha: Union[float, int], nu: int, i: int, q: int):
    T, k, r = F.shape
    IaF = np.hstack([np.eye(k), alpha * F_bar])
    if nu > 0:
        F_nu = F[nu: ]
        F_0nu = F[: -nu]
        E_nu = E[nu: ]
        E_0nu = E[: -nu]
    else:
        F_nu = F
        F_0nu = F
        E_nu = E
        E_0nu = E
    F_0nu_trans = F_0nu.transpose(0, 2, 1)
    e_nui = E_nu[:, i]
    e_0i = E_0nu[:, i]
    e_outer = np.einsum('ti, tj->tij', e_nui, e_0i)
    br = C.T @ e_outer @ C
    bl = br @ F_0nu_trans
    tr = F_nu @ br
    tl = F_nu @ bl
    block = np.block([[tl, tr], [bl, br]]).sum(axis=0)
    block /= (q*T)
    D_Rnui = IaF @ block @ IaF.T
    return D_Rnui

def estimate_cov_Cj(Y: np.ndarray, F: np.ndarray, R: np.ndarray, C: np.ndarray, j:int, alpha: Union[float, int], m: int) -> np.ndarray:
    r"""
    Estimates the covariance matrix of the $j$-th row (0 indexed) of $\hat{C}$.

    Parameters
    ----------
    Y : (T, p, q) ndarray
        The observed data.
    F : (T, k, r) ndarray
        The estimated factors.
    R : (p, k) ndarray
        The estimated front loading matrix.
    C : (q, r) ndarray
        The estimated back loading matrix.
    j : int
        The row index.
    alpha : float
        The hyperparameter $\alpha \geq -1$. Should be the same as the one used in `estimate_alpha_PCA`.
    m : int
        A tuning parameter. Theoretically, it satisfies $m \rightarrow \infty$ and $m/(pT)^{\frac{1}{4}} \rightarrow 0$.

    Returns
    -------
    Sig_Cj : (r, r) ndarray
        The estimated covariance matrix of the $j$-th row of $\hat{C}$.
    """
    if not (alpha >= -1):
        raise ValueError("alpha must be greater than or equal to -1")
    T, p, q = Y.shape
    if m >= T:
        warnings.warn("m is greater than T, unexpected results may occur.")
    _, k, r = F.shape
    E = Y - R @ F @ C.T
    Y_tilde = _compute_Y_tilde(Y, alpha)
    YTY = (Y_tilde.transpose(0, 2, 1) @ Y_tilde).sum(axis=0) / (p*q*T)
    V, _ = eigh(YTY, subset_by_index=(q-r, q-1))
    V_inv = 1 / V
    V_inv = np.diag(V_inv)
    F_bar = np.mean(F, axis=0)
    middle = _compute_D_Cnuj(F, F_bar, R, E, alpha, 0, j, p)
    for nu in range(1, m+1):
        D_Rnui = _compute_D_Cnuj(F, F_bar, R, E, alpha, nu, j, p)
        middle += (1 - nu / (m+1)) * (D_Rnui + D_Rnui.T)
    Sig_Ri = V_inv @ middle @ V_inv
    return Sig_Ri

def _compute_D_Cnuj(F: np.ndarray, F_bar: np.ndarray, R: np.ndarray, E: np.ndarray, alpha: Union[float, int], nu: int, j: int, p: int):
    T, k, r = F.shape
    IaFt = np.hstack([np.eye(r), alpha * F_bar.T])
    if nu > 0:
        F_nu = F[nu: ]
        F_0nu = F[: -nu]
        E_nu = E[nu: ]
        E_0nu = E[: -nu]
    else:
        F_nu = F
        F_0nu = F
        E_nu = E
        E_0nu = E
    F_nu_trans = F_nu.transpose(0, 2, 1)
    e_nuj = E_nu[:, :, j]
    e_0j = E_0nu[:, :, j]
    e_outer = np.einsum('ti, tj->tij', e_nuj, e_0j)
    br = R.T @ e_outer @ R
    bl = br @ F_0nu
    tr = F_nu_trans @ br
    tl = tr @ F_0nu
    block = np.block([[tl, tr], [bl, br]]).sum(axis=0)
    block /= (p*T)
    D_Cnuj = IaFt @ block @ IaFt.T
    return D_Cnuj
