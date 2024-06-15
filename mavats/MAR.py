from typing import Tuple, Union

import numpy as np
from numpy import linalg as LA


def estimate_mar1(
    X: np.ndarray, method: str = "least_square", **kwargs
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    r"""
    Estimates the first order MAR model $X_t = A X_{t-1} B^T + E_t$
    from Chen, Xiao, Yang 2021 (https://doi.org/10.1016/j.jeconom.2020.07.015)

    Parameters
    ----------
    X : (T, m, n) ndarray
        The observed data.
    method : str, optional
        The estimation method. Must be either "proj", "least_square" or "mle".
        Note MLE assumes Cov(vec($E_t$)) = $\Sigma_c \otimes \Sigma_r$
        where $\Sigma_c \in \mathbb{R}^{n \times n}$ and $\Sigma_r \in \mathbb{R}^{m \times m}$.
    **kwargs
        Additional arguments for the estimation method.

    Returns
    -------
    A : (m, m) ndarray
        The estimated left matrix.
    B : (n, n) ndarray
        The estimated right matrix.
    If method is "mle":
    Sigc : (m, m) ndarray
        The estimated covariance $\Sigma_c$
    Sigr : (n, n) ndarray
        The estimated covariance $\Sigma_r$

    """
    if method == "proj":
        return _proj_estimate(X)
    elif method == "least_square":
        niter = kwargs.get("niter", 50)
        A_init = kwargs.get("A_init", None)
        B_init = kwargs.get("B_init", None)
        init_method = kwargs.get("init_method", "proj")
        return _least_square_estimate(X, niter, A_init, B_init, init_method)
    elif method == "mle":
        niter = kwargs.get("niter", 50)
        A_init = kwargs.get("A_init", None)
        B_init = kwargs.get("B_init", None)
        init_method = kwargs.get("init_method", "proj")
        Sigc_init = kwargs.get("Sigc_init", None)
        Sigr_init = kwargs.get("Sigr_init", None)
        return _mle_estimate(
            X, niter, A_init, B_init, init_method, Sigc_init, Sigr_init
        )
    else:
        raise ValueError('method must be either "proj", "least_square" or "mle"')


def estimate_residual_cov(X: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""
    Estimates the covariance of $\text{Vec}(E_t)$ where $X_t = A X_{t-1} B^T + E_t$
    by computing the sample covariance of the vectorized residuals.

    Parameters
    ----------
    X : (T, m, n) ndarray
        The observed data.
    A : (m, m) ndarray
        The estimated left matrix.
    B : (n, n) ndarray
        The estimated right matrix.

    Returns
    -------
    Sigma : (m*n, m*n) ndarray
        The estimated covariance matrix.

    """
    T, m, n = X.shape
    X_tp1 = X[1:]
    X_t = X[:-1]
    X_tp1_hat = A @ X_t @ B.T
    res = X_tp1 - X_tp1_hat
    Sigma = np.cov(res.transpose(0, 2, 1).reshape(T - 1, -1).T)
    return Sigma


def _proj_estimate(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T, m, n = X.shape
    X_flat = X.transpose(0, 2, 1).reshape(T, -1)
    X_flat_tp1, X_flat_t = X_flat[1:], X_flat[:-1]
    Phi = LA.lstsq(X_flat_tp1, X_flat_t, rcond=None)[0].T
    Phi_tilde = _rearrange_phi(Phi, m, n)

    U, S, Vh = LA.svd(Phi_tilde, full_matrices=False)
    d = S[0]
    u = U[:, 0]
    v = Vh[0, :]
    A = u.reshape(m, m, order="F")
    B = v.reshape(n, n, order="F").T * d
    A, B = _normalize(A, B)
    return A, B


def _normalize(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norm_A = LA.norm(A)
    A /= norm_A
    B *= norm_A
    return A, B


def _rearrange_phi(Phi: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Rearranges the mn x mn matrix Phi into an $m^2 \times n^2$ matrix as specified in the paper.
    """
    col_splits = np.split(Phi, n, axis=1)
    blocks = [np.split(col, n, axis=0) for col in col_splits]
    return np.column_stack(
        [np.ravel(block, order="F") for row in blocks for block in row]
    )


def _initialize_AB(
    X: np.ndarray,
    A_init: Union[np.ndarray, None],
    B_init: Union[np.ndarray, None],
    init_method: str = "proj",
) -> Tuple[np.ndarray, np.ndarray]:
    T, m, n = X.shape
    if A_init is None or B_init is None:
        if init_method == "proj":
            A_init, B_init = _proj_estimate(X)
        elif init_method == "random":
            A_init = np.random.randn(m, m)
            B_init = np.random.randn(n, n)
        else:
            raise ValueError('init_method must be either "proj" or "random"')
    return A_init, B_init


def _least_square_estimate(
    X: np.ndarray,
    niter: int = 50,
    A_init: Union[np.ndarray, None] = None,
    B_init: Union[np.ndarray, None] = None,
    init_method: str = "proj",
) -> Tuple[np.ndarray, np.ndarray]:
    T, m, n = X.shape
    A_init, B_init = _initialize_AB(X, A_init, B_init, init_method)
    A = A_init
    B = B_init
    X_tp1 = X[1:]
    X_t = X[:-1]

    for i in range(niter):
        A, B = _update_lse(X_tp1, X_t, A, B)

    return A, B


def _update_lse(
    X_tp1: np.ndarray, X_t: np.ndarray, A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    X_tp1_transpose = X_tp1.transpose(0, 2, 1)
    X_t_transpose = X_t.transpose(0, 2, 1)

    XBXT = np.sum(X_tp1 @ B @ X_t_transpose, axis=0)
    XBTBXT = np.sum(X_t @ B.T @ B @ X_t_transpose, axis=0)

    A = np.linalg.solve(XBTBXT, XBXT)

    XTAX = np.sum(X_tp1_transpose @ A @ X_t, axis=0)
    XTATAX = np.sum(X_t_transpose @ A.T @ A @ X_t, axis=0)

    B = np.linalg.solve(XTATAX, XTAX)

    A, B = _normalize(A, B)

    return A, B


def _mle_estimate(
    X: np.ndarray,
    niter: int = 50,
    A_init: Union[np.ndarray, None] = None,
    B_init: Union[np.ndarray, None] = None,
    init_method: str = "proj",
    Sigc_init: Union[np.ndarray, None] = None,
    Sigr_init: Union[np.ndarray, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, m, n = X.shape
    A_init, B_init = _initialize_AB(X, A_init, B_init, init_method)
    A = A_init
    B = B_init
    if Sigc_init is None or Sigr_init is None:
        Sigc_init = np.eye(n)
        Sigr_init = np.eye(m)
    Sigc = Sigc_init
    Sigr = Sigr_init
    X_tp1 = X[1:]
    X_t = X[:-1]

    for i in range(niter):
        A, B, Sigc, Sigr = _update_mle(X_tp1, X_t, A, B, Sigc, Sigr)

    return A, B, Sigc, Sigr


def _update_mle(
    X_tp1: np.ndarray,
    X_t: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Sigc: np.ndarray,
    Sigr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = _update_A_mle(X_tp1, X_t, A, B, Sigc, Sigr)
    B = _update_B_mle(X_tp1, X_t, A, B, Sigc, Sigr)
    R = _compute_R(X_tp1, X_t, A, B)
    Sigc = _update_Sigc_mle(X_tp1, X_t, A, B, R, Sigc, Sigr)
    Sigr = _update_Sigr_mle(X_tp1, X_t, A, B, R, Sigc, Sigr)

    A, B = _normalize(A, B)
    Sigr, Sigc = _normalize(Sigr, Sigc)

    return A, B, Sigc, Sigr


def _update_A_mle(
    X_tp1: np.ndarray,
    X_t: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Sigc: np.ndarray,
    Sigr: np.ndarray,
) -> np.ndarray:
    X_tp1_transpose = X_tp1.transpose(0, 2, 1)
    X_t_transpose = X_t.transpose(0, 2, 1)
    Sigc_inv = LA.inv(Sigc)

    YSBX = np.sum(X_tp1 @ Sigc_inv @ B @ X_t_transpose, axis=0)
    XBSBX = np.sum(X_t @ B.T @ Sigc_inv @ B @ X_t_transpose, axis=0)

    A = np.linalg.solve(XBSBX, YSBX)
    return A


def _update_B_mle(
    X_tp1: np.ndarray,
    X_t: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Sigc: np.ndarray,
    Sigr: np.ndarray,
) -> np.ndarray:
    X_tp1_transpose = X_tp1.transpose(0, 2, 1)
    X_t_transpose = X_t.transpose(0, 2, 1)
    Sigr_inv = LA.inv(Sigr)
    YSAX = np.sum(X_tp1_transpose @ Sigr_inv @ A @ X_t, axis=0)
    XASAX = np.sum(X_t_transpose @ A.T @ Sigr_inv @ A @ X_t, axis=0)

    B = np.linalg.solve(XASAX, YSAX)
    return B


def _compute_R(
    X_tp1: np.ndarray, X_t: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    return X_tp1 - A @ X_t @ B.T


def _update_Sigc_mle(
    X_tp1: np.ndarray,
    X_t: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    Sigc: np.ndarray,
    Sigr: np.ndarray,
) -> np.ndarray:
    m, _ = A.shape
    T, _, _ = X_t.shape
    R_transpose = R.transpose(0, 2, 1)
    Sigr_inv = LA.inv(Sigr)
    RSR = np.sum(R_transpose @ Sigr_inv @ R, axis=0)
    Sigc = RSR / (T * m)
    return Sigc


def _update_Sigr_mle(
    X_tp1: np.ndarray,
    X_t: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    Sigc: np.ndarray,
    Sigr: np.ndarray,
) -> np.ndarray:
    n, _ = B.shape
    T, _, _ = X_t.shape
    R_transpose = R.transpose(0, 2, 1)
    Sigc_inv = LA.inv(Sigc)
    RSR = np.sum(R @ Sigc_inv @ R_transpose, axis=0)
    Sigr = RSR / (T * n)
    return Sigr
