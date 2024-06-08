import numpy as np
import pytest

from mavats.alphaPCA import (
    _compute_D_Cnuj,
    _compute_D_Rnui,
    _compute_loading,
    _compute_Y_tilde,
    estimate_alpha_PCA,
    estimate_cov_Cj,
    estimate_cov_Ri,
)


def test_estimate_alpha_PCA():
    T, p, q = 10, 5, 6
    alpha = 0.5
    Y = np.random.randn(T, p, q)
    k, r = 3, 4
    F, S, R, C = estimate_alpha_PCA(Y, alpha, k, r)

    assert F.shape == (T, k, r)
    assert S.shape == (T, p, q)
    assert R.shape == (p, k)
    assert C.shape == (q, r)
    assert np.allclose(R.T @ R / p, np.eye(k), atol=1e-5)
    assert np.allclose(C.T @ C / q, np.eye(r), atol=1e-5)


def test_compute_Y_tilde():
    T, p, q = 10, 5, 6
    alpha = 0.5
    Y = np.random.randn(T, p, q)
    Y_tilde = _compute_Y_tilde(Y, alpha)
    assert Y_tilde.shape == (T, p, q)


def test_compute_loading():
    p = 5
    M = np.random.randn(p, p)
    M = M.T @ M  # Make it symmetric positive definite
    k = 3
    loading = _compute_loading(M, k)
    assert loading.shape == (p, k)
    assert np.allclose(loading.T @ loading, np.eye(k), atol=1e-5)


def test_estimate_cov_Ri():
    T, p, q = 10, 5, 6
    alpha = 0.5
    k, r = 3, 4
    Y = np.random.randn(T, p, q)
    F, S, R, C = estimate_alpha_PCA(Y, alpha, k, r)
    m = 3
    i = 1
    Sig_Ri = estimate_cov_Ri(Y, F, R, C, i, alpha, m)
    assert Sig_Ri.shape == (k, k)


def test_estimate_cov_Cj():
    T, p, q = 10, 5, 6
    alpha = 0.5
    k, r = 3, 4
    Y = np.random.randn(T, p, q)
    F, S, R, C = estimate_alpha_PCA(Y, alpha, k, r)
    m = 3
    j = 1
    Sig_Cj = estimate_cov_Cj(Y, F, R, C, j, alpha, m)
    assert Sig_Cj.shape == (r, r)


def test_compute_D_Rnui():
    T, p, q = 10, 5, 6
    alpha = 0.5
    k, r = 3, 4
    Y = np.random.randn(T, p, q)
    F, S, R, C = estimate_alpha_PCA(Y, alpha, k, r)
    E = Y - R @ F @ C.T
    F_bar = np.mean(F, axis=0)
    nu = 1
    i = 1
    D_Rnui = _compute_D_Rnui(F, F_bar, C, E, alpha, nu, i, q)
    assert D_Rnui.shape == (k, k)


def test_compute_D_Cnuj():
    T, p, q = 10, 5, 6
    alpha = 0.5
    k, r = 3, 4
    Y = np.random.randn(T, p, q)
    F, S, R, C = estimate_alpha_PCA(Y, alpha, k, r)
    E = Y - R @ F @ C.T
    F_bar = np.mean(F, axis=0)
    nu = 1
    j = 1
    D_Cnuj = _compute_D_Cnuj(F, F_bar, R, E, alpha, nu, j, p)
    assert D_Cnuj.shape == (r, r)


if __name__ == "__main__":
    pytest.main()
