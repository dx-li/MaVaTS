import numpy as np
import pytest

from mavats.MAR import (
    _compute_R,
    _initialize_AB,
    _least_square_estimate,
    _mle_estimate,
    _normalize,
    _proj_estimate,
    _rearrange_phi,
    _update_A_mle,
    _update_B_mle,
    _update_lse,
    _update_mle,
    _update_Sigc_mle,
    _update_Sigr_mle,
    estimate_mar1,
    estimate_residual_cov,
)


def test_estimate_mar1_least_square():
    T, m, n = 10, 3, 4
    X = np.random.randn(T, m, n)
    A, B = estimate_mar1(X, method="least_square", niter=10)
    assert A.shape == (m, m)
    assert B.shape == (n, n)


def test_estimate_mar1_proj():
    T, m, n = 10, 3, 4
    X = np.random.randn(T, m, n)
    A, B = estimate_mar1(X, method="proj")
    assert A.shape == (m, m)
    assert B.shape == (n, n)


def test_estimate_mar1_mle():
    T, m, n = 10, 3, 4
    X = np.random.randn(T, m, n)
    A, B, Sigc, Sigr = estimate_mar1(X, method="mle", niter=10)
    assert A.shape == (m, m)
    assert B.shape == (n, n)
    assert Sigc.shape == (n, n)
    assert Sigr.shape == (m, m)


def test_estimate_residual_cov():
    T, m, n = 10, 3, 4
    X = np.random.randn(T, m, n)
    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    Sigma = estimate_residual_cov(X, A, B)
    assert Sigma.shape == (m * n, m * n)


def test_proj_estimate():
    T, m, n = 10, 3, 4
    X = np.random.randn(T, m, n)
    A, B = _proj_estimate(X)
    assert A.shape == (m, m)
    assert B.shape == (n, n)


def test_normalize():
    A = np.random.randn(3, 3)
    B = np.random.randn(4, 4)
    A_norm, B_norm = _normalize(A, B)
    assert np.isclose(np.linalg.norm(A_norm), 1)


def test_rearrange_phi():
    m, n = 3, 4
    Phi = np.random.randn(m * n, m * n)
    Phi_tilde = _rearrange_phi(Phi, m, n)
    assert Phi_tilde.shape == (m * m, n * n)


def test_initialize_AB():
    T, m, n = 10, 3, 4
    X = np.random.randn(T, m, n)
    A, B = _initialize_AB(X, None, None)
    assert A.shape == (m, m)
    assert B.shape == (n, n)


def test_least_square_estimate():
    T, m, n = 10, 3, 4
    X = np.random.randn(T, m, n)
    A, B = _least_square_estimate(X, niter=10)
    assert A.shape == (m, m)
    assert B.shape == (n, n)


def test_update_lse():
    T, m, n = 10, 3, 4
    X_tp1 = np.random.randn(T - 1, m, n)
    X_t = np.random.randn(T - 1, m, n)
    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    A, B = _update_lse(X_tp1, X_t, A, B)
    assert A.shape == (m, m)
    assert B.shape == (n, n)


def test_mle_estimate():
    T, m, n = 10, 3, 4
    X = np.random.randn(T, m, n)
    A, B, Sigc, Sigr = _mle_estimate(X, niter=10)
    assert A.shape == (m, m)
    assert B.shape == (n, n)
    assert Sigc.shape == (n, n)
    assert Sigr.shape == (m, m)


def test_update_mle():
    T, m, n = 10, 3, 4
    X_tp1 = np.random.randn(T - 1, m, n)
    X_t = np.random.randn(T - 1, m, n)
    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    Sigc = np.eye(n)
    Sigr = np.eye(m)
    A, B, Sigc, Sigr = _update_mle(X_tp1, X_t, A, B, Sigc, Sigr)
    assert A.shape == (m, m)
    assert B.shape == (n, n)
    assert Sigc.shape == (n, n)
    assert Sigr.shape == (m, m)


def test_update_A_mle():
    T, m, n = 10, 3, 4
    X_tp1 = np.random.randn(T - 1, m, n)
    X_t = np.random.randn(T - 1, m, n)
    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    Sigc = np.eye(n)
    Sigr = np.eye(m)
    A = _update_A_mle(X_tp1, X_t, A, B, Sigc, Sigr)
    assert A.shape == (m, m)


def test_update_B_mle():
    T, m, n = 10, 3, 4
    X_tp1 = np.random.randn(T - 1, m, n)
    X_t = np.random.randn(T - 1, m, n)
    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    Sigc = np.eye(n)
    Sigr = np.eye(m)
    B = _update_B_mle(X_tp1, X_t, A, B, Sigc, Sigr)
    assert B.shape == (n, n)


def test_compute_R():
    T, m, n = 10, 3, 4
    X_tp1 = np.random.randn(T - 1, m, n)
    X_t = np.random.randn(T - 1, m, n)
    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    R = _compute_R(X_tp1, X_t, A, B)
    assert R.shape == (T - 1, m, n)


def test_update_Sigc_mle():
    T, m, n = 10, 3, 4
    X_tp1 = np.random.randn(T - 1, m, n)
    X_t = np.random.randn(T - 1, m, n)
    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    R = _compute_R(X_tp1, X_t, A, B)
    Sigc = np.eye(n)
    Sigr = np.eye(m)
    Sigc = _update_Sigc_mle(X_tp1, X_t, A, B, R, Sigc, Sigr)
    assert Sigc.shape == (n, n)


def test_update_Sigr_mle():
    T, m, n = 10, 3, 4
    X_tp1 = np.random.randn(T - 1, m, n)
    X_t = np.random.randn(T - 1, m, n)
    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    R = _compute_R(X_tp1, X_t, A, B)
    Sigc = np.eye(n)
    Sigr = np.eye(m)
    Sigr = _update_Sigr_mle(X_tp1, X_t, A, B, R, Sigc, Sigr)
    assert Sigr.shape == (m, m)


if __name__ == "__main__":
    pytest.main()
