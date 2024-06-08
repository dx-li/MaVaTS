import numpy as np
import pytest

from mavats.factormodel import (
    _compute_M,
    _compute_omega_hat,
    _compute_Q,
    _ensure_positive_eigenvecs,
    _estimate_k,
    estimate_factor_model,
)


def test_estimate_factor_model():
    k1, k2 = 3, 4
    p1, p2 = 3, 12
    R = np.eye(p1)
    T = 10
    X = np.zeros((T, p1, p2))
    C = np.random.randn(p2, k2) * 3
    for i in range(T):
        X[i] = R @ np.random.randn(k1, k2) @ C.T + np.random.randn(p1, p2)
    Z, S, Q1, Q2 = estimate_factor_model(X, 3, k1, k2)

    assert Z.shape == (T, k1, k2)
    assert S.shape == (T, p1, p2)
    assert Q1.shape == (p1, k1)
    assert Q2.shape == (p2, k2)
    assert np.allclose(Q1.T @ Q1, np.eye(k1), atol=1e-5)
    assert np.allclose(Q2.T @ Q2, np.eye(k2), atol=1e-5)


def test_compute_omega_hat():
    T, p1, p2 = 10, 3, 4
    X = np.random.randn(T, p1, p2)
    h = 2
    omega_hat = _compute_omega_hat(X, h)
    assert omega_hat.shape == (p2, p2, p1, p1)


def test_compute_M():
    T, p1, p2 = 10, 3, 4
    X = np.random.randn(T, p1, p2)
    h0 = 2
    M = _compute_M(X, h0)
    assert M.shape == (p1, p1)


def test_compute_Q():
    p = 5
    M = np.random.randn(p, p)
    M = M.T @ M  # Make it symmetric positive definite
    k = 3
    Q = _compute_Q(M, k)
    assert Q.shape == (p, k)
    assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-5)


def test_estimate_k():
    w = np.array([10, 8, 6, 4, 2, 1])
    k = _estimate_k(w)
    assert k == 3


def test_ensure_positive_eigenvecs():
    v = np.array([[1, -2], [1, -1]])
    v_positive = _ensure_positive_eigenvecs(v)
    assert np.all(v_positive[:, 0] >= 0)
    assert np.all(v_positive[:, 1] >= 0)


if __name__ == "__main__":
    pytest.main()
