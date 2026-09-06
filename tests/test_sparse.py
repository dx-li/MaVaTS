"""Independent conditional posterior, numerical, and sparse recovery checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mavats.sparse import (
    _balance_coefficients,
    _balance_covariances,
    _coefficient_update,
    _covariance_mode,
    _log_posterior,
    _mixture,
    _scale_mode,
    fit_sparse_mar,
)


def _sample(size=1600, seed=4):
    rng = np.random.default_rng(seed)
    A, B = np.diag([0.8, 0.7, 0.65]), np.diag([0.7, 0.8])
    X = np.zeros((size + 100, 3, 2))
    for t in range(1, len(X)):
        X[t] = A @ X[t - 1] @ B.T + rng.normal(size=(3, 2))
    return X[100:], A, B


def test_mixture_matches_direct_bayes_and_handles_boundaries():
    coefficient = np.array([0, 0.1, -0.8, 10.0])
    p, logp = _mixture(coefficient, 0.23, 0.01, 4.0)
    spike = 0.77 * np.exp(-(coefficient**2) / 0.02) / np.sqrt(2 * np.pi * 0.01)
    slab = 0.23 * np.exp(-(coefficient**2) / 8.0) / np.sqrt(2 * np.pi * 4)
    assert_allclose(p, slab / (spike + slab))
    assert_allclose(logp, np.log(spike + slab).sum())
    assert_array_equal(_mixture(coefficient, 0, 0.01, 4)[0], 0)
    assert_array_equal(_mixture(coefficient, 1, 0.01, 4)[0], 1)
    assert np.isfinite(_mixture(np.array([1e5]), 0.1, 1e-8, 4)[1])


@pytest.mark.parametrize("singular", [False, True])
def test_coefficient_update_matches_independent_vector_gls(singular):
    rng = np.random.default_rng(35)
    Y, Z = rng.normal(size=(11, 2, 3)), rng.normal(size=(11, 2, 3))
    if singular:
        Z[:, 1] = 2 * Z[:, 0]
    B = rng.normal(size=(3, 3))
    R = np.array([[1.3, 0.7], [0.7, 1.8]])
    C = np.array([[1.1, 0.2, -0.1], [0.2, 0.8, 0.3], [-0.1, 0.3, 1.4]])
    precision = np.array([[0.4, 100], [20, 0.7]])
    actual = _coefficient_update(Y, Z, B, R, C, precision)
    # Full observation-space likelihood, independent of mode whitening/SVD.
    W = np.linalg.inv(np.kron(C, R))
    H = np.diag(precision.ravel(order="F"))
    score = np.zeros(4)
    for y, z in zip(Y, Z):
        design = np.kron((z @ B.T).T, np.eye(2))
        H += design.T @ W @ design
        score += design.T @ W @ y.ravel(order="F")
    expected = np.linalg.solve(H, score).reshape(2, 2, order="F")
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
    assert_allclose(H @ actual.ravel(order="F") - score, 0, atol=1e-12)


def test_covariance_and_gamma_updates_are_conditional_modes():
    rng = np.random.default_rng(2)
    E = rng.normal(size=(7, 2, 3))
    C = np.array([[1.3, 0.3, -0.1], [0.3, 0.8, 0.1], [-0.1, 0.1, 1]])
    omega = np.array([[0.7, 0.1], [0.1, 0.4]])
    xi, nu = 2.3, 4.5
    mode = _covariance_mode(E, C, omega, xi, nu)
    scatter = sum(e @ np.linalg.solve(C, e.T) for e in E) + xi * omega
    assert_allclose(mode, scatter / (7 * 3 + nu + 2 + 1))
    # Derivative w.r.t. the precision matrix vanishes at this IW mode.
    assert_allclose((7 * 3 + nu + 3) * mode - scatter, 0, atol=1e-13)
    dfs, omegas = (nu, 5), (omega, np.eye(3) / 3)
    scale = _scale_mode((mode, C), omegas, dfs, 1.7, 0.9)
    shape_minus_one = 0.7 + 0.5 * (2 * nu + 3 * 5)
    rate = 0.9 + 0.5 * (
        np.trace(np.linalg.solve(mode, omega)) + np.trace(np.linalg.solve(C, omegas[1]))
    )
    assert_allclose(shape_minus_one / scale - rate, 0, atol=1e-13)


def test_scale_acceleration_maximizes_surrogate_and_iw_prior():
    rng = np.random.default_rng(8)
    A, B = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    pa, pb = rng.uniform(1, 30, (2, 2)), rng.uniform(1, 30, (3, 3))
    a, b = _balance_coefficients(A, B, pa, pb)
    assert_allclose(np.kron(b, a), np.kron(B, A))
    assert_allclose(np.sum(pa * a**2), np.sum(pb * b**2))
    optimum = np.sum(pa * a**2) + np.sum(pb * b**2)
    for c in (0.5, 0.9, 1.1, 2):
        assert optimum < np.sum(pa * (c * a) ** 2) + np.sum(pb * (b / c) ** 2)
    covs = (np.array([[1.4, 0.3], [0.3, 0.8]]), np.diag([2.0, 0.6, 0.9]))
    omegas, dfs, xi = (np.eye(2) / 2, np.eye(3) / 3), (4, 5), 1.3
    balanced = _balance_covariances(covs, omegas, dfs, xi)
    assert_allclose(np.kron(*balanced[::-1]), np.kron(*covs[::-1]))

    def log_prior(covariances):
        return sum(
            -0.5 * (nu + len(cov) + 1) * np.linalg.slogdet(cov)[1]
            - 0.5 * xi * np.trace(np.linalg.solve(cov, omega))
            for cov, omega, nu in zip(covariances, omegas, dfs)
        )

    assert log_prior(balanced) >= log_prior(covs)
    for c in (0.5, 0.9, 1.1, 2):
        assert log_prior(balanced) > log_prior((balanced[0] * c, balanced[1] / c))


def test_log_posterior_matches_independent_density():
    from scipy.stats import beta, gamma, invwishart, multivariate_normal, norm

    rng = np.random.default_rng(1)
    Y, Z = rng.normal(size=(5, 2, 3)), rng.normal(size=(5, 2, 3))
    A, B = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    covs = (np.diag([0.8, 1.4]), np.diag([0.7, 1.2, 1.8]))
    omega, dfs = (np.eye(2) / 2, np.eye(3) / 3), (4, 5)

    def densities(A, theta, xi):
        ours = _log_posterior(
            Y, Z, A, B, covs, theta, xi, 0.02, 3, (1.3, 2.1), omega, dfs, (1.2, 0.7)
        )
        direct = sum(
            multivariate_normal.logpdf(
                e.ravel(order="F"), cov=np.kron(covs[1], covs[0])
            )
            for e in Y - A @ Z @ B.T
        )
        for coef, weight in zip((A, B), theta):
            direct += np.log(
                (1 - weight) * norm.pdf(coef, scale=np.sqrt(0.02))
                + weight * norm.pdf(coef, scale=np.sqrt(3))
            ).sum()
            direct += beta.logpdf(weight, 1.3, 2.1)
        direct += sum(
            invwishart.logpdf(cov, nu, scale=xi * o)
            for cov, nu, o in zip(covs, dfs, omega)
        )
        direct += gamma.logpdf(xi, 1.2, scale=1 / 0.7)
        return ours, direct

    first = densities(A, (0.3, 0.4), 1.4)
    second = densities(A * 1.2, (0.6, 0.2), 2.1)
    # Only constants depending on fixed hyperparameters are omitted.
    assert_allclose(first[0] - second[0], first[1] - second[1], atol=1e-12)


def test_sparse_recovery_forecast_monotonicity_and_no_mutation():
    X, A, B = _sample()
    snapshot = X.copy()
    fit = fit_sparse_mar(X, spike_variance=0.001, tol=1e-9)
    assert fit.converged
    assert fit.n_iter + 1 == len(fit.log_posterior_history)
    assert np.min(np.diff(fit.log_posterior_history)) >= -1e-9
    assert np.linalg.norm(fit.coefficients[0] - np.kron(B, A)) < 0.09
    assert_array_equal(fit.row_support, np.eye(3, dtype=bool))
    assert_array_equal(fit.column_support, np.eye(2, dtype=bool))
    assert_array_equal(X, snapshot)
    assert_allclose(fit.fitted_values + fit.residuals, X[1:])
    forecast = fit.forecast(2)
    assert_allclose(forecast[0], fit.A @ X[-1] @ fit.B.T)
    assert_allclose(forecast[1], fit.A @ forecast[0] @ fit.B.T)
    assert_allclose(fit.forecast(1, history=2 * X[-1:])[0], 2 * forecast[0])
    assert fit.is_stable
    assert np.isfinite(fit.log_likelihood)
    assert np.linalg.eigvalsh(fit.row_covariance).min() > 0
    assert np.linalg.eigvalsh(fit.column_covariance).min() > 0


def test_honest_iteration_limit_and_seeded_initialization():
    X, A, B = _sample(size=90)
    fit = fit_sparse_mar(X, initial=(A, B), max_iter=1, tol=1e-15)
    assert not fit.converged
    assert fit.n_iter == 1
    opposite = fit_sparse_mar(X, initial=(-A, -B), max_iter=1, tol=1e-15)
    assert_allclose(fit.A, opposite.A)
    assert_allclose(fit.B, opposite.B)


def test_nonsymmetric_sparse_recovery_with_correlated_innovations():
    rng = np.random.default_rng(7)
    A = np.array([[0.75, 0.3, 0], [0, 0.65, 0], [0, 0, 0.8]])
    B = np.array([[0.7, 0], [-0.3, 0.6]])
    R = np.array([[1, 0.3, 0.1], [0.3, 0.8, -0.2], [0.1, -0.2, 1.3]])
    C = np.array([[1.0, 0.4], [0.4, 1.5]])
    X = np.zeros((2300, 3, 2))
    for t in range(1, len(X)):
        X[t] = (
            A @ X[t - 1] @ B.T
            + np.linalg.cholesky(R) @ rng.normal(size=(3, 2)) @ np.linalg.cholesky(C).T
        )
    fit = fit_sparse_mar(X[100:], spike_variance=0.001, tol=1e-9)
    assert fit.converged
    assert_array_equal(fit.row_support, A != 0)
    assert_array_equal(fit.column_support, B != 0)
    assert np.linalg.norm(fit.coefficients[0] - np.kron(B, A)) < 0.09
    # Only the covariance Kronecker product is likelihood-identifiable.
    target = np.kron(C, R)
    actual = np.kron(fit.column_covariance, fit.row_covariance)
    assert np.linalg.norm(actual - target) / np.linalg.norm(target) < 0.07


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spike_variance": 0},
        {"slab_variance": 0.001},
        {"inclusion_prior": (0.5, 1)},
        {"inclusion_prior": (1,)},
        {"scale_prior": (0, 1)},
        {"scale_prior": (1, np.inf)},
        {"covariance_df": (1, 3)},
        {"covariance_scale": (np.eye(3), np.eye(3))},
        {"covariance_scale": (np.eye(3), -np.eye(2))},
        {"initial": (np.eye(2), np.eye(2))},
        {"initial": 3},
        {"covariance_scale": 2},
        {"max_iter": True},
        {"max_iter": 1.5},
        {"tol": 0},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        fit_sparse_mar(_sample(size=20)[0], **kwargs)


def test_scalar_and_rank_deficient_modes():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(80, 1, 1))
    fit = fit_sparse_mar(X)
    assert fit.forecast(2).shape == (2, 1, 1)
    assert np.isfinite(fit.log_posterior_history).all()
    with pytest.raises(ValueError, match="history"):
        fit.forecast(1, history=np.ones((2, 2, 2)))
    with pytest.raises(ValueError):
        fit.forecast(True)


@pytest.mark.parametrize(
    "X", [np.zeros((30, 2, 3)), np.ones((30, 2, 3)), np.full((3, 2, 2), np.nan)]
)
def test_degenerate_or_nonfinite_data_rejected(X):
    with pytest.raises(ValueError):
        fit_sparse_mar(X)
