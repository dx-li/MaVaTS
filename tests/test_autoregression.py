"""Numerical contracts using direct algebra and independently simulated series."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mavats.autoregression import (
    _left_update,
    _normalize,
    _rearrange_phi,
    _right_update,
    _whitener,
    fit_mar,
    nearest_kronecker_product,
    select_mar_rank,
)
from mavats.MAR import estimate_mar1, estimate_residual_cov


def series(size=3000, seed=24, intercept=None, separable=False):
    rng = np.random.default_rng(seed)
    A = np.array([[0.7, 0.3], [-0.1, 0.45]])
    B = np.array([[0.55, 0.2, -0.1], [0.05, 0.4, 0.15], [-0.2, 0.0, 0.35]])
    R = np.array([[1.0, 0.4], [0.4, 1.5]]) if separable else np.eye(2)
    C = (
        np.array([[1.0, 0.3, 0.1], [0.3, 0.8, -0.2], [0.1, -0.2, 0.6]])
        if separable
        else np.eye(3)
    )
    mean = np.zeros((2, 3)) if intercept is None else intercept
    result = np.zeros((size + 200, 2, 3))
    for t in range(1, len(result)):
        noise = (
            np.linalg.cholesky(R) @ rng.normal(size=(2, 3)) @ np.linalg.cholesky(C).T
        )
        result[t] = mean + A @ result[t - 1] @ B.T + noise
    return result[200:], A, B, R, C


def test_rearrangement_and_nearest_kronecker_exact_nonsymmetric():
    rng = np.random.default_rng(4)
    A, B = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    phi = np.kron(B, A)
    assert_allclose(
        _rearrange_phi(phi, 2, 3), np.outer(A.ravel(order="F"), B.ravel(order="F"))
    )
    a, b = nearest_kronecker_product(phi, (2, 3))
    assert_allclose(np.kron(b, a), phi, atol=3e-14)
    assert_allclose(np.linalg.norm(a), 1)
    assert a.flat[np.argmax(np.abs(a))] >= 0


def test_projection_is_frobenius_optimum_for_arbitrary_var():
    rng = np.random.default_rng(7)
    phi = rng.normal(size=(6, 6))
    a, b = nearest_kronecker_product(phi, (2, 3))
    spectrum = np.linalg.svd(_rearrange_phi(phi, 2, 3), compute_uv=False)
    assert_allclose(np.sum((phi - np.kron(b, a)) ** 2), np.sum(spectrum[1:] ** 2))


@pytest.mark.parametrize("method", ["projection", "als", "mle"])
def test_recovers_forward_nonsymmetric_dynamics(method):
    X, A, B, _, _ = series()
    fit = fit_mar(X, method=method)
    assert np.linalg.norm(np.kron(fit.B, fit.A) - np.kron(B, A)) < 0.10
    assert fit.converged
    assert fit.is_stable
    assert_allclose(fit.fitted_values + fit.residuals, X[1:])
    expected = fit.A @ X[-1] @ fit.B.T
    assert_allclose(fit.forecast(2)[0], expected)
    assert_allclose(fit.forecast(2)[1], fit.A @ expected @ fit.B.T)
    assert_allclose(
        fit.coefficients[0] @ X[-1].ravel(order="F"), expected.ravel(order="F")
    )


def test_coordinate_updates_satisfy_unweighted_normal_equations():
    rng = np.random.default_rng(78)
    Z, Y = rng.normal(size=(50, 2, 3)), rng.normal(size=(50, 2, 3))
    A, B = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    a = _left_update(Y, Z, B)
    design = Z @ B.T
    assert_allclose(
        np.sum((Y - a @ design) @ design.transpose(0, 2, 1), axis=0), 0, atol=1e-12
    )
    b = _right_update(Y, Z, A)
    design = A @ Z
    assert_allclose(
        np.sum((Y - design @ b.T).transpose(0, 2, 1) @ design, axis=0), 0, atol=1e-12
    )


def test_weighted_coordinate_updates_satisfy_likelihood_scores():
    rng = np.random.default_rng(23)
    Z, Y = rng.normal(size=(50, 2, 3)), rng.normal(size=(50, 2, 3))
    A, B = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    C = np.array([[2, 0.4, -0.1], [0.4, 1, 0.3], [-0.1, 0.3, 1.5]])
    R = np.array([[1, 0.3], [0.3, 2]])
    a = _left_update(Y, Z, B, whitening=_whitener(C))
    residual = Y - a @ Z @ B.T
    assert_allclose(
        np.sum(residual @ np.linalg.inv(C) @ B @ Z.transpose(0, 2, 1), axis=0),
        0,
        atol=2e-12,
    )
    b = _right_update(Y, Z, A, whitening=_whitener(R))
    residual = Y - A @ Z @ b.T
    assert_allclose(
        np.sum(residual.transpose(0, 2, 1) @ np.linalg.inv(R) @ A @ Z, axis=0),
        0,
        atol=2e-12,
    )


@pytest.mark.parametrize("ridge", [0, 10, 1000])
def test_als_objective_monotonic_and_invariant_penalty(ridge):
    X, _, _, _, _ = series(size=180)
    fit = fit_mar(X, order=2, ridge=ridge, tol=1e-11, max_iter=500)
    assert np.max(np.diff(fit.objective_history)) < 1e-8
    expected = np.sum(fit.residuals**2) + ridge * sum(
        np.sum(phi**2) for phi in fit.coefficients
    )
    assert_allclose(fit.objective_history[-1], expected / fit.data_scale**2)
    assert_allclose(np.linalg.norm(fit.left, axis=(1, 2)), 1)


def test_mle_recovers_covariance_and_loglike_matches_dense_gaussian():
    X, _, _, R, C = series(size=4000, separable=True)
    fit = fit_mar(X, method="mle", tol=1e-10, covariance_floor=0)
    assert fit.converged and not fit.covariance_regularized
    estimate = np.kron(fit.column_covariance, fit.row_covariance)
    truth = np.kron(C, R)
    assert np.linalg.norm(estimate - truth) / np.linalg.norm(truth) < 0.07
    assert np.max(np.diff(fit.objective_history)) < 1e-7
    residuals = fit.residuals.transpose(0, 2, 1).reshape(len(fit.residuals), -1)
    quadratic = np.sum(residuals * np.linalg.solve(estimate, residuals.T).T)
    expected = -0.5 * (
        len(residuals) * (6 * np.log(2 * np.pi) + np.linalg.slogdet(estimate)[1])
        + quadratic
    )
    assert_allclose(fit.log_likelihood, expected)
    assert_allclose(
        fit.log_likelihood,
        -fit.objective_history[-1] - fit.residuals.size * np.log(fit.data_scale),
    )


@pytest.mark.parametrize("method", ["als", "mle", "projection"])
def test_intercept_profile_centers_residuals(method):
    intercept = np.array([[1, -2, 3], [2, 0.5, -1]])
    X, _, _, _, _ = series(size=2000, intercept=intercept)
    fit = fit_mar(X, method=method, fit_intercept=True)
    assert_allclose(fit.residuals.mean(axis=0), 0, atol=2e-13)
    assert np.linalg.norm(fit.intercept - intercept) < 0.45
    assert_allclose(fit.forecast(1)[0], fit.intercept + fit.A @ X[-1] @ fit.B.T)


def test_mar2_scalar_matches_independent_ols_and_recursive_forecast():
    rng = np.random.default_rng(32)
    x = np.zeros(3000)
    for t in range(2, len(x)):
        x[t] = 0.4 + 0.55 * x[t - 1] - 0.3 * x[t - 2] + rng.normal()
    fit = fit_mar(x[:, None, None], order=2, fit_intercept=True, tol=1e-12)
    oracle = np.linalg.lstsq(
        np.column_stack((np.ones(len(x) - 2), x[1:-1], x[:-2])), x[2:], rcond=None
    )[0]
    assert_allclose(fit.intercept.item(), oracle[0], atol=1e-8)
    assert_allclose(fit.coefficients[:, 0, 0], oracle[1:], atol=1e-8)
    expected = [x[-2], x[-1]]
    for _ in range(4):
        expected.append(oracle[0] + oracle[1] * expected[-1] + oracle[2] * expected[-2])
    assert_allclose(fit.forecast(4)[:, 0, 0], expected[2:], atol=1e-8)
    roots = np.linalg.eigvals([[oracle[1], oracle[2]], [1, 0]])
    assert_allclose(fit.spectral_radius, np.max(np.abs(roots)), atol=1e-8)


def test_degenerate_and_collinear_designs_are_supported_by_als():
    zero = fit_mar(np.zeros((20, 2, 3)))
    assert_allclose(zero.forecast(3), 0)
    rng = np.random.default_rng(91)
    X = np.repeat(rng.normal(size=(30, 1, 1)), 6, axis=2).reshape(30, 2, 3)
    fit = fit_mar(X, ridge=0.1)
    assert np.isfinite(fit.forecast(3)).all()


def test_degenerate_mle_reports_undefined_covariance():
    with pytest.raises(ValueError, match="zero residual variance"):
        fit_mar(np.zeros((20, 2, 3)), method="mle")


def test_initialization_does_not_mutate_and_random_is_reproducible():
    X, A, B, _, _ = series(size=60)
    original = X.copy(), A.copy(), B.copy()
    fit_mar(X, initial=(A, B), max_iter=2)
    for value, saved in zip((X, A, B), original):
        assert_allclose(value, saved, atol=0)
    f1 = fit_mar(X, init="random", random_state=44)
    f2 = fit_mar(X, init="random", random_state=44)
    assert_allclose(f1.coefficients, f2.coefficients, atol=0)
    _, _ = _normalize(A, B)
    assert_allclose(A, original[1], atol=0)
    assert_allclose(B, original[2], atol=0)


def test_convergence_flag_and_residual_covariance_conventions():
    X, _, _, _, _ = series(size=60)
    fit = fit_mar(X, max_iter=1, tol=0)
    assert not fit.converged and fit.n_iter == 1
    flat = fit.residuals.transpose(0, 2, 1).reshape(59, -1)
    assert_allclose(fit.residual_covariance(ddof=1), np.cov(flat, rowvar=False))
    assert_allclose(estimate_residual_cov(X, fit.A, fit.B), np.cov(flat, rowvar=False))
    assert fit.log_likelihood is None


def test_auto_initialization_avoids_dense_projection_for_large_matrices(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("auto should not allocate the unrestricted VAR")

    monkeypatch.setattr("mavats.autoregression.nearest_kronecker_product", forbidden)
    X = np.random.default_rng(6).normal(size=(10, 17, 16))
    fit = fit_mar(X, max_iter=1)
    assert fit.forecast(1).shape == (1, 17, 16)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": 0},
        {"order": True},
        {"max_iter": -1},
        {"tol": -1},
        {"tol": np.nan},
        {"ridge": -1},
        {"method": "bad"},
        {"init": "bad"},
        {"fit_intercept": 1},
        {"method": "mle", "ridge": 1},
        {"covariance_floor": 1},
        {"initial": (np.eye(3), np.eye(3))},
    ],
)
def test_invalid_options_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        fit_mar(np.ones((20, 2, 3)), **kwargs)


@pytest.mark.parametrize(
    "X",
    [
        np.zeros((2, 3)),
        np.zeros((1, 2, 3)),
        np.full((5, 2, 3), np.nan),
        np.ones((5, 2, 3), dtype=complex),
    ],
)
def test_invalid_data(X):
    with pytest.raises(ValueError):
        fit_mar(X)


def test_forecast_rejects_invalid_history_and_horizon():
    fit = fit_mar(np.random.default_rng(3).normal(size=(20, 2, 3)), order=2)
    for horizon in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            fit.forecast(horizon)
    for history in (np.ones((1, 2, 3)), np.ones((5, 3, 2))):
        with pytest.raises(ValueError):
            fit.forecast(2, history=history)


def test_legacy_wrapper_rejects_unknown_arguments_and_matches_modern_fit():
    X, _, _, _, _ = series(size=100)
    with pytest.raises(TypeError, match="unexpected"):
        estimate_mar1(X, typo=3)
    with pytest.raises(ValueError, match="together"):
        estimate_mar1(X, A_init=np.eye(2))
    A, B = estimate_mar1(X, niter=200)
    fit = fit_mar(X)
    assert_allclose(np.kron(B, A), fit.coefficients[0])


def test_reduced_rank_update_matches_published_rrls_eigenproblem():
    rng = np.random.default_rng(13)
    Z = rng.normal(size=(100, 3, 2)) * np.array([1.0, 3.0, 9.0])[None, :, None]
    Y, B = rng.normal(size=(100, 3, 2)), rng.normal(size=(2, 2))
    D = Z @ B.T
    Sxx = np.sum(D @ D.transpose(0, 2, 1), axis=0)
    Syx = np.sum(Y @ D.transpose(0, 2, 1), axis=0)
    unrestricted = np.linalg.solve(Sxx, Syx.T).T
    _, U = np.linalg.eigh(unrestricted @ Syx.T)
    expected = U[:, -1:] @ U[:, -1:].T @ unrestricted
    actual = _left_update(Y, Z, B, rank=1)
    assert_allclose(actual, expected, atol=1e-13)
    assert np.linalg.matrix_rank(actual) == 1
    # Plain coefficient truncation is not reduced-rank regression.
    u, s, vh = np.linalg.svd(unrestricted)
    truncated = (u[:, :1] * s[:1]) @ vh[:1]
    assert np.sum((Y - actual @ D) ** 2) < np.sum((Y - truncated @ D) ** 2)


@pytest.mark.parametrize("ridge", [0, 2.0])
def test_reduced_rank_als_constraints_and_objective_monotonicity(ridge):
    X, _, _, _, _ = series(size=300)
    fit = fit_mar(X, ranks=(1, 2), order=2, ridge=ridge, max_iter=500)
    assert fit.ranks == (1, 2)
    assert all(np.linalg.matrix_rank(A) <= 1 for A in fit.left)
    assert all(np.linalg.matrix_rank(B) <= 2 for B in fit.right)
    assert np.max(np.diff(fit.objective_history)) < 1e-8
    assert fit.forecast(3).shape == (3, 2, 3)


def test_rrls_rank_selection_matches_explicit_paper_ebic():
    X, _, _, _, _ = series(size=100)
    selection = select_mar_rank(X, max_ranks=(2, 2))
    assert selection.scores.shape == (2, 2)
    T, m, n = X.shape
    for r in (1, 2):
        for s in (1, 2):
            fitted = fit_mar(X, ranks=(r, s))
            expected = np.log(np.sum(fitted.residuals**2) / (T * m * n)) + (
                np.log(T * n) * r * (2 * m - r) + np.log(T * m) * s * (2 * n - s)
            ) / (T * m * n)
            assert_allclose(selection.scores[r - 1, s - 1], expected)
    assert selection.ranks == tuple(
        i + 1 for i in np.unravel_index(np.argmin(selection.scores), (2, 2))
    )
    assert selection.result.ranks == selection.ranks


def test_reduced_rank_zero_case_and_validation():
    fit = fit_mar(np.zeros((10, 3, 4)), ranks=(1, 1))
    assert np.linalg.matrix_rank(fit.A) <= 1
    assert_allclose(fit.forecast(2), 0)
    for ranks in ((0, 1), (3, 4), (1,), 1):
        with pytest.raises(ValueError):
            fit_mar(np.zeros((10, 2, 3)), ranks=ranks)
    with pytest.raises(ValueError, match="only"):
        fit_mar(np.zeros((10, 2, 3)), ranks=(1, 1), method="projection")


@pytest.mark.parametrize("method", ["als", "projection", "mle"])
@pytest.mark.parametrize("scale", [1e-200, 1e-100, 1e100, 1e200])
def test_mar_fitting_and_stopping_are_invariant_to_data_units(method, scale):
    X, _, _, _, _ = series(size=160, separable=True)
    reference = fit_mar(X, method=method, fit_intercept=True, tol=1e-10)
    scaled = fit_mar(X * scale, method=method, fit_intercept=True, tol=1e-10)
    assert scaled.converged == reference.converged
    assert scaled.n_iter == reference.n_iter
    assert np.isfinite(scaled.objective_history).all()
    assert_allclose(scaled.coefficients, reference.coefficients, atol=2e-12)
    assert_allclose(scaled.intercept / scale, reference.intercept, atol=2e-12)
    assert_allclose(scaled.forecast(4) / scale, reference.forecast(4), atol=2e-12)
    assert_allclose(scaled.objective_history, reference.objective_history, rtol=2e-12)
    if method == "mle":
        assert_allclose(
            scaled.row_covariance / scale, reference.row_covariance, atol=2e-12
        )
        assert_allclose(
            scaled.column_covariance / scale, reference.column_covariance, atol=2e-12
        )
        assert np.isfinite(scaled.log_likelihood)
        assert_allclose(
            scaled.log_likelihood,
            reference.log_likelihood - scaled.residuals.size * np.log(scale),
            rtol=2e-12,
        )


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_mar_ridge_retains_original_units_after_internal_scaling(scale):
    X, _, _, _, _ = series(size=160)
    reference = fit_mar(X, ridge=17.0, ranks=(1, 2), order=2)
    scaled = fit_mar(X * scale, ridge=(17.0 * scale) * scale, ranks=(1, 2), order=2)
    assert_allclose(scaled.coefficients, reference.coefficients, atol=3e-12)
    assert_allclose(scaled.objective_history, reference.objective_history, rtol=3e-12)


def test_legacy_mle_preserves_covariance_normalization_in_ordinary_units():
    X, _, _, _, _ = series(size=100)
    _, _, col, row = estimate_mar1(X, method="mle", niter=200)
    modern = fit_mar(X, method="mle")
    assert_allclose(np.linalg.norm(row), 1.0)
    assert_allclose(
        np.kron(col, row), np.kron(modern.column_covariance, modern.row_covariance)
    )
