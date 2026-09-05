"""Objective, update, robustness, and equivariance tests for matrix Huber loss."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mavats.factors import fit_alpha_pca
from mavats.huber import fit_huber_factor


def _loss(X, row, column, threshold):
    total = 0.0
    norms = []
    for observation in X:
        residual = observation - row @ (row.T @ observation @ column) @ column.T
        norm = np.linalg.norm(residual, "fro")
        norms.append(norm)
        total += norm**2 if norm <= threshold else 2 * threshold * norm - threshold**2
    return total / len(X), np.asarray(norms)


def _leading(moment, rank):
    return np.linalg.eigh(moment)[1][:, -rank:]


def test_one_sweep_matches_independent_weighted_projection():
    X = np.random.default_rng(120).normal(size=(50, 5, 4))
    initial = (np.eye(5)[:, :2], np.eye(4)[:, :1])
    threshold = 3.0
    old_loss, norms = _loss(X, *initial, threshold)
    weights = np.minimum(1.0, threshold / norms)
    row_moment = sum(
        w * x @ initial[1] @ initial[1].T @ x.T for x, w in zip(X, weights)
    ) / len(X)
    col_moment = sum(
        w * x.T @ initial[0] @ initial[0].T @ x for x, w in zip(X, weights)
    ) / len(X)
    row, col = _leading(row_moment, 2), _leading(col_moment, 1)
    if _loss(X, row, col, threshold)[0] > old_loss:
        col_moment = sum(w * x.T @ row @ row.T @ x for x, w in zip(X, weights)) / len(X)
        col = _leading(col_moment, 1)
    fit = fit_huber_factor(X, (2, 1), threshold=threshold, initial=initial, max_iter=1)
    assert_allclose(fit.loadings[0] @ fit.loadings[0].T, row @ row.T, atol=2e-13)
    assert_allclose(fit.loadings[1] @ fit.loadings[1].T, col @ col.T, atol=2e-13)
    expected = _loss(X, row, col, threshold)[0] / fit.data_scale**2
    assert_allclose(fit.objective_history[-1], expected, rtol=1e-13)


def test_actual_matrix_huber_loss_and_final_weights():
    X = np.random.default_rng(121).normal(size=(70, 6, 5))
    fit = fit_huber_factor(X, (2, 2), threshold=4.0)
    objective, norms = _loss(X, *fit.loadings, 4.0)
    assert_allclose(
        fit.objective_history[-1], objective / fit.data_scale**2, rtol=1e-13
    )
    assert_allclose(fit.weights, np.minimum(1.0, 4.0 / norms), rtol=1e-13)
    assert np.all(np.diff(fit.objective_history) <= 0)
    assert fit.n_iter == len(fit.objective_history) - 1
    # Huber applied to scalar entries has a different objective.
    entrywise = np.where(
        np.abs(fit.residuals) <= 4.0, fit.residuals**2, 8 * np.abs(fit.residuals) - 16
    ).sum() / len(X)
    assert not np.isclose(objective, entrywise)


def test_default_threshold_is_fixed_initial_median():
    X = np.random.default_rng(122).normal(size=(80, 5, 4))
    initial = fit_alpha_pca(X, (2, 2))
    expected = np.median(np.linalg.norm(initial.residuals, axis=(1, 2)))
    fit = fit_huber_factor(X, (2, 2))
    assert_allclose(fit.threshold, expected, rtol=1e-13)
    explicit = fit_huber_factor(X, (2, 2), threshold=expected)
    assert_allclose(fit.signal, explicit.signal, atol=5e-12)


def test_exact_signal_and_transform_with_explicit_threshold():
    rng = np.random.default_rng(123)
    row = np.linalg.qr(rng.normal(size=(7, 2)))[0]
    col = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    X = row @ rng.normal(size=(60, 2, 2)) @ col.T
    fit = fit_huber_factor(X, (2, 2), threshold=1.0)
    assert_allclose(fit.signal, X, atol=2e-14)
    assert_allclose(fit.inverse_transform(fit.transform(X)), fit.signal, atol=2e-14)
    assert_allclose(fit.weights, 1.0)


@pytest.mark.parametrize("multiplier", [1e-100, 1e100])
@pytest.mark.parametrize("threshold", [None, 2.0])
def test_scale_equivariance(multiplier, threshold):
    X = np.random.default_rng(124).normal(size=(50, 5, 4))
    first = fit_huber_factor(X, (2, 2), threshold=threshold)
    second = fit_huber_factor(
        X * multiplier,
        (2, 2),
        threshold=None if threshold is None else threshold * multiplier,
    )
    assert_allclose(first.signal, second.signal / multiplier, atol=2e-9)
    assert_allclose(first.threshold, second.threshold / multiplier)
    assert_allclose(first.weights, second.weights, atol=2e-9)


def test_centering_uses_training_mean():
    X = np.random.default_rng(125).normal(size=(45, 4, 3)) + 20
    fit = fit_huber_factor(X, (1, 1), center=True)
    assert_allclose(fit.mean, X.mean(axis=0))
    assert_allclose(fit.inverse_transform(fit.transform(X)), fit.signal)


def test_maximum_iteration_status_and_no_input_mutation():
    X = np.random.default_rng(126).normal(size=(40, 6, 5))
    before = X.copy()
    fit = fit_huber_factor(X, (2, 2), max_iter=1, tol=1e-15)
    assert not fit.converged
    assert fit.stopping_reason == "max_iter"
    assert_allclose(X, before)


def test_descent_safeguard_handles_an_increasing_simultaneous_proposal():
    X = np.random.default_rng(2).normal(size=(12, 4, 4))
    initial = (np.eye(4)[:, :1], np.eye(4)[:, :1])
    fit = fit_huber_factor(X, (1, 1), threshold=2.0, initial=initial)
    assert fit.safeguard_steps >= 1
    assert fit.converged
    assert np.all(np.diff(fit.objective_history) <= 0)
    objective = _loss(X, *fit.loadings, 2.0)[0]
    assert_allclose(fit.objective_history[-1], objective / fit.data_scale**2)


def test_matrix_contamination_is_downweighted_and_loading_error_improves():
    rng = np.random.default_rng(128)
    row = np.linalg.qr(rng.normal(size=(8, 2)))[0]
    column = np.linalg.qr(rng.normal(size=(7, 2)))[0]
    signal = row @ (5 * rng.normal(size=(200, 2, 2))) @ column.T
    X = signal + 0.15 * rng.normal(size=signal.shape)
    X[:10] += 12 * rng.normal(size=X[:10].shape)
    fit = fit_huber_factor(X, (2, 2), threshold=2.0)
    ordinary = fit_alpha_pca(X, (2, 2))

    def error(result):
        return sum(
            np.linalg.norm(estimated @ estimated.T - actual @ actual.T)
            for estimated, actual in zip(result.loadings, (row, column))
        )

    assert error(fit) < 0.1
    assert error(fit) < error(ordinary) / 5
    assert np.median(fit.weights[:10]) < 0.1 * np.median(fit.weights[10:])
    assert fit.converged


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0},
        {"threshold": -1},
        {"threshold": np.inf},
        {"threshold": np.nan},
        {"threshold": True},
        {"tol": 0},
        {"max_iter": 0},
        {"initial": (np.ones((4, 2)), np.ones((3, 2)))},
    ],
)
def test_invalid_options(kwargs):
    X = np.random.default_rng(127).normal(size=(20, 4, 3))
    with pytest.raises(ValueError):
        fit_huber_factor(X, (2, 2), **kwargs)


def test_explicit_ranks_and_positive_auto_threshold_required():
    with pytest.raises(ValueError, match="ranks"):
        fit_huber_factor(np.ones((20, 4, 3)), None)
    with pytest.raises(ValueError, match="median residual"):
        fit_huber_factor(np.zeros((20, 4, 3)), (1, 1))
    fit = fit_huber_factor(np.zeros((20, 4, 3)), (1, 1), threshold=1.0)
    assert np.isfinite(fit.signal).all()
    assert_allclose(fit.objective_history, 0)
