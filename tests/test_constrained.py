"""Independent moment and invariance checks for known constraint spans."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mavats.constrained import fit_constrained_factor
from mavats.factors import fit_lagged_factor


def _projector(loading):
    return loading @ loading.T


def _explicit_loading(series, rank, lags):
    dimension, columns = series.shape[1:]
    moment = np.zeros((dimension, dimension))
    for lag in lags:
        for first in range(columns):
            for second in range(columns):
                covariance = np.zeros_like(moment)
                for time in range(len(series) - lag):
                    covariance += np.outer(
                        series[time, :, first], series[time + lag, :, second]
                    )
                covariance /= len(series) - lag
                moment += covariance @ covariance.T
    _, loading = np.linalg.eigh(moment)
    return loading[:, -rank:]


def test_constrained_matches_independent_lag_moments():
    rng = np.random.default_rng(900)
    X = rng.normal(size=(35, 6, 5))
    # These constraints are already orthonormal so the dense oracle needs no
    # knowledge of the implementation's orthogonalization or moment helpers.
    row = np.eye(6)[:, [0, 2, 4]]
    column = np.eye(5)[:, [1, 3]]
    reduced = np.array([row.T @ observation @ column for observation in X])
    expected_row = row @ _explicit_loading(reduced, 2, [1, 3])
    expected_col = column @ _explicit_loading(reduced.transpose(0, 2, 1), 1, [1, 3])
    fit = fit_constrained_factor(
        X, (2, 1), row_constraints=row, column_constraints=column, lags=[1, 3]
    )
    assert_allclose(_projector(fit.loadings[0]), _projector(expected_row), atol=2e-13)
    assert_allclose(_projector(fit.loadings[1]), _projector(expected_col), atol=2e-13)
    expected = np.array(
        [
            _projector(expected_row) @ observation @ _projector(expected_col)
            for observation in X
        ]
    )
    assert_allclose(fit.signal, expected, atol=3e-13)


def test_identity_constraints_reduce_to_unconstrained_estimator():
    X = np.random.default_rng(901).normal(size=(50, 5, 4))
    constrained = fit_constrained_factor(X, (2, 2), lags=2, center=True)
    ordinary = fit_lagged_factor(X, (2, 2), lags=2, center=True)
    assert_allclose(constrained.signal, ordinary.signal, atol=2e-13)


def test_reparameterizing_constraint_basis_preserves_estimates():
    rng = np.random.default_rng(902)
    X = rng.normal(size=(55, 7, 6))
    row = rng.normal(size=(7, 3))
    column = rng.normal(size=(6, 2))
    first = fit_constrained_factor(
        X, (2, 1), row_constraints=row, column_constraints=column
    )
    changed_row = row @ np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 3.0]])
    changed_col = column @ np.array([[2.0, 3.0], [0.0, 1.0]])
    # Different units per constraint column must not change a subspace.
    changed_row *= np.array([1e150, 1e-150, 1.0])
    second = fit_constrained_factor(
        X, (2, 1), row_constraints=changed_row, column_constraints=changed_col
    )
    assert_allclose(first.signal, second.signal, atol=2e-12)


def test_constraints_remove_noise_outside_known_spaces():
    rng = np.random.default_rng(903)
    row = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]) / np.sqrt(2)
    column = np.ones((3, 1)) / np.sqrt(3)
    t = np.arange(60)
    core = np.stack((np.sin(t / 4), np.cos(t / 7)), axis=1)[:, :, None]
    signal = row @ core @ column.T
    noise = rng.normal(size=signal.shape)
    noise = noise - _projector(row) @ noise
    fit = fit_constrained_factor(
        signal + noise, (2, 1), row_constraints=row, column_constraints=column
    )
    assert_allclose(fit.signal, signal, atol=3e-14)
    assert_allclose(fit.residuals, noise, atol=3e-14)


def test_centering_and_new_observation_transform():
    rng = np.random.default_rng(904)
    X = rng.normal(size=(30, 4, 3)) + 10
    constraints = np.eye(4)[:, :2]
    fit = fit_constrained_factor(X, (1, 1), row_constraints=constraints, center=True)
    assert_allclose(fit.mean, X.mean(axis=0))
    assert_allclose(fit.inverse_transform(fit.transform(X)), fit.signal)
    assert_allclose((np.eye(4) - _projector(constraints)) @ (fit.signal - fit.mean), 0)
    future = X[-2:] + 30
    expected = np.array(
        [fit.loadings[0].T @ (x - fit.mean) @ fit.loadings[1] for x in future]
    )
    assert_allclose(fit.transform(future), expected)


@pytest.mark.parametrize("multiplier", [1e-100, 1e100])
def test_data_scaling_preserves_fitted_signal(multiplier):
    X = np.random.default_rng(905).normal(size=(25, 4, 3))
    constraints = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    ordinary = fit_constrained_factor(X, (1, 1), row_constraints=constraints)
    scaled = fit_constrained_factor(X * multiplier, (1, 1), row_constraints=constraints)
    assert_allclose(scaled.signal / multiplier, ordinary.signal, atol=2e-13)
    assert np.isfinite(scaled.signal).all()


@pytest.mark.parametrize(
    "constraints",
    [
        np.ones((4, 2)),
        np.zeros((4, 1)),
        np.ones((3, 1)),
        np.empty((4, 0)),
        np.ones((4, 5)),
        np.full((4, 1), np.nan),
        np.ones((4, 1), dtype=complex),
        ["invalid"],
    ],
)
def test_invalid_constraints_raise(constraints):
    X = np.random.default_rng(906).normal(size=(15, 4, 3))
    with pytest.raises(ValueError, match="row_constraints"):
        fit_constrained_factor(X, (1, 1), row_constraints=constraints)


def test_rank_cannot_exceed_constraint_dimension():
    X = np.ones((15, 4, 3))
    with pytest.raises(ValueError, match="ranks"):
        fit_constrained_factor(X, (2, 1), row_constraints=np.ones((4, 1)))
