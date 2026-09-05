"""Matrix GARCH equations checked against independent dense Gaussian algebra."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mavats.volatility as volatility
from mavats.volatility import (
    MatrixGARCHParameters,
    MatrixGARCHState,
    _objective_gradient,
    _Parameterization,
    filter_matrix_garch,
    fit_matrix_garch,
    simulate_matrix_garch,
)


def parameters():
    return MatrixGARCHParameters(
        [[1, 0], [0.2, 0.8]],
        [[0.2, 0.02], [0.03, 0.15]],
        [[0.4, 0.05], [0, 0.3]],
        [[1, 0, 0], [0.2, 0.8, 0], [0.1, 0.05, 0.9]],
        [[0.2, 0.02, -0.01], [0.03, 0.1, 0.01], [0.01, -0.02, 0.15]],
        [[0.3, 0.01, 0.02], [0, 0.4, 0.01], [0.01, -0.02, 0.25]],
        0.3,
        0.12,
        0.6,
    )


def dense_oracle(X, p):
    """Direct equations (4)-(9), dense column-vectorized likelihood (12)."""
    m, n = X.shape[1:]
    s1, s2, y, previous = np.zeros((m, m)), np.zeros((n, n)), 0.0, np.zeros((m, n))
    covariances, rows, columns, traces, nll = [], [], [], [], []
    for x in X:
        s1 = (
            p.A0 @ p.A0.T + p.A1 @ (previous @ previous.T) @ p.A1.T + p.A2 @ s1 @ p.A2.T
        )
        s2 = (
            p.B0 @ p.B0.T + p.B1 @ (previous.T @ previous) @ p.B1.T + p.B2 @ s2 @ p.B2.T
        )
        y = p.w + p.alpha * np.trace(previous @ previous.T) + p.beta * y
        u, v = y * s1 / np.trace(s1), s2 / np.trace(s2)
        covariance = np.kron(v, u)
        vector = x.ravel(order="F")
        nll.append(
            0.5
            * (
                m * n * np.log(2 * np.pi)
                + np.linalg.slogdet(covariance)[1]
                + vector @ np.linalg.solve(covariance, vector)
            )
        )
        covariances.append(covariance)
        rows.append(u)
        columns.append(y * v)
        traces.append(y)
        previous = x
    return (
        np.array(covariances),
        np.array(rows),
        np.array(columns),
        np.array(traces),
        np.array(nll),
        s1,
        s2,
    )


def test_filter_and_likelihood_match_full_nonsquare_equation_oracle():
    X = np.random.default_rng(3).normal(size=(18, 2, 3))
    p = parameters()
    expected = dense_oracle(X, p)
    fit = filter_matrix_garch(X, p)
    assert_allclose([fit.covariance(t) for t in range(len(X))], expected[0], atol=2e-15)
    assert_allclose(fit.row_covariances, expected[1], atol=2e-15)
    assert_allclose(fit.column_covariances, expected[2], atol=2e-15)
    assert_allclose(fit.traces, expected[3], atol=2e-15)
    assert_allclose(fit.negative_log_likelihoods, expected[4], rtol=2e-14)
    assert_allclose(fit.state.row_shape, expected[5], atol=2e-15)
    assert_allclose(fit.state.column_shape, expected[6], atol=2e-15)
    assert_allclose(fit.log_likelihood, -expected[4].sum())
    assert_allclose(np.trace(fit.column_factors, axis1=1, axis2=2), 1)
    assert_allclose(np.trace(fit.row_covariances, axis1=1, axis2=2), fit.traces)


def test_full_recursive_gradient_matches_independent_dense_likelihood_derivative():
    X = np.random.default_rng(2).normal(size=(11, 2, 3)) * 0.4
    p = parameters()
    codec = _Parameterization(p.shape, "full")
    theta = codec.pack(p)
    value, gradient = _objective_gradient(X, p, codec)
    numerical = []
    for unit in np.eye(len(theta)):
        plus = dense_oracle(X, codec.unpack(theta + 1e-6 * unit))[4].mean()
        minus = dense_oracle(X, codec.unpack(theta - 1e-6 * unit))[4].mean()
        numerical.append((plus - minus) / 2e-6)
    assert_allclose(value, dense_oracle(X, p)[4].mean())
    assert_allclose(gradient, numerical, rtol=3e-5, atol=3e-9)
    assert len(theta) == 3 + (2 * 4 + 3 - 1) + (2 * 9 + 6 - 1)


def test_scalar_garch_reduction_and_exact_forecast():
    p = MatrixGARCHParameters([[1]], [[0]], [[0]], [[1]], [[0]], [[0]], 0.3, 0.2, 0.5)
    X = np.array([1.0, -0.5, 2.0, 0.1])[:, None, None]
    y, previous, expected = 0.0, 0.0, []
    for value in X[:, 0, 0]:
        y = 0.3 + 0.2 * previous**2 + 0.5 * y
        expected.append(y)
        previous = value
    fit = filter_matrix_garch(X, p)
    assert_allclose(fit.traces, expected)
    assert_allclose(fit.standardized_residuals[:, 0, 0], X[:, 0, 0] / np.sqrt(expected))
    forecast = fit.forecast_one()
    assert_allclose(forecast.covariance(), [[0.3 + 0.2 * 0.1**2 + 0.5 * y]])
    assert_allclose(forecast.column_covariance, forecast.row_covariance)
    assert p.spectral_bounds == (0, 0, 0.7)
    assert_allclose(p.sufficient_stationarity_bound, 0.7)


def test_prefix_causality_continuation_and_forecast_match_next_filter_state():
    X = np.random.default_rng(7).normal(size=(30, 2, 3))
    p = parameters()
    all_data = filter_matrix_garch(X, p)
    prefix = filter_matrix_garch(X[:17], p)
    suffix = filter_matrix_garch(X[17:], p, initial_state=prefix.state)
    assert_allclose(suffix.row_covariances, all_data.row_covariances[17:])
    assert_allclose(
        suffix.negative_log_likelihoods, all_data.negative_log_likelihoods[17:]
    )
    forecast = prefix.forecast_one()
    assert_allclose(forecast.row_covariance, suffix.row_covariances[0])
    assert_allclose(forecast.column_covariance, suffix.column_covariances[0])
    assert_allclose(forecast.trace, suffix.traces[0])
    changed = X.copy()
    changed[17:] *= 100
    altered = filter_matrix_garch(changed, p)
    assert_allclose(altered.row_covariances[:18], all_data.row_covariances[:18])
    assert_allclose(altered.column_factors[:18], all_data.column_factors[:18])


def test_simulation_states_and_whitened_innovations():
    p = parameters()
    sim = simulate_matrix_garch(3000, p, random_state=312, burnin=80)
    fitted = filter_matrix_garch(sim.observations, p, initial_state=sim.initial_state)
    assert_allclose(fitted.row_covariances, sim.row_covariances, atol=3e-15)
    assert_allclose(fitted.column_factors, sim.column_factors, atol=3e-15)
    assert_allclose(fitted.traces, sim.traces, atol=3e-15)
    z = fitted.standardized_residuals.reshape(3000, 6)
    assert_allclose(z.mean(axis=0), 0, atol=0.055)
    assert_allclose(z.T @ z / len(z), np.eye(6), atol=0.075)
    again = simulate_matrix_garch(10, p, random_state=312, burnin=80)
    assert_array_equal(again.observations, sim.observations[:10])


@pytest.mark.parametrize("factor", [1e-100, 1e100, -3.0])
def test_filter_change_of_units_with_correct_arch_parameter_map(factor):
    X = np.random.default_rng(9).normal(size=(20, 2, 3))
    p = parameters()
    q = replace(
        p, A1=p.A1 / abs(factor), B1=p.B1 / abs(factor), w=(p.w * factor) * factor
    )
    base, scaled = filter_matrix_garch(X, p), filter_matrix_garch(X * factor, q)
    assert_allclose(
        scaled.row_covariances / factor / factor, base.row_covariances, rtol=2e-14
    )
    assert_allclose(scaled.column_factors, base.column_factors, rtol=2e-14)
    assert_allclose(
        scaled.negative_log_likelihoods,
        base.negative_log_likelihoods + 6 * np.log(abs(factor)),
        atol=1e-11,
    )
    assert_allclose(
        scaled.forecast_one().covariance() / factor / factor,
        base.forecast_one().covariance(),
        rtol=3e-14,
    )


@pytest.mark.parametrize("name", ["A1", "A2", "B1", "B2"])
def test_independent_dynamic_matrix_sign_ambiguities(name):
    X = np.random.default_rng(7).normal(size=(20, 2, 3))
    p = parameters()
    first = filter_matrix_garch(X, p)
    second = filter_matrix_garch(X, replace(p, **{name: -getattr(p, name)}))
    assert_allclose(
        first.negative_log_likelihoods, second.negative_log_likelihoods, atol=1e-13
    )


def test_published_main_simulation_fails_sufficient_bound_but_passes_spectral_bounds():
    a0 = np.array([[1, 0, 0], [0.4, 0.4, 0], [0.4, 0.4, 0.4]])
    p = MatrixGARCHParameters(
        a0,
        0.3 * np.eye(3),
        0.6 * np.eye(3),
        a0,
        0.3 * np.eye(3),
        0.6 * np.eye(3),
        0.4,
        0.3,
        0.6,
    )
    assert_allclose(p.sufficient_stationarity_bound, 1.92)
    assert_allclose(p.spectral_bounds, (0.45, 0.45, 0.9))
    assert np.isfinite(simulate_matrix_garch(20, p, random_state=2).observations).all()


def test_full_fit_optimizes_offdiagonal_dynamics_and_reports_every_start():
    p = parameters()
    X = simulate_matrix_garch(65, p, random_state=15).observations
    result = fit_matrix_garch(
        X, dynamics="full", initial=p, n_starts=2, max_iter=8, random_state=24
    )
    assert result.parameters.A1.shape == (2, 2)
    assert result.parameters.B2.shape == (3, 3)
    assert np.any(result.parameters.A1 != np.diag(np.diag(result.parameters.A1)))
    assert len(result.runs) == 2
    assert not result.converged
    assert result.runs[result.selected_start].constraint_violation < 1e-6
    assert (
        result.objective
        <= np.mean(filter_matrix_garch(X, p).negative_log_likelihoods) + 1e-8
    )
    assert_allclose(result.log_likelihood, -len(X) * result.objective)
    assert_allclose(
        result.forecast_one().row_covariance,
        result.filtered.forecast_one().row_covariance,
    )
    future = X[-2:]
    assert_allclose(
        result.filter(future).traces,
        filter_matrix_garch(
            future, result.parameters, initial_state=result.filtered.state
        ).traces,
    )
    assert_allclose(
        result.filter(future, continue_history=False).traces,
        filter_matrix_garch(future, result.parameters).traces,
    )


def test_diagonal_fit_converges_with_feasibility_and_finite_covariances():
    p = MatrixGARCHParameters(
        np.eye(2),
        np.diag([0.25, 0.15]),
        np.diag([0.4, 0.3]),
        np.eye(2),
        np.diag([0.2, 0.15]),
        np.diag([0.3, 0.45]),
        0.2,
        0.1,
        0.6,
    )
    X = simulate_matrix_garch(100, p, random_state=4).observations
    result = fit_matrix_garch(X, dynamics="diagonal", n_starts=1, max_iter=150)
    assert result.converged
    assert np.linalg.eigvalsh(result.forecast_one().covariance())[0] > 0
    assert max(result.parameters.spectral_bounds) <= 1 - result.constraint_margin + 1e-6
    for a in (
        result.parameters.A1,
        result.parameters.A2,
        result.parameters.B1,
        result.parameters.B2,
    ):
        assert_array_equal(a, np.diag(np.diag(a)))
    assert result.runs[0].objective_history[-1] <= result.runs[0].objective_history[0]


def test_scalar_simulation_parameter_recovery():
    p = MatrixGARCHParameters([[1]], [[0]], [[0]], [[1]], [[0]], [[0]], 0.3, 0.2, 0.5)
    X = simulate_matrix_garch(1600, p, random_state=32).observations
    result = fit_matrix_garch(X, n_starts=1, max_iter=100)
    assert result.converged
    assert_allclose(
        [result.parameters.w, result.parameters.alpha, result.parameters.beta],
        [0.3, 0.2, 0.5],
        atol=0.13,
    )


def test_optimizer_exact_point_cache_owns_arrays_and_covers_callback(monkeypatch):
    evaluations = []
    original = volatility._objective_gradient

    def counted(X, p, codec, **kwargs):
        evaluations.append(codec.pack(p))
        return original(X, p, codec, **kwargs)

    def repeated_solver(fun, x0, *, callback, **kwargs):
        x = x0.copy()
        value, gradient = fun(x)
        expected_gradient = gradient.copy()
        gradient[:] = 12345  # A caller cannot corrupt the stored gradient.
        repeated_value, repeated_gradient = fun(x)
        assert repeated_value == value
        assert_array_equal(repeated_gradient, expected_gradient)
        callback(x)
        x[0] += 1e-6  # Mutating an earlier input must cause a new evaluation.
        value, gradient = fun(x)
        callback(x)
        return SimpleNamespace(
            x=x, success=True, status=0, message="test solver", nit=1, nfev=3
        )

    monkeypatch.setattr(volatility, "_objective_gradient", counted)
    monkeypatch.setattr(volatility, "minimize", repeated_solver)
    X = np.random.default_rng(23).normal(size=(12, 1, 1))
    result = fit_matrix_garch(X, n_starts=1)
    # Initial evaluation, repeated solver calls, callbacks and terminal audit
    # need only two likelihood recursions for the two distinct parameter points.
    assert len(evaluations) == 2
    assert result.converged
    assert_allclose(result.objective, np.mean(dense_oracle(X, result.parameters)[4]))


def test_sufficient_constraint_and_unconstrained_choices_are_explicit():
    p = MatrixGARCHParameters([[1]], [[0]], [[0]], [[1]], [[0]], [[0]], 0.3, 0.2, 0.5)
    X = simulate_matrix_garch(40, p, random_state=2).observations
    sufficient = fit_matrix_garch(X, constraint="sufficient", n_starts=1, max_iter=15)
    assert (
        sufficient.parameters.sufficient_stationarity_bound
        <= 1 - sufficient.constraint_margin + 1e-6
    )
    unrestricted = fit_matrix_garch(X, constraint="none", n_starts=1, max_iter=5)
    assert unrestricted.constraint == "none"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"w": 0},
        {"alpha": -1},
        {"beta": np.nan},
        {"w": True},
        {"A0": [[2, 0], [0, 1]]},
        {"A0": [[1, 0.1], [0, 1]]},
        {"A0": [[1, 0], [0, 0]]},
        {"A0": [[1, 0, 0], [0, 1, 0]]},
        {"A1": [[1j, 0], [0, 1]]},
        {"B2": [[1]]},
        {"A1": [[np.inf, 0], [0, 1]]},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        replace(parameters(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dynamics": "scalar"},
        {"constraint": "stationary"},
        {"n_starts": 0},
        {"n_starts": True},
        {"max_iter": 0},
        {"tol": 0},
        {"constraint_margin": 1},
        {"parameter_bound": 40},
        {"initial": np.eye(2)},
        {"dynamics": "diagonal", "initial": parameters()},
    ],
)
def test_invalid_fit_options(kwargs):
    X = np.random.default_rng(2).normal(size=(20, 2, 3))
    with pytest.raises(ValueError):
        fit_matrix_garch(X, **kwargs)


def test_invalid_state_data_and_boundary_inputs():
    X = np.random.default_rng(2).normal(size=(20, 2, 3))
    p = parameters()
    with pytest.raises(ValueError):
        filter_matrix_garch(X[:, :1], p)
    with pytest.raises(ValueError):
        filter_matrix_garch(X, p, initial_state="bad")
    state = MatrixGARCHState(-np.eye(2), np.eye(3), 1, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        filter_matrix_garch(X, p, initial_state=state)
    with pytest.raises(ValueError):
        fit_matrix_garch(np.zeros_like(X))
    with pytest.raises(ValueError):
        simulate_matrix_garch(0, p)
    with pytest.raises(ValueError):
        simulate_matrix_garch(3, p, burnin=-1)
    with pytest.raises(ValueError):
        MatrixGARCHParameters([[1]], [[0.2]], [[0]], [[1]], [[0]], [[0]], 1, 0.1, 0.5)
    with pytest.raises(FloatingPointError):
        filter_matrix_garch(X, replace(p, A2=np.eye(2) * 1e100))


def test_parameter_arrays_and_caller_data_are_preserved():
    p = parameters()
    X = np.random.default_rng(2).normal(size=(20, 2, 3))
    original = X.copy()
    filter_matrix_garch(X, p)
    assert_array_equal(X, original)
    with pytest.raises(ValueError):
        p.A0[0, 0] = 2


@pytest.mark.parametrize("multiple", [1, 2])
def test_unrepresentable_original_covariance_is_rejected(multiple):
    p = MatrixGARCHParameters(
        np.eye(2),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.eye(2),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        multiple * np.nextafter(0.0, 1.0),
        0,
        0,
    )
    X = np.zeros((1, 2, 2))
    with pytest.raises(FloatingPointError):
        filter_matrix_garch(X, p)
    valid = filter_matrix_garch(X, replace(p, w=1))
    forecast_origin = replace(
        valid,
        parameters=p,
        state=MatrixGARCHState(np.zeros((2, 2)), np.zeros((2, 2)), 0, X[0]),
    )
    with pytest.raises(FloatingPointError):
        forecast_origin.forecast_one()
    with pytest.raises(FloatingPointError):
        simulate_matrix_garch(1, p, burnin=0)


def test_large_finite_constant_variance_avoids_inactive_energy_overflow():
    p = MatrixGARCHParameters([[1]], [[0]], [[0]], [[1]], [[0]], [[0]], 1e308, 0, 0)
    result = simulate_matrix_garch(20, p, burnin=0, random_state=0)
    expected = np.sqrt(p.w) * np.random.default_rng(0).normal(size=(20, 1, 1))
    assert_array_equal(result.observations, expected)
    assert_array_equal(result.traces, np.full(20, p.w))
    filtered = filter_matrix_garch(result.observations, p)
    assert_allclose(filtered.traces, result.traces, rtol=1e-15)
    assert_allclose(filtered.standardized_residuals, expected / np.sqrt(p.w))


def test_weighted_energy_can_be_finite_when_unweighted_energy_overflows():
    p = MatrixGARCHParameters([[1]], [[0]], [[0]], [[1]], [[0]], [[0]], 1e307, 1e-4, 0)
    row, col, trace = volatility._next(
        p, np.zeros((1, 1)), np.zeros((1, 1)), 0, np.array([[1e155]])
    )
    assert_allclose(trace, 1.1e307, rtol=1e-15)
    assert_array_equal(row, [[1]])
    assert_array_equal(col, [[1]])


def test_tiny_units_multistart_falls_back_from_unrepresentable_spectral_proposal():
    X = np.random.default_rng(4).normal(size=(8, 2, 2)) * 1e-156
    one = fit_matrix_garch(X, max_iter=1, n_starts=1)
    two = fit_matrix_garch(X, max_iter=1, n_starts=2)
    assert_allclose(two.objective, one.objective)
    assert len(two.runs) == 2
    assert two.runs[1].start_fallback
    assert np.isfinite(two.filtered.row_covariances).all()
    assert np.all(np.linalg.eigvalsh(two.filtered.row_covariances) > 0)


def test_large_units_unconstrained_multistart_retains_physical_parameters():
    X = np.ones((8, 1, 1)) * 1e154
    p = MatrixGARCHParameters([[1]], [[0]], [[0]], [[1]], [[0]], [[0]], 1.7e308, 0, 0)
    one = fit_matrix_garch(
        X, initial=p, constraint="none", random_state=3, max_iter=1, n_starts=1
    )
    two = fit_matrix_garch(
        X, initial=p, constraint="none", random_state=3, max_iter=1, n_starts=2
    )
    assert two.objective <= one.objective + 1e-12
    assert len(two.runs) == 2
    assert np.isfinite(two.parameters.w)
    assert np.isfinite(two.filtered.traces).all()
    assert np.all(two.filtered.row_covariances > 0)
