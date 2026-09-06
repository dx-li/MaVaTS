"""Fit-level contracts for Tsay's conditional matrix ARMA model."""

from copy import deepcopy

import numpy as np
import pytest

from mavats.marma import (
    MARMAParameters,
    filter_marma,
    fit_marma,
    marma_diagnostics,
    simulate_marma,
)


def _scalar(ar=0.65, ma=0.25, intercept=0.15, covariance=True):
    return MARMAParameters(
        np.ones((1, 1, 1)),
        np.array([[[ar]]]),
        np.ones((1, 1, 1)),
        np.array([[[ma]]]),
        np.array([[intercept]]),
        np.ones((1, 1)) if covariance else None,
        np.ones((1, 1)) if covariance else None,
    )


def _dense(parameters, family):
    return np.array(
        [
            np.kron(b, a)
            for a, b in zip(
                getattr(parameters, family + "_left"),
                getattr(parameters, family + "_right"),
            )
        ]
    )


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_scalar_recovery_and_run_metadata(method):
    truth = _scalar()
    data = simulate_marma(1200, truth, random_state=813).observations
    result = fit_marma(data, method=method, n_starts=1, max_iter=200, tol=1e-9)
    assert result.converged
    np.testing.assert_allclose(_dense(result.parameters, "ar"), [[[0.65]]], atol=0.09)
    np.testing.assert_allclose(_dense(result.parameters, "ma"), [[[0.25]]], atol=0.11)
    np.testing.assert_allclose(result.parameters.intercept, [[0.15]], atol=0.09)
    run = result.runs[result.selected_start]
    assert run.optimizer == "SLSQP" and run.feasible and not run.failed
    assert np.all(run.constraint_residuals >= 0)
    assert run.n_iter > 0 and np.isfinite(run.gradient_norm)
    assert result.residuals.shape == (1199, 1, 1)
    assert np.isnan(result.filter_result.fitted_values[0]).all()
    if method == "mle":
        assert run.initialization_iterations > 0
        assert run.initialization_objective_history is not None
        assert result.log_likelihood is not None
    else:
        assert run.initialization_converged is None
        assert result.log_likelihood is None


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_white_noise_mean_and_covariance_oracle(method):
    data = np.random.default_rng(601).normal(size=(180, 2, 3)) + 0.8
    result = fit_marma(data, 0, 0, method=method, n_starts=1, tol=1e-10)
    np.testing.assert_allclose(
        result.parameters.intercept, data.mean(axis=0), atol=1e-9
    )
    np.testing.assert_allclose(
        result.forecast(3), np.repeat(data.mean(axis=0)[None], 3, axis=0), atol=1e-9
    )
    covariance = result.innovation_covariance
    if method == "ls":
        errors = np.array([x.ravel(order="F") for x in data - data.mean(axis=0)])
        np.testing.assert_allclose(
            covariance, errors.T @ errors / len(data), atol=1e-10
        )
    else:
        np.testing.assert_allclose(
            covariance,
            np.kron(
                result.parameters.column_covariance, result.parameters.row_covariance
            ),
            atol=1e-10,
        )
    np.testing.assert_allclose(
        result.forecast_covariance(3), np.repeat(covariance[None], 3, axis=0)
    )


def test_ar_only_scalar_matches_independent_least_squares():
    data = np.random.default_rng(920).normal(size=(140, 1, 1))
    result = fit_marma(data, 2, 0, n_starts=1, tol=1e-11)
    design = np.column_stack([data[1:-1, 0, 0], data[:-2, 0, 0], np.ones(138)])
    expected = np.linalg.lstsq(design, data[2:, 0, 0], rcond=None)[0]
    np.testing.assert_allclose(
        _dense(result.parameters, "ar").ravel(), expected[:2], atol=2e-7
    )
    np.testing.assert_allclose(
        result.parameters.intercept.ravel(), expected[2:], atol=2e-7
    )


def test_ma_only_recovery():
    truth = MARMAParameters([], [], [[[1.0]]], [[[0.55]]], [[0.0]], [[1.0]], [[1.0]])
    data = simulate_marma(1000, truth, random_state=777).observations
    result = fit_marma(data, 0, 1, intercept=False, n_starts=1, tol=1e-10)
    assert result.converged and result.diagnostics.is_invertible
    np.testing.assert_allclose(_dense(result.parameters, "ma"), [[[0.55]]], atol=0.08)


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_training_centering_and_units(method):
    data = simulate_marma(230, _scalar(), random_state=41).observations
    options = dict(method=method, n_starts=1, max_iter=200, tol=1e-10)
    baseline = fit_marma(data, **options)
    for multiplier, offset in [(1e-100, 0.0), (1e100, 0.0), (1.0, 1e6)]:
        changed = fit_marma(data * multiplier + offset, **options)
        for family in ("ar", "ma"):
            np.testing.assert_allclose(
                _dense(changed.parameters, family),
                _dense(baseline.parameters, family),
                rtol=2e-5,
                atol=2e-6,
            )
        np.testing.assert_allclose(
            (changed.forecast(4) - offset) / multiplier,
            baseline.forecast(4),
            rtol=2e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            changed.innovation_covariance / multiplier / multiplier,
            baseline.innovation_covariance,
            rtol=2e-5,
            atol=1e-5,
        )


def test_complete_history_forecast_and_prefix_causality():
    data = simulate_marma(100, _scalar(), random_state=291).observations
    result = fit_marma(data[:70], n_starts=1)
    prefix = data[:85]
    filtered = filter_marma(prefix, result.parameters)
    np.testing.assert_allclose(result.forecast(5, history=prefix), filtered.forecast(5))
    full = filter_marma(data, result.parameters)
    np.testing.assert_allclose(full.residuals[:85], filtered.residuals)
    changed = data.copy()
    changed[85:] += 50
    np.testing.assert_allclose(
        filter_marma(changed, result.parameters).residuals[:85], filtered.residuals
    )


def test_one_iteration_keeps_feasible_start_and_reports_nonconvergence():
    data = simulate_marma(100, _scalar(), random_state=14).observations
    result = fit_marma(data, n_starts=2, max_iter=1, tol=1e-13)
    assert len(result.runs) == 2
    assert not result.converged
    assert all(run.feasible for run in result.runs if not run.failed)
    assert result.diagnostics.is_stable and result.diagnostics.is_invertible


def test_unconverged_mle_warm_up_not_hidden():
    data = simulate_marma(100, _scalar(), random_state=18).observations
    result = fit_marma(data, method="mle", n_starts=1, max_iter=1)
    run = result.runs[0]
    assert run.initialization_converged is False
    assert run.initialization_iterations == 1
    assert (
        run.initialization_message and run.initialization_objective_history is not None
    )


def test_supplied_covariance_skips_mle_warm_up_and_preserves_inputs():
    data = simulate_marma(90, _scalar(), random_state=35).observations
    original = data.copy()
    parameters = _scalar()
    snapshot = deepcopy(parameters)
    result = fit_marma(data, method="mle", initial=parameters, n_starts=1, max_iter=100)
    assert result.runs[0].initialization_converged is None
    np.testing.assert_array_equal(data, original)
    for key in parameters.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(parameters, key), getattr(snapshot, key))


def test_local_generator_is_not_advanced_and_simulation_identity():
    generator = np.random.default_rng(10)
    before = deepcopy(generator.bit_generator.state)
    simulation = simulate_marma(70, _scalar(), random_state=generator)
    assert generator.bit_generator.state == before
    np.testing.assert_allclose(
        simulation.observations, simulation.conditional_mean + simulation.innovations
    )
    fit_marma(simulation.observations, n_starts=2, max_iter=2, random_state=generator)
    assert generator.bit_generator.state == before


def test_user_infeasible_start_is_rejected_not_contracted():
    data = np.random.default_rng(6).normal(size=(60, 1, 1))
    with pytest.raises(ValueError, match="initial parameters violate"):
        fit_marma(data, initial=_scalar(ma=1.1), n_starts=1)
    result = fit_marma(
        data,
        initial=_scalar(ma=1.01),
        n_starts=1,
        max_iter=1,
        enforce_admissibility=False,
    )
    assert result.runs[0].optimizer == "BFGS"
    assert not result.enforce_admissibility


@pytest.mark.parametrize(
    "options",
    [
        {"ar_order": -1},
        {"ma_order": True},
        {"n_starts": 0},
        {"max_iter": 0},
        {"tol": 0},
        {"tol": np.nan},
        {"method": "css"},
        {"intercept": 1},
        {"enforce_admissibility": 1},
        {"stability_margin": 0},
        {"stability_margin": 1},
        {"stability_margin": 1e-20},
        {"max_dense_dimension": 1},
    ],
)
def test_invalid_fit_controls(options):
    data = np.random.default_rng(63).normal(size=(30, 2, 2))
    with pytest.raises(ValueError):
        fit_marma(data, **options)


def test_invalid_series_and_dense_guard():
    with pytest.raises(ValueError, match="constant"):
        fit_marma(np.ones((20, 2, 2)))
    with pytest.raises(ValueError):
        fit_marma(np.full((20, 1, 1), np.nan))
    parameters = MARMAParameters(
        [np.eye(3)], [np.eye(3) * 0.2], [], [], np.zeros((3, 3))
    )
    with pytest.raises(ValueError, match="max_dense_dimension"):
        marma_diagnostics(parameters, max_dense_dimension=8)
    with pytest.raises(ValueError, match="spatial dimensions"):
        filter_marma(np.ones((10, 2, 3)), parameters)


def test_simulation_rejects_unstable_ar_but_allows_noninvertible_ma():
    with pytest.raises(ValueError, match="stable AR"):
        simulate_marma(10, _scalar(ar=1.1))
    simulation = simulate_marma(10, _scalar(ma=1.2), random_state=1)
    assert np.isfinite(simulation.observations).all()
