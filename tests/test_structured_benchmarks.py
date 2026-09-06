import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from benchmarks.structured_ar import (
    _fit_vector_vecm,
    _flatten,
    _OracleForecast,
    _reshape,
    forecast_scores,
)
from mavats.baselines import fit_var


def test_tensor_vector_adapter_preserves_fortran_spatial_order_not_time_order():
    X = np.arange(4 * 2 * 3 * 2).reshape(4, 2, 3, 2)
    assert_array_equal(_reshape(_flatten(X), X.shape[1:]), X)
    for t in range(len(X)):
        assert_array_equal(_flatten(X)[t, :, 0], X[t].ravel(order="F"))


def test_five_step_oracle_recurses_without_future_innovations():
    fit = _OracleForecast(np.array([0.5 * np.eye(6), 0.1 * np.eye(6)]), np.ones((2, 3)))
    history = np.arange(12).reshape(2, 2, 3)
    expected = list(history.astype(float))
    for _ in range(5):
        expected.append(1 + 0.5 * expected[-1] + 0.1 * expected[-2])
    assert_allclose(fit.forecast(5, history), expected[-5:])


def test_forecast_scoring_passes_only_past_at_common_origins():
    X = np.arange(20).reshape(20, 1, 1).astype(float)

    class Recorder:
        def __init__(self):
            self.origins = []

        def forecast(self, steps, history):
            origin = 10 + len(self.origins)
            assert_array_equal(history, X[origin - 2 : origin])
            self.origins.append(origin)
            return np.repeat(history[-1:], steps, axis=0)

    fit = Recorder()
    result = forecast_scores(fit, X, 10)
    assert fit.origins == list(range(10, 16))
    assert result["origins"] == 6
    assert result["one_step_mse"] == 1
    assert result["five_step_mse"] == 25


def test_full_rank_vector_vecm_equals_unrestricted_level_var():
    X = np.random.default_rng(42).normal(size=(80, 2, 3)).cumsum(axis=0)
    vecm = _fit_vector_vecm(X, rank=6)
    var = fit_var(X, order=2, intercept=True)
    assert_allclose(vecm.transitions, var.coefficients, atol=2e-14)
    assert_allclose(vecm.intercept.ravel(order="F"), var.intercept, atol=3e-14)
    assert_allclose(vecm.forecast(5, X[-2:]), var.forecast(5), atol=1e-13)


def test_vector_vecm_preserves_requested_rank_and_physical_scale():
    X = np.random.default_rng(27).normal(size=(100, 2, 3)).cumsum(axis=0)
    fit = _fit_vector_vecm(X, rank=2)
    scaled = _fit_vector_vecm(X * 1e120, rank=2)
    assert np.linalg.matrix_rank(fit.long_run_operator, tol=1e-12) == 2
    assert_allclose(fit.transitions, scaled.transitions, atol=1e-13)
    assert_allclose(fit.intercept, scaled.intercept / 1e120, atol=1e-13)
