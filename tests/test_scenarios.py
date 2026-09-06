import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from benchmarks.scenarios import additive_dynamic_data, monitoring_data


def test_additive_dynamic_design_has_separate_effects_and_oracle_mean():
    fit = additive_dynamic_data(50, shape=(4, 6), ranks=(1, 2), seed=18)
    assert fit.F.shape == (50, 4, 2) and fit.G.shape == (50, 6, 1)
    assert_allclose(fit.row_loadings.T @ fit.row_loadings, 4 * np.eye(1))
    assert_allclose(
        fit.column_loadings.T @ fit.column_loadings, 6 * np.eye(2), atol=1e-14
    )
    for t in range(1, 50):
        expected = fit.F[t - 1] @ fit.column_ar.T @ fit.column_loadings.T
        expected += fit.row_loadings @ fit.row_ar @ fit.G[t - 1].T
        assert_allclose(fit.conditional_mean[t], expected)
    assert_allclose(
        fit.signal,
        fit.F @ fit.column_loadings.T + fit.row_loadings @ fit.G.transpose(0, 2, 1),
    )
    assert_array_equal(
        fit.observations,
        additive_dynamic_data(50, shape=(4, 6), ranks=(1, 2), seed=18).observations,
    )


def test_monitoring_design_uses_no_future_values_before_prespecified_break():
    null, change = monitoring_data(30, 60, seed=8)
    for regime in ("space-switch", "factor-increase", "same-space-volatility"):
        changed, other = monitoring_data(30, 60, regime=regime, seed=8)
        assert other == change
        first = 30 + change - 1
        assert_array_equal(null[:first], changed[:first])
        assert not np.array_equal(null[first:], changed[first:])
    with pytest.raises(ValueError):
        monitoring_data(30, 60, change_step=0)
