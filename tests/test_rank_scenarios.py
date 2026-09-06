import numpy as np
import pytest
from numpy.testing import assert_allclose

from benchmarks.rank_scenarios import (
    population_factor_lag_moment,
    rank_factor_data,
)


@pytest.mark.parametrize(
    "regime",
    ["strong", "weak", "cancellation", "white-factors", "no-factors", "serial-noise"],
)
def test_rank_design_reconstruction_and_reproducibility(regime):
    data = rank_factor_data(40, shape=(6, 8), regime=regime)
    again = rank_factor_data(40, shape=(6, 8), regime=regime)
    assert_allclose(data.observations, again.observations, rtol=0, atol=0)
    assert_allclose(data.observations, data.signal + data.noise)
    row, column = data.loadings
    expected = np.array([row @ f @ column.T for f in data.factors])
    assert_allclose(data.signal, expected, atol=3e-15)
    assert data.noise_ar == (0.5 if regime == "serial-noise" else 0)


@pytest.mark.parametrize("mode", [0, 1])
def test_exact_population_cancellation_is_specific_to_inner_lag_one(mode):
    data = rank_factor_data(10, shape=(6, 8), regime="cancellation")
    inner1 = population_factor_lag_moment(data, 1, mode, "tipup")
    inner2 = population_factor_lag_moment(data, 2, mode, "tipup")
    outer1 = population_factor_lag_moment(data, 1, mode, "topup")
    assert np.linalg.norm(inner1) < 1e-24
    assert np.linalg.matrix_rank(inner2) == 2
    assert np.linalg.matrix_rank(outer1) == 2
    # Equal strengths imply the nonzero mode Gram eigenvalues coincide.
    assert_allclose(np.linalg.eigvalsh(outer1)[-2:], 2 * (6 * 8) ** 2 * 0.65**2)
    assert_allclose(np.linalg.eigvalsh(inner2)[-2:], 4 * (6 * 8) ** 2 * 0.65**4)


@pytest.mark.parametrize("regime", ["white-factors", "no-factors"])
def test_lag_methods_have_no_population_signal_when_temporal_information_absent(regime):
    data = rank_factor_data(10, regime=regime)
    for lag in (1, 2):
        for mode in (0, 1):
            for method in ("topup", "tipup"):
                assert_allclose(
                    population_factor_lag_moment(data, lag, mode, method), 0
                )
    assert data.ranks == ((0, 0) if regime == "no-factors" else (2, 2))


def test_weak_design_changes_only_loading_strength_with_paired_noise_and_cores():
    strong = rank_factor_data(50, regime="strong")
    weak = rank_factor_data(50, regime="weak")
    assert_allclose(strong.factors, weak.factors, rtol=0, atol=0)
    assert_allclose(strong.noise, weak.noise, rtol=0, atol=0)
    for dimension, original, shrunk in zip((10, 12), strong.loadings, weak.loadings):
        assert_allclose(shrunk[:, 0], original[:, 0])
        assert_allclose(np.linalg.norm(shrunk[:, 1]), dimension**0.25)


def test_design_does_not_consume_global_random_stream():
    original = np.random.get_state()
    try:
        np.random.seed(123)
        before = np.random.get_state()
        rank_factor_data(10)
        after = np.random.get_state()
        assert before[0] == after[0]
        assert_allclose(before[1], after[1], rtol=0, atol=0)
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(original)
