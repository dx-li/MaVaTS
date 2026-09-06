import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from benchmarks.structured_scenarios import (
    cointegrated_data,
    contaminated_factor_data,
    tensor_ar_data,
)


@pytest.mark.parametrize("regime", ["isotropic", "separable", "weak-adjustment"])
def test_cointegrated_design_matches_matrix_error_correction_and_i1_roots(regime):
    data = cointegrated_data(40, regime=regime)
    X = data.observations
    for t in range(2, len(X)):
        expected = X[t - 1] + data.A1 @ X[t - 1] @ data.A2.T
        expected += (
            data.short_run_left[0] @ (X[t - 1] - X[t - 2]) @ data.short_run_right[0].T
        )
        expected += data.intercept
        assert_allclose(data.conditional_mean[t], expected, atol=3e-14)
    assert_allclose(X, data.conditional_mean + data.innovations)
    roots = np.linalg.eigvals(data.companion)
    unit = np.abs(roots - 1) < 1e-10
    assert sum(unit) == 3 * 4 - 1 * 2
    assert max(abs(roots[~unit])) < 1
    assert np.linalg.matrix_rank(data.companion - np.eye(24), tol=1e-10) == 14
    assert_allclose(data.beta1.T @ data.intercept @ data.beta2, 0, atol=1e-16)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 2)])
@pytest.mark.parametrize("regime", ["isotropic", "separable", "nonseparable"])
def test_tensor_design_dense_vectorization_agrees_with_explicit_mode_indices(
    shape, regime
):
    data = tensor_ar_data(35, shape=shape, regime=regime)
    X = data.observations
    for t in range(2, len(X)):
        expected = np.zeros(shape)
        for lag, terms in enumerate(data.coefficients, 1):
            for term in terms:
                if len(shape) == 2:
                    expected += term[0] @ X[t - lag] @ term[1].T
                else:
                    expected += np.einsum("ia,jb,kc,abc->ijk", *term, X[t - lag])
        assert_allclose(data.conditional_mean[t], expected, atol=1e-15)
    assert_allclose(X, data.conditional_mean + data.innovations)
    assert max(abs(np.linalg.eigvals(data.companion))) < 1
    assert np.linalg.eigvalsh(data.covariance)[0] > 0
    assert_array_equal(X, tensor_ar_data(35, shape=shape, regime=regime).observations)


def test_structured_designs_reject_unknown_regimes():
    with pytest.raises(ValueError):
        cointegrated_data(40, regime="not-a-design")
    with pytest.raises(ValueError):
        tensor_ar_data(40, regime="not-a-design")


def test_contaminated_factor_comparisons_share_clean_signal_and_noise():
    clean = contaminated_factor_data(60)
    for regime in ("entry-outliers", "matrix-outliers"):
        data = contaminated_factor_data(60, regime=regime)
        assert_array_equal(clean.signal, data.signal)
        assert_array_equal(
            clean.observations[~data.contamination_mask],
            data.observations[~data.contamination_mask],
        )
        assert np.any(clean.observations != data.observations)
        if regime == "matrix-outliers":
            assert_array_equal(
                np.any(data.contamination_mask, axis=(1, 2)),
                np.all(data.contamination_mask, axis=(1, 2)),
            )
