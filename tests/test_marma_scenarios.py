import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from benchmarks.marma_scenarios import (
    marma_data,
    multiterm_factor_data,
    partial_factor_data,
)


@pytest.mark.parametrize(
    "regime", ["isotropic", "separable", "near-cancellation", "nonseparable"]
)
def test_dense_marma_design_matches_minus_ma_matrix_equation(regime):
    data = marma_data(45, regime=regime)
    X, E = data.observations, data.innovations
    expected = data.intercept + data.ar_left[0] @ X[:-1] @ data.ar_right[0].T
    expected -= data.ma_left[0] @ E[:-1] @ data.ma_right[0].T
    assert_allclose(data.conditional_mean[1:], expected, atol=2e-15)
    assert_allclose(X, data.conditional_mean + E, atol=2e-15)
    for transition in (data.ar_transition, data.ma_transition):
        assert max(abs(np.linalg.eigvals(transition))) < 1
    assert np.linalg.eigvalsh(data.covariance)[0] > 0
    assert (
        np.linalg.norm(
            data.ar_transition @ data.ma_transition
            - data.ma_transition @ data.ar_transition
        )
        > 1e-4
    )
    assert_array_equal(X, marma_data(45, regime=regime).observations)


def test_marma_near_cancellation_is_not_exact_and_nonseparable_covariance_is_not_kron():
    regular = marma_data(10)
    near = marma_data(10, regime="near-cancellation")
    gap = np.linalg.norm(near.ar_transition - near.ma_transition)
    assert 0 < gap < 0.1 * np.linalg.norm(regular.ar_transition - regular.ma_transition)
    other = marma_data(10, regime="nonseparable")
    assert (
        np.linalg.norm(
            other.covariance - np.kron(other.column_covariance, other.row_covariance)
        )
        > 0.1
    )


@pytest.mark.parametrize(
    "regime",
    [
        "full-interactions",
        "diagonal-only",
        "weak-complement",
        "misspecified-constraints",
    ],
)
def test_partial_design_has_four_shared_loading_blocks_and_orthogonal_energy(regime):
    data = partial_factor_data(40, regime=regime)
    row, column = data.loadings
    assert_allclose(data.signal, row @ data.factors @ column.T, atol=2e-15)
    blocks = [block for group in data.signal_blocks for block in group]
    assert_allclose(sum(blocks), data.signal, atol=2e-15)
    assert_allclose(
        sum(np.sum(b**2) for b in blocks), np.sum(data.signal**2), rtol=1e-14
    )
    assert_allclose(
        data.row_constraints.T @ data.row_constraints, np.eye(3), atol=1e-15
    )
    assert_allclose(
        data.column_constraints.T @ data.column_constraints, np.eye(4), atol=1e-15
    )
    if regime == "diagonal-only":
        assert_array_equal(data.signal_blocks[0][1], 0)
        assert_array_equal(data.signal_blocks[1][0], 0)


def test_partial_misspecification_changes_prior_not_data_and_requires_higher_complement_rank():
    correct = partial_factor_data(35)
    wrong = partial_factor_data(35, regime="misspecified-constraints")
    assert_array_equal(correct.observations, wrong.observations)
    for loadings, good, bad, rank in zip(
        correct.loadings,
        (correct.row_constraints, correct.column_constraints),
        (wrong.row_constraints, wrong.column_constraints),
        (2, 1),
    ):
        assert (
            np.linalg.matrix_rank(loadings - good @ (good.T @ loadings), tol=1e-12)
            == rank
        )
        assert (
            np.linalg.matrix_rank(loadings - bad @ (bad.T @ loadings), tol=1e-12)
            == rank + 1
        )


@pytest.mark.parametrize(
    "regime,cosine", [("orthogonal", 0), ("overlapping", 0.7), ("near-overlap", 0.98)]
)
def test_multiterm_design_has_controlled_overlap_and_identified_joint_scores(
    regime, cosine
):
    data = multiterm_factor_data(40, regime=regime)
    reconstructed = np.zeros_like(data.signal)
    design = []
    for (row, column), factor, component in zip(
        data.loadings, data.factors, data.component_signals
    ):
        assert_allclose(component, row @ factor @ column.T, atol=2e-15)
        reconstructed += component
        design.append(np.kron(column, row))
    assert_allclose(reconstructed, data.signal, atol=2e-15)
    assert np.linalg.matrix_rank(np.concatenate(design, axis=1)) == 2
    for mode in (0, 1):
        assert_allclose(
            data.constraints[0][mode].T @ data.constraints[1][mode],
            cosine * np.eye(2),
            atol=6e-16,
        )


@pytest.mark.parametrize(
    "generator", [marma_data, partial_factor_data, multiterm_factor_data]
)
def test_new_designs_reject_unknown_regimes(generator):
    with pytest.raises(ValueError, match="unknown"):
        generator(30, regime="not-a-design")
