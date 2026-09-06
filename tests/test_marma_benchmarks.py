import numpy as np
import pytest
from numpy.testing import assert_allclose

from benchmarks.advanced_matrix import _gaussian_score
from benchmarks.constrained_extensions import (
    _IndependentSum,
    _KnownLoadingProjection,
    run_suite,
)
from benchmarks.marma import _forecast_scores, _TrueMARMA
from benchmarks.marma_scenarios import marma_data, multiterm_factor_data


def test_marma_oracle_forecasts_recurse_in_matrix_space_with_only_past_innovation():
    data = marma_data(50)
    oracle = _TrueMARMA(data)
    origin, horizon = 25, 5
    value, noise = data.observations[origin - 1], data.innovations[origin - 1]
    expected = []
    for _ in range(horizon):
        value = (
            data.intercept
            + data.ar_left[0] @ value @ data.ar_right[0].T
            - data.ma_left[0] @ noise @ data.ma_right[0].T
        )
        expected.append(value)
        noise = np.zeros_like(noise)
    predictions = oracle.forecast(horizon, data.observations[:origin])
    assert_allclose(predictions, expected, atol=2e-15)
    assert_allclose(predictions[0], data.conditional_mean[origin], atol=2e-15)
    data.innovations[origin:] += 1000
    data.observations[origin:] -= 1000
    assert_allclose(oracle.forecast(horizon, data.observations[:origin]), predictions)


def test_marma_oracle_covariance_equals_dense_future_filter_solve():
    data = marma_data(25)
    h, d = 5, 6
    ar, ma = np.eye(h * d), np.eye(h * d)
    for t in range(1, h):
        ar[t * d : (t + 1) * d, (t - 1) * d : t * d] = -data.ar_transition
        ma[t * d : (t + 1) * d, (t - 1) * d : t * d] = -data.ma_transition
    transformation = np.linalg.solve(ar, ma)
    joint = transformation @ np.kron(np.eye(h), data.covariance) @ transformation.T
    expected = np.array(
        [joint[t * d : (t + 1) * d, t * d : (t + 1) * d] for t in range(h)]
    )
    assert_allclose(_TrueMARMA(data).forecast_covariance(h), expected, atol=2e-15)


def test_marma_scores_use_identical_origins_and_only_complete_past_prefixes():
    data = marma_data(35)
    seen = []

    class Predictor:
        def forecast(self, horizon, history):
            seen.append(len(history))
            assert_allclose(history, data.observations[: len(history)])
            return np.zeros((horizon, 2, 3))

    scores = _forecast_scores(Predictor(), data, 20)
    assert seen == list(range(20, 31))
    assert scores["origins"] == 11
    assert_allclose(scores["one_step_mse"], np.mean(data.observations[20:31] ** 2))
    assert_allclose(scores["five_step_mse"], np.mean(data.observations[24:35] ** 2))


def test_known_loading_projection_uses_joint_scores_for_overlapping_terms():
    data = multiterm_factor_data(35, regime="overlapping")
    projection = _KnownLoadingProjection(data.observations[:20], data.loadings)
    design = np.concatenate([np.kron(c, r) for r, c in data.loadings], axis=1)
    flat = np.stack([x.ravel(order="F") for x in data.observations[20:]])
    expected = np.linalg.solve(design.T @ design, design.T @ flat.T).T
    assert_allclose(projection.transform(data.observations[20:]), expected, atol=1e-15)
    assert_allclose(
        projection.transform(data.observations[20:25]), expected[:5], atol=1e-15
    )


def test_independent_sum_comparator_deliberately_does_not_adjust_overlap():
    row, column = np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])
    factor = np.arange(1.0, 21.0)
    X = factor[:, None, None] * (row @ column.T)
    fit = _IndependentSum(X, [(row, column), (row, column)])
    assert_allclose(fit.signal, 2 * X)
    assert_allclose(fit.inverse_transform(fit.transform(X[-3:])), 2 * X[-3:])


def test_constrained_smoke_preserves_ranks_and_identification_diagnostics():
    rows = run_suite(quick=True, repeats=1)
    assert len(rows) == 43
    assert all(row["status"] == "ok" for row in rows), rows
    for row in rows:
        assert row["held_out_task"] == "contemporaneous-denoising-not-forecasting"
        if row["task"] == "partial-constrained-factors":
            wrong = row["regime"] == "misspecified-constraints"
            assert row["generating_row_ranks"] == [1, 2]
            assert row["generating_column_ranks"] == [2, 1]
            assert row["true_supplied_projection_row_ranks"] == [1, 3 if wrong else 2]
            assert row["true_supplied_projection_column_ranks"] == [
                2,
                2 if wrong else 1,
            ]
        if row["method"] == "partial-auto-ranks":
            assert (
                len(row["selected_row_ranks"]) == len(row["selected_column_ranks"]) == 2
            )
            assert "group_rank_diagnostics" in row
        elif row["method"] == "multi-term-known-ranks":
            assert row["score_design_rank"] == 2
            assert row["score_condition"] >= 1
            assert len(row["identification_diagnostics"]) == 2


@pytest.mark.parametrize(
    "covariance",
    [np.diag([-1, -1, 1, 1, 1, 1]), np.zeros((6, 6)), np.tril(np.ones((6, 6)))],
)
def test_gaussian_score_rejects_indefinite_singular_or_asymmetric_covariance(
    covariance,
):
    with pytest.raises(ValueError, match="positive definiteness"):
        _gaussian_score(np.zeros((1, 2, 3)), covariance[None])


def test_gaussian_score_cholesky_matches_dense_density():
    data = marma_data(10)
    cov = data.covariance
    errors = np.stack([x.ravel(order="F") for x in data.innovations])
    expected = 0.5 * (
        6 * np.log(2 * np.pi)
        + np.linalg.slogdet(cov)[1]
        + np.mean(np.sum(errors * np.linalg.solve(cov, errors.T).T, axis=1))
    )
    assert_allclose(
        _gaussian_score(data.innovations, np.repeat(cov[None], 10, axis=0)), expected
    )
