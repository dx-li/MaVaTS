"""Equation oracles and identification tests for bilinear segmentation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mavats.decorrelation import (
    _connected_groups,
    _ratio_groups,
    fit_matrix_decorrelation,
)


def _oracle(oriented, max_lag):
    # Direct equations (5)--(7): intentionally construct every E_ij and every
    # signed lag. Do not reuse the estimator's SVD or streaming contractions.
    t, other, dim = oriented.shape
    marginal = sum(x.T @ x for x in oriented) / (t * other)
    values, vectors = np.linalg.eigh(marginal)
    whitener = (vectors / np.sqrt(values)) @ vectors.T
    moment = np.zeros((dim, dim))
    for lag in range(-max_lag, max_lag + 1):
        lo, hi = max(0, -lag), min(t, t - lag)
        for i in range(other):
            for j in range(other):
                selector = np.zeros((other, other))
                selector[i, j] = 1
                covariance = sum(
                    oriented[k + lag].T @ selector @ oriented[k] for k in range(lo, hi)
                ) / (hi - lo)
                v = whitener @ covariance @ whitener
                moment += v @ v.T / other**2
    return whitener, moment


def _correlation_oracle(partial, max_lag):
    t, other, dim = partial.shape
    variance = np.sum(partial**2, axis=0) / t
    result = np.zeros((dim, dim))
    for k in range(dim):
        for ell in range(dim):
            if k == ell:
                continue
            for lag in range(-max_lag, max_lag + 1):
                lo, hi = max(0, -lag), min(t, t - lag)
                for i in range(other):
                    for j in range(other):
                        covariance = sum(
                            partial[a + lag, i, k] * partial[a, j, ell]
                            for a in range(lo, hi)
                        ) / (hi - lo)
                        denominator = np.sqrt(variance[i, k] * variance[j, ell])
                        value = abs(covariance) / denominator if denominator else 0
                        result[k, ell] = max(result[k, ell], value)
    return result


@pytest.mark.parametrize("center", [True, False])
def test_full_signed_covariance_equation_oracle(center):
    X = np.random.default_rng(24).normal(size=(13, 2, 3)) + 0.7
    fit = fit_matrix_decorrelation(
        X, lags=2, correlation_lags=3, correlation_threshold=0.4, center=center
    )
    work = X - X.mean(axis=0) if center else X
    row_whitener, row_moment = _oracle(work.transpose(0, 2, 1), 2)
    column_whitener, column_moment = _oracle(work, 2)
    assert_allclose(fit.row_moment, row_moment, rtol=2e-13, atol=2e-13)
    assert_allclose(fit.column_moment, column_moment, rtol=2e-13, atol=2e-13)
    assert_allclose(
        fit.row_rotation.T @ row_moment @ fit.row_rotation,
        np.diag(fit.row_eigenvalues),
        atol=2e-13,
    )
    assert_allclose(
        fit.column_rotation.T @ column_moment @ fit.column_rotation,
        np.diag(fit.column_eigenvalues),
        atol=2e-13,
    )
    expected = (
        fit.row_rotation.T @ row_whitener @ work @ column_whitener @ fit.column_rotation
    )
    assert_allclose(fit.series, expected, atol=3e-14)
    assert_allclose(
        fit.row_correlations,
        _correlation_oracle(
            work.transpose(0, 2, 1) @ row_whitener @ fit.row_rotation, 3
        ),
        atol=3e-14,
    )
    assert_allclose(
        fit.column_correlations,
        _correlation_oracle(work @ column_whitener @ fit.column_rotation, 3),
        atol=3e-14,
    )


def test_transform_inverse_and_physical_mixing_equations():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(70, 3, 4)) + np.arange(12).reshape(3, 4)
    fit = fit_matrix_decorrelation(X)
    future = rng.normal(size=(7, 3, 4)) + 2
    for data in (X, future):
        latent = fit.transform(data)
        assert_allclose(
            latent,
            fit.row_transform @ (data - fit.mean) @ fit.column_transform,
            atol=2e-13,
        )
        assert_allclose(fit.inverse_transform(latent), data, atol=2e-13)
        assert_allclose(
            fit.row_mixing @ latent @ fit.column_mixing.T + fit.mean,
            data,
            atol=2e-13,
        )
        assert_allclose(fit.inverse_blocks(fit.blocks(data)), data, atol=2e-13)
    assert_allclose(fit.inverse_blocks(fit.blocks()), X, atol=2e-13)


@pytest.mark.parametrize("scale", [1e-150, 1e150, -1e-150, -1e150])
def test_global_scale_equivariance_and_extreme_scale_round_trip(scale):
    X = np.random.default_rng(5).normal(size=(100, 2, 3))
    fit = fit_matrix_decorrelation(X)
    scaled = fit_matrix_decorrelation(X * scale)
    assert_allclose(scaled.series * scale, fit.series, atol=3e-13)
    assert_allclose(scaled.row_moment, fit.row_moment, atol=3e-13)
    assert_allclose(scaled.column_moment, fit.column_moment, atol=3e-13)
    assert_allclose(scaled.row_correlations, fit.row_correlations, atol=3e-13)
    assert scaled.row_groups == fit.row_groups
    assert scaled.column_groups == fit.column_groups
    assert_allclose(scaled.inverse_transform(scaled.series) / scale, X, atol=3e-13)


def test_affine_offset_is_retained_without_mutating_input():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 2, 3))
    original = X.copy()
    offset = np.array([[2, 5, 8], [9, -1, -20]])
    first = fit_matrix_decorrelation(X)
    second = fit_matrix_decorrelation(X + offset)
    assert_allclose(first.series, second.series, atol=3e-13)
    assert_allclose(second.mean, first.mean + offset, atol=3e-14)
    assert_allclose(X, original, atol=0)


def _mixings():
    row = np.array([[1.0, 0.3, -0.2], [0.1, 1.0, 0.4], [0.2, -0.2, 1.0]])
    column = np.array([[1.0, 0.4, 0.1], [-0.2, 1.0, 0.3], [0.1, 0.2, 1.0]])
    return row, column


def test_planted_independent_ar_components_recover_all_nine_blocks():
    rng = np.random.default_rng(271)
    n = 10000
    phi = np.array([[0.92, 0.6, 0.2], [0.75, 0.4, 0.1], [0.55, 0.25, -0.1]])
    latent = np.zeros((n, 3, 3))
    noise = rng.normal(size=latent.shape) * np.sqrt(1 - phi**2)
    for t in range(1, n):
        latent[t] = phi * latent[t - 1] + noise[t]
    row, column = _mixings()
    X = row @ latent @ column.T
    fit = fit_matrix_decorrelation(X, correlation_lags=5, correlation_threshold=0.12)
    assert tuple(map(len, fit.row_groups)) == (1, 1, 1)
    assert tuple(map(len, fit.column_groups)) == (1, 1, 1)
    for truth, estimated in ((row, fit.row_mixing), (column, fit.column_mixing)):
        recovered = np.abs(np.linalg.solve(truth, estimated))
        recovered /= np.max(recovered, axis=0)
        assert_allclose(recovered, np.eye(3), atol=0.06)
    # Variances of the latent AR components are all one; contemporaneous PCA
    # cannot identify these directions. Temporal cross-products are essential.


@pytest.mark.parametrize("grouping", ["threshold", "ratio"])
def test_planted_dependent_rectangular_blocks_recover_subspaces(grouping):
    rng = np.random.default_rng(17)
    n = 12000
    latent = np.zeros((n, 3, 3))
    for rows in (slice(0, 2), slice(2, 3)):
        for columns in (slice(0, 2), slice(2, 3)):
            shape = (rows.stop - rows.start, columns.stop - columns.start)
            size = shape[0] * shape[1]
            transition = rng.uniform(-0.5, 0.5, (size, size))
            transition *= 0.8 / max(abs(np.linalg.eigvals(transition)))
            block = np.zeros((n, size))
            noise = rng.normal(size=block.shape)
            for t in range(1, n):
                block[t] = transition @ block[t - 1] + noise[t]
            latent[:, rows, columns] = block.reshape(n, *shape)
    row, column = _mixings()
    fit = fit_matrix_decorrelation(
        row @ latent @ column.T,
        lags=2,
        correlation_lags=5,
        correlation_threshold=0.15,
        grouping=grouping,
    )
    assert tuple(map(len, fit.row_groups)) == (2, 1)
    assert tuple(map(len, fit.column_groups)) == (2, 1)
    for truth, estimated in ((row, fit.row_mixing), (column, fit.column_mixing)):
        recovered = np.linalg.solve(truth, estimated)
        recovered /= np.linalg.norm(recovered, axis=0)
        assert np.max(abs(recovered[2, :2])) < 0.11
        assert np.max(abs(recovered[:2, 2])) < 0.11
    assert [[block.shape[1:] for block in row] for row in fit.blocks()] == [
        [(2, 2), (2, 1)],
        [(1, 2), (1, 1)],
    ]


def test_graph_grouping_transitivity_and_strict_threshold():
    correlations = np.array(
        [[0, 0.7, 0.1, 0.2], [0.7, 0, 0.8, 0], [0.1, 0.8, 0, 0], [0.2, 0, 0, 0]]
    )
    assert _connected_groups(correlations, 0.2) == ((0, 1, 2), (3,))
    assert _connected_groups(correlations, 0.8) == ((0,), (1,), (2,), (3,))


def test_ratio_equation_and_author_half_pair_search_endpoint():
    correlations = np.zeros((4, 4))
    correlations[np.triu_indices(4, k=1)] = [0.9, 0.8, 0.2, 0.19, 0.01, 0.001]
    correlations += correlations.T
    # Six pairs: authors search only j=1,2,3. Ratios are 1.125,4,1.053,
    # whereas extending the search exposes a larger ratio 19 at j=4.
    assert _ratio_groups(correlations, 0, None) == (((0, 1, 2), (3,)), 2, 3)
    assert _ratio_groups(correlations, 0, 5) == (((0, 1, 2, 3),), 4, 5)
    assert _ratio_groups(correlations, 0.5, 5) == (((0, 1, 2), (3,)), 2, 5)


def test_ratio_ridge_and_boundary_tie_conventions():
    zeros = np.zeros((4, 4))
    # The delta->0+ limit makes 0/0 equal one; the first ratio and then the
    # lexicographically first edge win. This selector does not test no edges.
    assert _ratio_groups(zeros, 0, None) == (((0, 1), (2,), (3,)), 1, 3)
    assert _ratio_groups(zeros, 1e300, None) == (((0, 1), (2,), (3,)), 1, 3)
    correlations = zeros.copy()
    correlations[2, 3] = correlations[3, 2] = 0.9
    assert _ratio_groups(correlations, 0, None) == (((0,), (1,), (2, 3)), 1, 3)
    assert _ratio_groups(np.zeros((1, 1)), 0, None) == (((0,),), 0, 0)
    with pytest.raises(ValueError, match="dimension two"):
        _ratio_groups(np.zeros((2, 2)), 0, None)


def test_ratio_fit_metadata_and_regrouped_equation_oracle():
    X = np.random.default_rng(116).normal(size=(60, 4, 5))
    result = fit_matrix_decorrelation(
        X, grouping="ratio", ratio_max_edges=(4, 7), ratio_delta=0.01
    )
    assert result.method == "matrix-decorrelation-ratio"
    assert result.grouping == "ratio"
    assert result.correlation_threshold is None
    assert result.ratio_max_edges == (4, 7)
    assert result.ratio_delta == 0.01
    for correlation, upper, count in (
        (result.row_correlations, 4, result.row_n_edges),
        (result.column_correlations, 7, result.column_n_edges),
    ):
        ordered = sorted(
            correlation[np.triu_indices(len(correlation), 1)], reverse=True
        )
        ratios = [(ordered[j] + 0.01) / (ordered[j + 1] + 0.01) for j in range(upper)]
        assert count == ratios.index(max(ratios)) + 1
    assert_allclose(result.inverse_blocks(result.blocks()), X, atol=3e-14)
    scaled = fit_matrix_decorrelation(
        X * 1e150, grouping="ratio", ratio_max_edges=(4, 7), ratio_delta=0.01
    )
    assert scaled.row_groups == result.row_groups
    assert scaled.column_groups == result.column_groups
    assert_allclose(scaled.series * 1e150, result.series, atol=3e-13)


@pytest.mark.parametrize("shape", [(30, 1, 1), (30, 1, 3), (30, 3, 1)])
def test_ratio_scalar_modes(shape):
    X = np.random.default_rng(14).normal(size=shape)
    result = fit_matrix_decorrelation(X, grouping="ratio")
    assert_allclose(result.inverse_transform(result.series), X, atol=3e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grouping": "invalid"},
        {"grouping": ["ratio"]},
        {"ratio_delta": -1},
        {"ratio_delta": np.nan},
        {"ratio_delta": True},
        {"ratio_max_edges": 0},
        {"ratio_max_edges": 1.5},
        {"ratio_max_edges": [1]},
        {"ratio_max_edges": [1, False]},
        {"grouping": "ratio", "ratio_max_edges": 3},
    ],
)
def test_invalid_ratio_tuning(kwargs):
    X = np.random.default_rng(14).normal(size=(30, 3, 3))
    with pytest.raises(ValueError):
        fit_matrix_decorrelation(X, **kwargs)


def test_ratio_two_coordinate_mode_needs_explicit_threshold_tuning():
    X = np.random.default_rng(14).normal(size=(30, 2, 3))
    with pytest.raises(ValueError, match="threshold grouping"):
        fit_matrix_decorrelation(X, grouping="ratio")
    X = X[:, :1, :]
    with pytest.raises(ValueError, match="singleton mode"):
        fit_matrix_decorrelation(X, grouping="ratio", ratio_max_edges=1)


def test_separate_thresholds_and_contiguous_reordering():
    X = np.random.default_rng(55).normal(size=(90, 3, 4))
    fit = fit_matrix_decorrelation(X, correlation_threshold=(0, 10))
    assert fit.row_groups == ((0, 1, 2),)
    assert fit.column_groups == ((0,), (1,), (2,), (3,))
    assert_allclose(fit.inverse_blocks(fit.blocks()), X, atol=2e-14)


@pytest.mark.parametrize("shape", [(100, 1, 1), (100, 1, 3), (100, 3, 1)])
def test_vector_and_scalar_boundary_cases(shape):
    X = np.random.default_rng(90).normal(size=shape)
    fit = fit_matrix_decorrelation(X)
    assert_allclose(fit.inverse_transform(fit.series), X, atol=2e-14)


def test_zero_variance_scalar_coordinates_with_full_rank_marginals():
    X = np.zeros((100, 2, 2))
    rng = np.random.default_rng(57)
    X[:, 0, 0] = rng.normal(size=100)
    X[:, 1, 1] = rng.normal(size=100)
    fit = fit_matrix_decorrelation(X)
    assert np.isfinite(fit.row_correlations).all()
    assert np.isfinite(fit.column_correlations).all()
    assert_allclose(fit.inverse_transform(fit.series), X, atol=2e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lags": 0},
        {"lags": True},
        {"lags": 1.0},
        {"lags": 40},
        {"correlation_lags": 0},
        {"correlation_lags": 40},
        {"correlation_threshold": -1},
        {"correlation_threshold": np.nan},
        {"correlation_threshold": [0.1]},
        {"correlation_threshold": [0.1, np.inf]},
        {"correlation_threshold": True},
        {"center": 1},
        {"rank_tol": 0},
        {"rank_tol": -0.1},
        {"rank_tol": 1},
        {"rank_tol": True},
        {"rank_tol": np.inf},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        fit_matrix_decorrelation(
            np.random.default_rng(2).normal(size=(40, 2, 3)), **kwargs
        )


@pytest.mark.parametrize(
    "X",
    [
        np.ones((30, 2, 3)),
        np.zeros((30, 2, 3)),
        np.ones((30, 2, 3), dtype=complex),
        np.full((30, 2, 3), np.nan),
    ],
)
def test_invalid_data_and_singular_marginals(X):
    with pytest.raises(ValueError):
        fit_matrix_decorrelation(X)


def test_rank_cutoff_rejects_nearly_singular_marginal_without_ridge():
    X = np.random.default_rng(8).normal(size=(30, 2, 3))
    X[:, 1] = X[:, 0] + 1e-10 * X[:, 1]
    with pytest.raises(ValueError, match="row marginal covariance"):
        fit_matrix_decorrelation(X)


def test_invalid_future_arrays_and_blocks():
    fit = fit_matrix_decorrelation(
        np.random.default_rng(3).normal(size=(30, 2, 3)), correlation_threshold=10
    )
    with pytest.raises(ValueError, match="observation dimensions"):
        fit.transform(np.ones((2, 3, 2)))
    with pytest.raises(ValueError, match="latent dimensions"):
        fit.inverse_transform(np.ones((2, 3, 2)))
    with pytest.raises(ValueError, match="partition"):
        fit.inverse_blocks(())
    with pytest.raises(ValueError, match="block dimensions"):
        fit.inverse_blocks([[np.ones((2, 2, 1))] * 3] * 2)
    blocks = [list(row) for row in fit.blocks()]
    blocks[0][0] = blocks[0][0][1:]
    with pytest.raises(ValueError, match="same number"):
        fit.inverse_blocks(blocks)
