"""Independent paper-moment, projection and joint-score regression oracles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mavats.constrained import (
    fit_constrained_factor,
    fit_multiterm_constrained_factor,
    fit_partial_constrained_factor,
)


def _moment(X, lags):
    d, columns = X.shape[1:]
    result = np.zeros((d, d))
    for h in lags:
        for i in range(columns):
            for j in range(columns):
                covariance = np.zeros((d, d))
                for t in range(len(X) - h):
                    covariance += np.outer(X[t, :, i], X[t + h, :, j])
                covariance /= len(X) - h
                result += covariance @ covariance.T
    return result


def _projector(A):
    return A @ A.T


def _leading(M, rank):
    return np.linalg.eigh(M)[1][:, -rank:]


def _data(seed=11, n=160, noise=0.0):
    rng = np.random.default_rng(seed)
    row = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    column = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    R, C = row[:, [0, 1, 3]], column[:, [0, 2, 3]]
    core = rng.normal(size=(n, 3, 3))
    for t in range(1, n):
        core[t] += 0.65 * core[t - 1]
    signal = R @ core @ C.T
    return (
        signal + noise * rng.normal(size=signal.shape),
        signal,
        row[:, :3],
        column[:, :2],
    )


def _fit_partial(X, row, column, **kwargs):
    return fit_partial_constrained_factor(
        X,
        row_constraints=row,
        column_constraints=column,
        row_ranks=(2, 1),
        column_ranks=(1, 2),
        **kwargs,
    )


def test_partial_matches_explicit_separate_moments_and_shared_spaces():
    X = np.random.default_rng(31).normal(size=(24, 7, 6))
    fit = _fit_partial(X, np.eye(7)[:, :3], np.eye(6)[:, :2], lags=[1, 3])
    row, col = fit.constraint_bases
    blocks = [[r.T @ (X / fit.data_scale) @ c for c in col] for r in row]
    for l in range(2):
        rm = sum(_moment(blocks[l][k], [1, 3]) for k in range(2))
        cm = sum(_moment(blocks[k][l].transpose(0, 2, 1), [1, 3]) for k in range(2))
        assert_allclose(sum(fit.block_moments[0][l]), rm, atol=1e-14)
        assert_allclose(
            sum(fit.block_moments[1][k][l] for k in range(2)), cm, atol=1e-14
        )
        expected_r = row[l] @ _leading(rm, fit.row_ranks[l])
        expected_c = col[l] @ _leading(cm, fit.column_ranks[l])
        offset_r, offset_c = sum(fit.row_ranks[:l]), sum(fit.column_ranks[:l])
        got_r = fit.loadings[0][:, offset_r : offset_r + fit.row_ranks[l]]
        got_c = fit.loadings[1][:, offset_c : offset_c + fit.column_ranks[l]]
        assert_allclose(_projector(got_r), _projector(expected_r), atol=2e-13)
        assert_allclose(_projector(got_c), _projector(expected_c), atol=2e-13)
        # Concatenation includes forbidden cross-group column pairs.
        concatenated = _moment(np.concatenate(blocks[l], axis=2), [1, 3])
        assert np.linalg.norm(concatenated - rm) > 0.01
    expected = _projector(fit.loadings[0]) @ X @ _projector(fit.loadings[1])
    assert_allclose(fit.signal, expected, atol=2e-14)


def test_noiseless_partial_recovers_all_interactions_and_pythagorean_identity():
    X, truth, row, col = _data()
    fit = _fit_partial(X, row, col, lags=2)
    assert_allclose(fit.signal, truth, atol=2e-13)
    signals = [s for block in fit.signal_blocks for s in block]
    assert_allclose(sum(signals), fit.signal, atol=1e-14)
    assert_allclose(
        sum(np.sum(s * s) for s in signals), np.sum(fit.signal**2), rtol=1e-14
    )
    for i in range(4):
        for j in range(i):
            assert_allclose(np.sum(signals[i] * signals[j]), 0, atol=1e-12)
    assert np.linalg.norm(fit.factor_blocks[0][1]) > 1
    assert np.linalg.norm(fit.factor_blocks[1][0]) > 1


def test_partial_diagonal_reduces_to_two_independent_constrained_terms():
    X, _, row, col = _data(noise=0.1)
    fit = _fit_partial(X, row, col, interactions=False)
    rows, cols = fit.constraint_bases
    independent = [
        fit_constrained_factor(
            X,
            (fit.row_ranks[l], fit.column_ranks[l]),
            row_constraints=rows[l],
            column_constraints=cols[l],
        )
        for l in range(2)
    ]
    assert_allclose(fit.signal, sum(f.signal for f in independent), atol=1e-12)
    assert_allclose(fit.factor_blocks[0][1], 0)
    assert_allclose(fit.factor_blocks[1][0], 0)
    changed = fit.factors.copy()
    changed[0, 0, fit.column_ranks[0]] = 1
    with pytest.raises(ValueError, match="zero cross blocks"):
        fit.inverse_transform(changed)


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_partial_data_and_constraint_gauge_invariance(scale):
    X, _, row, col = _data(noise=0.1)
    fit = _fit_partial(X, row, col)
    row2 = row @ np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 3.0]])
    row2 *= [1e200, 1e-200, -1.0]
    col2 = col @ np.array([[2.0, 1.0], [0.0, -3.0]])
    changed = _fit_partial(X * scale, row2, col2)
    assert_allclose(changed.signal / scale, fit.signal, atol=2e-12)
    for a, b in zip(fit.loadings, changed.loadings):
        assert_allclose(_projector(a), _projector(b), atol=2e-12)


def test_partial_empty_complements_explicit_zero_and_zero_moment_conventions():
    X = np.random.default_rng(32).normal(size=(30, 4, 3))
    full = fit_partial_constrained_factor(X, row_ranks=(2, 0), column_ranks=(1, 0))
    ordinary = fit_constrained_factor(X, (2, 1))
    assert_allclose(full.signal, ordinary.signal, atol=2e-13)
    empty = fit_partial_constrained_factor(
        X, row_constraints=np.empty((4, 0)), row_ranks=(0, 2), column_ranks=(1, 0)
    )
    assert_allclose(empty.signal, ordinary.signal, atol=2e-13)
    zero = fit_partial_constrained_factor(np.zeros_like(X))
    assert zero.ranks == (0, 0)
    assert zero.factors.shape == (30, 0, 0)
    assert_allclose(zero.inverse_transform(zero.transform(X)), 0)
    suppressed = fit_partial_constrained_factor(
        X, row_ranks=(0, 0), column_ranks=(0, 0)
    )
    assert_allclose(suppressed.signal, 0)
    with pytest.raises(ValueError, match="unidentified"):
        fit_partial_constrained_factor(np.zeros_like(X), row_ranks=(1, 0))


def _overlapping(seed=44, noise=0.0, n=120):
    rng = np.random.default_rng(seed)
    row1, col1 = np.eye(5)[:, :2], np.eye(6)[:, :3]
    row2 = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    ) / np.sqrt(2)
    col2 = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    ) / np.sqrt(2)
    loadings = [(row1[:, :1], col1[:, [0, 2]]), (row2, col2[:, :1])]
    cores = [rng.normal(size=(n, 1, 2)), rng.normal(size=(n, 2, 1))]
    for core in cores:
        for t in range(1, n):
            core[t] += 0.7 * core[t - 1]
    parts = [a @ f @ b.T for (a, b), f in zip(loadings, cores)]
    truth = sum(parts)
    X = truth + noise * rng.normal(size=truth.shape)
    return X, truth, [(row1, col1), (row2, col2)]


def test_overlapping_multiterm_noiseless_recovery_and_joint_dense_score_oracle():
    X, truth, constraints = _overlapping()
    fit = fit_multiterm_constrained_factor(X, constraints, [(1, 2), (2, 1)], lags=2)
    assert_allclose(fit.signal, truth, atol=3e-13)
    assert all(
        d["estimator"] == "competing-complements"
        for d in fit.identification_diagnostics
    )
    design = np.concatenate([np.kron(c, r) for r, c in fit.loadings], axis=1)
    flat = np.array([x.ravel(order="F") for x in X])
    expected = np.linalg.lstsq(design, flat.T, rcond=None)[0].T
    actual = np.concatenate(
        [np.array([f.ravel(order="F") for f in core]) for core in fit.factors], axis=1
    )
    assert_allclose(actual, expected, atol=2e-13)
    assert_allclose(sum(fit.component_signals), fit.signal, atol=2e-14)
    # Independent projections double-count overlapping terms.
    separate = sum(_projector(r) @ X @ _projector(c) for r, c in fit.loadings)
    assert np.linalg.norm(separate - truth) > 1


def test_multiterm_competing_projection_moments_match_nested_dense_oracle():
    X, _, constraints = _overlapping(noise=0.2, n=25)
    fit = fit_multiterm_constrained_factor(
        X, constraints, [(1, 2), (2, 1)], lags=[1, 3]
    )
    for j in range(2):
        row, col = fit.constraint_bases[j]
        other_row, other_col = fit.constraint_bases[1 - j]
        # Full projectors, not the implementation's complement coordinates.
        row_series = row.T @ (X / fit.data_scale) @ (np.eye(6) - _projector(other_col))
        col_series = (
            col.T
            @ (X / fit.data_scale).transpose(0, 2, 1)
            @ (np.eye(5) - _projector(other_row))
        )
        for k, (basis, series) in enumerate([(row, row_series), (col, col_series)]):
            M = _moment(series, [1, 3])
            expected = basis @ _leading(M, fit.ranks[j][k])
            assert_allclose(
                _projector(fit.loadings[j][k]), _projector(expected), atol=2e-12
            )
            assert_allclose(
                fit.eigenvalues[j][k],
                np.maximum(np.linalg.eigvalsh(M)[::-1], 0),
                atol=1e-13,
            )


@pytest.mark.parametrize("side", [0, 1])
def test_multiterm_one_sided_orthogonality_is_sufficient(side):
    X = np.random.default_rng(45).normal(size=(30, 5, 6))
    constraints = [
        (np.eye(5)[:, :2], np.eye(6)[:, :3]),
        (np.eye(5)[:, 2:4], np.eye(6)[:, 1:4]),
    ]
    if side == 1:
        constraints = [(c, r) for r, c in constraints]
        X = X.transpose(0, 2, 1)
    fit = fit_multiterm_constrained_factor(X, constraints, [(1, 2), (2, 1)])
    assert all(d["estimator"] == "direct" for d in fit.identification_diagnostics)
    independent = [
        fit_constrained_factor(X, rank, row_constraints=r, column_constraints=c)
        for (r, c), rank in zip(constraints, fit.ranks)
    ]
    assert_allclose(fit.signal, sum(f.signal for f in independent), atol=1e-12)


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_multiterm_constraint_and_observation_scaling(scale):
    X, _, constraints = _overlapping(noise=0.1)
    fit = fit_multiterm_constrained_factor(X, constraints, [(1, 2), (2, 1)])
    changed = []
    for r, c in constraints:
        changed.append(
            (
                r @ np.array([[1.0, 2.0], [0.0, -3.0]]) * [1e-200, 1e200],
                c * np.geomspace(1e-200, 1e200, c.shape[1]),
            )
        )
    scaled = fit_multiterm_constrained_factor(X * scale, changed, [(1, 2), (2, 1)])
    assert_allclose(scaled.signal / scale, fit.signal, atol=2e-12)


@pytest.mark.parametrize("kind", ["partial", "multi"])
def test_holdout_projection_has_no_future_scale_or_mean_leakage(kind):
    if kind == "partial":
        X, _, r, c = _data(noise=0.1)
        fit = _fit_partial(X[:100] + 4, r, c, center=True)
    else:
        X, _, constraints = _overlapping(noise=0.1)
        fit = fit_multiterm_constrained_factor(
            X[:100] + 4, constraints, [(1, 2), (2, 1)], center=True
        )
    test = X[100:102] + 4
    first = fit.transform(test[:1])
    appended = fit.transform(np.concatenate([test[:1], test[1:] * 1e140]))
    if kind == "partial":
        assert_allclose(first[0], appended[0], rtol=0, atol=0)
    else:
        for a, b in zip(first, appended):
            # BLAS may dispatch GEMV versus GEMM for singleton versus batched
            # right-hand sides; allow rounding, not future-dependent scaling.
            assert_allclose(a[0], b[0], rtol=2e-15, atol=2e-15)
    assert_allclose(
        fit.inverse_transform(fit.transform(X[:100] + 4)), fit.signal, atol=1e-13
    )


def test_multiterm_rank_loss_duplicate_spaces_and_all_omitted():
    X = np.random.default_rng(46).normal(size=(30, 4, 4))
    H = np.eye(4)[:, :2]
    with pytest.raises(ValueError, match="unidentified|rank deficient"):
        fit_multiterm_constrained_factor(X, [(H, H), (H, H)], [(1, 1), (1, 1)])
    # Surviving constraint dimension is one, insufficient for rank two.
    J = np.eye(4)[:, 1:3]
    with pytest.raises(ValueError, match="loses loading rank"):
        fit_multiterm_constrained_factor(X, [(H, H), (J, J)], [(2, 2), (1, 1)])
    omitted = fit_multiterm_constrained_factor(X, [(None, None)], [(0, 0)], center=True)
    assert omitted.score_design_rank == 0
    assert_allclose(
        omitted.signal, np.broadcast_to(X.mean(axis=0), X.shape), atol=1e-15
    )
    assert omitted.factors[0].shape == (len(X), 0, 0)


def test_multiterm_three_term_union_extension():
    X = np.random.default_rng(47).normal(size=(35, 6, 6))
    constraints = [
        (np.eye(6)[:, j : j + 2], np.eye(6)[:, j : j + 2]) for j in [0, 2, 4]
    ]
    fit = fit_multiterm_constrained_factor(X, constraints, [(1, 1)] * 3)
    assert len(fit.factors) == 3
    assert_allclose(fit.signal, sum(fit.component_signals), atol=1e-14)


def test_three_overlapping_terms_survive_union_complements():
    rng = np.random.default_rng(71)
    eye = np.eye(8)
    bases = [
        eye[:, :2],
        0.4 * eye[:, :2] + np.sqrt(0.84) * eye[:, 2:4],
        0.3 * eye[:, :2] + np.sqrt(0.91) * eye[:, 4:6],
    ]
    factors = rng.normal(size=(120, 3))
    for t in range(1, len(factors)):
        factors[t] += 0.6 * factors[t - 1]
    X = sum(
        h[:, :1] @ factors[:, j, None, None] @ h[:, :1].T for j, h in enumerate(bases)
    )
    fit = fit_multiterm_constrained_factor(X, [(h, h) for h in bases], [(1, 1)] * 3)
    assert_allclose(fit.signal, X, atol=3e-13)
    assert all(
        d["estimator"] == "competing-complements"
        for d in fit.identification_diagnostics
    )


def test_automatic_ranks_record_numerical_conventions():
    rng = np.random.default_rng(72)
    H = np.eye(8)[:, :4]
    row = np.eye(8)[:, [0, 4]]
    core = rng.normal(size=(120, 2, 2))
    for t in range(1, len(core)):
        core[t] += 0.7 * core[t - 1]
    X = row @ core @ row.T
    fit = fit_partial_constrained_factor(X, row_constraints=H, column_constraints=H)
    assert fit.row_ranks == fit.column_ranks == (1, 1)
    assert all(
        d["automatic"] and d["search_bound"] == 2
        for pair in fit.rank_diagnostics
        for d in pair
    )
    multi = fit_multiterm_constrained_factor(
        X, [(H, H), (np.eye(8)[:, 4:], np.eye(8)[:, 4:])]
    )
    assert multi.ranks == ((1, 1), (1, 1))


@pytest.mark.parametrize("kind", ["partial", "multi"])
def test_transform_rejects_physically_unrepresentable_scores(kind):
    X = np.arange(1, 31, dtype=float)[:, None, None] * np.ones((1, 2, 2))
    if kind == "partial":
        fit = fit_partial_constrained_factor(X, row_ranks=(1, 0), column_ranks=(1, 0))
    else:
        fit = fit_multiterm_constrained_factor(X, [(None, None)], [(1, 1)])
    with pytest.raises(FloatingPointError, match="scores"):
        fit.transform(np.full((1, 2, 2), 1e308))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(row_ranks=(1, 1)),
        dict(row_ranks=(True, 0)),
        dict(column_ranks=(4, 0)),
        dict(row_ranks=(-1, 0)),
        dict(interactions="yes"),
        dict(row_ranks=(1,)),
        dict(lags=[0]),
        dict(center=1),
    ],
)
def test_partial_invalid_options(kwargs):
    with pytest.raises(ValueError):
        fit_partial_constrained_factor(np.ones((20, 4, 3)), **kwargs)


@pytest.mark.parametrize(
    "constraints,ranks,kwargs",
    [
        ([], None, {}),
        ([(None,)], None, {}),
        ([(None, None)], [(1, 0)], {}),
        ([(None, None)], [(5, 1)], {}),
        ([(None, None)], [(1, 1), (1, 1)], {}),
        ([(None, None)], [(1, 1)], dict(max_design_elements=1)),
    ],
)
def test_multiterm_invalid_options(constraints, ranks, kwargs):
    with pytest.raises(ValueError):
        fit_multiterm_constrained_factor(
            np.random.default_rng(48).normal(size=(20, 4, 3)),
            constraints,
            ranks,
            **kwargs,
        )


@pytest.mark.parametrize("method", ["partial", "multi"])
def test_invalid_transform_and_inverse_shapes(method):
    X, _, row, col = _data()
    fit = (
        _fit_partial(X, row, col)
        if method == "partial"
        else fit_multiterm_constrained_factor(X, [(row, col)], [(1, 1)])
    )
    with pytest.raises(ValueError):
        fit.transform(np.ones((2, 4, 3)))
    with pytest.raises(ValueError):
        fit.transform(np.ones((2, 7, 6), dtype=complex))
    with pytest.raises(ValueError):
        fit.inverse_transform(np.ones((2, 1, 1)) if method == "partial" else [])


def test_no_mutation_of_observations_or_constraints():
    X, _, r, c = _data(noise=0.1)
    before = [a.copy() for a in (X, r, c)]
    _fit_partial(X, r, c)
    fit_multiterm_constrained_factor(X, [(r, c)], [(2, 1)])
    for a, b in zip((X, r, c), before):
        assert_allclose(a, b, rtol=0, atol=0)
