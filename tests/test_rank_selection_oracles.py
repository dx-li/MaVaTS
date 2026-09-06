"""Independent oracles for Han, Chen and Zhang (2022), EJS eqs. (1), (2), (6)–(8).

The tiny explicit index loops deliberately do not call the implementation's
unfolding, moment, projection, eigenvalue-selection, or penalty kernels.
Primary reference: https://doi.org/10.1214/22-EJS1991.
"""

import math
from decimal import Decimal, localcontext
from itertools import product

import numpy as np
import pytest

from mavats.rank_selection import _criterion as _implementation_criterion
from mavats.rank_selection import select_tensor_rank


def _moment(X, mode, method, lags):
    shape = X.shape[1:]
    d = shape[mode]
    other = [k for k in range(len(shape)) if k != mode]
    fibers = list(product(*(range(shape[k]) for k in other)))
    Y = np.empty((len(X), d, len(fibers)))
    for t, i, a in product(range(len(X)), range(d), range(len(fibers))):
        index = [0] * len(shape)
        index[mode] = i
        for k, value in zip(other, fibers[a]):
            index[k] = value
        Y[t, i, a] = X[(t, *index)]
    W = np.zeros((d, d))
    for h in lags:
        if method == "tipup":
            C = np.zeros((d, d))
            for i, j, a, t in product(
                range(d), range(d), range(len(fibers)), range(h, len(X))
            ):
                C[i, j] += Y[t - h, i, a] * Y[t, j, a] / (len(X) - h)
        else:
            C = np.zeros((d, len(fibers), d, len(fibers)))
            for i, a, j, b, t in product(
                range(d),
                range(len(fibers)),
                range(d),
                range(len(fibers)),
                range(h, len(X)),
            ):
                C[i, a, j, b] += Y[t - h, i, a] * Y[t, j, b] / (len(X) - h)
            C = C.reshape(d, -1)
        W += C @ C.T
    return W


def _penalty(shape, n, h0, mode, criterion, index, nu=0, c0=0.1):
    d = math.prod(shape)
    dk = shape[mode]
    if criterion == "ic":
        D = h0 * d ** (2 - 2 * nu)
        harmonic_log = math.log(d * n / (d + n))
        small_log = math.log(min(d, n))
        values = (
            D / n * harmonic_log,
            D * (1 / n + 1 / d) * harmonic_log,
            D / n * small_log,
            D * (1 / n + 1 / d) * small_log,
            D * (1 / n + 1 / d) * math.log(min(dk, n)),
        )
    else:
        values = (
            c0 * h0,
            h0 * d**2 / n**2,
            h0 * d**2 / (n**2 * dk**2),
            h0 * d**2 / (n**2 * dk**2) + h0 * dk**2 / n**2,
            h0 * d**2 / (n**2 * dk) + h0 * d * dk / n**2,
        )
    return values[index - 1]


def _criterion(eigenvalues, penalty, criterion, upper):
    candidates = np.arange(0 if criterion == "ic" else 1, upper + 1)
    if criterion == "ic":
        values = np.array(
            [math.fsum(eigenvalues[m:]) + m * penalty for m in candidates]
        )
    else:
        values = np.array(
            [
                (eigenvalues[m] + penalty) / (eigenvalues[m - 1] + penalty)
                for m in candidates
            ]
        )
    return int(candidates[np.argmin(values)]), candidates, values


def _eigensystem(W):
    values, vectors = np.linalg.eigh(W)
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0), vectors[:, order]


def _project_except(X, loadings, skip):
    shape = X.shape[1:]
    target = tuple(
        shape[k] if k == skip else loadings[k].shape[1] for k in range(len(shape))
    )
    output = np.zeros((len(X), *target))
    for destination in product(*(range(d) for d in target)):
        for source in product(*(range(d) for d in shape)):
            if source[skip] != destination[skip]:
                continue
            weight = math.prod(
                loadings[k][source[k], destination[k]]
                for k in range(len(shape))
                if k != skip
            )
            output[(slice(None), *destination)] += X[(slice(None), *source)] * weight
    return output


@pytest.mark.parametrize("method", ["topup", "tipup"])
@pytest.mark.parametrize("criterion", ["ic", "er"])
@pytest.mark.parametrize("penalty", [1, 2, 3, 4, 5])
def test_all_penalties_moments_and_criteria_match_dense_index_oracle(
    method,
    criterion,
    penalty,
):
    X = np.random.default_rng(581).normal(size=(9, 3, 4, 2))
    upper = (2, 3, 1)
    multiplier, nu, c0 = 0.31, 0.37, 0.23
    result = select_tensor_rank(
        X,
        method=method,
        criterion=criterion,
        penalty=penalty,
        lags=2,
        max_ranks=upper,
        penalty_multiplier=multiplier,
        nu=nu,
        c0=c0,
        center=False,
    )
    for k, diagnostic in enumerate(result.history[0]):
        expected, vectors = _eigensystem(_moment(X, k, method, (1, 2)))
        physical = diagnostic.eigenvalues * math.exp(diagnostic.log_eigenvalue_scale)
        np.testing.assert_allclose(physical, expected, rtol=2e-12, atol=2e-12)
        G = multiplier * _penalty(X.shape[1:], len(X), 2, k, criterion, penalty, nu, c0)
        assert diagnostic.log_penalty == pytest.approx(math.log(G), abs=2e-14)
        rank, candidates, scores = _criterion(expected, G, criterion, upper[k])
        assert result.ranks[k] == rank == diagnostic.rank
        np.testing.assert_array_equal(diagnostic.candidate_ranks, candidates)
        actual_scores = diagnostic.scores
        if criterion == "ic":
            actual_scores = actual_scores * math.exp(diagnostic.log_score_scale)
        np.testing.assert_allclose(actual_scores, scores, rtol=3e-12, atol=3e-12)
        Q = result.loadings[k]
        np.testing.assert_allclose(
            Q @ Q.T, vectors[:, :rank] @ vectors[:, :rank].T, atol=2e-11
        )


@pytest.mark.parametrize("method", ["topup", "tipup"])
@pytest.mark.parametrize("criterion", ["ic", "er"])
def test_one_sweep_uses_sequential_projected_moments_but_original_penalty_dimensions(
    method,
    criterion,
):
    X = np.random.default_rng(953).normal(size=(13, 4, 3, 2))
    upper, starting = (3, 2, 1), (2, 2, 1)
    penalty, multiplier = 5, 0.03
    Q = [
        _eigensystem(_moment(X, k, method, (1,)))[1][:, : starting[k]] for k in range(3)
    ]
    oracle = []
    for k in range(3):
        Z = _project_except(X, Q, k)
        values, vectors = _eigensystem(_moment(Z, k, method, (1,)))
        G = multiplier * _penalty(X.shape[1:], len(X), 1, k, criterion, penalty)
        rank, _, _ = _criterion(values, G, criterion, upper[k])
        Q[k] = vectors[:, :rank]
        oracle.append((Z.shape[1:], values, rank, G))
    result = select_tensor_rank(
        X,
        method=method,
        criterion=criterion,
        penalty=penalty,
        penalty_multiplier=multiplier,
        initial_ranks=starting,
        max_ranks=upper,
        iterative=True,
        max_iter=1,
        center=False,
    )
    assert result.n_iter == 1
    for k, diagnostic in enumerate(result.history[-1]):
        shape, values, rank, G = oracle[k]
        assert tuple(diagnostic.projected_shape) == shape
        assert diagnostic.rank == result.ranks[k] == rank
        assert diagnostic.log_penalty == pytest.approx(math.log(G))
        np.testing.assert_allclose(
            diagnostic.eigenvalues * math.exp(diagnostic.log_eigenvalue_scale),
            values,
            rtol=3e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            result.loadings[k] @ result.loadings[k].T, Q[k] @ Q[k].T, atol=2e-11
        )


@pytest.mark.parametrize("method", ["topup", "tipup"])
@pytest.mark.parametrize("criterion", ["ic", "er"])
@pytest.mark.parametrize("units", [1e-100, 1e100])
def test_scale_equivariance_requires_fourth_power_penalty_units(
    method, criterion, units
):
    X = np.random.default_rng(642).normal(size=(12, 4, 3))
    kwargs = dict(
        method=method,
        criterion=criterion,
        penalty=5,
        max_ranks=(3, 2),
        center=False,
        iterative=True,
        max_iter=2,
    )
    base = select_tensor_rank(X, penalty_multiplier=0.07, **kwargs)
    scaled = select_tensor_rank(
        X * units,
        log_penalty_multiplier=math.log(0.07) + 4 * math.log(units),
        **kwargs,
    )
    assert scaled.ranks == base.ranks
    for old_sweep, new_sweep in zip(base.history, scaled.history):
        for old, new in zip(old_sweep, new_sweep):
            assert old.rank == new.rank
            np.testing.assert_allclose(old.eigenvalues, new.eigenvalues, atol=2e-12)
            assert new.log_penalty - old.log_penalty == pytest.approx(
                4 * math.log(units)
            )
    np.testing.assert_allclose(scaled.signal / units, base.signal, atol=3e-11)


@pytest.mark.parametrize("method", ["topup", "tipup"])
def test_zero_factor_ic_and_positive_only_er_have_distinct_semantics(method):
    X = np.zeros((8, 3, 4))
    ic = select_tensor_rank(X, method=method, criterion="ic", iterative=True)
    er = select_tensor_rank(X, method=method, criterion="er", iterative=True)
    assert ic.ranks == (0, 0)
    assert er.ranks == (1, 1)
    np.testing.assert_array_equal(ic.signal, X)
    np.testing.assert_allclose(er.history[-1][0].scores, 1)


def test_tipup_signal_cancellation_does_not_cancel_topup_outer_products():
    X = np.zeros((3, 3, 2))
    X[0, 0] = [1, 1]
    X[1, 0] = [1, -1]
    assert np.linalg.norm(_moment(X, 0, "tipup", (1,))) == 0
    assert np.linalg.norm(_moment(X, 0, "topup", (1,))) > 0
    kwargs = dict(criterion="ic", penalty_multiplier=0.001, center=False)
    assert select_tensor_rank(X, method="tipup", **kwargs).ranks[0] == 0
    assert select_tensor_rank(X, method="topup", **kwargs).ranks[0] == 1


def test_ic_preserves_weak_positive_eigenvalue_below_dominant_leading_scale():
    # W=diag(1e240, 1, 0). Subtracting a leading cumulative sum from its
    # total erases the second eigenvalue; the exact IC with G=.5 chooses2.
    X = np.repeat(np.diag([1e60, 1, 0])[None], 4, axis=0)
    G0 = _penalty((3, 3), 4, 1, 0, "ic", 1)
    result = select_tensor_rank(
        X,
        criterion="ic",
        method="tipup",
        center=False,
        max_ranks=(2, 2),
        penalty_multiplier=0.5 / G0,
    )
    assert result.ranks == (2, 2)


def test_centering_removes_constant_rank_and_restores_mean_signal():
    X = np.repeat(np.arange(12, dtype=float).reshape(1, 3, 4), 9, axis=0)
    result = select_tensor_rank(X, criterion="ic", center=True, iterative=True)
    assert result.ranks == (0, 0)
    np.testing.assert_allclose(result.signal, X, atol=2e-14)


def test_conservative_initial_projection_ranks_are_distinct_from_selected_ranks():
    X = np.random.default_rng(718).normal(size=(15, 5, 4, 3))
    result = select_tensor_rank(X, criterion="er", iterative=True, max_iter=1)
    expected = tuple(
        min(d, 2 * r, r + 3) for d, r in zip(X.shape[1:], result.initial_ranks)
    )
    assert tuple(result.starting_ranks) == expected
    assert tuple(result.history[1][0].projected_shape) == (5, *expected[1:])


@pytest.mark.parametrize("method", ["topup", "tipup"])
@pytest.mark.parametrize("criterion", ["ic", "er"])
def test_noiseless_heterogeneous_ranks_and_loading_spaces_converge(method, criterion):
    rng = np.random.default_rng(425)
    truth = (2, 1, 2)
    loadings = [
        np.linalg.qr(rng.normal(size=(d, r)))[0] for d, r in zip((4, 3, 4), truth)
    ]
    factors = rng.normal(size=(60, *truth))
    for t in range(1, len(factors)):
        factors[t] += 0.75 * factors[t - 1]
    X = 5 * np.einsum("tabc,ia,jb,kc->tijk", factors, *loadings)
    result = select_tensor_rank(
        X,
        method=method,
        criterion=criterion,
        max_ranks=(3, 2, 3),
        iterative=True,
        max_iter=20,
        penalty_multiplier=0.001,
        center=True,
    )
    assert result.ranks == truth
    assert result.converged
    assert result.n_iter < 20
    for Q, U in zip(result.loadings, loadings):
        np.testing.assert_allclose(Q @ Q.T, U @ U.T, atol=1e-10)
    np.testing.assert_allclose(result.signal, X, atol=2e-10)


def test_weak_strength_tuning_changes_the_ic_penalty_not_the_moment_eigenvalues():
    X = np.repeat(np.diag([2, 0.2, 0])[None], 4, axis=0)
    multiplier = 0.01 / _penalty((3, 3), 4, 1, 0, "ic", 1)
    kwargs = dict(
        criterion="ic", max_ranks=(2, 2), center=False, penalty_multiplier=multiplier
    )
    strong = select_tensor_rank(X, nu=0, **kwargs)
    weak = select_tensor_rank(X, nu=1, **kwargs)
    assert strong.ranks == (1, 1)
    assert weak.ranks == (2, 2)
    for before, after in zip(strong.history[0], weak.history[0]):
        np.testing.assert_array_equal(before.eigenvalues, after.eigenvalues)
        assert after.log_penalty - before.log_penalty == pytest.approx(-2 * math.log(9))


@pytest.mark.parametrize("criterion", ["ic", "er"])
def test_paper_search_bound_excludes_full_spatial_dimension(criterion):
    with pytest.raises(ValueError):
        select_tensor_rank(np.ones((5, 3, 4)), criterion=criterion, max_ranks=(3, 2))


@pytest.mark.parametrize("log_penalty", [-800, 800])
def test_er_log_comparison_matches_high_precision_ratios_when_floats_round_to_one(
    log_penalty,
):
    values = np.array([4.0, 3.0, 0.0])
    with localcontext() as context:
        context.prec = 1000
        H = Decimal(log_penalty).exp()
        ratios = [
            (Decimal(int(b)) + H) / (Decimal(int(a)) + H)
            for a, b in zip(values[:-1], values[1:])
        ]
        expected = min(range(2), key=ratios.__getitem__) + 1
    result = _implementation_criterion(values, 0.0, log_penalty, "er", 2)
    assert result["rank"] == expected == 2
    if log_penalty > 0:
        np.testing.assert_array_equal(result["scores"], [1, 1])


def test_ic_tie_at_penalty_chooses_smallest_candidate_without_losing_weak_tail():
    result = _implementation_criterion(np.array([1e240, 1.0, 0.0]), 0, 0, "ic", 2)
    assert result["rank"] == 1
    lower = _implementation_criterion(np.array([1e240, 1.0, 0.0]), 0, -1, "ic", 2)
    assert lower["rank"] == 2
