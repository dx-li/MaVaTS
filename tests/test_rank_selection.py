"""Behavioral/regression tests; independent equation oracles live separately."""

import math

import numpy as np
import pytest

from mavats.rank_selection import _criterion, select_tensor_rank


def _factor_series(seed=85, n=320):
    rng = np.random.default_rng(seed)
    row = np.linalg.qr(rng.normal(size=(8, 2)))[0] * np.sqrt(8)
    col = np.linalg.qr(rng.normal(size=(9, 2)))[0] * 3
    core = np.zeros((n, 2, 2))
    for t in range(1, n):
        core[t] = 0.8 * core[t - 1] + rng.normal(size=(2, 2))
    signal = np.einsum("ia,tab,jb->tij", row, core, col)
    return signal + 0.4 * rng.normal(size=signal.shape), signal


@pytest.mark.parametrize("method", ["topup", "tipup"])
@pytest.mark.parametrize("criterion", ["ic", "er"])
@pytest.mark.parametrize("iterative", [False, True])
@pytest.mark.parametrize("penalty", range(1, 6))
def test_all_forty_procedures_recover_strong_dynamic_factors(
    method, criterion, iterative, penalty
):
    X, signal = _factor_series()
    fitted = select_tensor_rank(
        X, method=method, criterion=criterion, iterative=iterative, penalty=penalty
    )
    assert fitted.ranks == (2, 2)
    assert fitted.converged
    assert np.linalg.norm(fitted.signal - signal) / np.linalg.norm(signal) < 0.05
    assert fitted.n_iter == len(fitted.history) - 1
    assert fitted.diagnostics == fitted.history[-1]
    np.testing.assert_allclose(fitted.inverse_transform(fitted.factors), fitted.signal)
    np.testing.assert_allclose(fitted.transform(X), fitted.factors, atol=1e-13)
    np.testing.assert_allclose(fitted.signal + fitted.residuals, X, atol=2e-14)


def test_history_and_honest_nonconvergence():
    X, _ = _factor_series(n=70)
    fitted = select_tensor_rank(X, iterative=True, max_iter=1, tol=1e-15)
    assert not fitted.converged
    assert fitted.stop_reason == "max_iter"
    assert fitted.rank_history[0] == fitted.starting_ranks
    assert fitted.rank_history[-1] == fitted.ranks
    assert len(fitted.projector_changes) == 1
    assert fitted.projector_changes[0] > 1e-15
    assert len(fitted.history) == 2
    assert fitted.history[0][0].projected_shape == X.shape[1:]
    assert fitted.history[1][0].projected_shape == (8, fitted.starting_ranks[1])
    assert fitted.history[1][1].projected_shape == (fitted.ranks[0], 9)
    assert all(a.log_penalty == b.log_penalty for a, b in zip(*fitted.history))


@pytest.mark.parametrize("iterative", [False, True])
def test_zero_rank_core_roundtrip_and_mixed_raw_ranks(iterative):
    X, _ = _factor_series(n=25)
    fitted = select_tensor_rank(
        X, criterion="ic", max_ranks=(0, 2), iterative=iterative
    )
    assert fitted.ranks[0] == 0
    assert fitted.signal_ranks == (0, 0)
    assert fitted.factors.shape == (len(X), *fitted.ranks)
    assert fitted.converged
    expected = np.broadcast_to(X.mean(axis=0), X.shape)
    np.testing.assert_allclose(fitted.signal, expected, atol=1e-14)
    np.testing.assert_allclose(
        fitted.inverse_transform(fitted.transform(X)), expected, atol=1e-14
    )


def test_penalty_is_fixed_physical_not_a_scale_invariant_heuristic():
    X = np.repeat(np.diag([2.0, 1.0, 0.0])[None], 4, axis=0)
    small = select_tensor_rank(X, criterion="ic", center=False, max_ranks=(2, 2))
    large = select_tensor_rank(100 * X, criterion="ic", center=False, max_ranks=(2, 2))
    assert small.ranks == (0, 0)
    assert large.ranks == (2, 2)
    assert small.diagnostics[0].log_penalty == large.diagnostics[0].log_penalty


@pytest.mark.parametrize("criterion", ["ic", "er"])
@pytest.mark.parametrize("factor", [1e-200, -1e-200, 1e200, -1e200])
def test_extreme_physical_units_with_compensating_log_penalties(criterion, factor):
    X, _ = _factor_series(n=100)
    kwargs = dict(criterion=criterion, iterative=True, penalty=3)
    base = select_tensor_rank(X, **kwargs)
    fitted = select_tensor_rank(
        X * factor, log_penalty_multiplier=4 * math.log(abs(factor)), **kwargs
    )
    assert fitted.ranks == base.ranks
    np.testing.assert_allclose(
        fitted.signal / factor, base.signal, rtol=1e-10, atol=1e-10
    )
    for a, b in zip(fitted.loadings, base.loadings):
        np.testing.assert_allclose(a @ a.T, b @ b.T, atol=1e-11)


def test_per_mode_penalties_and_exclusive_log_alternative():
    X = np.repeat(np.diag([3.0, 1.0, 0.0])[None], 5, axis=0)
    kwargs = dict(criterion="ic", center=False, max_ranks=(2, 2))
    direct = select_tensor_rank(X, penalty_multiplier=(0.001, 10), **kwargs)
    logs = select_tensor_rank(X, log_penalty_multiplier=np.log([0.001, 10]), **kwargs)
    assert direct.ranks == logs.ranks == (2, 0)
    assert direct.log_penalty_multipliers == logs.log_penalty_multipliers


def test_extreme_er_rounding_does_not_hide_best_ratio():
    selected = _criterion(np.array([6.0, 5.0, 0.0]), 0.0, 1000.0, "er", 2)
    # Both displayed ratios round to one, but the second gap is larger.
    np.testing.assert_array_equal(selected["scores"], [1.0, 1.0])
    assert selected["rank"] == 2
    tied = _criterion(np.zeros(3), 0.0, 1000.0, "er", 2)
    assert tied["rank"] == 1


def test_ic_exact_boundary_tie_and_extreme_common_tail():
    assert _criterion(np.array([4.0, 1.0, 0.0]), 0, 0, "ic", 2)["rank"] == 1
    assert (
        _criterion(np.array([1e250, 1.0, 0.0]), 0, math.log(0.5), "ic", 2)["rank"] == 2
    )


def test_vector_special_case_and_small_sample_penalty_boundary():
    X = np.arange(36.0).reshape(9, 4)
    result = select_tensor_rank(X, iterative=True)
    assert result.ranks == (1,)
    assert result.converged
    assert result.signal.shape == X.shape
    with pytest.raises(ValueError, match="nonpositive"):
        select_tensor_rank(np.eye(2), criterion="ic", penalty=1)
    assert select_tensor_rank(np.eye(2), criterion="ic", penalty=3).ranks == (0,)


def test_transform_is_training_only_and_preserves_inputs():
    X, _ = _factor_series(n=80)
    copy = X.copy()
    result = select_tensor_rank(X[:60], iterative=True)
    once = result.transform(X[60:61])
    all_future = X[60:].copy()
    all_future[1:] = 1e100
    np.testing.assert_allclose(result.transform(all_future)[0], once[0], rtol=1e-13)
    np.testing.assert_array_equal(X, copy)


def test_extreme_future_amplitude_cannot_underflow_an_earlier_transform_or_inverse():
    train = np.zeros((5, 3, 4))
    train[:, 0, 0] = np.arange(-2.0, 3.0)
    train[:, 1, 1] = 1e-120
    fitted = select_tensor_rank(train)
    earlier = fitted.mean[None].copy()
    earlier[0, 0, 0] = 1e-100
    later = fitted.mean[None].copy()
    later[0, 0, 0] = 1e300
    once = fitted.transform(earlier)
    together = fitted.transform(np.concatenate((earlier, later)))
    assert once[0, 0, 0] != 0
    np.testing.assert_array_equal(together[:1], once)
    reconstructed_once = fitted.inverse_transform(once)
    reconstructed_together = fitted.inverse_transform(together)
    np.testing.assert_array_equal(reconstructed_together[:1], reconstructed_once)
    np.testing.assert_allclose(reconstructed_once, earlier, rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "pca"},
        {"criterion": "ratio"},
        {"penalty": 0},
        {"penalty": 6},
        {"penalty": True},
        {"lags": 0},
        {"lags": 10},
        {"lags": [1]},
        {"max_iter": 0},
        {"iterative": 1},
        {"center": 1},
        {"tol": 0},
        {"nu": -0.1},
        {"nu": 1.1},
        {"nu": np.nan},
        {"c0": 0},
        {"penalty_multiplier": 0},
        {"penalty_multiplier": [1]},
        {"penalty_multiplier": [1, np.inf]},
        {"log_penalty_multiplier": [1]},
        {"log_penalty_multiplier": np.inf},
        {"penalty_multiplier": [1, 1], "log_penalty_multiplier": 0},
        {"max_ranks": [0, 1]},
        {"max_ranks": [1]},
        {"max_ranks": 1},
        {"max_ranks": [3, 1]},
        {"max_ranks": [1.0, 1]},
        {"initial_ranks": [1, 1]},
        {"initial_ranks": [4, 1], "iterative": True},
        {"initial_ranks": [0, 1], "iterative": True},
        {"initial_ranks": [1], "iterative": True},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        select_tensor_rank(np.ones((10, 3, 4)), **kwargs)


@pytest.mark.parametrize(
    "X",
    [
        np.ones(5),
        np.ones((1, 3, 4)),
        np.ones((5, 1, 3)),
        np.ones((5, 0, 3)),
        np.ones((5, 3, 4)) * np.inf,
        np.ones((5, 3, 4), dtype=complex),
    ],
)
def test_invalid_data(X):
    with pytest.raises(ValueError):
        select_tensor_rank(X)


def test_invalid_transforms():
    result = select_tensor_rank(np.ones((5, 3, 4)))
    with pytest.raises(ValueError):
        result.transform(np.ones((2, 4, 3)))
    with pytest.raises(ValueError):
        result.inverse_transform(np.ones((2, 2, 2)))
    with pytest.raises(ValueError):
        result.inverse_transform(np.full((2, *result.ranks), np.nan))


def test_physical_core_overflow_is_not_silently_returned():
    X = np.ones((5, 3, 4)) * 1e308
    with pytest.raises(ValueError, match="physical data units"):
        select_tensor_rank(X, center=False)
