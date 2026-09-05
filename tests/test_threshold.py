"""Independent equation oracles for the published 2022 threshold estimator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mavats.threshold import fit_threshold_factors


def explicit_moments(X, z, lower, upper, max_lag):
    # Equation (9): mask only z[t], retain actual time offsets and divide by T.
    output = []
    for regime in range(2):
        pair = []
        for data in (X, X.transpose(0, 2, 1)):
            n, p, q = data.shape
            moment = np.zeros((p, p))
            for h in range(1, max_lag + 1):
                for u in range(q):
                    for v in range(q):
                        omega = np.zeros((p, p))
                        for t in range(n - h):
                            active = z[t] < lower if regime == 0 else z[t] >= upper
                            if active:
                                omega += np.outer(data[t, :, u], data[t + h, :, v])
                        omega /= n
                        moment += omega @ omega.T
            pair.append(moment)
        output.append(pair)
    return output


def synthetic(seed=48, n=500, noise=0.035):
    rng = np.random.default_rng(seed)
    row = np.linalg.qr(rng.normal(size=(6, 4)))[0]
    column = np.linalg.qr(rng.normal(size=(5, 4)))[0]
    loadings = ((row[:, :1], column[:, :2]), (row[:, 1:3], column[:, 2:3]))
    z = rng.uniform(-1, 1, n)
    cores = np.zeros((n, 2, 2))
    for t in range(1, n):
        cores[t] = 0.8 * cores[t - 1] + rng.normal(size=(2, 2))
    signal = np.empty((n, 6, 5))
    for t in range(n):
        i = int(z[t] >= 0)
        a, b = loadings[i]
        signal[t] = a @ cores[t, : a.shape[1], : b.shape[1]] @ b.T
    return signal + noise * rng.normal(size=signal.shape), z, loadings, signal


def test_known_threshold_equations_and_time_alignment():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(19, 3, 4))
    z = rng.normal(size=19)
    result = fit_threshold_factors(X, z, ((2, 1), (1, 2)), threshold=0, max_lag=3)
    oracle = explicit_moments(X / result.data_scale, z, 0, 0, 3)
    for i in range(2):
        for mode in range(2):
            values, vectors = np.linalg.eigh(oracle[i][mode])
            k = result.ranks[i][mode]
            expected = vectors[:, -k:] @ vectors[:, -k:].T
            q = result.loadings[i][mode]
            assert_allclose(q @ q.T, expected, atol=3e-14)
            assert_allclose(result.eigenvalues[i][mode], values[::-1], atol=1e-15)
    assert result.candidates.size == result.candidate_scores.size == 0
    assert result.reference_loadings is None
    assert_allclose(result.reconstruct(X, z), result.signal)


def test_profile_matches_extreme_complement_spectral_norm_oracle():
    rng = np.random.default_rng(92)
    X = rng.normal(size=(23, 3, 4))
    z = np.linspace(-1, 1, len(X))[rng.permutation(len(X))]
    candidates = np.array([-0.3, 0.0, 0.4])
    result = fit_threshold_factors(
        X,
        z,
        ((1, 2), (2, 1)),
        threshold_bounds=(-0.6, 0.6),
        candidates=candidates,
        max_lag=2,
    )
    work = X / result.data_scale
    extreme = explicit_moments(work, z, -0.6, 0.6, 2)
    complements = []
    for i in range(2):
        pair = []
        for mode in range(2):
            _, vectors = np.linalg.eigh(extreme[i][mode])
            k = result.ranks[i][mode]
            pair.append(vectors[:, :-k])
            q = result.reference_loadings[i][mode]
            assert_allclose(q @ q.T, vectors[:, -k:] @ vectors[:, -k:].T, atol=1e-14)
        complements.append(pair)
    scores = []
    for candidate in candidates:
        moments = explicit_moments(work, z, candidate, candidate, 2)
        scores.append(
            sum(
                np.linalg.norm(
                    complements[i][m].T @ moments[i][m] @ complements[i][m], 2
                )
                for i in range(2)
                for m in range(2)
            )
        )
    assert_allclose(result.candidate_scores, scores, rtol=1e-13)
    assert result.threshold == candidates[np.argmin(scores)]


def test_recovery_unequal_regime_ranks_and_auto_selection():
    X, z, loadings, signal = synthetic()
    result = fit_threshold_factors(X, z, threshold_bounds=(-0.6, 0.6), max_lag=2)
    assert result.ranks == ((1, 2), (2, 1))
    assert abs(result.threshold) < 0.035
    assert np.mean((result.signal - signal) ** 2) < 0.001
    for i in range(2):
        for estimated, actual in zip(result.loadings[i], loadings[i]):
            assert np.linalg.norm(estimated @ estimated.T - actual @ actual.T) < 0.03
        assert result.factors[i].shape == (len(result.indices[i]), *result.ranks[i])
    assert_allclose(result.inverse_transform(result.factors, z), result.signal)
    assert_allclose(result.signal + result.residuals, X)


def test_noiseless_known_threshold_exact_common_component():
    X, z, _, signal = synthetic(n=150, noise=0)
    result = fit_threshold_factors(X, z, ((1, 2), (2, 1)), threshold=0)
    assert_allclose(result.signal, signal, atol=1e-13)


@pytest.mark.parametrize("multiplier", [1e-120, 1e120, -7.0])
def test_extreme_scale_invariance(multiplier):
    X, z, _, _ = synthetic(n=100)
    kwargs = dict(ranks=((1, 2), (2, 1)), candidates=[-0.2, 0, 0.2])
    first = fit_threshold_factors(X, z, **kwargs)
    second = fit_threshold_factors(multiplier * X, z, **kwargs)
    assert first.threshold == second.threshold
    assert_allclose(first.candidate_scores, second.candidate_scores, rtol=1e-12)
    assert_allclose(first.signal, second.signal / multiplier, atol=1e-12)


def test_threshold_covariate_affine_units_and_equality():
    X, z, _, _ = synthetic(n=100)
    z[0] = 0
    first = fit_threshold_factors(X, z, ((1, 2), (2, 1)), threshold=0)
    second = fit_threshold_factors(X, 3 * z + 9, ((1, 2), (2, 1)), threshold=9)
    assert_array_equal(first.regimes, second.regimes)
    assert first.regimes[0] == 1
    assert_allclose(first.signal, second.signal)


def test_transform_out_of_sample_uses_training_spaces_and_can_have_empty_regime():
    X, z, _, _ = synthetic(n=100)
    result = fit_threshold_factors(X[:80], z[:80], ((1, 2), (2, 1)), threshold=0)
    future_z = np.ones(20)
    future = X[80:]
    cores = result.transform(future, future_z)
    assert cores[0].shape == (0, 1, 2)
    a, b = result.loadings[1]
    expected = np.array([a @ a.T @ matrix @ b @ b.T for matrix in future])
    assert_allclose(result.inverse_transform(cores, future_z), expected, atol=1e-14)
    assert_allclose(result.reconstruct(future[:1], future_z[:1]), expected[:1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_lag": 0},
        {"max_lag": True},
        {"max_lag": 40},
        {"threshold": np.nan},
        {"threshold": 1e9},
        {"trim": (0, 0.9)},
        {"trim": (0.9, 0.1)},
        {"threshold_bounds": (0, 0)},
        {"threshold_bounds": (100, 200)},
        {"candidates": []},
        {"candidates": [[0]]},
        {"candidates": [np.nan]},
        {"threshold_bounds": (-0.5, 0.5), "candidates": [0.5]},
        {"threshold": 0, "candidates": [0]},
        {"ranks": (1, 2)},
        {"ranks": ((0, 1), (1, 1))},
        {"ranks": ((7, 1), (1, 1))},
        {"ranks": ((True, 1), (1, 1))},
        {"ranks": ((6, 5), (6, 5))},
    ],
)
def test_invalid_options(kwargs):
    X, z, _, _ = synthetic(n=40)
    with pytest.raises(ValueError):
        fit_threshold_factors(X, z, **kwargs)


@pytest.mark.parametrize(
    "bad_z",
    [np.zeros((20, 1)), np.ones(19), [np.nan] * 20, ["a"] * 20, [True] * 20, [1j] * 20],
)
def test_invalid_aligned_threshold_variable(bad_z):
    X, _, _, _ = synthetic(n=20)
    with pytest.raises(ValueError, match="z must"):
        fit_threshold_factors(X, bad_z)


def test_missing_extreme_information_and_numerical_rank_rejected():
    X, z, _, _ = synthetic(n=40, noise=0)
    with pytest.raises(ValueError, match="numerical"):
        fit_threshold_factors(X, z, ((2, 2), (2, 2)), threshold=0)
    with pytest.raises(ValueError, match="nonzero"):
        fit_threshold_factors(np.zeros_like(X), z, threshold=0)
    with pytest.raises(ValueError, match="lagged origin"):
        fit_threshold_factors(X, np.r_[np.zeros(39), 1], threshold=1)


def test_result_validation():
    X, z, _, _ = synthetic(n=40)
    result = fit_threshold_factors(X, z, ((1, 2), (2, 1)), threshold=0)
    with pytest.raises(ValueError):
        result.transform(X[:, :2], z)
    with pytest.raises(ValueError):
        result.transform(X, z[:-1])
    with pytest.raises(ValueError):
        result.classify(z[:, None])
    with pytest.raises(ValueError):
        result.inverse_transform(result.factors[:1], z)
    with pytest.raises(ValueError):
        result.inverse_transform(result.factors[::-1], z)


def test_singleton_mode_and_input_preservation():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 1, 3))
    z = rng.normal(size=40)
    original = X.copy()
    fit_threshold_factors(X, z, threshold=0)
    assert_array_equal(X, original)
