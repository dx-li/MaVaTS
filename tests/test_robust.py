import numpy as np
import pytest

from mavats.metrics import subspace_distance
from mavats.robust import fit_matrix_kendall, matrix_kendall
from mavats.simulation import simulate_factor


def test_kendall_matches_literal_pair_sum():
    x = np.random.default_rng(40).normal(size=(9, 3, 4))
    row, col = np.zeros((3, 3)), np.zeros((4, 4))
    for t in range(len(x)):
        for s in range(t):
            d = x[t] - x[s]
            row += d @ d.T / np.sum(d * d)
            col += d.T @ d / np.sum(d * d)
    actual = matrix_kendall(x, batch_size=3)
    np.testing.assert_allclose(actual[0], row / 36, atol=1e-14)
    np.testing.assert_allclose(actual[1], col / 36, atol=1e-14)
    assert np.trace(actual[0]) == pytest.approx(1)
    assert np.trace(actual[1]) == pytest.approx(1)


def test_kendall_scale_translation_and_transposition():
    x = np.random.default_rng(41).normal(size=(15, 3, 4))
    original = matrix_kendall(x)
    for altered in (x * 1e250, x * 1e-250, x + 100):
        for a, b in zip(original, matrix_kendall(altered)):
            np.testing.assert_allclose(a, b, atol=1e-13)
    transposed = matrix_kendall(x.transpose(0, 2, 1))
    np.testing.assert_allclose(original[0], transposed[1])


def test_exact_low_rank_recovery():
    data = simulate_factor(80, (5, 6), (2, 2), noise_std=0, random_state=5)
    fit = fit_matrix_kendall(data.observations, (2, 2))
    np.testing.assert_allclose(fit.signal, data.signal, atol=1e-12)
    for loading, truth in zip(fit.loadings, data.loadings):
        assert subspace_distance(loading, truth) < 3e-8


def test_approximation_reproducible_and_near_exact():
    x = np.random.default_rng(2).normal(size=(150, 2, 3))
    exact = matrix_kendall(x)
    approx = matrix_kendall(x, max_pairs=5000, random_state=7)
    repeated = matrix_kendall(x, max_pairs=5000, random_state=7)
    for a, b, c in zip(exact, approx, repeated):
        np.testing.assert_array_equal(b, c)
        np.testing.assert_allclose(a, b, atol=0.012)


def test_duplicate_convention_and_all_equal_rejection():
    x = np.array([[[0.0]], [[0.0]], [[1.0]]])
    with pytest.warns(RuntimeWarning, match="duplicate"):
        row, col = matrix_kendall(x)
    assert row[0, 0] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="distinct"):
        matrix_kendall(np.zeros((5, 2, 3)))
