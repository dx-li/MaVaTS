"""Verify the benchmark's model geometry and independent forecast oracles."""

import numpy as np
import pytest

from benchmarks.envelope import _KnownEMAR, _scenario


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize(
    "regime", ["high-immaterial", "high-material", "isotropic", "covariance-coupled"]
)
def test_scenario_geometry_and_stable_dense_companion(order, regime):
    data = _scenario(40, order=order, regime=regime, seed=43, shape=(4, 5))
    for coefficient, covariance, basis in zip(
        (data["left"], data["right"]), (data["row"], data["column"]), data["bases"]
    ):
        p = basis @ basis.T
        q = np.eye(len(p)) - p
        np.testing.assert_allclose(p @ coefficient, coefficient, atol=1e-13)
        assert np.linalg.norm(coefficient @ q) > 1e-5  # no hidden input envelope
        assert np.linalg.eigvalsh(covariance)[0] > 0
        np.testing.assert_allclose(np.trace(covariance), len(covariance), atol=1e-12)
        leakage = np.linalg.norm(p @ covariance @ q)
        assert leakage > 0.01 if regime == "covariance-coupled" else leakage < 1e-12
    oracle = _KnownEMAR(data)
    d = 20
    companion = np.zeros((order * d, order * d))
    companion[:d] = np.concatenate(oracle.coefficients, axis=1)
    companion[d:, :-d] = np.eye((order - 1) * d)
    assert max(abs(np.linalg.eigvals(companion))) < 1
    # Compare oracle impulse covariance with independent companion powers.
    noise = np.zeros_like(companion)
    noise[:d, :d] = oracle.covariance
    total = np.zeros_like(companion)
    power = np.eye(len(companion))
    expected = []
    for _ in range(5):
        total += power @ noise @ power.T
        expected.append(np.trace(total[:d, :d]) / d)
        power = companion @ power
    np.testing.assert_allclose(oracle.noise_floor(5), expected, atol=1e-12)
    # Independent matrix-form mean recursion, including intercept and lags.
    history = list(data["observations"][-order:].copy())
    for _ in range(5):
        value = data["intercept"].copy()
        for lag, (a, b) in enumerate(zip(data["left"], data["right"]), 1):
            value += a @ history[-lag] @ b.T
        history.append(value)
    np.testing.assert_allclose(
        oracle.forecast(5, data["observations"]), history[order:], atol=1e-12
    )
