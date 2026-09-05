"""Independent accepted-manuscript equation and sequential-state oracles."""

import json
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mavats.monitoring import MatrixFactorMonitor


def sample(seed=17):
    return np.random.default_rng(seed).normal(size=(28, 8, 6))


def zero_drift_sample(n=20):
    x = np.zeros((n, 4, 3))
    x[:, 0, 0] = 1
    return x


def equation_oracle(window, rank, projection_rank, power, epsilon):
    """Direct unscaled loop equations; does not call implementation helpers."""
    m, p, q = window.shape
    opposite = sum(matrix.T @ matrix for matrix in window) / (m * p)
    _, vectors = np.linalg.eigh(opposite)
    qhat = vectors[:, -projection_rank:]
    projected = [matrix @ qhat / np.sqrt(q) for matrix in window]
    covariance = sum(y @ y.T for y in projected) / m
    eigenvalue = np.linalg.eigvalsh(covariance)[-rank - 1]
    trace = np.trace(covariance)
    beta = np.log(p) / np.log(q * m)
    delta = epsilon + max(0, 1 - 1 / (2 * beta))
    normalized = p ** (1 - delta) * eigenvalue / trace
    return eigenvalue, trace, normalized, normalized**power


def score_oracle(y, cumulative, t, horizon, statistic, eta, delay):
    if statistic == "maximum":
        root = np.sqrt(2 * np.log(horizon))
        b = root - (np.log(np.log(horizon)) + np.log(4 * np.pi)) / (2 * root)
        a = b / (1 + b * b)
        return (y - b) / a
    if eta == 0.5:
        nested = np.log(np.log(horizon))
        a = np.sqrt(2 * nested)
        b = 2 * nested + 0.5 * np.log(nested) - 0.5 * np.log(np.pi)
        return a * abs(cumulative) / np.sqrt(t) - b
    reference = horizon if eta < 0.5 else delay
    return reference ** (eta - 0.5) * abs(cumulative) / t**eta


@pytest.mark.parametrize("mode", ["row", "column"])
@pytest.mark.parametrize(
    "statistic,eta,delay",
    [
        ("maximum", 0.25, None),
        ("partial_sum", 0, None),
        ("partial_sum", 0.25, None),
        ("partial_sum", 0.5, None),
        ("partial_sum", 0.75, 3),
    ],
)
def test_each_rolling_window_and_boundary_matches_independent_equations(
    mode, statistic, eta, delay
):
    x = sample()
    m, horizon = 12, 9
    monitor = MatrixFactorMonitor(
        x[:m],
        1,
        2,
        horizon,
        mode=mode,
        power=3,
        epsilon=0.07,
        statistic=statistic,
        eta=eta,
        delay=delay,
        critical_value=1e6,
    )
    noises = np.linspace(-1.3, 1.1, horizon)
    cumulative = 0
    for t, noise in enumerate(noises, start=1):
        window = x[t : m + t]
        oriented = window if mode == "row" else window.transpose(0, 2, 1)
        eig, trace, normalized, drift = equation_oracle(oriented, 1, 2, 3, 0.07)
        cumulative += drift + noise
        expected = score_oracle(
            drift + noise, cumulative, t, horizon, statistic, eta, delay
        )
        result = monitor.update(x[m + t - 1], noise=noise)
        assert_allclose(
            result.eigenvalue_scaled * result.data_scale**2, eig, rtol=2e-13
        )
        assert_allclose(result.trace_scaled * result.data_scale**2, trace, rtol=2e-13)
        assert_allclose(
            [
                result.normalized_eigenvalue,
                result.drift,
                result.partial_sum,
                result.statistic,
            ],
            [normalized, drift, cumulative, expected],
            rtol=2e-12,
        )
        assert result.monitoring_index == t
        assert result.observation_index == m + t
        assert result.window_start_index == t + 1
        assert result.eligible == (t >= (delay or 1))
        assert not result.alarm
        assert_array_equal(monitor.window, window)
    assert monitor.stop_reason == "horizon"


@pytest.mark.parametrize(
    "eta,delay,noise,alarm",
    [
        (0, None, [1, 1, 1], 3),
        (0.5, None, [0, 0, 30], 3),
        (0.75, 3, [30, 0, 0], 3),
    ],
)
def test_hand_partial_sum_crossings(eta, delay, noise, alarm):
    x = zero_drift_sample()
    critical = 0.5 if eta == 0 else 3
    monitor = MatrixFactorMonitor(
        x[:4],
        1,
        1,
        16,
        statistic="partial_sum",
        eta=eta,
        delay=delay,
        critical_value=critical,
    )
    steps = monitor.update_many(x[4:7], noise=noise)
    assert monitor.alarm_step == alarm
    assert monitor.alarm_observation == 4 + alarm
    assert steps[-1].alarm
    assert monitor.stop_reason == "alarm"
    # Eta=0 first two sums have score 0.25 and EXACTLY 0.5: strict > only.
    if eta == 0:
        assert steps[1].statistic == 0.5
        assert not steps[1].alarm


def test_maximum_is_one_sided_and_ignores_future_after_crossing():
    x = zero_drift_sample()
    monitor = MatrixFactorMonitor(x[:4], 1, 1, 10, critical_value=3)
    future = x[4:8].copy()
    future[-1] = np.nan
    steps = monitor.update_many(future, noise=[-100, 0, 100, np.nan])
    assert len(steps) == 3
    assert not steps[0].alarm
    assert monitor.alarm_step == 3
    assert monitor.update_many("not inspected") == ()
    with pytest.raises(RuntimeError, match="alarm"):
        monitor.update(x[0])


def test_horizon_consumes_only_available_prefix():
    x = zero_drift_sample()
    monitor = MatrixFactorMonitor(x[:4], 1, 1, 2, critical_value=100)
    assert len(monitor.update_many(x[4:8], noise=[0, 0, np.nan, np.nan])) == 2
    assert monitor.stop_reason == "horizon"
    assert monitor.alarm_step is None
    with pytest.raises(RuntimeError, match="horizon"):
        monitor.update(x[0])


@pytest.mark.parametrize("bitgen", ["PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"])
def test_prefix_chunk_and_strict_json_state_resume(bitgen):
    x = sample()
    rng = np.random.Generator(getattr(np.random, bitgen)(91))
    before = deepcopy(rng.bit_generator.state)
    monitor = MatrixFactorMonitor(
        x[:12], 1, 2, 12, critical_value=1e9, random_state=rng
    )
    monitor.update_many(x[12:15])
    snapshot = json.loads(json.dumps(monitor.state_dict(), allow_nan=False))
    resumed = MatrixFactorMonitor.from_state(snapshot)
    single = MatrixFactorMonitor(
        x[:12],
        1,
        2,
        12,
        critical_value=1e9,
        random_state=np.random.Generator(getattr(np.random, bitgen)(91)),
    )
    expected = tuple(single.update(matrix) for matrix in x[12:24])
    assert monitor.steps + monitor.update_many(x[15:24]) == expected
    assert (
        resumed.steps + resumed.update_many(x[15:19]) + resumed.update_many(x[19:24])
        == expected
    )
    assert json.dumps(
        rng.bit_generator.state, default=lambda a: a.tolist()
    ) == json.dumps(before, default=lambda a: a.tolist())
    # Neither detached state nor copied window can mutate a live monitor.
    snapshot["window"][0][0][0] = 1e9
    detached = single.window
    detached[:] = 0
    assert_array_equal(single.window, x[12:24])


def test_zero_drift_state_and_supplied_draw_do_not_advance_rng():
    x = zero_drift_sample()
    monitor = MatrixFactorMonitor(x[:4], 1, 1, 10, critical_value=1e6, random_state=83)
    result = monitor.update(x[4], noise=0)
    assert result.drift == 0 and result.log_drift == -np.inf
    restored = MatrixFactorMonitor.from_state(
        json.loads(json.dumps(monitor.state_dict(), allow_nan=False))
    )
    assert restored.update(x[5]).noise == np.random.default_rng(83).standard_normal()
    assert restored.steps[-1] == monitor.update(x[5])


def test_planted_new_factor_and_identical_prechange_prefix():
    rng = np.random.default_rng(120)
    m, horizon, p, q = 30, 40, 20, 12
    row = np.linalg.qr(rng.normal(size=(p, 2)))[0] * np.sqrt(p)
    column = np.linalg.qr(rng.normal(size=(q, 2)))[0] * np.sqrt(q)
    factors = rng.normal(size=(m + horizon, 2))
    null = np.einsum("i,t,j->tij", row[:, 0], factors[:, 0], column[:, 0])
    null += 0.2 * rng.normal(size=(m + horizon, p, q))
    changed = null.copy()
    changed[m + 10 :] += np.einsum(
        "i,t,j->tij", row[:, 1], factors[m + 10 :, 1], column[:, 0]
    )
    baseline = MatrixFactorMonitor(null[:m], 1, 2, horizon, random_state=22)
    alternative = MatrixFactorMonitor(changed[:m], 1, 2, horizon, random_state=22)
    baseline.update_many(null[m:])
    alternative.update_many(changed[m:])
    assert baseline.steps[:10] == alternative.steps[:10]
    assert baseline.stop_reason == "horizon"
    assert alternative.alarm_step is not None
    assert 11 <= alternative.alarm_step <= 20


@pytest.mark.parametrize("scale", [1e-150, -1e-150, 1e150, -1e150])
def test_scale_equivariance_without_covariance_overflow(scale):
    x = sample()
    kwargs = dict(
        rank=1,
        projection_rank=2,
        horizon=10,
        power=8,
        critical_value=1e8,
        random_state=82,
    )
    reference = MatrixFactorMonitor(x[:12], **kwargs).update_many(x[12:22])
    actual = MatrixFactorMonitor(x[:12] * scale, **kwargs).update_many(x[12:22] * scale)
    assert_allclose(
        [s.statistic for s in actual], [s.statistic for s in reference], rtol=1e-12
    )
    assert_allclose(
        [s.log_drift for s in actual], [s.log_drift for s in reference], rtol=1e-12
    )
    assert [s.alarm for s in actual] == [s.alarm for s in reference]


def test_overflowed_drift_is_a_finite_diagnostic_and_correct_delayed_crossing():
    x = sample(2)
    monitor = MatrixFactorMonitor(
        x[:12],
        1,
        2,
        10,
        power=1e5,
        statistic="partial_sum",
        eta=0.75,
        delay=3,
        critical_value=1e300,
    )
    results = monitor.update_many(x[12:18], noise=np.zeros(6))
    assert len(results) == 3
    assert results[0].saturated
    assert results[-1].alarm
    assert all(np.isfinite(s.statistic) and np.isfinite(s.partial_sum) for s in results)
    restored = MatrixFactorMonitor.from_state(
        json.loads(json.dumps(monitor.state_dict(), allow_nan=False))
    )
    assert restored.steps == monitor.steps


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((3, 2)),
        np.full((8, 6), np.nan),
        np.ones((8, 6), dtype=complex),
        [["x"] * 6] * 8,
    ],
)
def test_invalid_update_is_transactional(invalid):
    x = sample()
    monitor = MatrixFactorMonitor(x[:12], 1, 2, 10, critical_value=1e9, random_state=1)
    before = monitor.state_dict()
    with pytest.raises(ValueError):
        monitor.update(invalid)
    assert monitor.state_dict() == before
    assert monitor.update(x[12]).noise == np.random.default_rng(1).standard_normal()


def test_invalid_draw_and_zero_trace_do_not_advance_state():
    x = zero_drift_sample()
    x[:3] = 0
    monitor = MatrixFactorMonitor(x[:4], 1, 1, 10, random_state=1)
    for noise in [np.nan, np.inf, True, [1]]:
        before = monitor.state_dict()
        with pytest.raises(ValueError):
            monitor.update(x[4], noise=noise)
        assert monitor.state_dict() == before
    # Shift the only nonzero matrix out of the window without drawing noise.
    monitor.update_many(np.zeros((3, 4, 3)), noise=[0, 0, 0])
    before = monitor.state_dict()
    with pytest.raises(ValueError, match="zero trace"):
        monitor.update(np.zeros((4, 3)))
    assert monitor.state_dict() == before


@pytest.mark.parametrize(
    "override",
    [
        {"rank": 0},
        {"rank": 8},
        {"rank": True},
        {"projection_rank": 6},
        {"projection_rank": 0},
        {"horizon": 1},
        {"horizon": 2.1},
        {"mode": "tensor"},
        {"statistic": "max"},
        {"power": 1},
        {"epsilon": 0},
        {"epsilon": 1},
        {"eta": 1},
        {"eta": -0.1},
        {"alpha": 0},
        {"alpha": 1},
        {"critical_value": np.inf},
        {"delay": 1},
        {"statistic": "partial_sum", "critical_value": None},
        {"statistic": "partial_sum", "critical_value": 0},
        {"statistic": "partial_sum", "eta": 0.5, "horizon": 2},
        {"statistic": "partial_sum", "eta": 0.75, "delay": 11},
    ],
)
def test_invalid_configuration(override):
    kwargs = dict(rank=1, projection_rank=2, horizon=10, critical_value=3)
    kwargs.update(override)
    with pytest.raises(ValueError):
        MatrixFactorMonitor(sample()[:12], **kwargs)


def test_zero_training_and_unsupported_window_rank():
    with pytest.raises(ValueError, match="zero trace"):
        MatrixFactorMonitor(np.zeros((4, 8, 6)), 1, 2, 10)
    with pytest.raises(ValueError, match="cannot support"):
        MatrixFactorMonitor(sample()[:2], 3, 1, 10)


def test_default_gumbel_and_renyi_delay():
    x = sample()[:12]
    monitor = MatrixFactorMonitor(x, 1, 2, 20, alpha=0.1)
    assert_allclose(monitor.critical_value, -np.log(-np.log(0.9)))
    assert monitor.calibration == "asymptotic-gumbel"
    renyi = MatrixFactorMonitor(
        x, 1, 2, 20, statistic="partial_sum", eta=0.75, critical_value=3
    )
    assert renyi.delay == int(np.log(20))


@pytest.mark.parametrize(
    "corruption", ["schema", "index", "noise", "extra_step", "generator"]
)
def test_invalid_checkpoint(corruption):
    x = zero_drift_sample()
    monitor = MatrixFactorMonitor(x[:4], 1, 1, 10, critical_value=3)
    monitor.update(x[4], noise=100)
    state = monitor.state_dict()
    if corruption == "schema":
        state["schema_version"] = 0
    elif corruption == "index":
        state["steps"][0]["monitoring_index"] = 3
    elif corruption == "noise":
        state["steps"][0]["noise"] = np.nan
    elif corruption == "extra_step":
        state["steps"].append(deepcopy(state["steps"][0]))
    else:
        state["rng_state"]["bit_generator"] = "exec"
    with pytest.raises(ValueError):
        MatrixFactorMonitor.from_state(state)
