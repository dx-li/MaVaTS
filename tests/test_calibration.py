import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import binom, norm

from mavats.calibration import calibrate_monitor


@pytest.mark.parametrize(
    "horizon,alpha", [(2, 0.05), (100, 0.05), (10000, 1e-12), (50, 0.9)]
)
def test_exact_maximum_gaussian_reference(horizon, alpha):
    fit = calibrate_monitor(horizon, alpha=alpha)
    root = np.sqrt(2 * np.log(horizon))
    b = root - (np.log(np.log(horizon)) + np.log(4 * np.pi)) / (2 * root)
    a = b / (1 + b * b)
    boundary = b + a * fit.critical_value
    actual = -np.expm1(horizon * norm.logcdf(boundary))
    assert_allclose(actual, alpha, rtol=2e-13)
    assert fit.replications == 0 and fit.order_index is None
    assert fit.quantile_interval == (fit.critical_value, fit.critical_value)


@pytest.mark.parametrize("eta", [0, 0.25, 0.5, 0.75, 0.9])
def test_partial_sum_against_explicit_path_oracle_and_batch_independence(eta):
    horizon, repeats, alpha = 15, 99, 0.1
    draws = np.random.default_rng(42).normal(size=(repeats, horizon))
    statistics = []
    for path in draws:
        running = 0.0
        maximum = -np.inf
        for t, draw in enumerate(path, 1):
            running += draw
            if eta > 0.5 and t < 4:
                continue
            if eta == 0.5:
                ll = np.log(np.log(horizon))
                score = np.sqrt(2 * ll) * abs(running) / np.sqrt(t) - (
                    2 * ll + 0.5 * np.log(ll) - 0.5 * np.log(np.pi)
                )
            else:
                score = (
                    (4 if eta > 0.5 else horizon) ** (eta - 0.5) * abs(running) / t**eta
                )
            maximum = max(maximum, score)
        statistics.append(maximum)
    values = np.sort(statistics)
    kwargs = dict(
        statistic="partial_sum",
        eta=eta,
        delay=4 if eta > 0.5 else None,
        alpha=alpha,
        replications=repeats,
        random_state=42,
    )
    fit = calibrate_monitor(horizon, batch_size=10, **kwargs)
    other = calibrate_monitor(horizon, batch_size=200, **kwargs)
    assert fit == other
    assert fit.order_index == 90
    assert_allclose(fit.critical_value, values[89])
    low = int(binom.ppf(0.025, repeats, 0.9))
    high = int(binom.ppf(0.975, repeats, 0.9)) + 1
    assert_allclose(fit.quantile_interval, [values[low - 1], values[high - 1]])
    assert fit.monitor_kwargs["critical_value"] == fit.critical_value


def test_calibration_reference_coverage_on_independent_paths():
    fit = calibrate_monitor(
        30, statistic="partial_sum", eta=0.25, replications=20000, random_state=183
    )
    draws = np.random.default_rng(849).normal(size=(20000, 30))
    score = np.max(
        30 ** (-0.25) * abs(np.cumsum(draws, axis=1)) / np.arange(1, 31) ** 0.25, axis=1
    )
    rate = np.mean(score > fit.critical_value)
    assert abs(rate - 0.05) < 0.012
    assert fit.quantile_interval[0] < fit.critical_value < fit.quantile_interval[1]


def test_global_rng_unchanged_and_uncertain_tail_is_explicit():
    np.random.seed(735)
    before = np.random.get_state()
    fit = calibrate_monitor(
        10, statistic="partial_sum", alpha=0.05, replications=20, random_state=13
    )
    assert np.isinf(fit.quantile_interval[1])
    assert_array_equal(before[1], np.random.get_state()[1])


@pytest.mark.parametrize(
    "statistic,eta",
    [
        ("maximum", 0.25),
        ("partial_sum", 0),
        ("partial_sum", 0.5),
        ("partial_sum", 0.75),
    ],
)
def test_calibration_options_match_monitor_and_reference_score(statistic, eta):
    from mavats.monitoring import MatrixFactorMonitor

    fit = calibrate_monitor(
        12, statistic=statistic, eta=eta, replications=99, random_state=83
    )
    monitor = MatrixFactorMonitor(np.ones((5, 3, 2)), 1, 1, **fit.monitor_kwargs)
    draws = np.random.default_rng(18).normal(size=12) * 0.05
    steps = monitor.update_many(np.ones((12, 3, 2)), noise=draws)
    assert len(steps) == 12
    if statistic == "maximum":
        root = np.sqrt(2 * np.log(12))
        b = root - (np.log(np.log(12)) + np.log(4 * np.pi)) / (2 * root)
        expected = (max(draws) - b) / (b / (1 + b * b))
    else:
        from mavats.calibration import _reference_supremum

        expected = _reference_supremum(draws[None], eta, fit.delay)[0]
    eligible = [step.statistic for step in steps if step.eligible]
    assert_allclose(max(eligible), expected, atol=1e-12)


@pytest.mark.parametrize(
    "horizon,kwargs",
    [
        (True, {}),
        (1, {}),
        (1000001, {}),
        (10, {"alpha": 0}),
        (10, {"alpha": 1}),
        (10, {"alpha": np.nan}),
        (10, {"confidence": 1}),
        (10, {"eta": -0.1}),
        (10, {"eta": 1}),
        (10, {"delay": 11}),
        (10, {"delay": 1.5}),
        (10, {"statistic": "invalid"}),
        (2, {"statistic": "partial_sum", "eta": 0.5}),
        (10, {"statistic": "partial_sum", "alpha": 0.001, "replications": 20}),
        (10, {"batch_size": 0}),
        (10, {"replications": True}),
    ],
)
def test_calibration_invalid_inputs(horizon, kwargs):
    with pytest.raises(ValueError):
        calibrate_monitor(horizon, **kwargs)
