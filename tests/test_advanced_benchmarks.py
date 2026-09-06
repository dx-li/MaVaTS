import numpy as np
from numpy.testing import assert_allclose

from benchmarks.advanced_matrix import _gaussian_score
from benchmarks.monitoring import run_study, summarize


def test_covariance_score_uses_column_vectorization_and_gaussian_constants():
    observations = np.arange(24.0).reshape(4, 2, 3) / 10
    diagonal = np.arange(1.0, 7.0)
    covariances = np.repeat(np.diag(diagonal)[None], 4, axis=0)
    independent = []
    for x in observations:
        value = 0.0
        for j in range(3):
            for i in range(2):
                variance = diagonal[i + 2 * j]
                value += 0.5 * (np.log(2 * np.pi * variance) + x[i, j] ** 2 / variance)
        independent.append(value)
    assert_allclose(_gaussian_score(observations, covariances), np.mean(independent))


def test_monitoring_summary_retains_failures_prealarms_and_censoring():
    def row(alarm, pre, after, delay):
        return dict(
            regime="space-switch",
            method="test",
            status="ok",
            alarm=alarm,
            pre_event_alarm=pre,
            detected_after_event=after,
            post_event_delay=delay,
        )

    rows = [
        row(True, True, False, None),
        row(True, False, True, 3),
        row(False, False, False, None),
        dict(regime="space-switch", method="test", status="error"),
    ]
    s = summarize(rows)[0]
    assert s["total"] == 4 and s["successful"] == 3
    assert s["alarm_rate"] == 2 / 3
    assert s["at_risk_at_change"] == 2 and s["post_event_detections"] == 1
    assert s["right_censored"] == 1
    assert s["detection_rate_conditional_on_no_pre_alarm"] == 0.5
    assert s["mean_delay_conditional_on_detection"] == 3
    failed = summarize([dict(regime="null", method="test", status="error")])[0]
    assert failed["alarm_rate"] is None and failed["alarm_wilson_95"] is None


def test_small_monitoring_study_reuses_paired_data_and_keeps_stop_indices():
    rows, calibrations = run_study(repeats=1, quick=True, calibration_repeats=99)
    assert len(rows) == 20 and len(calibrations) == 4
    assert all(row["status"] == "ok" for row in rows), rows
    assert len({row["seed"] for row in rows}) == 1
    assert len({row["noise_seed"] for row in rows}) == 1
    for row in rows:
        assert row["consumed"] <= row["horizon"]
        if row["alarm"]:
            assert row["alarm_step"] == row["consumed"]
            assert row["stop_reason"] == "alarm"
        else:
            assert row["consumed"] == row["horizon"]
            assert row["post_event_delay"] is None
