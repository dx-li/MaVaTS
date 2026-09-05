"""Replication-level uncertainty and end-to-end benchmark recording contracts."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from benchmarks import inference as inference_benchmark
from benchmarks.inference import run_study, summary
from benchmarks.matrix_extensions import run_suite


def _coverage(coverage, *, status="ok", replicate=0):
    row = dict(
        task="coverage",
        n=100,
        regime="isotropic",
        method="als",
        replicate=replicate,
        status=status,
    )
    if status == "ok":
        row.update(
            coverage=coverage,
            entry_coverage=np.full((4, 4), coverage).tolist(),
            mean_width=1 + coverage,
            assumptions_satisfied=True,
        )
    else:
        row["error"] = "synthetic numerical failure"
    return row


def test_coverage_mc_error_uses_series_not_operator_entries():
    rows = [_coverage(0), _coverage(1, replicate=1)]
    record = summary(rows)[0]
    assert record["successful"] == record["total"] == 2
    assert record["mean_rate"] == 0.5
    # Two independent series with coverage rates 0 and 1 give SE=.5.
    # Treating their 32 strongly dependent entries as trials is incorrect.
    assert_allclose(record["monte_carlo_se"], 0.5)
    assert_allclose(record["per_entry_coverage"], np.full((4, 4), 0.5))
    assert_allclose(record["mean_interval_width"], 1.5)
    assert record["assumptions_satisfied"] is True


def test_summary_retains_failure_denominator_without_imputing_coverage():
    rows = [_coverage(0), _coverage(1, replicate=1), _coverage(None, status="error")]
    before = json.dumps(rows, sort_keys=True)
    record = summary(rows)[0]
    assert record["successful"] == 2
    assert record["total"] == 3
    assert record["rates_condition_on_success"] is True
    assert record["mean_rate"] == 0.5
    assert_allclose(record["monte_carlo_se"], 0.5)
    assert json.dumps(rows, sort_keys=True) == before
    failed = summary([_coverage(None, status="error")])[0]
    assert failed["successful"] == 0
    assert failed["total"] == 1
    assert failed["mean_rate"] is None
    assert failed["monte_carlo_se"] is None
    assert "per_entry_coverage" not in failed
    single = summary([_coverage(1)])[0]
    assert single["monte_carlo_se"] is None
    assert summary([]) == []


@pytest.mark.parametrize("rejected", [False, True])
def test_specification_wilson_bounds_remain_informative_at_endpoints(rejected):
    count = 20
    rows = [
        dict(
            task="specification",
            n=100,
            regime="null",
            method="wald",
            status="ok",
            rejected=rejected,
            replicate=i,
        )
        for i in range(count)
    ]
    record = summary(rows)[0]
    assert record["mean_rate"] == float(rejected)
    assert record["monte_carlo_se"] == 0
    z = 1.959963984540054
    endpoint_width = z**2 / (count + z**2)
    expected = [1 - endpoint_width, 1] if rejected else [0, endpoint_width]
    assert_allclose(record["binomial_wilson_95"], expected, atol=2e-16)


def test_summary_separates_methods_regimes_and_sample_sizes():
    baseline = _coverage(1)
    rows = [baseline]
    for field, value in (("method", "mle"), ("regime", "separable"), ("n", 200)):
        rows.append(dict(baseline, **{field: value}))
    records = summary(rows)
    assert len(records) == 4
    assert all(record["total"] == 1 for record in records)


def test_small_inference_study_records_comparable_runs_and_assumptions():
    rows = run_study(repeats=1, sample_sizes=(80,), seed=7123)
    assert rows
    assert {row["task"] for row in rows} == {"coverage", "specification"}
    assert all(row["status"] == "ok" for row in rows), rows
    assert all(
        row["n"] == 80 and row["replicate"] == 0 and row["seed"] == 7123 for row in rows
    )
    coverage = [row for row in rows if row["task"] == "coverage"]
    assert {row["regime"] for row in coverage} == {
        "isotropic",
        "separable",
        "nonseparable",
    }
    assert {row["method"] for row in coverage} >= {"projection", "als", "mle"}
    for row in coverage:
        indicators = np.asarray(row["entry_coverage"])
        assert indicators.shape == (4, 4)
        assert set(np.unique(indicators)) <= {0, 1}
        assert_allclose(row["coverage"], indicators.mean())
        assert row["mean_width"] > 0
        assert row["assumptions_satisfied"] == (
            not (row["method"] == "mle" and row["regime"] == "nonseparable")
        )
    specification = [row for row in rows if row["task"] == "specification"]
    assert {row["regime"] for row in specification} == {"null", "alternative"}
    for row in specification:
        assert 0 <= row["pvalue"] <= 1
        assert row["rejected"] == (row["pvalue"] < 0.05)
    # Benchmark reports must serialize strictly, without NaN/Infinity.
    json.dumps(dict(results=rows, summary=summary(rows)), allow_nan=False)


def test_inference_study_preserves_numerical_failure_rows(monkeypatch):
    def fail_fit(*args, **kwargs):
        raise FloatingPointError("injected solver failure")

    monkeypatch.setattr(inference_benchmark, "fit_mar", fail_fit)
    rows = run_study(repeats=1, sample_sizes=(80,), seed=7123)
    coverage = [row for row in rows if row["task"] == "coverage"]
    assert coverage
    assert all(row["status"] == "error" for row in coverage)
    assert all("injected solver failure" in row["error"] for row in coverage)
    assert all(row["seconds"] >= 0 for row in coverage)
    assert all("coverage" not in row for row in coverage)
    assert all(record["successful"] == 0 for record in summary(coverage))
    assert sum(record["total"] for record in summary(coverage)) == len(coverage)
    assert any(row["task"] == "specification" and row["status"] == "ok" for row in rows)


def test_quick_matrix_extension_suite_records_methods_and_scores():
    rows = run_suite(quick=True, repeats=1, seed=48)
    assert rows
    assert all(row["status"] == "ok" for row in rows), rows
    assert {row["task"] for row in rows} == {
        "sparse-forecast",
        "threshold-factor",
        "decorrelation-forecast",
    }
    assert {row["method"] for row in rows} >= {
        "sparse-emvs",
        "mar-als",
        "mar-mle",
        "var",
        "threshold-estimated",
        "threshold-known-oracle",
        "global-factor-union-ranks",
        "decorrelated-block-var",
    }
    for row in rows:
        assert row["replicate"] == 0 and row["seed"] == 48
        assert row["fit_seconds"] >= 0
        assert row["n_train"] > 0
        assert isinstance(row["warnings"], list)
        if row["task"] == "threshold-factor":
            assert row["held_out_signal_error"] >= 0
        else:
            assert row["forecast_mse"] >= 0
    sparse = next(row for row in rows if row["method"] == "sparse-emvs")
    for metric in ("support_error", "true_positive_rate", "false_positive_rate"):
        assert 0 <= sparse[metric] <= 1
    block = next(row for row in rows if row["method"] == "decorrelated-block-var")
    assert block["round_trip_error"] < 1e-10
    json.dumps(rows, allow_nan=False)
