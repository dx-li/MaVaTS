import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import rank_stability as benchmark
from benchmarks.rank_scenarios import rank_factor_data


def _tiny_settings():
    return dict(
        n=20,
        shape=(6, 8),
        c_grid=np.geomspace(0.01, 100, 5),
        spatial_subsets=(
            (tuple(range(4)), tuple(range(6))),
            (tuple(range(5)), tuple(range(7))),
            (tuple(range(6)), tuple(range(8))),
        ),
        time_prefixes=(10, 15, 20),
        max_ranks=(3, 3),
    )


def test_standard_and_quick_protocol_grids():
    for quick, n, points, rows, cols, times, caps in [
        (False, 300, 25, [6, 8, 10], [8, 10, 12], [100, 200, 300], (4, 4)),
        (True, 70, 9, [4, 5, 6], [6, 7, 8], [30, 50, 70], (3, 3)),
    ]:
        setup = benchmark._settings(quick)
        assert setup["n"] == n
        assert setup["max_ranks"] == caps
        assert list(setup["time_prefixes"]) == times
        assert [len(s[0]) for s in setup["spatial_subsets"]] == rows
        assert [len(s[1]) for s in setup["spatial_subsets"]] == cols
        np.testing.assert_allclose(setup["c_grid"], np.geomspace(0.01, 100, points))


@pytest.mark.parametrize("iterative", [False, True])
def test_real_tiny_path_retains_all_cells_and_population_variance(iterative):
    data = rank_factor_data(25, shape=(6, 8))
    row = benchmark._run_case(data, _tiny_settings(), iterative=iterative, metadata={})
    assert row["status"] == "ok"
    ranks = np.asarray(row["ranks"])
    assert ranks.shape == (2, 5, 3)
    np.testing.assert_allclose(row["variances"], ranks.var(axis=2, ddof=0))
    np.testing.assert_array_equal(row["full_sample_ranks"], ranks[:, :, -1])
    assert np.shape(row["cell_converged"]) == (5, 3)
    assert row["cell_failures"] == 0
    assert row["fixed_c1"]["status"] == "ok"
    assert row["plateau_options"] == dict(
        variance_tolerance=0, minimum_grid_points=2, allow_unconverged=False
    )
    if row["selection_complete"]:
        assert (
            row["final_refit"]["penalty_multipliers"]
            == row["selected_penalty_multipliers"]
        )
    else:
        assert row["final_refit"]["status"] == "not-run"
    json.dumps(row, allow_nan=False)


def test_no_admissible_choice_is_not_failure_and_never_refits(monkeypatch):
    data = rank_factor_data(25, shape=(6, 8), regime="no-factors")
    settings = _tiny_settings()
    settings["c_grid"] = np.array([1e10, 1e11])
    calls = []
    original = benchmark._fit_score

    def traced(*args):
        calls.append(args[-1])
        return original(*args)

    monkeypatch.setattr(benchmark, "_fit_score", traced)
    row = benchmark._run_case(data, settings, iterative=False, metadata={})
    assert row["status"] == "ok"
    assert not row["selection_complete"]
    assert row["selected_penalty_multipliers"] == [None, None]
    assert calls == [1.0]
    assert row["final_refit"] == dict(
        status="not-run", reason="incomplete_modewise_plateau_choice"
    )
    assert benchmark._execution_errors([row]) == 0


def test_failed_cells_are_retained_and_nan_variances_become_null(monkeypatch):
    import mavats.rank_selection as selector

    original = selector.select_tensor_rank

    def fail_small(X, **kwargs):
        if len(X) == 10:
            raise ValueError("deliberate subsample failure")
        return original(X, **kwargs)

    monkeypatch.setattr(selector, "select_tensor_rank", fail_small)
    data = rank_factor_data(25, shape=(6, 8))
    row = benchmark._run_case(data, _tiny_settings(), iterative=False, metadata={})
    assert row["status"] == "ok"
    assert row["cell_failures"] == 5
    assert row["variances"] == [[None] * 5, [None] * 5]
    assert not row["selection_complete"]
    assert all("deliberate subsample" in cells[0]["error"] for cells in row["cells"])
    assert benchmark._execution_errors([row]) == 5
    json.dumps(row, allow_nan=False)


def test_unconverged_cells_disqualify_choices_but_are_not_execution_errors(monkeypatch):
    import mavats.rank_selection as selector

    original = selector.select_tensor_rank

    def unfinished_small(X, **kwargs):
        result = original(X, **kwargs)
        if len(X) == 10:
            result.converged = False
        return result

    monkeypatch.setattr(selector, "select_tensor_rank", unfinished_small)
    data = rank_factor_data(25, shape=(6, 8))
    row = benchmark._run_case(data, _tiny_settings(), iterative=False, metadata={})
    assert row["cell_unconverged"] == 5
    assert not row["selection_complete"]
    assert not row["all_cells_converged"]
    assert benchmark._execution_errors([row]) == 0


def test_complete_modewise_choice_triggers_joint_refit_and_retains_rank_disagreement(
    monkeypatch,
):
    from mavats.rank_stability import TensorRankStabilityResult

    original = TensorRankStabilityResult.choose_plateaus

    def selected(self, **kwargs):
        choice = original(self, **kwargs)
        return replace(
            choice,
            selected_indices=(0, 4),
            penalty_multipliers=(0.01, 100.0),
            selected_ranks=(1, 2),
        )

    monkeypatch.setattr(TensorRankStabilityResult, "choose_plateaus", selected)
    data = rank_factor_data(25, shape=(6, 8))
    row = benchmark._run_case(data, _tiny_settings(), iterative=True, metadata={})
    assert row["selection_complete"]
    assert row["final_refit"]["penalty_multipliers"] == [0.01, 100.0]
    assert row["final_refit"]["status"] == "ok"
    assert row["selected_path_ranks"] == [1, 2]
    assert row["joint_refit_matches_selected_path_ranks"] == (
        row["final_refit"]["ranks"] == [1, 2]
    )


def test_tuning_and_fit_receive_only_training_observations(monkeypatch):
    data = rank_factor_data(25, shape=(6, 8))
    calls = []
    path_function, fit_function = (
        benchmark.tensor_rank_stability,
        benchmark.select_tensor_rank,
    )

    def path_spy(X, *args, **kwargs):
        calls.append(X.copy())
        return path_function(X, *args, **kwargs)

    def fit_spy(X, **kwargs):
        calls.append(X.copy())
        return fit_function(X, **kwargs)

    monkeypatch.setattr(benchmark, "tensor_rank_stability", path_spy)
    monkeypatch.setattr(benchmark, "select_tensor_rank", fit_spy)
    benchmark._run_case(data, _tiny_settings(), iterative=False, metadata={})
    assert len(calls) >= 2
    for received in calls:
        np.testing.assert_array_equal(received, data.observations[:20])


def test_score_failure_preserves_fitting_time(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "select_tensor_rank",
        lambda *a, **k: SimpleNamespace(converged=True, n_iter=0),
    )
    monkeypatch.setattr(
        benchmark,
        "_scores",
        lambda *a: (_ for _ in ()).throw(ValueError("score failed")),
    )
    clock = iter([10, 12, 100])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(clock))
    row = benchmark._fit_score(None, None, 0, {}, 1)
    assert row["status"] == "error"
    assert row["failure_phase"] == "score"
    assert row["fit_seconds"] == 2


def test_path_failure_keeps_fixed_comparator(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "tensor_rank_stability",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("path failed")),
    )
    data = rank_factor_data(25, shape=(6, 8))
    row = benchmark._run_case(data, _tiny_settings(), iterative=False, metadata={})
    assert row["status"] == "error"
    assert row["failure_phase"] == "path"
    assert row["fixed_c1"]["status"] == "ok"
    assert row["final_refit"]["status"] == "not-run"


def test_run_suite_record_count_and_assumption_labels(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_run_case",
        lambda data, settings, iterative, metadata: dict(metadata, iterative=iterative),
    )
    quick = benchmark.run_suite(quick=True)
    assert len(quick) == 8
    assert set(row["regime"] for row in quick) == {
        "strong",
        "weak",
        "cancellation",
        "no-factors",
    }
    for row in quick:
        assert row["n_train"] == 70
        assert row["informative_tipup"] == (row["regime"] in ("strong", "weak"))
        assert row["positive_rank_plateau_assumption"] == (
            row["regime"] != "no-factors"
        )
        assert row["held_out_task"] == "contemporaneous-denoising-not-forecasting"
    standard = benchmark.run_suite(repeats=2)
    assert len(standard) == 16
    assert {row["seed"] for row in standard} == {2573, 2574}


def test_json_sanitization_is_recursive():
    converted = benchmark._json_safe(
        dict(
            x=np.array([np.nan, np.inf, -np.inf, 2.0]), y=(np.int64(3), np.bool_(True))
        )
    )
    assert converted == dict(x=[None, None, None, 2.0], y=[3, True])
    json.dumps(converted, allow_nan=False)


@pytest.mark.parametrize(
    "change, error, expected_exit",
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_cli_strict_json_provenance_and_exit_policy(
    monkeypatch, tmp_path, change, error, expected_exit
):
    output = tmp_path / "report.json"
    rows = [
        dict(
            status="ok",
            selection_complete=False,
            cell_unconverged=1,
            cell_failures=int(error),
            variances=[[np.nan]],
            fixed_c1=dict(status="ok"),
            final_refit=dict(status="not-run"),
        )
    ]
    monkeypatch.setattr(benchmark, "run_suite", lambda **kwargs: rows)
    monkeypatch.setattr(benchmark, "environment", lambda: {})
    monkeypatch.setattr(
        benchmark, "source_snapshot", lambda *args: {"before": "digest"}
    )
    monkeypatch.setattr(
        benchmark,
        "source_provenance",
        lambda snapshot: dict(
            source_sha256=snapshot,
            source_changed_during_run=["changed.py"] if change else [],
        ),
    )
    monkeypatch.setattr(
        "sys.argv", ["rank_stability", "--quick", "--output", str(output)]
    )
    if expected_exit:
        with pytest.raises(SystemExit, match="1"):
            benchmark.main()
    else:
        benchmark.main()
    report = json.loads(output.read_text())
    assert report["configuration"]["repeats"] == 1
    assert report["source_sha256"] == {"before": "digest"}
    assert report["results"][0]["variances"] == [[None]]
