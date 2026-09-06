from benchmarks import run


def test_scoring_failure_preserves_already_measured_fit_only_time(monkeypatch):
    times = iter([0.0, 2.0, 10.0])
    monkeypatch.setattr(run, "perf_counter", lambda: next(times))

    def failed_score(fit):
        raise ValueError("scoring failed after successful fit")

    result = run._measure("test", object, failed_score, {})
    assert result["status"] == "error"
    assert result["failure_phase"] == "score"
    assert result["fit_seconds"] == 2.0
    assert "scoring failed" in result["error"]


def test_fitting_failure_retains_elapsed_attempt_time(monkeypatch):
    times = iter([1.0, 4.5])
    monkeypatch.setattr(run, "perf_counter", lambda: next(times))

    def failed_fit():
        raise ValueError("fit failed")

    result = run._measure("test", failed_fit, lambda fit: {}, {})
    assert result["status"] == "error"
    assert result["failure_phase"] == "fit"
    assert result["fit_seconds"] == 3.5
