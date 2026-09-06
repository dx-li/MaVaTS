"""Prevent observational examples, scientific coverage and figures from drifting."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import mavats
from examples.air_quality.cases import CASES
from examples.air_quality.data import DATA, load, preprocess

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "docs/gallery"


def test_catalog_covers_every_public_fit_and_selection_procedure():
    expected = {n for n in mavats.__all__ if n.startswith(("fit_", "select_"))}
    expected.update(
        {
            "mar_inference",
            "mar_specification_test",
            "tensor_rank_stability",
            "MatrixFactorMonitor",
            "calibrate_monitor",
        }
    )
    assert {c.api for c in CASES} == expected
    assert len({c.id for c in CASES}) == len(CASES)
    for case in CASES:
        assert case.question and case.caution
        assert callable(getattr(mavats, case.api))


def test_bundled_observations_have_pinned_provenance_and_complete_calendar():
    dates, values, counts = load()
    assert values.shape == counts.shape == (365, 4, 3, 2)
    assert np.all(np.diff(dates).astype(int) == 1)
    assert str(dates[273]) == "2014-10-01"
    assert np.all((counts >= 0) & (counts <= 12))
    assert np.array_equal(counts == 0, np.isnan(values))
    assert np.all(values[np.isfinite(values)] >= 0)
    provenance = json.loads((DATA / "provenance.json").read_text())
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["doi"] == "https://doi.org/10.24432/C5RK5G"


@pytest.mark.parametrize("tensor", [False, True])
def test_preprocessing_never_learns_from_heldout_values(tensor):
    _, values, counts = load()
    x, mask, parameters = preprocess(values, counts, tensor=tensor)
    changed = values.copy()
    changed[273:] = np.where(
        np.isfinite(changed[273:]), changed[273:] * 9 + 100, np.nan
    )
    y, changed_mask, other = preprocess(changed, counts, tensor=tensor)
    np.testing.assert_array_equal(x[:273], y[:273])
    np.testing.assert_array_equal(mask, changed_mask)
    for name in ("mean", "median", "scale"):
        np.testing.assert_array_equal(parameters[name], other[name])
    assert np.isfinite(x).all()
    assert (~mask).any()
    if tensor:
        np.testing.assert_array_equal(mask, counts >= 9)
    else:
        np.testing.assert_array_equal(mask, counts.sum(axis=-1) >= 18)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_every_case_has_auditable_page_figure_and_numerical_results(case):
    report = json.loads((GALLERY / f"{case.id}.json").read_text())
    assert report["id"] == case.id
    assert report["status"] in {"completed", "not_converged", "rejected"}
    page = (GALLERY / f"{case.id}.md").read_text(encoding="utf-8")
    assert "## Question and applicability" in page
    assert case.caution in page
    assert "../citations.md#" in page
    assert "No causal, health or regulatory conclusion follows" in page
    assert (GALLERY / f"{case.id}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with np.load(GALLERY / f"{case.id}.npz", allow_pickle=False) as arrays:
        if "metric_value" in report:
            observed, prediction, mask = (
                arrays[n] for n in ("observed", "prediction", "mask")
            )
            assert mask.dtype == bool
            assert int(mask.sum()) == report["observed_target_cells"]
            mse = np.mean((observed[mask] - prediction[mask]) ** 2)
            assert mse == pytest.approx(report["metric_value"], rel=1e-12)
            if case.kind == "forecast":
                for array, field in (
                    ("last", "last_mse"),
                    ("training_mean", "training_mean_mse"),
                ):
                    score = np.mean((observed[mask] - arrays[array][mask]) ** 2)
                    assert score == pytest.approx(report[field], rel=1e-12)
        if report["status"] == "rejected":
            assert report["error"]
            assert "metric_value" not in report


def test_gallery_fingerprints_match_the_recorded_source_and_data():
    provenance = json.loads((GALLERY / "provenance.json").read_text())
    assert set(provenance["methods"]) == {c.id for c in CASES}
    for source, sha in provenance["source_sha256"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == sha, source
    assert provenance["data"] == json.loads((DATA / "provenance.json").read_text())
    summary = json.loads((GALLERY / "summary.json").read_text())
    assert {r["id"] for r in summary} == {c.id for c in CASES}
    for report in summary:
        assert report == json.loads((GALLERY / f"{report['id']}.json").read_text())


def test_new_documentation_relative_links_have_real_targets():
    paths = [ROOT / "docs/real-world-examples.md", *GALLERY.glob("*.md")]
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https://", "#")):
                continue
            assert (path.parent / target.split("#", 1)[0]).exists(), (path, target)


def test_gallery_smoke_executes_with_optional_plotting_dependency(tmp_path):
    pytest.importorskip("matplotlib")
    from examples.air_quality.__main__ import build_case

    dates, values, counts = load()
    case = next(c for c in CASES if c.id == "projected-pca")
    report = build_case(case, tmp_path, dates, values, counts)
    assert report["status"] == "completed"
    assert report["metric"] == "reconstruction_discrepancy"
    assert report["metric_value"] > 0


def test_scores_exclude_imputed_targets():
    pytest.importorskip("matplotlib")
    from examples.air_quality.__main__ import mse

    observed = np.array([1.0, 9.0, 3.0])
    predicted = np.array([2.0, 0.0, 5.0])
    mask = np.array([True, False, True])
    assert mse(observed, predicted, mask) == 2.5
    observed[1] = 1e100
    assert mse(observed, predicted, mask) == 2.5
