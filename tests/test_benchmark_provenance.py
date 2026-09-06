import hashlib

from benchmarks.run import source_provenance, source_snapshot


def test_snapshot_records_starting_source_and_detects_modified_or_missing_files(
    tmp_path, monkeypatch
):
    from benchmarks import run

    root = tmp_path
    (root / "benchmarks").mkdir()
    (root / "mavats").mkdir()
    helper = root / "benchmarks/run.py"
    helper.write_text("helper")
    model = root / "mavats/model.py"
    model.write_text("original")
    model_key = str(model.relative_to(root))
    helper_key = str(helper.relative_to(root))
    monkeypatch.setattr(run, "__file__", str(helper))
    snapshot = source_snapshot(helper)
    assert source_provenance(snapshot)["source_changed_during_run"] == []
    model.write_text("modified")
    report = source_provenance(snapshot)
    assert report["source_sha256"][model_key] == hashlib.sha256(b"original").hexdigest()
    assert report["source_changed_during_run"] == [model_key]
    helper.unlink()
    assert source_provenance(snapshot)["source_changed_during_run"] == [
        helper_key,
        model_key,
    ]
