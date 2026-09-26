import json
import shutil
from pathlib import Path

import pytest

from marine_echo.serving.release import build_diagnostic, run_registry


@pytest.mark.parametrize("attempt", ["failed_row", "completed_top_level"])
def test_working_preview_preserves_registry_and_current_evidence(
    tmp_path: Path, attempt: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    for name in (
        "configs/experiments.json",
        "data/manifests/mosaic_azfp_down_2020/source_inventory.json",
        "evidence/data/diagnostic_raw_replay_2020-02-17.json",
        "reports/active/protocol.json",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    (tmp_path / "web/dist").mkdir(parents=True)
    (tmp_path / "web/dist/index.html").write_text("<html>Test fixture</html>")
    registry = run_registry(root)
    registry["test_opened"] = True
    if attempt == "failed_row":
        registry["runs"][0].update(status="FAILED", updates=2, failure_log="preserved.log")
    else:
        registry["status"] = "COMPLETED"
    (tmp_path / "reports/active/training_registry.json").write_text(json.dumps(registry))
    summary = {
        "schema_version": "1.0",
        "evidence_scope": "TRAIN_QC_ONLY_NOT_BENCHMARK",
        "limitations": ["Test fixture scope only"],
        "forecast_unavailability_reason": "No eligible reviewed corpus or completed benchmark.",
    }
    (tmp_path / "evidence/continuation").mkdir()
    (tmp_path / "evidence/continuation/public_summary.json").write_text(json.dumps(summary))
    output = tmp_path / "preview"
    build_diagnostic(tmp_path, output)
    catalog = json.loads((output / "artifacts/catalog.json").read_text())
    evidence = json.loads((output / "artifacts/research.json").read_text())
    assert catalog["experiments"] == registry["runs"]
    assert evidence["run_registry"] == registry
    assert evidence["gates"]["G2_EXPERIMENT"] == "INCOMPLETE_RECORDED_ATTEMPTS"
    assert evidence["continuation"] == summary
    assert evidence["forecast_unavailability_reason"] == summary["forecast_unavailability_reason"]
    assert "Test fixture scope only" in evidence["limitations"]
    assert all(model["status"] != "AVAILABLE" for model in catalog["models"])


@pytest.mark.parametrize(
    "corruption",
    ["empty", "boolean_count", "boolean_updates", "duplicate", "identity", "missing_status"],
)
def test_registry_validation_rejects_false_unattempted_claim(corruption: str) -> None:
    from marine_echo.serving.release import validate_recorded_registry

    root = Path(__file__).resolve().parents[2]
    registry = run_registry(root)
    if corruption == "empty":
        registry["runs"] = []
    elif corruption == "boolean_count":
        registry["completed_benchmark_runs"] = False
    elif corruption == "boolean_updates":
        registry["runs"][0]["updates"] = False
    elif corruption == "duplicate":
        registry["runs"][1] = registry["runs"][0].copy()
    elif corruption == "identity":
        registry["runs"][0]["seed"] = 13
    else:
        del registry["status"]
    with pytest.raises(ValueError):
        validate_recorded_registry(root, registry)


def test_release_rejects_existing_directory_before_reading_sources(tmp_path: Path) -> None:
    output = tmp_path / "published"
    output.mkdir()
    (output / "stale.js").write_text("old")
    with pytest.raises(FileExistsError):
        build_diagnostic(tmp_path / "missing-source", output)
    assert (output / "stale.js").read_text() == "old"


def test_failed_build_does_not_publish_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "published"
    with pytest.raises(FileNotFoundError):
        build_diagnostic(tmp_path / "missing-source", output)
    assert not output.exists()
