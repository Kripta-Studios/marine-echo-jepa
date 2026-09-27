"""AEON study evidence must remain separate from historical MOSAiC evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marine_echo.serving.aeon_study import (
    build_aeon_calibration_report,
    build_aeon_development_report,
    build_aeon_research,
)
from marine_echo.serving.api import create_app

ROOT = Path(__file__).resolve().parents[2]
REVIEWS = (
    "AEON_VALIDATION_RESCORE_OUTCOME_REVIEW_20260927.json",
    "AEON_HYBRID_OUTCOME_REVIEW_20260927.json",
    "AEON_SOTA_SUPERVISED_OUTCOME_REVIEW_20260927.json",
    "AEON_FORWARD_OUTCOME_REVIEW_20260927.json",
    "AEON_CHRONOS2_OUTCOME_REVIEW_20260927.json",
)


def test_reviewed_development_report_labels_source_and_unopened_partitions() -> None:
    study = build_aeon_development_report(ROOT)
    assert study["study_id"] == "aeon3_geb_2024_hourly_sv_v1"
    assert study["target"]["frequency_hz"] == 38000
    assert study["target"]["source_variable"] == "Sv_mean"
    assert study["target"]["calibration_claim"] == "SOURCE_REPORTED_CONDITIONED_NOT_INDEPENDENTLY_FIELD_VERIFIED"
    assert study["source_time_basis"] == "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC"
    assert study["assessment_partition"] == "validation"
    assert study["calibration_outcomes"] == "NOT_OPENED_FOR_THIS_REPORT"
    assert study["retrospective_test_outcomes"] == "NOT_OPENED_FOR_THIS_REPORT"
    assert study["final_evaluation"] is False
    assert study["development"]["core"]["issued_rows"] == 1219
    assert len(study["development"]["core"]["slot_primary_pinball_db"]) == 17
    assert study["development"]["post_hoc_supervised"]["family"] == "LightGBM"
    assert study["development"]["forward_ema"]["primary_pinball_db"] == pytest.approx(0.6533145184149417)
    assert study["development"]["chronos2"]["primary_pinball_db"] == pytest.approx(0.6935176150003005)
    assert study["development"]["chronos2"]["status"] == "INDEPENDENTLY_REVIEWED_POST_HOC_ZERO_SHOT_DEVELOPMENT"
    assert study["selection"] == "NOT_PERFORMED_IN_THIS_REPORT"
    assert study["cached_forecasts"] == 0


def test_calibration_report_requires_exact_reviewed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_source = ROOT / "orchestration/reviews/AEON_CALIBRATION_OUTCOME_REVIEW_20260927.json"
    review_path = tmp_path / "orchestration/reviews/AEON_CALIBRATION_OUTCOME_REVIEW_20260927.json"
    review_path.parent.mkdir(parents=True)
    shutil.copyfile(review_source, review_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    artifact = {
        "status": "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING",
        "study_id": "aeon3_geb_2024_hourly_sv_v1",
        "partition": "calibration", "test_access": "PROHIBITED",
        "selection_freeze_sha256": review["selection_freeze_sha256"],
        "forecast_manifest_sha256": review["calibration_forecast_plan_sha256"],
        "config_sha256": review["config_sha256"],
        "issued_row_ids": [f"fixture-{index}" for index in range(810)],
        "models": {},
    }
    for name, model in review["models"].items():
        artifact["models"][name] = {
            "selection_classification": model["selection_classification"],
            "adjustment_db": model["adjustment_db_by_horizon"],
            "eligible_days_per_horizon": [34, 34, 34],
            "eligible_rows_per_horizon": review["calibration_support"]["eligible_scored_rows_per_horizon"],
            "raw_interval_metrics": {
                "primary_daily_mean_pinball_db": model["raw_primary_daily_mean_pinball_db"],
                "daily_mean_pinball_db_per_horizon": model["raw_daily_mean_pinball_db_per_horizon"],
                "coverage90_per_horizon": model["raw_coverage90_per_horizon"],
            },
            "widened_interval_metrics": {
                "primary_daily_mean_pinball_db": model["widened_primary_daily_mean_pinball_db"],
                "coverage90_per_horizon": model["widened_coverage90_per_horizon"],
            },
        }
    artifact_path = tmp_path / "calibration.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    monkeypatch.setattr("marine_echo.serving.aeon_study._CAL_ARTIFACT_SHA", digest)
    review["artifact_sha256"] = digest
    review_path.write_text(json.dumps(review), encoding="utf-8")
    monkeypatch.setattr(
        "marine_echo.serving.aeon_study._CAL_REVIEW_SHA",
        hashlib.sha256(review_path.read_bytes()).hexdigest(),
    )
    summary = build_aeon_calibration_report(tmp_path, artifact_path)
    assert summary["issued_rows"] == 810
    assert summary["eligible_days_per_horizon"] == [34, 34, 34]
    assert summary["models"]["core_direct_equal_three_seed_ensemble"]["raw_primary_pinball_db"] == pytest.approx(0.6530512691964927)
    artifact["models"]["post_hoc_lightgbm"]["adjustment_db"][0] = -1
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact digest"):
        build_aeon_calibration_report(tmp_path, artifact_path)


@pytest.mark.parametrize("changed_review", REVIEWS)
def test_tampered_review_bytes_cannot_change_development_metric(
    tmp_path: Path, changed_review: str,
) -> None:
    folder = tmp_path / "orchestration/reviews"
    folder.mkdir(parents=True)
    for name in REVIEWS:
        shutil.copyfile(ROOT / "orchestration/reviews" / name, folder / name)
    changed = folder / changed_review
    review = json.loads(changed.read_text(encoding="utf-8"))
    review["status"] = "UNREVIEWED_MUTATION"
    changed.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="review digest"):
        build_aeon_development_report(tmp_path)


def test_release_adds_aeon_route_without_changing_v1_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base_release(_: Path, destination: Path) -> dict[str, str]:
        artifacts = destination / "artifacts"
        artifacts.mkdir(parents=True)
        (destination / "web").mkdir()
        catalog = {
            "schema_version": "1.0", "release_class": "OFFLINE_RESEARCH_ENGINEERING_ONLY",
            "datasets": [{"id": "mosaic"}],
            "models": [{"id": "direct", "status": "UNAVAILABLE"}],
            "experiments": [{"run_id": "v1", "status": "BLOCKED", "metrics": None}],
            "artifacts": {}, "exports": [], "forecasts": {},
        }
        (artifacts / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        (destination / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        return {"status": catalog["release_class"]}

    monkeypatch.setattr("marine_echo.serving.aeon_study.build_v2_research", base_release)
    destination = tmp_path / "aeon-release"
    result = build_aeon_research(ROOT, destination)
    assert result["status"] == "OFFLINE_RESEARCH_MIXED_STUDIES_DEVELOPMENT_ONLY"
    catalog = json.loads((destination / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["experiments"] == [{"run_id": "v1", "status": "BLOCKED", "metrics": None}]
    assert catalog["forecasts"] == {}
    sums = (destination / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert result["packaged_asset_count"] == len(sums)
    for line in sums:
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    client = TestClient(create_app(destination / "artifacts"))
    response = client.get("/api/v1/studies/aeon")
    assert response.status_code == 200
    assert response.json()["final_evaluation"] is False
    assert client.post("/api/v1/forecast", json={"dataset_id": "mosaic", "model_id": "direct", "cutoff": "2020-02-17T12:00:00Z"}).status_code == 503
