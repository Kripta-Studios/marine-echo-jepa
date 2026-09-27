"""AEON study evidence must remain separate from historical MOSAiC evidence."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marine_echo.serving.aeon_study import build_aeon_development_report, build_aeon_research
from marine_echo.serving.api import create_app


ROOT = Path(__file__).resolve().parents[2]
REVIEWS = (
    "AEON_VALIDATION_RESCORE_OUTCOME_REVIEW_20260927.json",
    "AEON_HYBRID_OUTCOME_REVIEW_20260927.json",
    "AEON_SOTA_SUPERVISED_OUTCOME_REVIEW_20260927.json",
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
    assert study["development"]["forward_ema"] == "PENDING_INDEPENDENT_OUTCOME_REVIEW"
    assert study["development"]["chronos2"] == "PENDING_INDEPENDENT_OUTCOME_REVIEW"
    assert study["selection"] == "NOT_PERFORMED_IN_THIS_REPORT"
    assert study["cached_forecasts"] == 0


def test_tampered_review_bytes_cannot_change_development_metric(tmp_path: Path) -> None:
    folder = tmp_path / "orchestration/reviews"
    folder.mkdir(parents=True)
    for name in REVIEWS:
        shutil.copyfile(ROOT / "orchestration/reviews" / name, folder / name)
    changed = folder / REVIEWS[0]
    review = json.loads(changed.read_text(encoding="utf-8"))
    review["primary_daily_mean_pinball_db_by_slot"]["direct_seed7"] = 0.0
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
