"""The offline research artifact must preserve reviewed row and gate identity."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marine_echo.serving.api import create_app
from marine_echo.serving.raw_development import build_raw_development_report
from marine_echo.serving.release import build_v2_research

ROOT = Path(__file__).resolve().parents[2]


def test_reviewed_raw_report_exposes_all_issued_rows_and_negative_result() -> None:
    report = build_raw_development_report(ROOT)
    assert report["status"] == "REAL_TRAIN_DEVELOPMENT_ONLY"
    assert report["calibrated"] is False
    assert report["final_evaluation"] is False
    assert len(report["rows"]) == 212
    assert [row["eligible_rows"] for row in report["horizons"]] == [176, 140, 135]
    assert [row["target_days"] for row in report["horizons"]] == [13, 13, 13]
    assert all(
        row["ridge"]["daily_mean_pinball_code"] < row["direct_neural"]["daily_mean_pinball_code"]
        for row in report["horizons"]
    )
    assert all(len(row["horizons"]) == 3 for row in report["rows"])


def test_offline_v2_release_serves_reviewed_rows_and_preserves_v1_block(tmp_path: Path) -> None:
    output = tmp_path / "v2-research"
    result = build_v2_research(ROOT, output)
    assert result["raw_development_rows"] == 212
    assert result["calibrated_benchmark_runs"] == 0
    assert (output / "web/index.html").is_file()
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["release_class"] == "OFFLINE_RESEARCH_ENGINEERING_ONLY"
    assert len(catalog["experiments"]) == 25
    assert all(row["status"] == "BLOCKED" and row["updates"] == 0 for row in catalog["experiments"])
    client = TestClient(create_app(output / "artifacts", output / "web"))
    assert client.get("/health").json()["release_class"] == "OFFLINE_RESEARCH_ENGINEERING_ONLY"
    evidence = client.get("/api/v1/evidence/research").json()
    assert evidence["release_class"] == "OFFLINE_RESEARCH_ENGINEERING_ONLY"
    assert evidence["gates"]["G2_EXPERIMENT"] == "BLOCKED_CALIBRATED_CORE_0_OF_25"
    report = client.get("/api/v1/evidence/raw-development").json()
    assert len(report["rows"]) == 212
    assert client.get("/api/v1/exports/raw-development-export").json() == report
    assert (
        client.post(
            "/api/v1/forecast",
            json={
                "dataset_id": catalog["datasets"][0]["id"],
                "model_id": "ridge",
                "cutoff": "2020-02-17T12:00:00Z",
            },
        ).status_code
        == 503
    )


def test_v2_release_rejects_existing_destination_before_source_read(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "owner.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_v2_research(tmp_path / "missing-source", output)
    assert (output / "owner.txt").read_text(encoding="utf-8") == "preserve"


def test_raw_report_rejects_prediction_byte_change(tmp_path: Path) -> None:
    review_relative = "orchestration/reviews/V2_RAW_RESPONSE_DEVELOPMENT_RESULT_20260927.json"
    run_relative = "outputs/raw-response-development-v1-deterministic-20260927"
    review = tmp_path / review_relative
    review.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / review_relative, review)
    run = tmp_path / run_relative
    shutil.copytree(ROOT / run_relative, run)
    path = run / "ridge-assessment-predictions.npz"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="prediction bytes changed"):
        build_raw_development_report(tmp_path)
