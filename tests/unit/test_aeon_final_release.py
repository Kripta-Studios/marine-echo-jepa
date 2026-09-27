"""Synthetic-only checks for the reviewed AEON final-release boundary."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from marine_echo.serving import aeon_final
from marine_echo.serving.api import create_app


def _write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    source = "a" * 64
    selection = "b" * 64
    cal = "c" * 64
    pretest = "d" * 64
    row_ids = ["row-1", "row-2"]
    candidate = tmp_path / "candidate.json"
    candidate_sha = _write(candidate, {
        "classification": "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED",
        "numeric_test_outcome_access": "PROHIBITED",
        "source_archive_sha256": source,
        "candidate_rows": [
            {"row_id": row, "cutoff_interval_id": 100 + index,
             "cutoff_source_timestamp": f"2025-01-06 0{index}:00:00",
             "target_interval_ids": [101 + index, 103 + index, 106 + index]}
            for index, row in enumerate(row_ids)
        ],
    })
    score_dir = tmp_path / "score"
    score_dir.mkdir()
    models = {}
    for name in aeon_final.MODEL_IDS:
        file = score_dir / f"{name}-forecast.npz"
        with file.open("wb") as stream:
            np.savez_compressed(stream, row_ids=np.asarray(row_ids),
                                quantiles_db=np.tile(np.asarray([-4., -3., -2., -1., 0.]), (2, 3, 1)))
        models[name] = {
            "forecast_artifact_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "selection_classification": "CORE" if name != "post_hoc_lightgbm" else "POST_HOC",
            "raw_metrics": {"primary_daily_mean_pinball_db": 0.7,
                            "daily_mean_pinball_db_per_horizon": [0.6, 0.7, 0.8],
                            "eligible_days_per_horizon": [49, 50, 51],
                            "coverage90_per_horizon": [0.8, 0.81, 0.82]},
            "widened_interval_metrics": {"coverage90_per_horizon": [0.9, 0.9, 0.9]},
        }
    score = score_dir / "test-score.json"
    score_sha = _write(score, {
        "status": "COMPLETED_AEON_RETROSPECTIVE_TEST_SCORING",
        "classification": "RETROSPECTIVE_EVALUATION_NOT_SEALED",
        "study_id": "aeon3_geb_2024_hourly_sv_v1", "partition": "test",
        "source_archive_sha256": source,
        "selection_freeze_sha256": selection, "calibration_artifact_sha256": cal,
        "pretest_freeze_sha256": pretest, "candidate_contract_sha256": candidate_sha,
        "issued_row_ids": row_ids, "models": models,
        "comparisons": {name: {"baseline_model_id": aeon_final.MODEL_IDS[0],
                               "primary_relative_loss_change": -0.02,
                               "paired_95_percent_interval_db": [-0.03, 0.01],
                               "passes_full_unnarrowed_promotion_rule": False}
                        for name in aeon_final.MODEL_IDS[1:]},
    })
    review = tmp_path / "review.json"
    review_sha = _write(review, {
        "status": "APPROVED_AEON_RETROSPECTIVE_TEST_OUTCOME",
        "reviewer_session": "/root/aeon_reviewer",
        "artifact_sha256": {"test_score": score_sha, **{
            f"{name}_forecast": value["forecast_artifact_sha256"] for name, value in models.items()}},
        "frozen_lineage_sha256": {
            "source_archive": source, "selection_freeze": selection,
            "calibration_artifact": cal, "pretest_freeze": pretest,
            "metadata_candidate_report": candidate_sha,
        },
        "support": {"issued_rows": 2, "eligible_source_dates_per_horizon": [49, 50, 51]},
        "models": {name: {
            "selection_classification": value["selection_classification"],
            "raw_primary_daily_mean_pinball_db": 0.7,
            "raw_daily_mean_pinball_db_per_horizon": [0.6, 0.7, 0.8],
            "raw_coverage90_per_horizon": [0.8, 0.81, 0.82],
            "widened_coverage90_per_horizon": [0.9, 0.9, 0.9],
        } for name, value in models.items()},
        "comparisons": {f"{name}_minus_direct": {
            "primary_relative_loss_change": -0.02,
            "paired_95_percent_interval_db": [-0.03, 0.01],
            "passes_full_unnarrowed_promotion_rule": False,
        } for name in aeon_final.MODEL_IDS[1:]},
        "independent_execution": {
            "new_test_evaluator_run": False,
            "selection_or_tuning_after_test_access": False,
        },
        "classification": "RETROSPECTIVE_EVALUATION_NOT_SEALED",
    })
    monkeypatch.setattr(aeon_final, "TEST_SCORE_SHA256", score_sha)
    monkeypatch.setattr(aeon_final, "TEST_REVIEW_SHA256", review_sha)
    monkeypatch.setattr(aeon_final, "SOURCE_SHA256", source)
    monkeypatch.setattr(aeon_final, "SELECTION_SHA256", selection)
    monkeypatch.setattr(aeon_final, "CAL_SHA256", cal)
    monkeypatch.setattr(aeon_final, "PRETEST_SHA256", pretest)
    monkeypatch.setattr(aeon_final, "EXPECTED_ISSUED_ROWS", 2)
    return score, candidate, review, score_dir


def test_reviewed_report_and_truth_free_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, candidate, review, forecasts = _fixture(tmp_path, monkeypatch)
    report, replay = aeon_final.load_reviewed_test(score, candidate, review, forecasts)
    assert report["issued_rows"] == 2
    assert report["eligible_days_per_horizon"] == [49, 50, 51]
    assert report["jepa_value_gate"] == "PASSED_FROZEN_WITHIN_STUDY_RETROSPECTIVE_GATE"
    assert replay["rows"][0]["predictions"][aeon_final.MODEL_IDS[0]][0] == [-4., -3., -2., -1., 0.]
    assert "truth" not in json.dumps(replay).lower()


def test_no_review_pin_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, candidate, review, forecasts = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(aeon_final, "TEST_REVIEW_SHA256", None)
    with pytest.raises(ValueError, match="pinned independent TEST review"):
        aeon_final.load_reviewed_test(score, candidate, review, forecasts)


def test_changed_review_or_forecast_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, candidate, review, forecasts = _fixture(tmp_path, monkeypatch)
    review.write_text(review.read_text() + " ")
    with pytest.raises(ValueError, match="review digest"):
        aeon_final.load_reviewed_test(score, candidate, review, forecasts)
    monkeypatch.setattr(aeon_final, "TEST_REVIEW_SHA256", hashlib.sha256(review.read_bytes()).hexdigest())
    forecast = forecasts / f"{aeon_final.MODEL_IDS[0]}-forecast.npz"
    forecast.write_bytes(forecast.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="forecast digest"):
        aeon_final.load_reviewed_test(score, candidate, review, forecasts)


def test_changed_lineage_or_row_order_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, candidate, review, forecasts = _fixture(tmp_path, monkeypatch)
    value = json.loads(score.read_text())
    value["selection_freeze_sha256"] = "wrong"
    monkeypatch.setattr(aeon_final, "TEST_SCORE_SHA256", _write(score, value))
    with pytest.raises(ValueError, match="lineage"):
        aeon_final.load_reviewed_test(score, candidate, review, forecasts)


def test_replay_api_is_bounded_and_kept_off_utc_forecast_route(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    study_sha = _write(artifact / "study.json", {
        "retrospective_test_outcomes": "INDEPENDENTLY_REVIEWED_RETROSPECTIVE_TEST",
        "retrospective_test": {"test_score_sha256": "a" * 64,
                               "models": {name: {} for name in aeon_final.MODEL_IDS}},
    })
    replay_sha = _write(artifact / "replay.json", {
        "classification": "REVIEWED_RETROSPECTIVE_SOURCE_CLOCK_REPLAY_NOT_LIVE",
        "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
        "target": "source-conditioned Sv_mean",
        "test_score_sha256": "a" * 64,
        "horizon_source_interval_steps": [1, 3, 6],
        "quantile_levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "rows": [{"row_id": "first", "cutoff_interval_id": 101,
                  "cutoff_source_timestamp": "2025-01-06 01:00:00",
                  "predictions": {name: [[-4, -3, -2, -1, 0]] * 3
                                  for name in aeon_final.MODEL_IDS}}],
    })
    _write(artifact / "catalog.json", {
        "release_class": "REVIEW_PENDING_FIXTURE",
        "datasets": [], "models": [],
        "artifacts": {
            "aeon-study": {"path": "study.json", "kind": "aeon-study", "sha256": study_sha},
            "aeon-test-replay": {"path": "replay.json", "kind": "aeon-test-replay", "sha256": replay_sha},
        },
    })
    client = TestClient(create_app(artifact))
    model = aeon_final.MODEL_IDS[0]
    response = client.get("/api/v1/studies/aeon/replay", params={"model_id": model})
    assert response.status_code == 200
    assert response.json()["rows"][0]["cutoff_source_timestamp"] == "2025-01-06 01:00:00"
    assert "truth" not in response.text.lower()
    assert client.get("/api/v1/studies/aeon/replay", params={"model_id": model, "limit": 25}).status_code == 422
    assert client.get("/api/v1/studies/aeon/replay", params={"model_id": "unknown"}).status_code == 404
    assert client.post("/api/v1/forecast", json={"dataset_id": "aeon", "model_id": model,
        "cutoff": "2025-01-06T01:00:00Z"}).status_code == 404
    (artifact / "replay.json").unlink()
    assert TestClient(create_app(artifact)).get(
        "/api/v1/studies/aeon/replay", params={"model_id": model},
    ).status_code == 503
