"""Synthetic, outcome-free checks for the AEON validation comparison gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training.aeon_validation_compare import (
    SOURCE_CONTRACTS,
    _comparison_code_sha256,
    _compare_arrays,
    _load_sources,
    _source_report_status,
    _validate_manifest,
    _verify_outer_approval,
)


@pytest.mark.parametrize("dependency", ["aeon.py", "aeon_rescore.py"])
def test_comparison_approval_digest_binds_imported_contracts(
    monkeypatch: pytest.MonkeyPatch, dependency: str,
) -> None:
    baseline = _comparison_code_sha256()
    original = Path.read_bytes

    def changed_read(path: Path) -> bytes:
        content = original(path)
        return content + b"dependency_changed" if path.name == dependency else content

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    assert _comparison_code_sha256() != baseline


def _forecast(value: float, rows: int = 36) -> np.ndarray:
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    return np.broadcast_to(value + offsets, (rows, 3, 5)).copy()


def test_comparison_is_paired_and_reports_no_winner() -> None:
    truth = np.zeros((36, 3))
    mask = np.ones((36, 3), dtype=bool)
    dates = np.repeat(np.array(["2024-11-01", "2024-11-02"], dtype="datetime64[D]"), 18)
    times = np.broadcast_to(dates[:, None], (36, 3)).copy()
    forecasts = {
        "direct_seed7": _forecast(0.2),
        "direct_seed13": _forecast(0.3),
        "direct_seed23": _forecast(0.4),
        "ema_jepa_seed7": _forecast(0.5),
        "ema_jepa_seed13": _forecast(0.6),
        "ema_jepa_seed23": _forecast(0.7),
    }
    result = _compare_arrays(truth, mask, times, forecasts)
    assert result["model_selection"] == "NOT_PERFORMED"
    assert result["comparisons"]["core_ema_equal_three_seed_ensemble"]["point_difference_db"] > 0
    assert result["comparisons"]["core_direct_equal_three_seed_ensemble"]["point_difference_db"] == 0
    assert result["comparisons"]["core_ema_equal_three_seed_ensemble"]["bootstrap_48h"]["draws"] == 2000
    assert result["nonempty_48h_source_date_blocks"] == 1
    assert result["distinct_eligible_source_dates"] == 2
    assert len(result["comparisons"]["core_ema_equal_three_seed_ensemble"]["paired_daily_differences_db_by_horizon"]) == 3


def test_outer_review_binds_exact_code_config_and_source_hashes(tmp_path: Path) -> None:
    config = tmp_path / "input.json"
    config.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    source_hashes = {"core": "a" * 64, "hybrid": "b" * 64}
    approval = {
        "status": "APPROVED_AEON_VALIDATION_COMPARISON_EXECUTION",
        "reviewer_session": "/root/aeon_reviewer",
        "comparison_code_sha256": "c" * 64,
        "comparison_input_sha256": digest,
        "source_report_sha256": source_hashes,
        "calibration_access": "PROHIBITED",
        "test_access": "PROHIBITED",
    }
    _verify_outer_approval(approval, "c" * 64, digest, source_hashes)
    for changed in (
        {**approval, "status": "PROPOSED"},
        {**approval, "comparison_code_sha256": "d" * 64},
        {**approval, "source_report_sha256": {"core": "a" * 64}},
        {**approval, "test_access": "ALLOWED"},
    ):
        with pytest.raises(ValueError, match="independent approval"):
            _verify_outer_approval(changed, "c" * 64, digest, source_hashes)


def test_rejects_mismatched_forecast_geometry() -> None:
    times = np.full((36, 3), np.datetime64("2024-11-01"))
    with pytest.raises(ValueError, match="identical validation support"):
        _compare_arrays(np.zeros((36, 3)), np.ones((36, 3), bool), times, {"bad": np.zeros((35, 3, 5))})


def test_incomplete_candidate_manifest_fails_before_prediction_access() -> None:
    manifest = {
        "schema_version": "1.0",
        "study_id": "aeon3_geb_2024_hourly_sv_v1",
        "phase": "validation_comparison_only",
        "status": "PROPOSED_FOR_INDEPENDENT_COMPARISON_REVIEW",
        "calibration_access": "PROHIBITED",
        "test_access": "PROHIBITED",
        "assessment_partition": "validation",
        "validation_source_date_bounds_inclusive": ["2024-10-08", "2024-11-30"],
        "validation_issued_rows": 1219,
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "cohort_sha256": "a" * 64,
        "validation_row_sha256": "b" * 64,
        "source_archive_sha256": "c" * 64,
        "sources": {"core": {}},
    }
    with pytest.raises(ValueError, match="finite validation contract"):
        _validate_manifest(manifest)


def test_all_source_groups_need_exact_reviewed_report_and_slot_hashes(tmp_path: Path) -> None:
    config = {
        "schema_version": "1.0",
        "study_id": "aeon3_geb_2024_hourly_sv_v1",
        "phase": "validation_comparison_only",
        "status": "PROPOSED_FOR_INDEPENDENT_COMPARISON_REVIEW",
        "calibration_access": "PROHIBITED",
        "test_access": "PROHIBITED",
        "assessment_partition": "validation",
        "validation_source_date_bounds_inclusive": ["2024-10-08", "2024-11-30"],
        "validation_issued_rows": 1219,
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "cohort_sha256": "a" * 64,
        "validation_row_sha256": "b" * 64,
        "source_archive_sha256": "c" * 64,
        "sources": {},
    }
    for kind, (review_status, report_key, slots) in SOURCE_CONTRACTS.items():
        report = {
            "status": _source_report_status(kind),
            "test_access": "PROHIBITED",
            "validation_row_sha256": "b" * 64,
            "cohort_sha256": "a" * 64,
            "source_archive_sha256": "c" * 64,
            "slots": {slot: {"prediction_sha256": "d" * 64, "slot_sha256": "e" * 64} for slot in slots},
            "prediction_sha256": "d" * 64,
        }
        if kind == "hybrid":
            report["slots"]["raw_only_hgb"] = {"prediction_sha256": "d" * 64}
        report_path = tmp_path / f"{kind}-report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
        review = {
            "status": review_status,
            "reviewer_session": "/root/aeon_reviewer",
            "artifact_sha256": {report_key: report_sha},
            "test_access": "PROHIBITED",
        }
        review_path = tmp_path / f"{kind}-review.json"
        review_path.write_text(json.dumps(review), encoding="utf-8")
        predictions = {
            slot: {
                "path": str(tmp_path / f"{kind}-{slot}.npz"),
                "sha256": "d" * 64,
                **({"slot_report": {"path": str(tmp_path / f"{slot}.json"), "sha256": "e" * 64}} if kind == "forward" else {}),
            }
            for slot in slots
        }
        config["sources"][kind] = {
            "report": {"path": str(report_path), "sha256": report_sha},
            "review": {"path": str(review_path), "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest()},
            "predictions": predictions,
        }
    reports, hashes = _load_sources(config)
    assert set(reports) == set(SOURCE_CONTRACTS)
    assert hashes["core"] == config["sources"]["core"]["report"]["sha256"]
    config["sources"]["forward"]["predictions"]["forward_ema_seed7"]["slot_report"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="slot report differs"):
        _load_sources(config)
