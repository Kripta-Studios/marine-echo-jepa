"""Synthetic-only tests for the gated AEON final-evaluation runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest

from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.training import aeon_final_evaluation, aeon_forecast_adapters, aeon_windows
from marine_echo.training.aeon_final_evaluation import (
    _comparison_gates,
    _selection,
    _test_forecast_plan,
    adapter_composite_sha256,
    artifact_sha256,
    execute_calibration,
    execute_retrospective_test,
    reader_composite_sha256,
)
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window(index: int, partition: str, day_zero: str) -> AeonHourlyWindow:
    cutoff = np.datetime64(day_zero, "us") + np.timedelta64(index, "h")
    targets = cutoff + np.asarray([1, 3, 6], dtype="timedelta64[h]")
    return AeonHourlyWindow(
        row_id=hashlib.sha256(f"row:{partition}:{index}".encode()).hexdigest(),
        partition=partition,
        cutoff_source_timestamp=cutoff,
        cutoff_interval_id=10_000 + index,
        context_db=np.full((24, 4), -70.0),
        context_mask=np.ones((24, 4), dtype=bool),
        context_interval_ids=np.arange(9977 + index, 10_001 + index),
        context_source_timestamps=cutoff - np.arange(23, -1, -1).astype("timedelta64[h]"),
        target_interval_ids=10_000 + index + np.asarray([1, 3, 6]),
        target_source_timestamps=targets,
        target_db=np.zeros(3),
        target_mask=np.ones(3, dtype=bool),
        target_qc_status=("OBSERVED_SOURCE_PRODUCT",) * 3,
        source_archive_sha256="a" * 64,
        past_members=("past.csv",),
        target_members=("target.csv",),
    )


class _Reader:
    def __init__(self, rows: list[AeonHourlyWindow]) -> None:
        self.rows = rows
        self.opened = False
        self.opens = 0

    def iter_windows(self):  # type: ignore[no-untyped-def]
        self.opened = True
        self.opens += 1
        yield from self.rows


def _contract(tmp_path: Path, partition: str, rows: list[AeonHourlyWindow]):
    config = tmp_path / "config.json"
    _write_json(config, {
        "schema_version": "1.0", "study_id": "aeon3_geb_2024_hourly_sv_v1",
        "minimum_eligible_dates": {"calibration": 12, "test": 20},
        "minimum_anchors_per_date": 18, "horizons_hours": [1, 3, 6],
        "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "incremental_loss_gate": 0.05, "per_horizon_regression_guard": 0.10,
    })
    selection = tmp_path / "selection.json"
    selection_sha = _write_json(selection, {
        "schema_version": "1.0", "status": "FROZEN_AEON_MODEL_SELECTION",
        "study_id": "aeon3_geb_2024_hourly_sv_v1", "test_access": "PROHIBITED",
        "selection_rule_review_sha256": "dd3dcbdf5e93d42e021abbaf4bf310bec0b40ea52d53411802e00770840127b2",
        "neutral_comparison_sha256": "f4a0f4771868866ff7c330f4033c5ca9da7c6c36cdb19a1758cf2f383146757b",
        "neutral_comparison_outcome_review_sha256": "53a3a164915eea4ab3bc8851c096bbc42821cb05330eb4c2829039df83df3e07",
        "development_selection_record_review_sha256": "527bf68313cac2354ced78d8cf2673153a92e2f2d08845451ee9afaccfab47c8",
        "adapter_composite_sha256": adapter_composite_sha256(),
        "corrected_validation_rescore_sha256": "32cea9f8141fbad220a3e47d9e840039ad23b4f8e893efbcfb90f52944214c46",
        "corrected_validation_outcome_review_sha256": "16bb9ef931f5e2d8f8a3322fa609c5cb1c769f99f5e4a89a5ef519a0fb52f44f",
        "validation_row_sha256": "9ce5ed6d60c4082efd3f342b3ee09ddadc5cf6286be6d6a3e953ab0f6166a99f",
        "models": [
            {"model_id": "baseline", "role": "baseline", "adapter": "SYNTHETIC_FIXTURE",
             "selection_classification": "CORE_CONVENTIONAL_SELECTION",
             "component_artifacts": [{"artifact_id": "baseline-model", "sha256": "3" * 64}],
             "ensemble_weights": [1.0], "adapter_options": {}},
            {"model_id": "core_jepa", "role": "candidate", "adapter": "SYNTHETIC_FIXTURE",
             "selection_classification": "CORE_JEPA_SELECTION",
             "component_artifacts": [{"artifact_id": "core-jepa-models", "sha256": "4" * 64}],
             "ensemble_weights": [1.0], "adapter_options": {}},
            {"model_id": "candidate", "role": "candidate", "adapter": "SYNTHETIC_FIXTURE",
             "selection_classification": "POST_HOC_DEVELOPMENT_SELECTION",
             "component_artifacts": [{"artifact_id": "post-hoc-model", "sha256": "5" * 64}],
             "ensemble_weights": [1.0], "adapter_options": {}},
        ],
    })
    predictions = []
    model_specs = (
        ("baseline", "baseline", "CORE_CONVENTIONAL_SELECTION", "3" * 64),
        ("core_jepa", "candidate", "CORE_JEPA_SELECTION", "4" * 64),
        ("candidate", "candidate", "POST_HOC_DEVELOPMENT_SELECTION", "5" * 64),
    )
    for model, role, classification, component_sha in model_specs:
        predictions.append({
            "model_id": model, "role": role,
            "selection_classification": classification,
            "adapter": "SYNTHETIC_FIXTURE",
            "options": {},
            "ensemble_weights": [1.0],
            "component_artifacts": [{
                "artifact_id": (
                    "baseline-model" if model == "baseline" else
                    "core-jepa-models" if model == "core_jepa" else "post-hoc-model"
                ),
                "path": f"{model}.fixture", "sha256": component_sha,
            }],
        })
    candidate = None
    candidate_sha = None
    manifest = tmp_path / f"{partition}-forecasts.json"
    manifest_sha = _write_json(manifest, {
        "schema_version": "1.0",
        "status": f"FROZEN_AEON_{partition.upper()}_FORECAST_ADAPTER_PLAN",
        "study_id": "aeon3_geb_2024_hourly_sv_v1", "partition": partition,
        "selection_freeze_sha256": selection_sha,
        "candidate_contract_sha256": candidate_sha,
        "adapter_composite_sha256": adapter_composite_sha256(),
        "models": predictions,
    })
    return config, selection, selection_sha, candidate, candidate_sha, manifest, manifest_sha


def _candidate_contract(tmp_path: Path, rows: list[AeonHourlyWindow]) -> tuple[Path, str]:
    candidate = tmp_path / "candidate.json"
    digest = _write_json(candidate, {
        "status": "METADATA_CANDIDATE_UNIVERSE_PENDING_INDEPENDENT_REVIEW",
        "classification": "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED",
        "numeric_test_outcome_access": "PROHIBITED",
        "source_archive_sha256": "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde",
        "partition_start_source_date_inclusive": "2025-01-06",
        "partition_end_source_date_exclusive": "2025-03-01",
        "candidate_cutoff_interval_ids": [r.cutoff_interval_id for r in rows],
        "candidate_rows": [
            {"cutoff_interval_id": row.cutoff_interval_id,
             "cutoff_source_timestamp": str(row.cutoff_source_timestamp),
             "row_id": row.row_id, "numeric_issuance_status": "UNKNOWN",
             "target_scoring_status": "UNKNOWN"} for row in rows
        ],
        "actual_issued_rows": "UNKNOWN_NUMERIC_QC_NOT_OPENED",
        "actual_scored_rows": "UNKNOWN_NUMERIC_QC_NOT_OPENED",
    })
    return candidate, digest


def _pretest(
    tmp_path: Path, candidate_sha: str, selection_sha: str, calibration_sha: str,
    config_sha: str, forecast_manifest_sha: str,
) -> tuple[Path, str]:
    path = tmp_path / "pretest-freeze.json"
    value = {
        "schema_version": "1.0", "status": "FROZEN_AEON_RETROSPECTIVE_TEST_PREACCESS",
        "study_id": "aeon3_geb_2024_hourly_sv_v1",
        "source_archive_sha256": "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde",
        "metadata_candidate_report_sha256": candidate_sha,
        "metadata_candidate_review_sha256": "9d9c6ca4580d6a4ead4117a00a1a872710bf6e323eb17d491837e24c1011bebe",
        "selection_freeze_sha256": selection_sha,
        "calibration_artifact_sha256": calibration_sha, "config_sha256": config_sha,
        "calibration_outcome_review_sha256": "7b61385b29836be53d0e2c5a0b5e3f5db7e2dcb8983e534d17711ed0b44c1428",
        "forecast_manifest_sha256": forecast_manifest_sha,
        "runner_code_sha256": artifact_sha256(Path(aeon_final_evaluation.__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_composite_sha256": reader_composite_sha256(),
        "adapter_composite_sha256": adapter_composite_sha256(),
        "issued_row_rule": "EXACT_24_PRIOR_INTERVAL_IDS_OBSERVED_38KHZ",
        "primary_metric": "RAW_FIVE_QUANTILE_ELIGIBLE_TARGET_DATE_PINBALL",
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "test_access": "PROHIBITED_PENDING_INDEPENDENT_APPROVAL",
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    digest = artifact_sha256(path)
    return path, digest


def _calibration_artifact(
    tmp_path: Path, selection_sha: str, config: Path,
    adjustments: dict[str, list[float]],
) -> tuple[Path, str]:
    classifications = {
        "baseline": "CORE_CONVENTIONAL_SELECTION",
        "core_jepa": "CORE_JEPA_SELECTION",
        "candidate": "POST_HOC_DEVELOPMENT_SELECTION",
    }
    path = tmp_path / "calibration.json"
    value = {
        "status": "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING",
        "study_id": "aeon3_geb_2024_hourly_sv_v1", "partition": "calibration",
        "test_access": "PROHIBITED",
        "source_archive_sha256": "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde",
        "selection_freeze_sha256": selection_sha, "forecast_manifest_sha256": "6" * 64,
        "config_sha256": artifact_sha256(config), "runner_review_sha256": "7" * 64,
        "runner_code_sha256": artifact_sha256(Path(aeon_final_evaluation.__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_composite_sha256": reader_composite_sha256(),
        "adapter_composite_sha256": adapter_composite_sha256(),
        "reader_review_sha256": "8" * 64,
        "issued_row_ids": [hashlib.sha256(b"calibration-row").hexdigest()],
        "models": {
            model_id: {
                "adjustment_db": adjustment,
                "eligible_days_per_horizon": [12, 12, 12],
                "eligible_rows_per_horizon": [216, 216, 216],
                "all_scored_days_per_horizon": [12, 12, 12],
                "eligible_source_dates_by_horizon": [
                    [f"2024-12-{day:02d}" for day in range(1, 13)] for _ in range(3)
                ],
                "selection_classification": classifications[model_id],
                "raw_interval_metrics": {}, "widened_interval_metrics": {},
            }
            for model_id, adjustment in adjustments.items()
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    digest = artifact_sha256(path)
    return path, digest


def _bind_candidate(manifest: Path, candidate_sha: str) -> str:
    value = json.loads(manifest.read_text())
    value["candidate_contract_sha256"] = candidate_sha
    return _write_json(manifest, value)


def _fixture_forecaster(rows: list[AeonHourlyWindow]) -> dict[str, np.ndarray]:
    base = np.broadcast_to(np.array([-2., -1., 0., 1., 2.]), (len(rows), 3, 5)).copy()
    return {"baseline": base, "core_jepa": base * 0.75, "candidate": base * 0.5}


def _review(tmp_path: Path, partition: str, bindings: dict[str, str]) -> tuple[Path, str]:
    reader_review = tmp_path / f"{partition}-reader-review.json"
    _write_json(reader_review, {
        "partition": partition,
        "reviewer_session": "/root/aeon_reviewer",
        "partition_access": (
            "CALIBRATION_FIXTURE_ONLY" if partition == "calibration"
            else "RETROSPECTIVE_TEST_FIXTURE_ONLY"
        ),
    })
    path = tmp_path / f"{partition}-review.json"
    digest = _write_json(path, {
        "status": ("APPROVED_AEON_CALIBRATION_RUNNER_FIXTURE" if partition == "calibration"
                   else "APPROVED_AEON_RETROSPECTIVE_TEST_RUNNER_FIXTURE"),
        "data_kind": "SYNTHETIC_FIXTURE", "partition": partition,
        "reviewer_session": "/root/aeon_reviewer",
        "partition_access": (
            "CALIBRATION_FIXTURE_ONLY" if partition == "calibration"
            else "RETROSPECTIVE_TEST_FIXTURE_ONLY"
        ),
        "runner_code_sha256": artifact_sha256(Path(aeon_final_evaluation.__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_composite_sha256": reader_composite_sha256(),
        "adapter_composite_sha256": adapter_composite_sha256(),
        **bindings,
    })
    return path, digest


def _reader_review(tmp_path: Path, partition: str) -> tuple[Path, str]:
    path = tmp_path / f"{partition}-reader-review.json"
    if not path.exists():
        _write_json(path, {
            "partition": partition,
            "reviewer_session": "/root/aeon_reviewer",
            "partition_access": (
                "CALIBRATION_FIXTURE_ONLY" if partition == "calibration"
                else "RETROSPECTIVE_TEST_FIXTURE_ONLY"
            ),
        })
    return path, artifact_sha256(path)


def test_calibration_is_hash_gated_and_requires_twelve_eligible_dates(tmp_path: Path) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, manifest_sha = _contract(
        tmp_path, "calibration", rows
    )
    reader = _Reader(rows)
    review, review_sha = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config),
        "reader_review_sha256": _reader_review(tmp_path, "calibration")[1],
    })
    result = execute_calibration(
        archive=tmp_path, reader_review_path=_reader_review(tmp_path, "calibration")[0],
        reader_review_sha256=_reader_review(tmp_path, "calibration")[1],
        runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
        selection_freeze_path=selection, forecast_manifest_path=manifest,
        output=tmp_path / "calibration-output", fixture_reader=reader,
        fixture_forecaster=_fixture_forecaster,
    )
    assert result["status"] == "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING"
    assert result["models"]["baseline"]["eligible_days_per_horizon"] == [12, 12, 12]
    assert reader.opened


def test_gate_failure_happens_before_any_partition_rows_are_read(tmp_path: Path) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, manifest_sha = _contract(
        tmp_path, "calibration", rows
    )
    reader = _Reader(rows)
    review, review_sha = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": "0" * 64,
        "reader_review_sha256": _reader_review(tmp_path, "calibration")[1],
    })
    with pytest.raises(ValueError, match="review bindings"):
        execute_calibration(
            archive=tmp_path, reader_review_path=_reader_review(tmp_path, "calibration")[0],
            reader_review_sha256=_reader_review(tmp_path, "calibration")[1],
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=reader,
        )
    assert not reader.opened


def test_calibration_component_substitution_precedes_reader(tmp_path: Path) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, _ = _contract(
        tmp_path, "calibration", rows
    )
    changed = json.loads(manifest.read_text())
    changed["models"][0]["component_artifacts"][0]["sha256"] = "f" * 64
    manifest_sha = _write_json(manifest, changed)
    reader = _Reader(rows)
    review, review_sha = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config),
        "reader_review_sha256": _reader_review(tmp_path, "calibration")[1],
    })
    with pytest.raises(ValueError, match="adapter plan differs"):
        execute_calibration(
            archive=tmp_path, reader_review_path=_reader_review(tmp_path, "calibration")[0],
            reader_review_sha256=_reader_review(tmp_path, "calibration")[1],
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=reader,
        )
    assert not reader.opened


@pytest.mark.parametrize("bad_options", [
    {"family": "ema_jepa", "device": "cpu"},
    {"family": "direct", "device": "tpu"},
])
def test_adapter_option_substitution_precedes_calibration_reader(
    tmp_path: Path, bad_options: dict[str, str],
) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, _ = _contract(
        tmp_path, "calibration", rows
    )
    changed = json.loads(manifest.read_text())
    changed["models"][0]["options"] = bad_options
    manifest_sha = _write_json(manifest, changed)
    reader = _Reader(rows)
    review, review_sha = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config),
        "reader_review_sha256": _reader_review(tmp_path, "calibration")[1],
    })
    with pytest.raises(ValueError, match="adapter plan differs"):
        execute_calibration(
            archive=tmp_path, reader_review_path=_reader_review(tmp_path, "calibration")[0],
            reader_review_sha256=_reader_review(tmp_path, "calibration")[1],
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=reader,
            fixture_forecaster=_fixture_forecaster,
        )
    assert not reader.opened


def test_missing_real_component_bytes_precede_reader_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, manifest_sha = _contract(
        tmp_path, "calibration", rows
    )
    reader = _Reader(rows)
    review, review_sha = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config),
        "reader_review_sha256": _reader_review(tmp_path, "calibration")[1],
    })
    real_entry = {
        "model_id": "core_direct_equal_three_seed_ensemble",
        "adapter": "core_neural_ensemble",
        "options": {"family": "direct", "device": "cpu"},
        "component_artifacts": [{
            "artifact_id": "missing-model", "path": "missing.joblib", "sha256": "a" * 64,
        }],
    }
    original = aeon_final_evaluation._preflight_forecast_plan

    def missing_preflight(*_args: object, **_kwargs: object) -> None:
        original([real_entry], tmp_path, fixture=False)

    monkeypatch.setattr(aeon_final_evaluation, "_preflight_forecast_plan", missing_preflight)
    with pytest.raises(FileNotFoundError):
        execute_calibration(
            archive=tmp_path, reader_review_path=_reader_review(tmp_path, "calibration")[0],
            reader_review_sha256=_reader_review(tmp_path, "calibration")[1],
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=reader,
            fixture_forecaster=_fixture_forecaster,
        )
    assert not reader.opened


def test_real_review_rejects_synthetic_selection_before_reader(tmp_path: Path) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, manifest_sha = _contract(
        tmp_path, "calibration", rows
    )
    reader = _Reader(rows)
    review, _ = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config),
        "reader_review_sha256": _reader_review(tmp_path, "calibration")[1],
    })
    value = json.loads(review.read_text())
    value.update(status="APPROVED_AEON_CALIBRATION_ACCESS", data_kind="REAL",
                 partition_access="CALIBRATION_NUMERIC_ACCESS_APPROVED")
    review_sha = _write_json(review, value)
    with pytest.raises(ValueError, match="selection fixture mode"):
        execute_calibration(
            archive=tmp_path, reader_review_path=_reader_review(tmp_path, "calibration")[0],
            reader_review_sha256=_reader_review(tmp_path, "calibration")[1],
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=reader,
        )
    assert not reader.opened


@pytest.mark.parametrize(("field", "value"), [
    ("reviewer_session", "/reviewer/arbitrary"),
    ("partition_access", "RETROSPECTIVE_TEST_FIXTURE_ONLY"),
])
def test_reader_review_requires_trusted_partition_approval(
    tmp_path: Path, field: str, value: str,
) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, manifest_sha = _contract(
        tmp_path, "calibration", rows
    )
    reader = _Reader(rows)
    review, _ = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config),
        "reader_review_sha256": _reader_review(tmp_path, "calibration")[1],
    })
    reader_path, _ = _reader_review(tmp_path, "calibration")
    reader_value = json.loads(reader_path.read_text())
    reader_value[field] = value
    reader_sha = _write_json(reader_path, reader_value)
    review_value = json.loads(review.read_text())
    review_value["reader_review_sha256"] = reader_sha
    review_sha = _write_json(review, review_value)
    with pytest.raises(ValueError, match="trusted partition-specific approval"):
        execute_calibration(
            archive=tmp_path, reader_review_path=reader_path,
            reader_review_sha256=reader_sha, runner_review_path=review,
            runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=reader,
        )
    assert not reader.opened


def test_code_hash_mutation_is_rejected_before_partition_read(tmp_path: Path) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, manifest_sha = _contract(
        tmp_path, "calibration", rows
    )
    reader = _Reader(rows)
    review, _ = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config),
        "reader_review_sha256": _reader_review(tmp_path, "calibration")[1],
    })
    doc = json.loads(review.read_text())
    doc["runner_code_sha256"] = "0" * 64
    review_sha = _write_json(review, doc)
    with pytest.raises(ValueError, match="review bindings"):
        execute_calibration(
            archive=tmp_path, reader_review_path=_reader_review(tmp_path, "calibration")[0],
            reader_review_sha256=_reader_review(tmp_path, "calibration")[1],
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=reader,
        )
    assert not reader.opened


def test_reader_dependency_mutation_changes_composite_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = reader_composite_sha256()
    mutated = tmp_path / "aeon_windows.py"
    mutated.write_bytes(Path(aeon_windows.__file__).read_bytes() + b"\n# mutation\n")
    monkeypatch.setattr(aeon_windows, "__file__", str(mutated))
    assert reader_composite_sha256() != original


def test_adapter_dependency_mutation_changes_composite_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = adapter_composite_sha256()
    dependency = aeon_forecast_adapters.aeon_ssl
    mutated = tmp_path / "aeon_ssl.py"
    mutated.write_bytes(Path(dependency.__file__).read_bytes() + b"\n# mutation\n")
    monkeypatch.setattr(dependency, "__file__", str(mutated))
    assert adapter_composite_sha256() != original


def test_adapter_composite_is_independent_of_installed_distribution_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = adapter_composite_sha256()
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "arbitrary-environment")
    assert adapter_composite_sha256() == original


def test_selection_categories_and_component_set_are_exact(tmp_path: Path) -> None:
    rows = [_window(i, "test", "2025-01-06") for i in range(20 * 24)]
    _, selection_path, selection_sha, _, _, plan, _ = _contract(tmp_path, "test", rows)
    candidate, candidate_sha = _candidate_contract(tmp_path, rows)
    del candidate
    _bind_candidate(plan, candidate_sha)
    selection, _ = _selection(selection_path)
    changed = json.loads(plan.read_text())
    changed["models"][1]["component_artifacts"][0]["sha256"] = "f" * 64
    _write_json(plan, changed)
    with pytest.raises(ValueError, match="differs from the frozen selection"):
        _test_forecast_plan(plan, selection, selection_sha, candidate_sha)

    changed["models"][1]["component_artifacts"][0]["sha256"] = "4" * 64
    changed["models"][1]["options"] = {"family": "direct", "device": "tpu"}
    _write_json(plan, changed)
    with pytest.raises(ValueError, match="differs from the frozen selection"):
        _test_forecast_plan(plan, selection, selection_sha, candidate_sha)

    invalid = json.loads(selection_path.read_text())
    invalid["models"][1]["selection_classification"] = "POST_HOC_DEVELOPMENT_SELECTION"
    _write_json(selection_path, invalid)
    with pytest.raises(ValueError, match="at least one core JEPA"):
        _selection(selection_path)

    unsupported = selection.copy()
    unsupported["models"] = [dict(model) for model in selection["models"]]
    unsupported["models"][1]["adapter"] = "hybrid_raw_latent_hgb_ensemble"
    _write_json(selection_path, unsupported)
    with pytest.raises(ValueError, match="hybrid selection is unsupported"):
        _selection(selection_path)


def test_selection_binds_reviewed_components_and_equal_weights(tmp_path: Path) -> None:
    rows = [_window(i, "test", "2025-01-06") for i in range(20 * 24)]
    _, selection_path, _, _, _, _, _ = _contract(tmp_path, "test", rows)
    selection = json.loads(selection_path.read_text())
    thirds = [1 / 3, 1 / 3, 1 / 3]
    selection["models"] = [
        {
            "model_id": "core_direct_equal_three_seed_ensemble",
            "role": "baseline",
            "selection_classification": "CORE_CONVENTIONAL_SELECTION",
            "adapter": "core_neural_ensemble",
            "component_artifacts": [
                {"artifact_id": "direct_seed7", "sha256": "a616f9160f359e54a6da5d87dc3b7a46edfee46e449e2944af7a69be8ae25c95"},
                {"artifact_id": "direct_seed13", "sha256": "eb6ca62f4c68024b43ea8b2102a0754d646623fd91cbf1ad31fb72218faf48dc"},
                {"artifact_id": "direct_seed23", "sha256": "ee3ae09ebf05bc41ec762e946a1d0046be33fff4ea47472a992893a294054761"},
            ],
            "ensemble_weights": thirds,
            "adapter_options": {"family": "direct", "device": "cpu"},
        },
        {
            "model_id": "core_ema_equal_three_seed_ensemble",
            "role": "candidate",
            "selection_classification": "CORE_JEPA_SELECTION",
            "adapter": "core_neural_ensemble",
            "component_artifacts": [
                {"artifact_id": "ema_jepa_seed7", "sha256": "f1fc41860e21cb2c47868488b051c68cd951487d717fc0fe033e63b8dfcc664c"},
                {"artifact_id": "ema_jepa_seed13", "sha256": "59cb3d1e6d79ca2d9807f24c5e816d0e5c89fad0e274e703d6e9da5b44024633"},
                {"artifact_id": "ema_jepa_seed23", "sha256": "4a7c63931a31a48f60323b944110df9292856f39cb822ce0b891371a10a16a15"},
            ],
            "ensemble_weights": thirds,
            "adapter_options": {"family": "ema_jepa", "device": "cpu"},
        },
        {
            "model_id": "post_hoc_lightgbm",
            "role": "candidate",
            "selection_classification": "POST_HOC_DEVELOPMENT_SELECTION",
            "adapter": "lightgbm",
            "component_artifacts": [
                {"artifact_id": "lightgbm_model", "sha256": "164389731cd5d2e0694d1fb5228196196e7701fc73269e79ef84d85520fb168b"},
                {"artifact_id": "lightgbm_recipe", "sha256": "3feee4089ce790c66adb189ff82ad4e006c350822a23aaa494e86afaef884eea"},
            ],
            "ensemble_weights": [1.0],
            "adapter_options": {"recipe_sha256": "3feee4089ce790c66adb189ff82ad4e006c350822a23aaa494e86afaef884eea"},
        },
    ]
    _write_json(selection_path, selection)
    _selection(selection_path)

    selection["models"][0]["adapter_options"]["family"] = "ema_jepa"
    _write_json(selection_path, selection)
    with pytest.raises(ValueError, match="reviewed model family"):
        _selection(selection_path)

    selection["models"][0]["adapter_options"] = {"family": "direct", "device": "tpu"}
    _write_json(selection_path, selection)
    with pytest.raises(ValueError, match="explicit CPU or CUDA"):
        _selection(selection_path)

    selection["models"][0]["adapter_options"] = {"family": "direct", "device": "cpu"}
    selection["models"][1]["ensemble_weights"] = [0.34, 0.33, 0.33]
    _write_json(selection_path, selection)
    with pytest.raises(ValueError, match="ensemble weights"):
        _selection(selection_path)

    selection["models"][1]["ensemble_weights"] = thirds
    selection["models"][0]["component_artifacts"][0]["sha256"] = "f" * 64
    _write_json(selection_path, selection)
    with pytest.raises(ValueError, match="reviewed selection rule"):
        _selection(selection_path)


def test_incremental_gate_is_separate_from_full_horizon_promotion() -> None:
    result = _comparison_gates(
        baseline_primary=1.0, candidate_primary=0.8,
        baseline_horizons=np.asarray([1.0, 1.0, 1.0]),
        candidate_horizons=np.asarray([0.5, 0.5, 1.11]),
        paired_interval=[-0.3, -0.01],
    )
    assert result["passes_incremental_loss_gate"] is True
    assert result["passes_per_horizon_ten_percent_regression_guard"] is False
    assert result["passes_full_unnarrowed_promotion_rule"] is False


def test_calibration_adapter_output_shape_is_checked(tmp_path: Path) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, manifest_sha = _contract(
        tmp_path, "calibration", rows
    )
    review, review_sha = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha, "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config),
        "reader_review_sha256": _reader_review(tmp_path, "calibration")[1],
    })
    def malformed(issued: list[AeonHourlyWindow]) -> dict[str, np.ndarray]:
        result = _fixture_forecaster(issued)
        result["baseline"] = result["baseline"][:-1]
        return result

    with pytest.raises(ValueError, match="invalid quantiles"):
        execute_calibration(
            archive=tmp_path, reader_review_path=_reader_review(tmp_path, "calibration")[0],
            reader_review_sha256=_reader_review(tmp_path, "calibration")[1],
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=_Reader(rows),
            fixture_forecaster=malformed,
        )


def test_test_scoring_requires_candidate_and_calibration_hashes(tmp_path: Path) -> None:
    rows = [_window(i, "test", "2025-01-06") for i in range(20 * 24)]
    config, selection, selection_sha, _, _, manifest, _ = _contract(
        tmp_path, "test", rows
    )
    calibration, calibration_sha = _calibration_artifact(tmp_path, selection_sha, config, {
        "baseline": [0., 0., 0.], "core_jepa": [0.05, 0.05, 0.05],
        "candidate": [0.1, 0.1, 0.1],
    })
    candidate, candidate_sha = _candidate_contract(tmp_path, rows)
    manifest_sha = _bind_candidate(manifest, candidate_sha)
    pretest, pretest_sha = _pretest(
        tmp_path, candidate_sha, selection_sha, calibration_sha, artifact_sha256(config),
        manifest_sha,
    )
    review, review_sha = _review(tmp_path, "test", {
        "selection_freeze_sha256": selection_sha, "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config), "pretest_freeze_sha256": pretest_sha,
        "calibration_artifact_sha256": calibration_sha,
        "reader_review_sha256": _reader_review(tmp_path, "test")[1],
    })
    result = execute_retrospective_test(
        archive=tmp_path, reader_review_path=_reader_review(tmp_path, "test")[0],
        reader_review_sha256=_reader_review(tmp_path, "test")[1],
        runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
        selection_freeze_path=selection, candidate_contract_path=candidate,
        calibration_artifact_path=calibration, pretest_freeze_path=pretest,
        forecast_manifest_path=manifest,
        output=tmp_path / "test-output", fixture_reader=(reader := _Reader(rows)),
        fixture_forecaster=_fixture_forecaster,
    )
    assert result["status"] == "COMPLETED_AEON_RETROSPECTIVE_TEST_SCORING"
    assert result["classification"] == "RETROSPECTIVE_EVALUATION_NOT_SEALED"
    assert len(result["comparisons"]["candidate"]["bootstrap_candidate_minus_baseline_db"]) == 2000
    assert reader.opens == 1
    assert result["models"]["baseline"]["selection_classification"] == "CORE_CONVENTIONAL_SELECTION"
    assert result["models"]["candidate"]["selection_classification"] == "POST_HOC_DEVELOPMENT_SELECTION"
    comparison = result["comparisons"]["candidate"]
    assert comparison["passes_prespecified_five_percent_point_improvement"] is True
    assert comparison["passes_paired_95_percent_interval_strictly_favoring_candidate"] is True
    assert comparison["passes_per_horizon_ten_percent_regression_guard"] is True
    assert comparison["passes_incremental_loss_gate"] is True
    assert comparison["passes_full_unnarrowed_promotion_rule"] is True
    with np.load(tmp_path / "test-output" / "baseline-forecast.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"row_ids", "quantiles_db"}


def test_existing_test_output_rejects_before_reader(tmp_path: Path) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    reader = _Reader([_window(0, "test", "2025-01-06")])
    missing = tmp_path / "not-opened.json"
    with pytest.raises(FileExistsError, match="output already exists"):
        execute_retrospective_test(
            archive=tmp_path, reader_review_path=missing, reader_review_sha256="0" * 64,
            runner_review_path=missing, runner_review_sha256="0" * 64,
            config_path=missing, selection_freeze_path=missing,
            candidate_contract_path=missing, calibration_artifact_path=missing,
            pretest_freeze_path=missing, forecast_manifest_path=missing,
            output=output, fixture_reader=reader, fixture_forecaster=_fixture_forecaster,
        )
    assert not reader.opened


@pytest.mark.parametrize("mutation", ["missing_model", "invalid_adjustment"])
def test_malformed_calibration_rejects_before_test_reader(
    tmp_path: Path, mutation: str,
) -> None:
    rows = [_window(i, "test", "2025-01-06") for i in range(20 * 24)]
    config, selection, selection_sha, _, _, manifest, _ = _contract(tmp_path, "test", rows)
    calibration, _ = _calibration_artifact(tmp_path, selection_sha, config, {
        "baseline": [0., 0., 0.], "core_jepa": [0., 0., 0.],
        "candidate": [0., 0., 0.],
    })
    value = json.loads(calibration.read_text())
    if mutation == "missing_model":
        del value["models"]["core_jepa"]
    else:
        value["models"]["core_jepa"]["adjustment_db"] = [0., -0.1, 0.]
    _write_json(calibration, value)
    reader = _Reader(rows)
    missing = tmp_path / "not-opened.json"
    with pytest.raises(ValueError, match="calibration"):
        execute_retrospective_test(
            archive=tmp_path, reader_review_path=missing, reader_review_sha256="0" * 64,
            runner_review_path=missing, runner_review_sha256="0" * 64,
            config_path=config, selection_freeze_path=selection,
            candidate_contract_path=missing, calibration_artifact_path=calibration,
            pretest_freeze_path=missing, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=reader,
            fixture_forecaster=_fixture_forecaster,
        )
    assert not reader.opened


def test_test_candidate_contract_rejects_unfrozen_issued_row(tmp_path: Path) -> None:
    rows = [_window(i, "test", "2025-01-06") for i in range(20 * 24)]
    config, selection, selection_sha, _, _, manifest, _ = _contract(
        tmp_path, "test", rows
    )
    calibration, calibration_sha = _calibration_artifact(tmp_path, selection_sha, config, {
        "baseline": [0., 0., 0.], "core_jepa": [0., 0., 0.],
        "candidate": [0., 0., 0.],
    })
    candidate, _ = _candidate_contract(tmp_path, rows)
    doc = json.loads(candidate.read_text())
    doc["candidate_cutoff_interval_ids"] = doc["candidate_cutoff_interval_ids"][:-1]
    doc["candidate_rows"] = doc["candidate_rows"][:-1]
    candidate_sha = _write_json(candidate, doc)
    manifest_sha = _bind_candidate(manifest, candidate_sha)
    pretest, pretest_sha = _pretest(
        tmp_path, candidate_sha, selection_sha, calibration_sha, artifact_sha256(config),
        manifest_sha,
    )
    review, review_sha = _review(tmp_path, "test", {
        "selection_freeze_sha256": selection_sha, "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config), "pretest_freeze_sha256": pretest_sha,
        "calibration_artifact_sha256": calibration_sha,
        "reader_review_sha256": _reader_review(tmp_path, "test")[1],
    })
    with pytest.raises(ValueError, match="candidate universe"):
        execute_retrospective_test(
            archive=tmp_path, reader_review_path=_reader_review(tmp_path, "test")[0],
            reader_review_sha256=_reader_review(tmp_path, "test")[1],
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, candidate_contract_path=candidate,
            calibration_artifact_path=calibration, pretest_freeze_path=pretest,
            forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=_Reader(rows),
            fixture_forecaster=_fixture_forecaster,
        )


def test_test_candidate_matching_includes_exact_row_id(tmp_path: Path) -> None:
    rows = [_window(i, "test", "2025-01-06") for i in range(20 * 24)]
    config, selection, selection_sha, _, _, manifest, _ = _contract(tmp_path, "test", rows)
    calibration, calibration_sha = _calibration_artifact(tmp_path, selection_sha, config, {
        "baseline": [0., 0., 0.], "core_jepa": [0., 0., 0.],
        "candidate": [0., 0., 0.],
    })
    candidate, _ = _candidate_contract(tmp_path, rows)
    candidate_doc = json.loads(candidate.read_text())
    candidate_doc["candidate_rows"][0]["row_id"] = "f" * 64
    candidate_sha = _write_json(candidate, candidate_doc)
    manifest_sha = _bind_candidate(manifest, candidate_sha)
    pretest, pretest_sha = _pretest(
        tmp_path, candidate_sha, selection_sha, calibration_sha, artifact_sha256(config),
        manifest_sha,
    )
    review, review_sha = _review(tmp_path, "test", {
        "selection_freeze_sha256": selection_sha, "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config), "pretest_freeze_sha256": pretest_sha,
        "calibration_artifact_sha256": calibration_sha,
        "reader_review_sha256": _reader_review(tmp_path, "test")[1],
    })
    reader = _Reader(rows)
    with pytest.raises(ValueError, match="outside the metadata-only candidate universe"):
        execute_retrospective_test(
            archive=tmp_path, reader_review_path=_reader_review(tmp_path, "test")[0],
            reader_review_sha256=_reader_review(tmp_path, "test")[1],
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, candidate_contract_path=candidate,
            calibration_artifact_path=calibration, pretest_freeze_path=pretest,
            forecast_manifest_path=manifest, output=tmp_path / "never",
            fixture_reader=reader, fixture_forecaster=_fixture_forecaster,
        )
    assert reader.opens == 1
