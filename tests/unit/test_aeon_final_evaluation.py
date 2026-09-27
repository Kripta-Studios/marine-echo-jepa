"""Synthetic-only tests for the gated AEON final-evaluation runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training.aeon_final_evaluation import (
    artifact_sha256,
    execute_calibration,
    execute_retrospective_test,
    adapter_composite_sha256,
    _selection,
    _test_forecast_plan,
    _comparison_gates,
    reader_composite_sha256,
)
from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.training import aeon_final_evaluation
from marine_echo.training import aeon_windows
from marine_echo.training import aeon_forecast_adapters
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
        "adapter_composite_sha256": adapter_composite_sha256(),
        "corrected_validation_rescore_sha256": "32cea9f8141fbad220a3e47d9e840039ad23b4f8e893efbcfb90f52944214c46",
        "corrected_validation_outcome_review_sha256": "16bb9ef931f5e2d8f8a3322fa609c5cb1c769f99f5e4a89a5ef519a0fb52f44f",
        "validation_row_sha256": "9ce5ed6d60c4082efd3f342b3ee09ddadc5cf6286be6d6a3e953ab0f6166a99f",
        "models": [
            {"model_id": "baseline", "role": "baseline", "adapter": "SYNTHETIC_FIXTURE",
             "selection_classification": "CORE_CONVENTIONAL_SELECTION",
             "component_artifacts": [{"artifact_id": "baseline-model", "sha256": "3" * 64}]},
            {"model_id": "core_jepa", "role": "candidate", "adapter": "SYNTHETIC_FIXTURE",
             "selection_classification": "CORE_JEPA_SELECTION",
             "component_artifacts": [{"artifact_id": "core-jepa-models", "sha256": "4" * 64}]},
            {"model_id": "candidate", "role": "candidate", "adapter": "SYNTHETIC_FIXTURE",
             "selection_classification": "POST_HOC_DEVELOPMENT_SELECTION",
             "component_artifacts": [{"artifact_id": "post-hoc-model", "sha256": "5" * 64}]},
        ],
    })
    predictions = []
    model_specs = (
        ("baseline", "baseline", "CORE_CONVENTIONAL_SELECTION", "3" * 64, 0.0),
        ("core_jepa", "candidate", "CORE_JEPA_SELECTION", "4" * 64, 0.1),
        ("candidate", "candidate", "POST_HOC_DEVELOPMENT_SELECTION", "5" * 64, 0.2),
    )
    for model, role, classification, component_sha, shift in model_specs:
        if partition == "calibration":
            artifact = tmp_path / f"{partition}-{model}.npz"
            q = np.broadcast_to(np.array([-2., -1., 0., 1., 2.]) + shift,
                                (len(rows), 3, 5)).copy()
            np.savez_compressed(
                artifact, row_ids=np.asarray([r.row_id for r in rows]), quantiles_db=q
            )
            predictions.append({"model_id": model, "artifact": artifact.name,
                                "artifact_sha256": artifact_sha256(artifact)})
        else:
            predictions.append({
                "model_id": model, "role": role,
                "selection_classification": classification,
                "adapter": "SYNTHETIC_FIXTURE",
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
        "schema_version": "1.0", "status": (
            "FROZEN_AEON_PARTITION_FORECASTS" if partition == "calibration"
            else "FROZEN_AEON_TEST_FORECAST_ADAPTER_PLAN"
        ),
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
    digest = _write_json(path, {
        "schema_version": "1.0", "status": "FROZEN_AEON_RETROSPECTIVE_TEST_PREACCESS",
        "study_id": "aeon3_geb_2024_hourly_sv_v1",
        "source_archive_sha256": "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde",
        "metadata_candidate_report_sha256": candidate_sha,
        "metadata_candidate_review_sha256": "9d9c6ca4580d6a4ead4117a00a1a872710bf6e323eb17d491837e24c1011bebe",
        "selection_freeze_sha256": selection_sha,
        "calibration_artifact_sha256": calibration_sha, "config_sha256": config_sha,
        "forecast_manifest_sha256": forecast_manifest_sha,
        "runner_code_sha256": artifact_sha256(Path(aeon_final_evaluation.__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_composite_sha256": reader_composite_sha256(),
        "adapter_composite_sha256": adapter_composite_sha256(),
        "issued_row_rule": "EXACT_24_PRIOR_INTERVAL_IDS_OBSERVED_38KHZ",
        "primary_metric": "RAW_FIVE_QUANTILE_ELIGIBLE_TARGET_DATE_PINBALL",
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "test_access": "PROHIBITED_PENDING_INDEPENDENT_APPROVAL",
    })
    return path, digest


def _bind_candidate(manifest: Path, candidate_sha: str) -> str:
    value = json.loads(manifest.read_text())
    value["candidate_contract_sha256"] = candidate_sha
    return _write_json(manifest, value)


def _fixture_forecaster(rows: list[AeonHourlyWindow]) -> dict[str, np.ndarray]:
    base = np.broadcast_to(np.array([-2., -1., 0., 1., 2.]), (len(rows), 3, 5)).copy()
    return {"baseline": base, "core_jepa": base * 0.75, "candidate": base * 0.5}


def _review(tmp_path: Path, partition: str, bindings: dict[str, str]) -> tuple[Path, str]:
    path = tmp_path / f"{partition}-review.json"
    digest = _write_json(path, {
        "status": ("APPROVED_AEON_CALIBRATION_RUNNER_FIXTURE" if partition == "calibration"
                   else "APPROVED_AEON_RETROSPECTIVE_TEST_RUNNER_FIXTURE"),
        "data_kind": "SYNTHETIC_FIXTURE", "partition": partition,
        "runner_code_sha256": artifact_sha256(Path(aeon_final_evaluation.__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_composite_sha256": reader_composite_sha256(),
        "adapter_composite_sha256": adapter_composite_sha256(),
        **bindings,
    })
    return path, digest


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
        "reader_review_sha256": "8" * 64,
    })
    result = execute_calibration(
        archive=tmp_path, reader_review_path=review, reader_review_sha256="8" * 64,
        runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
        selection_freeze_path=selection, forecast_manifest_path=manifest,
        output=tmp_path / "calibration-output", fixture_reader=reader,
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
        "reader_review_sha256": "8" * 64,
    })
    with pytest.raises(ValueError, match="review bindings"):
        execute_calibration(
            archive=tmp_path, reader_review_path=review, reader_review_sha256="8" * 64,
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
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
        "reader_review_sha256": "8" * 64,
    })
    doc = json.loads(review.read_text())
    doc["runner_code_sha256"] = "0" * 64
    review_sha = _write_json(review, doc)
    with pytest.raises(ValueError, match="review bindings"):
        execute_calibration(
            archive=tmp_path, reader_review_path=review, reader_review_sha256="8" * 64,
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


def test_forecasts_must_match_exact_issued_row_order(tmp_path: Path) -> None:
    rows = [_window(i, "calibration", "2024-12-01") for i in range(12 * 24)]
    config, selection, selection_sha, _, _, manifest, manifest_sha = _contract(
        tmp_path, "calibration", rows
    )
    with np.load(tmp_path / "calibration-baseline.npz", allow_pickle=False) as saved:
        ids, q = saved["row_ids"][::-1], saved["quantiles_db"][::-1]
    np.savez_compressed(tmp_path / "calibration-baseline.npz", row_ids=ids, quantiles_db=q)
    doc = json.loads(manifest.read_text())
    doc["models"][0]["artifact_sha256"] = artifact_sha256(tmp_path / "calibration-baseline.npz")
    manifest_sha = _write_json(manifest, doc)
    review, review_sha = _review(tmp_path, "calibration", {
        "selection_freeze_sha256": selection_sha, "forecast_manifest_sha256": manifest_sha,
        "config_sha256": artifact_sha256(config),
        "reader_review_sha256": "8" * 64,
    })
    with pytest.raises(ValueError, match="row order"):
        execute_calibration(
            archive=tmp_path, reader_review_path=review, reader_review_sha256="8" * 64,
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=_Reader(rows),
        )


def test_test_scoring_requires_candidate_and_calibration_hashes(tmp_path: Path) -> None:
    rows = [_window(i, "test", "2025-01-06") for i in range(20 * 24)]
    config, selection, selection_sha, _, _, manifest, _ = _contract(
        tmp_path, "test", rows
    )
    calibration = tmp_path / "calibration.json"
    calibration_sha = _write_json(calibration, {
        "status": "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING",
        "study_id": "aeon3_geb_2024_hourly_sv_v1", "selection_freeze_sha256": selection_sha,
        "models": {"baseline": {"adjustment_db": [0., 0., 0.]},
                   "core_jepa": {"adjustment_db": [0.05, 0.05, 0.05]},
                   "candidate": {"adjustment_db": [0.1, 0.1, 0.1]}},
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
        "reader_review_sha256": "8" * 64,
    })
    result = execute_retrospective_test(
        archive=tmp_path, reader_review_path=review, reader_review_sha256="8" * 64,
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


def test_test_candidate_contract_rejects_unfrozen_issued_row(tmp_path: Path) -> None:
    rows = [_window(i, "test", "2025-01-06") for i in range(20 * 24)]
    config, selection, selection_sha, _, _, manifest, _ = _contract(
        tmp_path, "test", rows
    )
    calibration = tmp_path / "calibration.json"
    calibration_sha = _write_json(calibration, {
        "status": "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING",
        "study_id": "aeon3_geb_2024_hourly_sv_v1", "selection_freeze_sha256": selection_sha,
        "models": {"baseline": {"adjustment_db": [0., 0., 0.]},
                   "core_jepa": {"adjustment_db": [0., 0., 0.]},
                   "candidate": {"adjustment_db": [0., 0., 0.]}},
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
        "reader_review_sha256": "8" * 64,
    })
    with pytest.raises(ValueError, match="candidate universe"):
        execute_retrospective_test(
            archive=tmp_path, reader_review_path=review, reader_review_sha256="8" * 64,
            runner_review_path=review, runner_review_sha256=review_sha, config_path=config,
            selection_freeze_path=selection, candidate_contract_path=candidate,
            calibration_artifact_path=calibration, pretest_freeze_path=pretest,
            forecast_manifest_path=manifest,
            output=tmp_path / "never", fixture_reader=_Reader(rows),
            fixture_forecaster=_fixture_forecaster,
        )
