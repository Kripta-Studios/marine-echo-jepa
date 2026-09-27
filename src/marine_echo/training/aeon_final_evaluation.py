"""Hash-gated AEON interval calibration and retrospective TEST scoring.

This runner consumes forecast-only artifacts produced by separately reviewed
model adapters. It never selects a model and it validates every freeze before
constructing the real CAL/TEST reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from marine_echo.evaluation.aeon import (
    apply_interval_widening,
    calibrate_interval_widening,
    daily_pinball,
    paired_48h_bootstrap,
)
from marine_echo.training.aeon_evaluation_reader import AeonEvaluationReader
from marine_echo.training.aeon_windows import AeonHourlyWindow


_STUDY = "aeon3_geb_2024_hourly_sv_v1"
_SOURCE_SHA256 = "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde"
_VALIDATION_ROW_SHA256 = "9ce5ed6d60c4082efd3f342b3ee09ddadc5cf6286be6d6a3e953ab0f6166a99f"
_CORRECTED_RESCORE_SHA256 = "32cea9f8141fbad220a3e47d9e840039ad23b4f8e893efbcfb90f52944214c46"
_RESCORE_OUTCOME_REVIEW_SHA256 = "16bb9ef931f5e2d8f8a3322fa609c5cb1c769f99f5e4a89a5ef519a0fb52f44f"
_HEX = set("0123456789abcdef")


class _WindowReader(Protocol):
    def iter_windows(self) -> Any: ...


def artifact_sha256(path: Path) -> str:
    """Return the byte-level SHA-256 used by every runner gate."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _json(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    path = path.resolve(strict=True)
    actual = artifact_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"AEON artifact digest differs: {path.name}.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"AEON JSON contract is not an object: {path.name}.")
    return value, actual


def _config(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _json(path)
    expected = {
        "schema_version": "1.0",
        "study_id": _STUDY,
        "minimum_eligible_dates": {"calibration": 12, "test": 20},
        "minimum_anchors_per_date": 18,
        "horizons_hours": [1, 3, 6],
        "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "incremental_loss_gate": 0.05,
        "per_horizon_regression_guard": 0.10,
    }
    if value != expected:
        raise ValueError("AEON final-evaluation config differs from the frozen protocol.")
    return value, digest


def _selection(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _json(path)
    models = value.get("models")
    if (
        set(value) != {
            "schema_version", "status", "study_id", "test_access",
            "corrected_validation_rescore_sha256",
            "corrected_validation_outcome_review_sha256", "validation_row_sha256", "models",
        }
        or
        value.get("schema_version") != "1.0"
        or value.get("status") != "FROZEN_AEON_MODEL_SELECTION"
        or value.get("study_id") != _STUDY
        or value.get("test_access") != "PROHIBITED"
        or value.get("validation_row_sha256") != _VALIDATION_ROW_SHA256
        or value.get("corrected_validation_rescore_sha256") != _CORRECTED_RESCORE_SHA256
        or value.get("corrected_validation_outcome_review_sha256") != _RESCORE_OUTCOME_REVIEW_SHA256
        or not isinstance(models, list)
        or len(models) < 2
    ):
        raise ValueError("AEON model-selection freeze is incomplete or unreviewed.")
    ids: set[str] = set()
    roles: list[str] = []
    for model in models:
        if (
            not isinstance(model, dict)
            or set(model) != {"model_id", "role", "checkpoint_sha256", "predictor_code_sha256"}
            or not isinstance(model.get("model_id"), str)
            or not model["model_id"]
            or model["model_id"] in ids
            or model.get("role") not in ("baseline", "candidate")
            or not _digest(model.get("checkpoint_sha256"))
            or not _digest(model.get("predictor_code_sha256"))
        ):
            raise ValueError("AEON selected-model identity or artifact binding is invalid.")
        ids.add(model["model_id"])
        roles.append(model["role"])
    if roles.count("baseline") != 1 or "candidate" not in roles:
        raise ValueError("AEON selection needs one baseline and at least one candidate.")
    return value, digest


def _safe_artifact(directory: Path, name: object, digest: object) -> Path:
    if not isinstance(name, str) or not _digest(digest):
        raise ValueError("AEON forecast artifact reference is malformed.")
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in (".", ".."):
        raise ValueError("AEON forecast artifact escapes its manifest directory.")
    path = (directory / relative).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()) or artifact_sha256(path) != digest:
        raise ValueError("AEON forecast artifact digest or location differs.")
    return path


def _forecast_manifest(
    path: Path,
    partition: str,
    selection: dict[str, Any],
    selection_sha256: str,
    candidate_sha256: str | None,
) -> tuple[list[tuple[str, str, Path]], str]:
    value, digest = _json(path)
    entries = value.get("models")
    expected_ids = [model["model_id"] for model in selection["models"]]
    if (
        value.get("schema_version") != "1.0"
        or value.get("status") != "FROZEN_AEON_PARTITION_FORECASTS"
        or value.get("study_id") != _STUDY
        or value.get("partition") != partition
        or value.get("selection_freeze_sha256") != selection_sha256
        or value.get("candidate_contract_sha256") != candidate_sha256
        or not isinstance(entries, list)
        or [entry.get("model_id") for entry in entries if isinstance(entry, dict)] != expected_ids
    ):
        raise ValueError("AEON forecast manifest differs from the frozen selection or partition.")
    result = []
    roles = {model["model_id"]: model["role"] for model in selection["models"]}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"model_id", "artifact", "artifact_sha256"}:
            raise ValueError("AEON forecast manifest model entry is malformed.")
        result.append((entry["model_id"], roles[entry["model_id"]], _safe_artifact(
            path.parent, entry["artifact"], entry["artifact_sha256"]
        )))
    return result, digest


def _review(
    path: Path,
    expected_sha256: str,
    partition: str,
    bindings: dict[str, str],
) -> tuple[dict[str, Any], bool]:
    value, _ = _json(path, expected_sha256)
    fixture = value.get("data_kind") == "SYNTHETIC_FIXTURE"
    expected_status = {
        ("calibration", True): "APPROVED_AEON_CALIBRATION_RUNNER_FIXTURE",
        ("test", True): "APPROVED_AEON_RETROSPECTIVE_TEST_RUNNER_FIXTURE",
        ("calibration", False): "APPROVED_AEON_CALIBRATION_ACCESS",
        ("test", False): "APPROVED_AEON_RETROSPECTIVE_TEST_ACCESS",
    }[(partition, fixture)]
    code_bindings = {
        "runner_code_sha256": artifact_sha256(Path(__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_code_sha256": artifact_sha256(Path(AeonEvaluationReader.__init__.__code__.co_filename)),
    }
    if (
        value.get("status") != expected_status
        or value.get("partition") != partition
        or any(value.get(key) != digest for key, digest in bindings.items())
        or any(value.get(key) != digest for key, digest in code_bindings.items())
    ):
        raise ValueError("AEON runner review bindings differ from the frozen artifacts.")
    return value, fixture


def _reader(
    *, archive: Path, review_path: Path, review_sha256: str, partition: str,
    fixture: bool, fixture_reader: _WindowReader | None,
) -> _WindowReader:
    if fixture_reader is not None:
        if not fixture:
            raise ValueError("An injected AEON reader is permitted for synthetic fixtures only.")
        return fixture_reader
    return AeonEvaluationReader(
        archive, review_path=review_path, review_sha256=review_sha256,
        partition=partition, fixture_only=fixture,
    )


def _rows(reader: _WindowReader, partition: str) -> list[AeonHourlyWindow]:
    rows = list(reader.iter_windows())
    ids = [row.row_id for row in rows]
    cutoffs = [row.cutoff_source_timestamp for row in rows]
    if (
        not rows
        or any(row.partition != partition for row in rows)
        or len(set(ids)) != len(ids)
        or any(right <= left for left, right in zip(cutoffs, cutoffs[1:]))
    ):
        raise ValueError("AEON issued rows are empty, duplicated, reordered or cross-partition.")
    return rows


def _forecast(path: Path, rows: list[AeonHourlyWindow]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"row_ids", "quantiles_db"}:
            raise ValueError("AEON forecast-only artifact schema differs.")
        row_ids = saved["row_ids"].copy()
        quantiles = saved["quantiles_db"].copy()
    expected = np.asarray([row.row_id for row in rows])
    if not np.array_equal(row_ids, expected):
        raise ValueError("AEON forecast row order differs from exact issued rows.")
    if (
        quantiles.shape != (len(rows), 3, 5)
        or not np.isfinite(quantiles).all()
        or (np.diff(quantiles, axis=-1) < 0).any()
    ):
        raise ValueError("AEON forecast quantiles are nonfinite, nonmonotone or mis-shaped.")
    return quantiles.astype(np.float64, copy=False)


def _targets(rows: list[AeonHourlyWindow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([row.target_db for row in rows]),
        np.stack([row.target_mask for row in rows]),
        np.stack([row.target_source_timestamps for row in rows]),
    )


def _require_eligible_floor(metrics: dict[str, object], minimum: int, partition: str) -> None:
    days = metrics.get("eligible_days_per_horizon")
    if (
        not isinstance(days, list)
        or len(days) != 3
        or not all(isinstance(value, int) for value in days)
        or min(days) < minimum
    ):
        raise ValueError(
            f"AEON {partition} has fewer than {minimum} eligible dates per horizon."
        )


def _write_once(output: Path, filename: str, result: dict[str, Any]) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        target = output / filename
        if not target.is_file() or json.loads(target.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing AEON final-evaluation output differs.")
        return result
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        (stage / filename).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    except BaseException:
        if stage.exists():
            for child in stage.iterdir():
                child.unlink()
            stage.rmdir()
        raise
    return result


def execute_calibration(
    *, archive: Path, reader_review_path: Path, reader_review_sha256: str,
    runner_review_path: Path, runner_review_sha256: str, config_path: Path,
    selection_freeze_path: Path, forecast_manifest_path: Path, output: Path,
    fixture_reader: _WindowReader | None = None,
) -> dict[str, Any]:
    """Fit only nonnegative CAL interval widening for every frozen model."""
    config, config_sha = _config(config_path)
    selection, selection_sha = _selection(selection_freeze_path)
    forecasts, manifest_sha = _forecast_manifest(
        forecast_manifest_path, "calibration", selection, selection_sha, None
    )
    _, fixture = _review(runner_review_path, runner_review_sha256, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
    })
    reader = _reader(
        archive=archive, review_path=reader_review_path, review_sha256=reader_review_sha256,
        partition="calibration", fixture=fixture, fixture_reader=fixture_reader,
    )
    rows = _rows(reader, "calibration")
    truth, observed, times = _targets(rows)
    result: dict[str, Any] = {
        "status": "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING",
        "study_id": _STUDY,
        "partition": "calibration",
        "test_access": "PROHIBITED",
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "runner_review_sha256": runner_review_sha256,
        "runner_code_sha256": artifact_sha256(Path(__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_code_sha256": artifact_sha256(Path(AeonEvaluationReader.__init__.__code__.co_filename)),
        "issued_row_ids": [row.row_id for row in rows],
        "models": {},
    }
    for model_id, _, artifact in forecasts:
        prediction = _forecast(artifact, rows)
        calibration = calibrate_interval_widening(
            truth, prediction, observed, times, partition="calibration"
        )
        _require_eligible_floor(calibration, 12, "calibration")
        widened = apply_interval_widening(prediction, np.asarray(calibration["adjustment_db"]))
        result["models"][model_id] = {
            **calibration,
            "forecast_artifact_sha256": artifact_sha256(artifact),
            "raw_interval_metrics": daily_pinball(truth, prediction, observed, times),
            "widened_interval_metrics": daily_pinball(truth, widened, observed, times),
        }
    return _write_once(output, "calibration.json", result)


def _candidate(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _json(path)
    ids = value.get("candidate_cutoff_interval_ids")
    rows = value.get("candidate_rows")
    if (
        value.get("status") != "METADATA_CANDIDATE_UNIVERSE_PENDING_INDEPENDENT_REVIEW"
        or value.get("classification") != "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED"
        or value.get("numeric_test_outcome_access") != "PROHIBITED"
        or value.get("source_archive_sha256") != _SOURCE_SHA256
        or value.get("partition_start_source_date_inclusive") != "2025-01-06"
        or value.get("partition_end_source_date_exclusive") != "2025-03-01"
        or value.get("actual_issued_rows") != "UNKNOWN_NUMERIC_QC_NOT_OPENED"
        or value.get("actual_scored_rows") != "UNKNOWN_NUMERIC_QC_NOT_OPENED"
        or not isinstance(ids, list)
        or not ids
        or len(ids) != len(set(ids))
        or not all(isinstance(item, int) for item in ids)
        or not isinstance(rows, list)
        or len(rows) != len(ids)
        or [row.get("cutoff_interval_id") for row in rows if isinstance(row, dict)] != ids
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("cutoff_source_timestamp"), str)
            or not isinstance(row.get("row_id"), str)
            or row.get("numeric_issuance_status") != "UNKNOWN"
            or row.get("target_scoring_status") != "UNKNOWN"
            for row in rows
        )
    ):
        raise ValueError("AEON metadata-only TEST candidate contract is invalid.")
    return value, digest


def _pretest_freeze(
    path: Path, *, candidate_sha: str, selection_sha: str, calibration_sha: str,
    config_sha: str,
) -> tuple[dict[str, Any], str]:
    value, digest = _json(path)
    expected = {
        "schema_version": "1.0",
        "status": "FROZEN_AEON_RETROSPECTIVE_TEST_PREACCESS",
        "study_id": _STUDY,
        "source_archive_sha256": _SOURCE_SHA256,
        "metadata_candidate_report_sha256": candidate_sha,
        "metadata_candidate_review_sha256": value.get("metadata_candidate_review_sha256"),
        "selection_freeze_sha256": selection_sha,
        "calibration_artifact_sha256": calibration_sha,
        "config_sha256": config_sha,
        "runner_code_sha256": artifact_sha256(Path(__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_code_sha256": artifact_sha256(Path(AeonEvaluationReader.__init__.__code__.co_filename)),
        "issued_row_rule": "EXACT_24_PRIOR_INTERVAL_IDS_OBSERVED_38KHZ",
        "primary_metric": "RAW_FIVE_QUANTILE_ELIGIBLE_TARGET_DATE_PINBALL",
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "test_access": "PROHIBITED_PENDING_INDEPENDENT_APPROVAL",
    }
    if value != expected or not _digest(value.get("metadata_candidate_review_sha256")):
        raise ValueError("AEON pretest freeze differs from the one-way reviewed dependency graph.")
    return value, digest


def execute_retrospective_test(
    *, archive: Path, reader_review_path: Path, reader_review_sha256: str,
    runner_review_path: Path, runner_review_sha256: str, config_path: Path,
    selection_freeze_path: Path, candidate_contract_path: Path,
    calibration_artifact_path: Path, pretest_freeze_path: Path,
    forecast_manifest_path: Path, output: Path,
    fixture_reader: _WindowReader | None = None,
) -> dict[str, Any]:
    """Score frozen raw forecasts once after the exact TEST access approval."""
    config, config_sha = _config(config_path)
    selection, selection_sha = _selection(selection_freeze_path)
    calibration, calibration_sha = _json(calibration_artifact_path)
    if (
        calibration.get("status") != "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING"
        or calibration.get("study_id") != _STUDY
        or calibration.get("selection_freeze_sha256") != selection_sha
        or not isinstance(calibration.get("models"), dict)
    ):
        raise ValueError("AEON calibration artifact differs from the frozen selection.")
    candidate, candidate_sha = _candidate(candidate_contract_path)
    _, pretest_sha = _pretest_freeze(
        pretest_freeze_path, candidate_sha=candidate_sha, selection_sha=selection_sha,
        calibration_sha=calibration_sha, config_sha=config_sha,
    )
    forecasts, manifest_sha = _forecast_manifest(
        forecast_manifest_path, "test", selection, selection_sha, candidate_sha
    )
    _, fixture = _review(runner_review_path, runner_review_sha256, "test", {
        "selection_freeze_sha256": selection_sha,
        "config_sha256": config_sha,
        "calibration_artifact_sha256": calibration_sha,
        "pretest_freeze_sha256": pretest_sha,
    })
    reader = _reader(
        archive=archive, review_path=reader_review_path, review_sha256=reader_review_sha256,
        partition="test", fixture=fixture, fixture_reader=fixture_reader,
    )
    rows = _rows(reader, "test")
    candidate_pairs = [
        (row["cutoff_interval_id"], row["cutoff_source_timestamp"])
        for row in candidate["candidate_rows"]
    ]
    frozen = set(candidate_pairs)
    issued = {(row.cutoff_interval_id, str(row.cutoff_source_timestamp)) for row in rows}
    if not issued <= frozen:
        raise ValueError("AEON issued row falls outside the metadata-only candidate universe.")
    truth, observed, times = _targets(rows)
    model_results: dict[str, Any] = {}
    raw: dict[str, np.ndarray] = {}
    baseline_id = next(model["model_id"] for model in selection["models"] if model["role"] == "baseline")
    for model_id, _, artifact in forecasts:
        prediction = _forecast(artifact, rows)
        model_calibration = calibration["models"].get(model_id)
        if not isinstance(model_calibration, dict):
            raise ValueError("AEON calibration is missing a frozen selected model.")
        adjustment = np.asarray(model_calibration.get("adjustment_db"), dtype=np.float64)
        widened = apply_interval_widening(prediction, adjustment)
        raw_metrics = daily_pinball(truth, prediction, observed, times)
        _require_eligible_floor(raw_metrics, 20, "TEST")
        raw[model_id] = prediction
        model_results[model_id] = {
            "forecast_artifact_sha256": artifact_sha256(artifact),
            "raw_metrics": raw_metrics,
            "widened_interval_metrics": daily_pinball(truth, widened, observed, times),
        }
    baseline_metrics = model_results[baseline_id]["raw_metrics"]
    comparisons: dict[str, Any] = {}
    for model in selection["models"]:
        model_id = model["model_id"]
        if model_id == baseline_id:
            continue
        boot = paired_48h_bootstrap(truth, raw[baseline_id], raw[model_id], observed, times)
        base_h = np.asarray(baseline_metrics["daily_mean_pinball_db_per_horizon"])
        candidate_h = np.asarray(model_results[model_id]["raw_metrics"]["daily_mean_pinball_db_per_horizon"])
        base_primary = float(baseline_metrics["primary_daily_mean_pinball_db"])
        candidate_primary = float(model_results[model_id]["raw_metrics"]["primary_daily_mean_pinball_db"])
        relative_horizon = [
            float(candidate_value / base_value - 1.0) if base_value > 0 else None
            for base_value, candidate_value in zip(base_h, candidate_h, strict=True)
        ]
        daily_differences = []
        base_daily = baseline_metrics["daily_pinball_db_by_horizon_date_quantile"]
        candidate_daily = model_results[model_id]["raw_metrics"][
            "daily_pinball_db_by_horizon_date_quantile"
        ]
        for horizon, (base_rows, candidate_rows) in zip(
            (1, 3, 6), zip(base_daily, candidate_daily, strict=True), strict=True
        ):
            if [row["source_date"] for row in base_rows] != [
                row["source_date"] for row in candidate_rows
            ]:
                raise ValueError("AEON compared models differ in eligible TEST date support.")
            daily_differences.append({
                "horizon_hours": horizon,
                "candidate_minus_baseline_pinball_db": [
                    {
                        "source_date": base_row["source_date"],
                        "difference_db": float(np.mean(candidate_row["quantile_losses_db"]) - np.mean(base_row["quantile_losses_db"])),
                    }
                    for base_row, candidate_row in zip(base_rows, candidate_rows, strict=True)
                ],
            })
        comparisons[model_id] = {
            "baseline_model_id": baseline_id,
            "primary_relative_loss_change": (
                candidate_primary / base_primary - 1.0 if base_primary > 0 else None
            ),
            "passes_prespecified_five_percent_improvement_gate": candidate_primary <= 0.95 * base_primary,
            "relative_loss_change_per_horizon": relative_horizon,
            "passes_per_horizon_ten_percent_regression_guard": bool(
                np.all(candidate_h <= 1.10 * base_h)
            ),
            "daily_paired_differences": daily_differences,
            "paired_95_percent_interval_db": np.quantile(boot, [0.025, 0.975]).tolist(),
            "bootstrap_candidate_minus_baseline_db": boot.tolist(),
        }
    result: dict[str, Any] = {
        "status": "COMPLETED_AEON_RETROSPECTIVE_TEST_SCORING",
        "classification": "RETROSPECTIVE_EVALUATION_NOT_SEALED",
        "study_id": _STUDY,
        "partition": "test",
        "selection_freeze_sha256": selection_sha,
        "candidate_contract_sha256": candidate_sha,
        "calibration_artifact_sha256": calibration_sha,
        "pretest_freeze_sha256": pretest_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "runner_review_sha256": runner_review_sha256,
        "runner_code_sha256": artifact_sha256(Path(__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_code_sha256": artifact_sha256(Path(AeonEvaluationReader.__init__.__code__.co_filename)),
        "issued_row_ids": [row.row_id for row in rows],
        "scored_target_interval_ids_by_row": [
            [int(value) if row.target_mask[index] else None
             for index, value in enumerate(row.target_interval_ids)] for row in rows
        ],
        "target_qc_status_by_row": [list(row.target_qc_status) for row in rows],
        "candidate_not_issued": [
            {"cutoff_interval_id": interval, "cutoff_source_timestamp": timestamp,
             "reason": "FAILED_FROZEN_24_PRIOR_38KHZ_ISSUANCE_RULE"}
            for interval, timestamp in candidate_pairs if (interval, timestamp) not in issued
        ],
        "models": model_results,
        "comparisons": comparisons,
    }
    return _write_once(output, "test-score.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "score-test"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reader-review", type=Path, required=True)
    parser.add_argument("--reader-review-sha256", required=True)
    parser.add_argument("--runner-review", type=Path, required=True)
    parser.add_argument("--runner-review-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path, required=True)
    parser.add_argument("--forecast-manifest", type=Path, required=True)
    parser.add_argument("--candidate-contract", type=Path)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--pretest-freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common = dict(
        archive=args.archive, reader_review_path=args.reader_review,
        reader_review_sha256=args.reader_review_sha256,
        runner_review_path=args.runner_review, runner_review_sha256=args.runner_review_sha256,
        config_path=args.config, selection_freeze_path=args.selection_freeze,
        forecast_manifest_path=args.forecast_manifest, output=args.output,
    )
    if args.stage == "calibrate":
        if (
            args.candidate_contract is not None
            or args.calibration_artifact is not None
            or args.pretest_freeze is not None
        ):
            parser.error("calibrate does not accept TEST-only artifacts")
        result = execute_calibration(**common)
    else:
        if (
            args.candidate_contract is None
            or args.calibration_artifact is None
            or args.pretest_freeze is None
        ):
            parser.error(
                "score-test requires --candidate-contract, --calibration-artifact and "
                "--pretest-freeze"
            )
        result = execute_retrospective_test(
            **common, candidate_contract_path=args.candidate_contract,
            calibration_artifact_path=args.calibration_artifact,
            pretest_freeze_path=args.pretest_freeze,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
