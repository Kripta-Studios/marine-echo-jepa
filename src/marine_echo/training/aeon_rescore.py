"""Independently rescore all saved AEON validation rows before model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.training.aeon_campaign import campaign_slots
from marine_echo.training.aeon_development import _sha256


_ROW_FIELDS = (
    "row_ids",
    "cutoff_source_timestamps",
    "target_source_timestamps",
    "target_interval_ids",
    "truth_db",
    "target_mask",
    "target_qc_status",
    "past_members",
    "target_members",
)


def _artifact(directory: Path, name: object, expected_sha256: object) -> Path:
    if not isinstance(name, str) or not isinstance(expected_sha256, str):
        raise ValueError("AEON rescore artifact name or digest is malformed.")
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in (".", ".."):
        raise ValueError("AEON rescore artifact escapes its slot directory.")
    path = (directory / relative).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()) or _sha256(path) != expected_sha256:
        raise ValueError("AEON rescore artifact digest or location differs.")
    return path


def _row_digest(rows: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for field in _ROW_FIELDS:
        values = np.ascontiguousarray(rows[field])
        digest.update(field.encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _same_rows(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return all(
        np.array_equal(left[field], right[field], equal_nan=left[field].dtype.kind in "fM")
        for field in _ROW_FIELDS
    )


def _load_prediction(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(_ROW_FIELDS) | {"quantiles_db"}:
            raise ValueError("AEON saved validation prediction schema differs.")
        rows = {field: saved[field].copy() for field in _ROW_FIELDS}
        quantiles = saved["quantiles_db"].copy()
    if (
        len(set(rows["row_ids"].tolist())) != len(rows["row_ids"])
        or quantiles.shape != (len(rows["row_ids"]), 3, 5)
        or rows["truth_db"].shape != (len(rows["row_ids"]), 3)
        or rows["target_mask"].shape != rows["truth_db"].shape
        or rows["target_source_timestamps"].shape != rows["truth_db"].shape
    ):
        raise ValueError("AEON validation row identities or dimensions differ.")
    return rows, quantiles


def _date_hashes(metrics: dict[str, Any]) -> list[str]:
    days = metrics["eligible_source_dates_by_horizon"]
    if not isinstance(days, list) or len(days) != 3:
        raise ValueError("AEON protocol score lacks exact eligible source dates.")
    return [
        hashlib.sha256(("\n".join(dates) + "\n").encode("ascii")).hexdigest()
        for dates in days
    ]


def rescore_campaign(
    campaign_output: Path,
    config_path: Path,
    correction_review_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Score immutable final validation predictions; never rank or retrain models."""
    campaign_output = campaign_output.resolve(strict=True)
    manifest_path = campaign_output / "manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sha256 = _sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    correction_sha256 = _sha256(correction_review_path)
    correction = json.loads(correction_review_path.read_text(encoding="utf-8"))
    expected_slots = campaign_slots(config)
    entries = manifest.get("slots")
    if (
        manifest.get("status") != "COMPLETED_TRAIN_VALIDATION_CAMPAIGN_NOT_FINAL_EVALUATION"
        or manifest.get("config_sha256") != config_sha256
        or not isinstance(entries, dict)
        or set(entries) != {slot.run_id for slot in expected_slots}
        or any(entry.get("status") != "DONE" for entry in entries.values())
        or correction.get("disposition")
        != "APPROVE_EXECUTION_ONLY_SELECTION_REQUIRES_PROTOCOL_RESCORE"
        or correction.get("campaign_code_sha256") != manifest.get("code_sha256")
        or correction.get("config_sha256") != config_sha256
        or correction.get("test_access") != "PROHIBITED"
    ):
        raise ValueError("AEON rescore needs all 17 completed, hash-bound development slots.")
    score_code_path = Path(daily_pinball.__code__.co_filename)
    report: dict[str, Any] = {
        "status": "PROTOCOL_VALIDATION_RESCORE_PENDING_INDEPENDENT_SELECTION_REVIEW",
        "study_id": config["study_id"],
        "classification": "DEVELOPMENT_NOT_FINAL_EVALUATION",
        "test_access": "PROHIBITED",
        "campaign_manifest_sha256": manifest_sha256,
        "campaign_config_sha256": config_sha256,
        "campaign_code_sha256": manifest["code_sha256"],
        "correction_review_sha256": correction_sha256,
        "evaluation_code_sha256": _sha256(score_code_path),
        "rescore_code_sha256": _sha256(Path(__file__)),
        "original_metric_status": "NON_PROTOCOL_DIAGNOSTIC",
        "model_selection": "NOT_PERFORMED_PENDING_INDEPENDENT_REVIEW",
        "slots": {},
    }
    first_rows: dict[str, np.ndarray] | None = None
    reference_date_hashes: list[str] | None = None
    for slot in expected_slots:
        directory = campaign_output / slot.run_id
        entry = entries[slot.run_id]
        slot_path = _artifact(directory, "slot.json", entry.get("slot_sha256"))
        recorded = json.loads(slot_path.read_text(encoding="utf-8"))
        if (
            recorded.get("run_id") != slot.run_id
            or recorded.get("family") != slot.family
            or recorded.get("seed") != slot.seed
            or recorded.get("primary_daily_mean_pinball_db")
            != entry.get("primary_daily_mean_pinball_db")
        ):
            raise ValueError("AEON slot identity or original metric differs from the ledger.")
        prediction_path = _artifact(
            directory, recorded.get("prediction_path"), recorded.get("prediction_sha256")
        )
        _artifact(directory, recorded.get("model_path"), recorded.get("model_sha256"))
        for artifact in recorded.get("checkpoints", []) + recorded.get("validation_checks", []):
            _artifact(directory, artifact.get("path"), artifact.get("sha256"))
        rows, quantiles = _load_prediction(prediction_path)
        if first_rows is None:
            first_rows = rows
            report["validation_row_sha256"] = _row_digest(rows)
            report["issued_rows"] = len(rows["row_ids"])
        elif not _same_rows(first_rows, rows):
            raise ValueError("AEON compared slots differ in validation rows, masks or truth.")
        score = daily_pinball(
            rows["truth_db"],
            quantiles,
            rows["target_mask"],
            rows["target_source_timestamps"],
        )
        date_hashes = _date_hashes(score)
        if reference_date_hashes is None:
            reference_date_hashes = date_hashes
            report["eligible_source_dates_by_horizon"] = score[
                "eligible_source_dates_by_horizon"
            ]
            report["eligible_source_date_sha256_by_horizon"] = date_hashes
        elif date_hashes != reference_date_hashes:
            raise ValueError("AEON compared slots differ in eligible source-date support.")
        report["slots"][slot.run_id] = {
            "family": slot.family,
            "seed": slot.seed,
            "prediction_sha256": recorded["prediction_sha256"],
            "original_all_scored_day_metric_status": "NON_PROTOCOL_DIAGNOSTIC",
            "original_all_scored_day_primary_pinball_db": recorded[
                "primary_daily_mean_pinball_db"
            ],
            "protocol_validation_metrics": score,
        }
    output = output.resolve()
    if output.exists():
        existing = output / "rescore.json"
        if not existing.is_file():
            raise FileExistsError("AEON rescore output exists without its report.")
        old = json.loads(existing.read_text(encoding="utf-8"))
        if old != report:
            raise ValueError("AEON rescore output differs from current campaign artifacts.")
        return old
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        (stage / "rescore.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    except BaseException:
        if stage.exists() and stage.parent == output.parent:
            (stage / "rescore.json").unlink(missing_ok=True)
            stage.rmdir()
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--correction-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = rescore_campaign(
        arguments.campaign_output,
        arguments.config,
        arguments.correction_review,
        arguments.output,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
