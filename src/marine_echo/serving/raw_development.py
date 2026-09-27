"""Publish the exact reviewed raw-code development rows as research evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

RUN_DIRECTORY = "outputs/raw-response-development-v1-deterministic-20260927"
REVIEW_RECORD = "orchestration/reviews/V2_RAW_RESPONSE_DEVELOPMENT_RESULT_20260927.json"
QUANTILES = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _utc(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="s") + "Z"


def _verified_saved_rows(path: Path, digest: str) -> dict[str, np.ndarray]:
    if _sha256(path) != digest:
        raise ValueError(f"Reviewed development prediction bytes changed: {path.name}")
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "study_id",
            "quantity",
            "cutoffs",
            "target_times",
            "targets_code",
            "target_mask",
            "forecast_quantiles_code",
            "quantile_levels",
            "reviewed_rows_sha256",
        }
        if set(saved.files) != required:
            raise ValueError("Development prediction schema changed")
        return {key: saved[key].copy() for key in required}


def _metrics(rows: dict[str, np.ndarray], horizon: int) -> dict[str, Any]:
    mask = rows["target_mask"][:, horizon]
    truth = rows["targets_code"][mask, horizon]
    forecast = rows["forecast_quantiles_code"][mask, horizon]
    errors = truth[:, None] - forecast
    pinball = np.maximum(QUANTILES * errors, (QUANTILES - 1) * errors).mean(axis=1)
    days = rows["target_times"][mask, horizon].astype("datetime64[D]")
    return {
        "eligible_rows": int(mask.sum()),
        "target_days": len(np.unique(days)),
        "issued_but_unscored": int((~mask).sum()),
        "daily_mean_pinball_code": float(
            np.mean([pinball[days == day].mean() for day in np.unique(days)])
        ),
        "mae_code_median": float(np.abs(truth - forecast[:, 2]).mean()),
        "empirical_90pct_interval_coverage": float(
            ((truth >= forecast[:, 0]) & (truth <= forecast[:, 4])).mean()
        ),
    }


def build_raw_development_report(root: Path) -> dict[str, Any]:
    """Expose every reviewed issued row without selecting favorable examples."""
    review = json.loads((root / REVIEW_RECORD).read_text(encoding="utf-8"))
    if (
        review.get("disposition") != "APPROVE_RAW_RESPONSE_DEVELOPMENT_RESULT"
        or review.get("reviewer_session") != "/root/v2_reviewer"
    ):
        raise ValueError("Distinct raw development result review is required")
    directory = root / RUN_DIRECTORY
    result_path = directory / "result.json"
    if _sha256(result_path) != review["result_sha256"]:
        raise ValueError("Reviewed development result changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("study_id") != "raw_response_development_v1"
        or result.get("run_id") != "raw_response_development_deterministic_20260927"
        or result.get("calibrated_core_slots_consumed") != 0
        or result.get("direct_resources", {}).get("resume_equivalent") is not True
        or result.get("direct_resources", {}).get("updates") != 128
        or result.get("review_sha256") != review.get("review_binding_sha256")
    ):
        raise ValueError("Reviewed development identity or completion changed")
    for label, key in (
        ("direct-checkpoint-64.pt", "checkpoint_64_sha256"),
        ("direct-checkpoint-128.pt", "checkpoint_128_sha256"),
    ):
        if _sha256(directory / label) != review[key]:
            raise ValueError("Reviewed direct checkpoint changed")
    ridge = _verified_saved_rows(
        directory / "ridge-assessment-predictions.npz", review["ridge_prediction_sha256"]
    )
    direct = _verified_saved_rows(
        directory / "direct-assessment-predictions.npz", review["direct_prediction_sha256"]
    )
    if (
        len(ridge["cutoffs"]) != 212
        or ridge["targets_code"].shape != (212, 3)
        or ridge["forecast_quantiles_code"].shape != (212, 3, 5)
        or direct["forecast_quantiles_code"].shape != (212, 3, 5)
        or ridge["study_id"].item() != "raw_response_development_v1"
        or ridge["quantity"].item() != "complete_positive_azfp_backscatter_r_code_mean"
        or ridge["reviewed_rows_sha256"].item() != result["assessment_rows_sha256"]
        or not np.array_equal(ridge["quantile_levels"], QUANTILES)
    ):
        raise ValueError("Reviewed raw-code prediction cohort changed")
    for key in (
        "cutoffs",
        "target_times",
        "targets_code",
        "target_mask",
        "quantile_levels",
        "reviewed_rows_sha256",
        "study_id",
        "quantity",
    ):
        if not np.array_equal(ridge[key], direct[key], equal_nan=ridge[key].dtype.kind in "fc"):
            raise ValueError(f"Development model cohorts differ: {key}")
    if (
        not np.isfinite(ridge["forecast_quantiles_code"]).all()
        or not np.isfinite(direct["forecast_quantiles_code"]).all()
        or (np.diff(ridge["forecast_quantiles_code"], axis=-1) < 0).any()
        or (np.diff(direct["forecast_quantiles_code"], axis=-1) < 0).any()
    ):
        raise ValueError("Invalid saved forecast quantiles")
    horizon_reports = []
    for index, hours in enumerate((1, 3, 6)):
        ridge_metrics = _metrics(ridge, index)
        direct_metrics = _metrics(direct, index)
        for family, metrics in (("ridge", ridge_metrics), ("direct_neural", direct_metrics)):
            reported = result["models"][family]["metrics"]["horizons"][str(hours)]
            for key, value in metrics.items():
                if not np.isclose(value, reported[key], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"Saved {family} metric differs: {hours}h/{key}")
        horizon_reports.append(
            {
                "horizon_hours": hours,
                "eligible_rows": ridge_metrics["eligible_rows"],
                "target_days": ridge_metrics["target_days"],
                "ridge": ridge_metrics,
                "direct_neural": direct_metrics,
            }
        )
    rows = []
    for row_index in range(212):
        row_horizons = []
        for hindex, hours in enumerate((1, 3, 6)):
            eligible = bool(ridge["target_mask"][row_index, hindex])
            row_horizons.append(
                {
                    "horizon_hours": hours,
                    "target_start_utc": _utc(ridge["target_times"][row_index, hindex]),
                    "eligible": eligible,
                    "truth_code": float(ridge["targets_code"][row_index, hindex])
                    if eligible
                    else None,
                    "ridge_quantiles_code": ridge["forecast_quantiles_code"][row_index, hindex]
                    .astype(float)
                    .tolist(),
                    "direct_quantiles_code": direct["forecast_quantiles_code"][row_index, hindex]
                    .astype(float)
                    .tolist(),
                }
            )
        rows.append({"cutoff_utc": _utc(ridge["cutoffs"][row_index]), "horizons": row_horizons})
    return {
        "study_id": "raw_response_development_v1",
        "run_id": result["run_id"],
        "status": "REAL_TRAIN_DEVELOPMENT_ONLY",
        "quantity": "complete_positive_azfp_backscatter_r_code_mean",
        "unit": "transformed AZFP response code (not calibrated Sv)",
        "calibrated": False,
        "final_evaluation": False,
        "comparison_label": "Ridge outperformed the direct neural model on this fixed TRAIN-development assessment at all three horizons.",
        "horizons": horizon_reports,
        "rows": rows,
        "source_result_sha256": review["result_sha256"],
        "direct_checkpoint_128_sha256": review["checkpoint_128_sha256"],
        "review_record_sha256": _sha256(root / REVIEW_RECORD),
        "limitations": [
            "All rows are fixed TRAIN-development assessment; the April segment is not a sealed holdout or final evaluation.",
            "These transformed complete-positive instrument response codes are not calibrated Sv, acoustic backscatter in physical units, fish, biomass, or catch.",
            "The 25-slot calibrated v2 campaign and JEPA comparisons were not executed because the reviewed calibrated target lacks April support.",
            "Interval coverage here is descriptive on overlapping development rows and is not calibrated uncertainty or a superiority interval.",
            "No commercial or operational Marine validation was performed.",
        ],
    }
