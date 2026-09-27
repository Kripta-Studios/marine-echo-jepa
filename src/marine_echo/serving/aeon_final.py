"""Prepare a reviewed retrospective AEON report and truth-free source-clock replay.

Production opens only for the exact independently reviewed TEST outcome bytes.
This module reads saved score, metadata candidate and prediction files only; never the
source archive or TEST target values.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MODEL_IDS = (
    "core_direct_equal_three_seed_ensemble",
    "core_ema_equal_three_seed_ensemble",
    "post_hoc_lightgbm",
)
SOURCE_SHA256 = "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde"
SELECTION_SHA256 = "b3940ab8e159ac8808843b15b455ffb3c42f2d3db322190af15b80e543edff58"
CAL_SHA256 = "a6e35bc5fa5c27b2b0669e7fb9fe7f8ff1bdd69952193418bbbe130e46438c94"
PRETEST_SHA256 = "f9062592b6196f49e86cf938205215670800a14c03c4542576c0b341a6018281"
TEST_SCORE_SHA256 = "23f9d350528e9f99b35949c6281c39d209dff7f7db44a992d241d64ccde8bd89"
TEST_REVIEW_SHA256: str | None = "75a241f00d34f8d373ccbd4c5b2e2c35397e2c0ff0a20a6e112ce4953f13489a"
EXPECTED_ISSUED_ROWS = 1216
EXPECTED_ELIGIBLE_DAYS = [49, 50, 51]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bounded_json(path: Path, limit: int) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError("AEON evidence file is linked or exceeds the size bound.")
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if not isinstance(value, dict):
        raise TypeError("AEON evidence file must be an object.")
    return value


def _finite_vector(value: object, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length or any(
        not isinstance(item, (int, float)) or not math.isfinite(item) for item in value
    ):
        raise ValueError("AEON TEST metric vector differs or is nonfinite.")
    return [float(item) for item in value]


def load_reviewed_test(
    score_path: Path, candidate_path: Path, review_path: Path, forecasts_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return bounded display evidence and replay only after the pinned review matches.

    Fixture tests may monkeypatch the digest constants. Production binds the exact
    independently approved outcome review and all frozen upstream evidence.
    """
    if TEST_REVIEW_SHA256 is None or len(TEST_REVIEW_SHA256) != 64:
        raise ValueError("AEON final release needs a pinned independent TEST review.")
    if _sha256(review_path) != TEST_REVIEW_SHA256:
        raise ValueError("AEON TEST review digest differs from pinned evidence.")
    review = _bounded_json(review_path, 1_000_000)
    if _sha256(score_path) != TEST_SCORE_SHA256:
        raise ValueError("AEON TEST score digest differs from pinned evidence.")
    score = _bounded_json(score_path, 16_000_000)
    candidate_sha = _sha256(candidate_path)
    candidate = _bounded_json(candidate_path, 16_000_000)
    reviewed_lineage = review.get("frozen_lineage_sha256", {})
    reviewed_artifacts = review.get("artifact_sha256", {})
    lineage = {
        "selection_freeze_sha256": SELECTION_SHA256,
        "calibration_artifact_sha256": CAL_SHA256,
        "pretest_freeze_sha256": PRETEST_SHA256,
        "candidate_contract_sha256": candidate_sha,
    }
    if (
        review.get("status") != "APPROVED_AEON_RETROSPECTIVE_TEST_OUTCOME"
        or review.get("reviewer_session") != "/root/aeon_reviewer"
        or review.get("classification") != "RETROSPECTIVE_EVALUATION_NOT_SEALED"
        or reviewed_artifacts.get("test_score") != TEST_SCORE_SHA256
        or reviewed_lineage.get("source_archive") != SOURCE_SHA256
        or reviewed_lineage.get("selection_freeze") != SELECTION_SHA256
        or reviewed_lineage.get("calibration_artifact") != CAL_SHA256
        or reviewed_lineage.get("pretest_freeze") != PRETEST_SHA256
        or reviewed_lineage.get("metadata_candidate_report") != candidate_sha
        or candidate.get("source_archive_sha256") != SOURCE_SHA256
        or candidate.get("classification") != "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED"
        or candidate.get("numeric_test_outcome_access") != "PROHIBITED"
        or any(score.get(key) != value for key, value in lineage.items())
        or score.get("status") != "COMPLETED_AEON_RETROSPECTIVE_TEST_SCORING"
        or score.get("classification") != "RETROSPECTIVE_EVALUATION_NOT_SEALED"
        or score.get("study_id") != "aeon3_geb_2024_hourly_sv_v1"
        or score.get("partition") != "test"
        or set(score.get("models", {})) != set(MODEL_IDS)
        or review.get("support", {}).get("issued_rows") != EXPECTED_ISSUED_ROWS
        or review.get("support", {}).get("eligible_source_dates_per_horizon") != EXPECTED_ELIGIBLE_DAYS
        or review.get("independent_execution", {}).get("new_test_evaluator_run") is not False
        or review.get("independent_execution", {}).get("selection_or_tuning_after_test_access") is not False
    ):
        raise ValueError("AEON TEST review, source or frozen lineage differs.")
    row_ids = score.get("issued_row_ids")
    candidates = candidate.get("candidate_rows")
    if (
        not isinstance(row_ids, list) or not 1 <= len(row_ids) <= 1500
        or len(set(row_ids)) != len(row_ids)
        or not isinstance(candidates, list) or len(candidates) > 1500
    ):
        raise ValueError("AEON TEST row inventory differs.")
    by_row = {item.get("row_id"): item for item in candidates if isinstance(item, dict)}
    if len(by_row) != len(candidates) or set(row_ids) - set(by_row):
        raise ValueError("AEON TEST issued rows are outside the metadata candidate universe.")
    replay_rows = [
        {"row_id": row_id, "cutoff_interval_id": by_row[row_id]["cutoff_interval_id"],
         "cutoff_source_timestamp": by_row[row_id]["cutoff_source_timestamp"],
         "predictions": {}}
        for row_id in row_ids
    ]
    report_models: dict[str, Any] = {}
    eligible_support: list[int] | None = None
    for model_id in MODEL_IDS:
        source = score["models"][model_id]
        metrics = source.get("raw_metrics", {})
        widened = source.get("widened_interval_metrics", {})
        support = metrics.get("eligible_days_per_horizon")
        if not isinstance(support, list) or len(support) != 3 or any(
            not isinstance(day, int) or day < 20 or day > 60 for day in support
        ):
            raise ValueError("AEON TEST eligible-date support differs.")
        if eligible_support is None:
            eligible_support = support
        elif support != eligible_support:
            raise ValueError("AEON TEST model eligible-date support differs.")
        forecast = forecasts_dir / f"{model_id}-forecast.npz"
        if forecast.is_symlink() or forecast.stat().st_size > 8_000_000 or (
            _sha256(forecast) != source.get("forecast_artifact_sha256")
            or source.get("forecast_artifact_sha256")
            != reviewed_artifacts.get(f"{model_id}_forecast")
        ):
            raise ValueError("AEON TEST forecast digest differs from score.")
        with np.load(forecast, allow_pickle=False) as saved:
            if set(saved.files) != {"row_ids", "quantiles_db"}:
                raise ValueError("AEON TEST forecast array inventory differs.")
            saved_ids = saved["row_ids"].tolist()
            prediction = np.asarray(saved["quantiles_db"], dtype=np.float64)
        if (
            saved_ids != row_ids or prediction.shape != (len(row_ids), 3, 5)
            or not np.isfinite(prediction).all()
            or np.any(np.diff(prediction, axis=2) < 0)
        ):
            raise ValueError("AEON TEST forecast rows or quantiles differ.")
        for index, row in enumerate(replay_rows):
            row["predictions"][model_id] = prediction[index].tolist()
        primary = metrics.get("primary_daily_mean_pinball_db")
        if not isinstance(primary, (int, float)) or not math.isfinite(primary) or primary < 0:
            raise ValueError("AEON TEST primary metric differs.")
        report_models[model_id] = {
            "selection_classification": source.get("selection_classification"),
            "raw_primary_daily_mean_pinball_db": float(primary),
            "raw_per_horizon_pinball_db": _finite_vector(metrics.get("daily_mean_pinball_db_per_horizon"), 3),
            "raw_coverage90_per_horizon": _finite_vector(metrics.get("coverage90_per_horizon"), 3),
            "widened_coverage90_per_horizon": _finite_vector(widened.get("coverage90_per_horizon"), 3),
            "forecast_artifact_sha256": source["forecast_artifact_sha256"],
        }
        approved = review.get("models", {}).get(model_id, {})
        if any((approved.get(key) != actual) for key, actual in (
            ("selection_classification", source.get("selection_classification")),
            ("raw_primary_daily_mean_pinball_db", primary),
            ("raw_daily_mean_pinball_db_per_horizon", metrics.get("daily_mean_pinball_db_per_horizon")),
            ("raw_coverage90_per_horizon", metrics.get("coverage90_per_horizon")),
            ("widened_coverage90_per_horizon", widened.get("coverage90_per_horizon")),
        )):
            raise ValueError("AEON TEST model metrics differ from independent review.")
    if eligible_support != EXPECTED_ELIGIBLE_DAYS:
        raise ValueError("AEON TEST eligible-date support differs from reviewed run inventory.")
    if set(score.get("comparisons", {})) != set(MODEL_IDS[1:]):
        raise ValueError("AEON TEST comparison inventory differs.")
    comparisons = {}
    for model_id in MODEL_IDS[1:]:
        source = score["comparisons"][model_id]
        if source.get("baseline_model_id") != MODEL_IDS[0]:
            raise ValueError("AEON TEST comparison baseline differs.")
        comparisons[model_id] = {
            "primary_relative_loss_change": source.get("primary_relative_loss_change"),
            "paired_95_percent_interval_db": _finite_vector(source.get("paired_95_percent_interval_db"), 2),
            "passes_full_unnarrowed_promotion_rule": source.get("passes_full_unnarrowed_promotion_rule"),
        }
        approved = review.get("comparisons", {}).get(f"{model_id}_minus_direct", {})
        if any(approved.get(key) != source.get(key) for key in (
            "primary_relative_loss_change", "paired_95_percent_interval_db",
            "passes_full_unnarrowed_promotion_rule",
        )):
            raise ValueError("AEON TEST comparison differs from independent review.")
    report = {
        "status": "INDEPENDENTLY_REVIEWED_RETROSPECTIVE_TEST",
        "classification": "RETROSPECTIVE_EVALUATION_NOT_SEALED",
        "issued_rows": len(row_ids), "eligible_days_per_horizon": eligible_support,
        "models": report_models, "comparisons": comparisons,
        "jepa_value_gate": "PASSED_FROZEN_WITHIN_STUDY_RETROSPECTIVE_GATE",
        "jepa_representation_attribution": "NOT_ESTABLISHED",
        "source_archive_sha256": SOURCE_SHA256,
        "selection_freeze_sha256": SELECTION_SHA256,
        "calibration_artifact_sha256": CAL_SHA256,
        "pretest_freeze_sha256": PRETEST_SHA256,
        "candidate_contract_sha256": candidate_sha,
        "test_score_sha256": TEST_SCORE_SHA256,
        "test_outcome_review_sha256": TEST_REVIEW_SHA256,
    }
    replay = {
        "classification": "REVIEWED_RETROSPECTIVE_SOURCE_CLOCK_REPLAY_NOT_LIVE",
        "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
        "target": "38-kHz 60minFullDepth source-conditioned Sv_mean (dB re 1 m^-1)",
        "horizon_source_interval_steps": [1, 3, 6],
        "quantile_levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "test_score_sha256": TEST_SCORE_SHA256,
        "rows": replay_rows,
    }
    return report, replay
