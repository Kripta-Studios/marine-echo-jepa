"""Package independently reviewed AEON development evidence beside historical v1/v2."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from marine_echo.serving.aeon_development_supplements import (
    load_reviewed_expanded,
    load_reviewed_scale,
    reviewed_expanded_provenance,
    reviewed_scale_provenance,
)
from marine_echo.serving.aeon_external_study import (
    load_reviewed_external_study,
    reviewed_external_files,
)
from marine_echo.serving.aeon_final import load_reviewed_test
from marine_echo.serving.release import build_v2_research

_REVIEW_FILES = {
    "core": (
        "AEON_VALIDATION_RESCORE_OUTCOME_REVIEW_20260927.json",
        "16bb9ef931f5e2d8f8a3322fa609c5cb1c769f99f5e4a89a5ef519a0fb52f44f",
        "APPROVED_AEON_VALIDATION_RESCORE_OUTCOME_NO_SELECTION",
    ),
    "hybrid": (
        "AEON_HYBRID_OUTCOME_REVIEW_20260927.json",
        "5244af6f1c9ba0308f1acaf44b1d5b197fb568dc02b9bd00c41ac57823bf9021",
        "APPROVED_AEON_HYBRID_OUTCOME_NO_SELECTION",
    ),
    "post_hoc_supervised": (
        "AEON_SOTA_SUPERVISED_OUTCOME_REVIEW_20260927.json",
        "d08932b0c88f322ceb573ef2ee30198aed0ceb3c5611b6e865e0e562f640138d",
        "APPROVED_AEON_POST_HOC_SUPERVISED_OUTCOME_NO_SELECTION",
    ),
    "forward_ema": (
        "AEON_FORWARD_OUTCOME_REVIEW_20260927.json",
        "d3f142e1f01bb84b22155faa9f1cedb40d499b5aa784865cc9191416c6cfbed6",
        "APPROVED_AEON_FORWARD_OUTCOME_NO_SELECTION",
    ),
    "chronos2": (
        "AEON_CHRONOS2_OUTCOME_REVIEW_20260927.json",
        "11f1c8dd25cd485d296e9502a94c8247cf57f442f4cbb4d00538ce172c6bb541",
        "APPROVED_AEON_CHRONOS2_OUTCOME_NO_SELECTION",
    ),
}
_SOURCE_SHA = "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde"
_ROW_SHA = "9ce5ed6d60c4082efd3f342b3ee09ddadc5cf6286be6d6a3e953ab0f6166a99f"
_CAL_ARTIFACT_SHA = "a6e35bc5fa5c27b2b0669e7fb9fe7f8ff1bdd69952193418bbbe130e46438c94"
_CAL_REVIEW_SHA = "7b61385b29836be53d0e2c5a0b5e3f5db7e2dcb8983e534d17711ed0b44c1428"
_CAL_REVIEW_NAME = "AEON_CALIBRATION_OUTCOME_REVIEW_20260927.json"
_CAL_MODELS = (
    "core_direct_equal_three_seed_ensemble",
    "core_ema_equal_three_seed_ensemble",
    "post_hoc_lightgbm",
)
_CORE_SLOTS = {
    "persistence", "seasonal_24_source_intervals", "ridge", "hist_gradient_boosting",
    *(f"{family}_seed{seed}" for family in ("direct", "ema_jepa", "shared_sigreg") for seed in (7, 13, 23)),
    "random_encoder_ema_seed7", "random_encoder_shared_sigreg_seed7",
    "temporally_shuffled_pretrain_target_ema_seed7",
    "temporally_shuffled_pretrain_target_shared_sigreg_seed7",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _review(root: Path, key: str) -> tuple[dict[str, Any], str]:
    name, expected, status = _REVIEW_FILES[key]
    path = root / "orchestration/reviews" / name
    if _sha256(path) != expected:
        raise ValueError(f"AEON {key} review digest differs from approved evidence.")
    record = json.loads(path.read_text(encoding="utf-8"))
    execution = record.get("independent_execution", {})
    numeric_access = execution.get(
        "calibration_or_test_outcomes_opened",
        execution.get("calibration_or_test_numeric_access"),
    )
    if (
        record.get("status") != status
        or record.get("reviewer_session") != "/root/aeon_reviewer"
        or record.get("test_access") != "PROHIBITED"
        or numeric_access is not False
    ):
        raise ValueError(f"AEON {key} lacks a distinct development outcome review.")
    return record, expected


def _finite_scores(scores: object, expected_keys: set[str] | None = None) -> dict[str, float]:
    if not isinstance(scores, dict) or (expected_keys is not None and set(scores) != expected_keys):
        raise ValueError("AEON reviewed score inventory differs.")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in scores.values()):
        raise ValueError("AEON reviewed score must be finite and nonnegative.")
    return {str(key): float(value) for key, value in scores.items()}


def build_aeon_calibration_report(root: Path, calibration_artifact: Path) -> dict[str, Any]:
    """Expose exact reviewed CAL interval evidence without ranking models or reading TEST."""
    review_path = root / "orchestration/reviews" / _CAL_REVIEW_NAME
    if _sha256(review_path) != _CAL_REVIEW_SHA:
        raise ValueError("AEON CAL review digest differs from approved evidence.")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if _sha256(calibration_artifact) != _CAL_ARTIFACT_SHA:
        raise ValueError("AEON CAL artifact digest differs from approved evidence.")
    artifact = json.loads(calibration_artifact.read_text(encoding="utf-8"))
    support = review.get("calibration_support", {})
    if (
        review.get("status") != "APPROVED_AEON_CALIBRATION_OUTCOME_NO_SELECTION"
        or review.get("reviewer_session") != "/root/aeon_reviewer"
        or review.get("classification") != "CALIBRATION_ONLY_NOT_MODEL_SELECTION_OR_FINAL_EVALUATION"
        or review.get("test_access") != "PROHIBITED"
        or review.get("independent_execution", {}).get("test_numeric_access") is not False
        or review.get("artifact_sha256") != _CAL_ARTIFACT_SHA
        or review.get("source_archive_sha256") != _SOURCE_SHA
        or artifact.get("status") != "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING"
        or artifact.get("study_id") != "aeon3_geb_2024_hourly_sv_v1"
        or artifact.get("partition") != "calibration"
        or artifact.get("test_access") != "PROHIBITED"
        or artifact.get("selection_freeze_sha256") != review.get("selection_freeze_sha256")
        or artifact.get("forecast_manifest_sha256") != review.get("calibration_forecast_plan_sha256")
        or artifact.get("config_sha256") != review.get("config_sha256")
        or support.get("issued_rows") != 810
        or support.get("unique_row_ids") != 810
        or support.get("eligible_source_dates_per_horizon") != [34, 34, 34]
        or not isinstance(artifact.get("issued_row_ids"), list)
        or len(artifact["issued_row_ids"]) != 810
        or len(set(artifact["issued_row_ids"])) != 810
        or set(artifact.get("models", {})) != set(_CAL_MODELS)
        or set(review.get("models", {})) != set(_CAL_MODELS)
    ):
        raise ValueError("AEON CAL review, artifact or support contract differs.")
    models: dict[str, dict[str, Any]] = {}
    for name in _CAL_MODELS:
        recorded = artifact["models"][name]
        approved = review["models"][name]
        raw = recorded.get("raw_interval_metrics", {})
        widened = recorded.get("widened_interval_metrics", {})
        expected_pairs = (
            (recorded.get("selection_classification"), approved.get("selection_classification")),
            (recorded.get("adjustment_db"), approved.get("adjustment_db_by_horizon")),
            (recorded.get("eligible_days_per_horizon"), [34, 34, 34]),
            (recorded.get("eligible_rows_per_horizon"), support.get("eligible_scored_rows_per_horizon")),
            (raw.get("primary_daily_mean_pinball_db"), approved.get("raw_primary_daily_mean_pinball_db")),
            (raw.get("daily_mean_pinball_db_per_horizon"), approved.get("raw_daily_mean_pinball_db_per_horizon")),
            (raw.get("coverage90_per_horizon"), approved.get("raw_coverage90_per_horizon")),
            (widened.get("primary_daily_mean_pinball_db"), approved.get("widened_primary_daily_mean_pinball_db")),
            (widened.get("coverage90_per_horizon"), approved.get("widened_coverage90_per_horizon")),
        )
        if any(left != right for left, right in expected_pairs):
            raise ValueError(f"AEON CAL {name} metrics differ from independent review.")
        finite = [
            *recorded["adjustment_db"], raw["primary_daily_mean_pinball_db"],
            *raw["daily_mean_pinball_db_per_horizon"], *raw["coverage90_per_horizon"],
            widened["primary_daily_mean_pinball_db"], *widened["coverage90_per_horizon"],
        ]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in finite):
            raise ValueError("AEON CAL metrics are nonfinite.")
        models[name] = {
            "selection_classification": recorded["selection_classification"],
            "raw_primary_pinball_db": raw["primary_daily_mean_pinball_db"],
            "raw_per_horizon_pinball_db": raw["daily_mean_pinball_db_per_horizon"],
            "raw_coverage90_per_horizon": raw["coverage90_per_horizon"],
            "interval_widening_db_by_horizon": recorded["adjustment_db"],
            "widened_in_sample_coverage90_per_horizon": widened["coverage90_per_horizon"],
        }
    return {
        "status": "INDEPENDENTLY_REVIEWED_CALIBRATION_ONLY",
        "classification": "CALIBRATION_ONLY_NOT_MODEL_SELECTION_OR_FINAL_EVALUATION",
        "issued_rows": 810,
        "eligible_days_per_horizon": [34, 34, 34],
        "models": models,
        "artifact_sha256": _CAL_ARTIFACT_SHA,
        "outcome_review_sha256": _CAL_REVIEW_SHA,
        "selection_freeze_sha256": review["selection_freeze_sha256"],
        "retrospective_test_outcomes": "NOT_OPENED_FOR_THIS_REPORT",
    }


def build_aeon_development_report(root: Path) -> dict[str, Any]:
    """Use only exact approved outcome reviews; no source ZIP or CAL/TEST values."""
    core, core_sha = _review(root, "core")
    hybrid, hybrid_sha = _review(root, "hybrid")
    supervised, supervised_sha = _review(root, "post_hoc_supervised")
    forward, forward_sha = _review(root, "forward_ema")
    chronos, chronos_sha = _review(root, "chronos2")
    core_scores = _finite_scores(core.get("primary_daily_mean_pinball_db_by_slot"), _CORE_SLOTS)
    hybrid_scores = _finite_scores(hybrid.get("primary_daily_mean_pinball_db"))
    required_hybrid = {
        "raw_only_hgb", "ema_hybrid_seed7", "ema_hybrid_seed13", "ema_hybrid_seed23",
        "shared_hybrid_seed7", "shared_hybrid_seed13", "shared_hybrid_seed23",
        "random_encoder_hybrid_seed7", "shuffled_ema_hybrid_seed7",
        "shuffled_shared_hybrid_seed7", "ema_hybrid_equal_three_seed_ensemble",
        "shared_hybrid_equal_three_seed_ensemble", "core_direct_equal_three_seed_ensemble",
        "core_ema_equal_three_seed_ensemble", "core_shared_equal_three_seed_ensemble",
    }
    if set(hybrid_scores) != required_hybrid:
        raise ValueError("AEON reviewed hybrid/ensemble score inventory differs.")
    core_support = core.get("support", {})
    supervised_support = supervised.get("lineage", {})
    supervised_metrics = supervised.get("verified_metrics", {})
    if (
        core_support.get("issued_rows") != 1219
        or core_support.get("validation_row_sha256") != _ROW_SHA
        or core_support.get("eligible_days_per_horizon") != [50, 50, 50]
        or supervised_support.get("source_archive_sha256") != _SOURCE_SHA
        or supervised_support.get("validation_row_sha256") != _ROW_SHA
        or supervised_support.get("validation_issued_rows") != 1219
        or supervised_metrics.get("eligible_days_per_horizon") != [50, 50, 50]
        or core.get("artifact_sha256", {}).get("rescore_report")
        != hybrid.get("artifact_sha256", {}).get("validation_rescore")
        or core.get("artifact_sha256", {}).get("rescore_report")
        != supervised.get("artifact_sha256", {}).get("validation_rescore")
        or abs(core_scores["hist_gradient_boosting"] - hybrid_scores["raw_only_hgb"]) > 5e-12
    ):
        raise ValueError("AEON reviewed development cohort or metric support differs.")
    sota_score = _finite_scores({
        "primary_daily_mean_pinball_db": supervised_metrics.get("primary_daily_mean_pinball_db")
    })["primary_daily_mean_pinball_db"]
    forward_slots = forward.get("slots", {})
    expected_forward = {
        *(f"forward_ema_seed{seed}" for seed in (7, 13, 23)),
        "random_encoder_seed7", "temporally_shuffled_future_target_seed7",
    }
    forward_scores = _finite_scores(
        {name: slot.get("primary_daily_mean_pinball_db") for name, slot in forward_slots.items()},
        expected_forward,
    )
    forward_ensemble = forward.get("fixed_three_seed_forward_ensemble", {})
    chronos_metrics = chronos.get("validation_metrics", {})
    forward_primary = _finite_scores({
        "primary_daily_mean_pinball_db": forward_ensemble.get("primary_daily_mean_pinball_db")
    })["primary_daily_mean_pinball_db"]
    chronos_primary = _finite_scores({
        "primary_daily_mean_pinball_db": chronos_metrics.get("primary_daily_mean_pinball_db")
    })["primary_daily_mean_pinball_db"]
    if (
        forward.get("validation_row_sha256") != _ROW_SHA
        or forward.get("validation_issued_rows") != 1219
        or forward.get("eligible_source_dates_per_horizon") != [50, 50, 50]
        or forward.get("cohort_sha256") != supervised_support.get("cohort_sha256")
        or forward.get("sota_claim") != "NOT_ESTABLISHED"
        or forward.get("classification") != "POST_HOC_DEVELOPMENT_NOT_FINAL_EVALUATION"
        or forward_ensemble.get("method") != "arithmetic_mean_of_three_saved_forecasts_then_monotone_quantile_sort"
        or abs(forward_ensemble.get("direct_three_seed_reference_primary_daily_mean_pinball_db", -1) - hybrid_scores["core_direct_equal_three_seed_ensemble"]) > 5e-12
        or chronos.get("source_archive_sha256") != _SOURCE_SHA
        or chronos.get("validation_row_sha256") != _ROW_SHA
        or chronos.get("cohort_sha256") != supervised_support.get("cohort_sha256")
        or chronos_metrics.get("issued_rows") != 1219
        or chronos_metrics.get("eligible_source_dates_per_horizon") != [50, 50, 50]
        or chronos.get("sota_claim") != "NOT_ESTABLISHED"
        or chronos.get("fit_behavior") != "FROZEN_ZERO_SHOT_NO_TRAINING"
        or chronos.get("corrected_validation_rescore_sha256") != core["artifact_sha256"]["rescore_report"]
    ):
        raise ValueError("AEON reviewed forward or Chronos support differs from development cohort.")
    return {
        "schema_version": "1.0",
        "study_id": "aeon3_geb_2024_hourly_sv_v1",
        "title": "AEON3 Georges Basin hourly acoustic development",
        "classification": "REVIEWED_TRAIN_VALIDATION_DEVELOPMENT_NOT_FINAL_EVALUATION",
        "source": {
            "publisher": "Figshare AEON AZFP Integrated Sv products, version 2, file 61937281",
            "archive_sha256": _SOURCE_SHA,
            "site": "AEON3 Georges Basin fixed lander",
            "source_identifier": "55144; physical serial binding not independently verified",
        },
        "target": {
            "source_variable": "Sv_mean", "frequency_hz": 38000,
            "product": "60minFullDepth", "nominal_layer_m": [0, 200],
            "unit": "dB re 1 m^-1, source-reported conditioned volume backscattering strength",
            "calibration_claim": "SOURCE_REPORTED_CONDITIONED_NOT_INDEPENDENTLY_FIELD_VERIFIED",
            "horizon_source_interval_steps": [1, 3, 6],
        },
        "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
        "assessment_partition": "validation",
        "validation_row_sha256": _ROW_SHA,
        "calibration_outcomes": "NOT_OPENED_FOR_THIS_REPORT",
        "retrospective_test_outcomes": "NOT_OPENED_FOR_THIS_REPORT",
        "final_evaluation": False,
        "selection": "NOT_PERFORMED_IN_THIS_REPORT",
        "cached_forecasts": 0,
        "development": {
            "core": {
                "status": "INDEPENDENTLY_REVIEWED", "issued_rows": 1219,
                "eligible_days_per_horizon": [50, 50, 50],
                "slot_primary_pinball_db": core_scores,
                "rescore_report_sha256": core["artifact_sha256"]["rescore_report"],
                "outcome_review_sha256": core_sha,
            },
            "hybrid": {
                "status": "INDEPENDENTLY_REVIEWED", "primary_pinball_db": hybrid_scores,
                "report_sha256": hybrid["artifact_sha256"]["hybrid_report"],
                "outcome_review_sha256": hybrid_sha,
            },
            "post_hoc_supervised": {
                "family": "LightGBM", "status": "INDEPENDENTLY_REVIEWED_POST_HOC_DEVELOPMENT",
                "primary_pinball_db": sota_score,
                "per_horizon_pinball_db": supervised_metrics["daily_mean_pinball_db_per_horizon"],
                "report_sha256": supervised["artifact_sha256"]["report"],
                "outcome_review_sha256": supervised_sha,
            },
            "forward_ema": {
                "status": "INDEPENDENTLY_REVIEWED_POST_HOC_DEVELOPMENT",
                "primary_pinball_db": forward_primary,
                "per_horizon_pinball_db": forward_ensemble["daily_mean_pinball_db_per_horizon"],
                "individual_seed_primary_pinball_db": {
                    key: forward_scores[key] for key in sorted(forward_scores) if key.startswith("forward_ema_seed")
                },
                "control_primary_pinball_db": {
                    key: forward_scores[key] for key in sorted(forward_scores) if not key.startswith("forward_ema_seed")
                },
                "manifest_sha256": forward["manifest_sha256"],
                "outcome_review_sha256": forward_sha,
            },
            "chronos2": {
                "status": "INDEPENDENTLY_REVIEWED_POST_HOC_ZERO_SHOT_DEVELOPMENT",
                "primary_pinball_db": chronos_primary,
                "per_horizon_pinball_db": chronos_metrics["daily_mean_pinball_db_per_horizon"],
                "model_revision": chronos["model"]["revision"],
                "manifest_sha256": chronos["manifest_sha256"],
                "outcome_review_sha256": chronos_sha,
            },
        },
        "limitations": [
            "Retrospective TRAIN/validation development scores are not a final evaluation or a state-of-the-art claim.",
            "The source clock timezone and product availability latency are unknown; source dates are not UTC dates.",
            "Publisher conditioning includes filtering and manual exclusions; exact calibration coefficients and processing settings are not independently verified.",
            "The acoustic response does not establish species, biomass, catch or operational savings.",
            "No AEON forecasts are cached or served by this release.",
        ],
    }


def build_aeon_research(
    root: Path, output: Path, *, calibration_artifact: Path | None = None,
    web_dist: Path | None = None,
    test_score: Path | None = None, test_candidate: Path | None = None,
    test_review: Path | None = None, test_forecasts: Path | None = None,
    external_transfer: bool = False,
    scale_output: Path | None = None, expanded_output: Path | None = None,
) -> dict[str, Any]:
    """Build a new offline package; retain v1/v2 artifacts and attach AEON evidence."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("AEON release output already exists.")
    if calibration_artifact is not None and (
        web_dist is None or not (web_dist / "index.html").is_file()
    ):
        raise ValueError("CAL study package requires an explicitly built web/dist.")
    final_inputs = (test_score, test_candidate, test_review, test_forecasts)
    if any(value is not None for value in final_inputs) and (
        any(value is None for value in final_inputs)
        or calibration_artifact is None or web_dist is None
    ):
        raise ValueError("AEON TEST package requires all reviewed evidence, CAL and built web assets.")
    if external_transfer and not all(value is not None for value in final_inputs):
        raise ValueError("AEON external package requires the reviewed retrospective study.")
    study = build_aeon_development_report(root)
    if calibration_artifact is not None:
        study["calibration"] = build_aeon_calibration_report(root, calibration_artifact)
        study["calibration_outcomes"] = "INDEPENDENTLY_REVIEWED_CALIBRATION_ONLY"
        study["classification"] = "REVIEWED_DEVELOPMENT_AND_CALIBRATION_NOT_FINAL_EVALUATION"
        study["title"] = "AEON3 Georges Basin hourly acoustic research"
        study["limitations"].append(
            "CAL interval coverage is in-sample and does not establish retrospective TEST or external coverage."
        )
    replay: dict[str, Any] | None = None
    if all(value is not None for value in final_inputs):
        assert test_score is not None and test_candidate is not None
        assert test_review is not None and test_forecasts is not None
        report, replay = load_reviewed_test(test_score, test_candidate, test_review, test_forecasts)
        study["retrospective_test"] = report
        study["retrospective_test_outcomes"] = report["status"]
        study["classification"] = "REVIEWED_RETROSPECTIVE_TEST_NOT_SEALED"
        study["assessment_partition"] = "retrospective_test"
        study["final_evaluation"] = True
        study["selection"] = "FROZEN_BEFORE_TEST_NO_RELEASE_RERANKING"
        study["cached_forecasts"] = len(replay["rows"]) * len(report["models"])
        study["limitations"] = [
            limit for limit in study["limitations"]
            if limit != "No AEON forecasts are cached or served by this release."
        ]
        study["limitations"].extend([
            "Retrospective TEST is not a sealed or external replication.",
            "Saved source-clock predictions are historical replay, not live forecasts or UTC-time service.",
            "The frozen within-study retrospective EMA-JEPA forecast-family gate passed. Attribution to learned JEPA representations, state of the art, external generalization and business validation remain unestablished.",
        ])
    if external_transfer:
        study["external_transfer"] = load_reviewed_external_study(root)
        study["limitations"].extend([
            "The primary cross-site external target is metadata-ineligible; no primary acoustic model result exists.",
            "The negative JEPA comparison is descriptive same-site prior-year transfer, not sealed confirmation or cross-site replication.",
        ])
    if scale_output is not None:
        study["scaling_development"] = load_reviewed_scale(root, scale_output)
        study["limitations"].append(
            "The post-hoc same-cohort 30k seed-7 endpoints worsened validation loss; "
            "the stage-2 seeds and 50k extension were not run or authorized."
        )
    if expanded_output is not None:
        study["expanded_train_development"] = load_reviewed_expanded(root, expanded_output)
        study["limitations"].append(
            "The post-hoc expanded-TRAIN seed-7 comparison reuses the previously inspected "
            "prior-year deployment as TRAIN. Its validation result is not external or sealed, "
            "and added deployment effects cannot be attributed to data volume alone."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".aeon-release-stage-", dir=output.parent))
    base = stage / "package"
    try:
        build_v2_research(root, base)
        if web_dist is not None:
            web_target = base / "web"
            if not web_target.resolve().is_relative_to(base.resolve()):
                raise ValueError("AEON web staging path escaped the release.")
            shutil.rmtree(web_target)
            shutil.copytree(web_dist, web_target)
        artifacts = base / "artifacts"
        catalog_path = artifacts / "catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if (
            catalog.get("release_class") != "OFFLINE_RESEARCH_ENGINEERING_ONLY"
            or catalog.get("forecasts") != {}
            or "aeon-study" in catalog.get("artifacts", {})
            or "aeon3_geb_2024_hourly_sv_v1" in catalog.get("studies", [])
        ):
            raise ValueError("AEON base release has unexpected historical contract.")
        study_path = artifacts / "aeon-study.json"
        study_path.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        catalog["artifacts"]["aeon-study"] = {
            "path": study_path.name, "sha256": _sha256(study_path), "kind": "aeon-study",
        }
        if replay is not None:
            replay_path = artifacts / "aeon-test-replay.json"
            replay_path.write_text(json.dumps(replay, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            catalog["artifacts"]["aeon-test-replay"] = {
                "path": replay_path.name, "sha256": _sha256(replay_path),
                "kind": "aeon-test-replay",
            }
        if external_transfer:
            source_files = reviewed_external_files(root)
            pinned = study["external_transfer"]["provenance_sha256"]
            if set(source_files) != set(pinned):
                raise ValueError("AEON external packaged provenance inventory differs from review.")
            for name, source in source_files.items():
                if _sha256(source) != pinned[name]:
                    raise ValueError("AEON external packaged provenance differs from reviewed source.")
                destination = base / "provenance/external_transfer" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if _sha256(destination) != pinned[name]:
                    raise ValueError("AEON external packaged provenance differs from reviewed source.")
        for key, directory, inventory in (
            ("scaling_development", scale_output, reviewed_scale_provenance),
            ("expanded_train_development", expanded_output, reviewed_expanded_provenance),
        ):
            if directory is None:
                continue
            sources = inventory(root, directory)
            pinned = study[key]["provenance_sha256"]
            if set(sources) != set(pinned):
                raise ValueError(f"AEON {key} provenance inventory differs from review.")
            for name, source in sources.items():
                if _sha256(source) != pinned[name]:
                    raise ValueError(f"AEON {key} provenance digest differs from review: {name}")
                destination = base / "provenance" / key / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if _sha256(destination) != pinned[name]:
                    raise ValueError(f"AEON {key} packaged provenance differs: {name}")
        catalog["studies"] = ["mosaic_v1_v2_historical", study["study_id"]]
        if external_transfer:
            catalog["studies"].append(study["external_transfer"]["study_id"])
        for key in ("scaling_development", "expanded_train_development"):
            if key in study:
                catalog["studies"].append(study[key]["study_id"])
        catalog["release_class"] = (
            "OFFLINE_RESEARCH_MIXED_STUDIES_TEST_REVIEWED_RELEASE_REVIEW_PENDING"
            if replay is not None else "OFFLINE_RESEARCH_MIXED_STUDIES_DEVELOPMENT_ONLY"
        )
        for path in (catalog_path, base / "catalog.json"):
            path.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        packaged_files = sorted(
            (path for path in base.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(base).as_posix(),
        )
        asset_hashes = {
            path.relative_to(base).as_posix(): _sha256(path) for path in packaged_files
        }
        (base / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in asset_hashes.items()),
            encoding="utf-8",
        )
        base.rename(output)
        return {
            "status": catalog["release_class"], "output": str(output),
            "aeon_study_sha256": _sha256(output / "artifacts/aeon-study.json"),
            "packaged_asset_count": len(asset_hashes),
            "historical_registry_rows": len(catalog["experiments"]),
            "aeon_cached_forecasts": study["cached_forecasts"],
        }
    finally:
        if stage.exists():
            resolved = stage.resolve()
            if not resolved.is_relative_to(output.parent.resolve()):
                raise ValueError("AEON release staging path escaped output parent.")
            shutil.rmtree(resolved)
