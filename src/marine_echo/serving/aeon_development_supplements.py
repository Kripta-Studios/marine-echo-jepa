"""Build-time binding for the independently reviewed AEON 30k development result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

_STUDY = "aeon3_geb_2024_hourly_sv_scale_30k_development_v1"
_CLASSIFICATION = "POST_HOC_TRAIN_VALIDATION_DEVELOPMENT_NOT_FINAL_EVALUATION"
_REVIEW_NAME = "AEON_SCALE_30K_OUTCOME_REVIEW_20260928.json"
_REVIEW_SHA256 = "6d55cdff9801619f3be1acc8fb18380bf4946f966163f4a7b4b2a269ad11252a"
_SLOTS = ("direct_seed7", "ema_jepa_seed7")
_CONFIG = "configs/aeon_scale_30k.json"
_PROTOCOL = "docs/adr/0013-aeon-scaling-development.md"
_EXPANDED_STUDY = "aeon3_geb_expanded_train_3k_development_v1"
_EXPANDED_CLASSIFICATION = "POST_HOC_TRAIN_VALIDATION_DEVELOPMENT_NOT_EXTERNAL_EVALUATION"
_EXPANDED_REVIEW_NAME = "AEON_EXPANDED_3K_OUTCOME_REVIEW_20260928.json"
_EXPANDED_REVIEW_SHA256 = "3ba0452b63496c448ab51799a76d296d4b363abd4771953a1a776d925799e9f0"
_EXPANDED_CONFIG = "configs/aeon_expanded_3k.json"
_EXPANDED_PROTOCOL = "docs/adr/0014-aeon-expanded-data-development.md"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bounded(root: Path, relative: str, max_bytes: int) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("AEON scale evidence path is unsafe.")
    path = root / candidate
    if (
        path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
        or path.stat().st_size > max_bytes
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"AEON scale evidence path is unsafe: {relative}")
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(
        ValueError("AEON scale evidence contains nonfinite JSON.")
    ))
    if not isinstance(value, dict):
        raise ValueError("AEON scale evidence must be a JSON object.")
    return value


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("AEON scale reviewed score is not finite.")
    return float(value)


def reviewed_scale_provenance(root: Path, output: Path) -> dict[str, Path]:
    """Return reviewed records and final binaries; never include interim checkpoints."""
    return {
        "outcome-review.json": _bounded(root, f"orchestration/reviews/{_REVIEW_NAME}", 128_000),
        "config.json": _bounded(root, _CONFIG, 128_000),
        "protocol.md": _bounded(root, _PROTOCOL, 128_000),
        "manifest.json": _bounded(output, "manifest.json", 128_000),
        **{f"{slot}/slot.json": _bounded(output, f"{slot}/slot.json", 256_000) for slot in _SLOTS},
        **{
            f"{slot}/validation-predictions.npz": _bounded(
                output, f"{slot}/validation-predictions.npz", 8_000_000
            ) for slot in _SLOTS
        },
        **{
            f"{slot}/checkpoint-supervised-{step}.pt": _bounded(
                output, f"{slot}/checkpoint-supervised-{step}.pt", 8_000_000
            ) for slot, step in (("direct_seed7", 30000), ("ema_jepa_seed7", 15000))
        },
    }


def load_reviewed_scale(root: Path, output: Path) -> dict[str, Any]:
    """Validate reviewed endpoint identities and emit a compact offline-safe summary."""
    files = reviewed_scale_provenance(root, output)
    if _sha256(files["outcome-review.json"]) != _REVIEW_SHA256:
        raise ValueError("AEON scale outcome review digest differs from pinned approval.")
    review = _json(files["outcome-review.json"])
    if _sha256(files["manifest.json"]) != review.get("manifest_sha256"):
        raise ValueError("AEON scale manifest digest differs from reviewed evidence.")
    config = _json(files["config.json"])
    manifest = _json(files["manifest.json"])
    if (
        review.get("study_id") != _STUDY
        or review.get("classification") != _CLASSIFICATION
        or review.get("status") != "APPROVED_AEON_SCALE_30K_STAGE1_OUTCOME_STAGE2_NOT_AUTHORIZED"
        or review.get("reviewer_session") != "/root/scale_reviewer"
        or review.get("calibration_access") != "PROHIBITED"
        or review.get("test_access") != "PROHIBITED"
        or review.get("stage_2") != "NOT_RUN_BY_PREDECLARED_ONE_PERCENT_GATE"
        or review.get("fifty_thousand_updates") != "NOT_AUTHORIZED_BY_THIS_REVIEW"
        or review.get("manifest_status") != "STAGE_2_NOT_AUTHORIZED_BY_PREDECLARED_RULE"
        or _sha256(files["manifest.json"]) != review.get("manifest_sha256")
        or _sha256(files["config.json"]) != review.get("config_sha256")
        or _sha256(files["protocol.md"]) != config.get("protocol_sha256")
        or config.get("study_id") != _STUDY
        or config.get("classification") != _CLASSIFICATION
        or config.get("fit_partition") != "train"
        or config.get("assessment_partition") != "validation"
        or config.get("calibration_access") != "PROHIBITED"
        or config.get("test_access") != "PROHIBITED"
        or manifest.get("study_id") != _STUDY
        or manifest.get("status") != review.get("manifest_status")
        or manifest.get("config_sha256") != review.get("config_sha256")
        or manifest.get("code_sha256") != review.get("code_sha256")
        or manifest.get("cohort_sha256") != review.get("cohort_sha256")
        or manifest.get("review_sha256") != review.get("prefit_review_sha256")
        or set(manifest.get("slots", {})) != set(_SLOTS)
        or review.get("validation_rows") != 1219
        or review.get("eligible_validation_rows_by_horizon") != [1194, 1192, 1189]
        or review.get("eligible_validation_dates_by_horizon") != [50, 50, 50]
        or review.get("eligible_date_sha256_by_horizon") != [
            "5d26e8b41caa05ad7189fc1a402b977ff368a7c13b9ab4fecb1694f2f6c10ae1"
        ] * 3
    ):
        raise ValueError("AEON scale review, protocol, manifest or support differs.")
    summaries = {}
    for name in _SLOTS:
        slot_path = files[f"{name}/slot.json"]
        reviewed = review["slots"][name]
        entry = manifest["slots"][name]
        if _sha256(slot_path) != reviewed["slot_sha256"]:
            raise ValueError(f"AEON scale slot digest differs: {name}")
        slot = _json(slot_path)
        family = name.removesuffix("_seed7")
        step = 30000 if family == "direct" else 15000
        metrics = slot.get("protocol_validation_metrics", {})
        checks = slot.get("protocol_validation_checks", [])
        checkpoints = slot.get("checkpoints", [])
        if (
            entry != {"status": "DONE", "run_id": name, "slot_sha256": reviewed["slot_sha256"]}
            or _sha256(slot_path) != reviewed["slot_sha256"]
            or slot.get("run_id") != name or slot.get("family") != family or slot.get("seed") != 7
            or slot.get("selection") != "FINAL_ENDPOINT_ONLY_INTERMEDIATE_CHECKS_DIAGNOSTIC"
            or slot.get("final_endpoint_step") != step
            or slot.get("prediction_path") != "validation-predictions.npz"
            or slot.get("model_path") != f"checkpoint-supervised-{step}.pt"
            or slot.get("prediction_sha256") != reviewed["prediction_sha256"]
            or slot.get("model_sha256") != reviewed["endpoint_model_sha256"]
            or metrics.get("issued_rows") != review["validation_rows"]
            or metrics.get("eligible_scored_rows_per_horizon") != review["eligible_validation_rows_by_horizon"]
            or metrics.get("eligible_days_per_horizon") != review["eligible_validation_dates_by_horizon"]
            or slot.get("eligible_source_date_sha256_by_horizon") != review["eligible_date_sha256_by_horizon"]
            or _finite(metrics.get("primary_daily_mean_pinball_db"))
            != _finite(reviewed["corrected_final_daily_mean_pinball_db"])
            or _finite(slot.get("primary_daily_mean_pinball_db"))
            != _finite(reviewed["corrected_final_daily_mean_pinball_db"])
            or not isinstance(checks, list) or len(checks) != step // 2500
            or checks[-1].get("step") != step
            or checks[-1].get("sha256") != reviewed["prediction_sha256"]
            or _finite(checks[-1].get("protocol_primary_daily_mean_pinball_db"))
            != _finite(reviewed["corrected_final_daily_mean_pinball_db"])
            or not isinstance(checkpoints, list)
            or not any(item.get("phase") == "supervised" and item.get("step") == step
                       and item.get("sha256") == reviewed["endpoint_model_sha256"] for item in checkpoints)
        ):
            raise ValueError(f"AEON scale reviewed slot differs: {name}")
        prediction = _bounded(output, f"{name}/{slot['prediction_path']}", 8_000_000)
        model = _bounded(output, f"{name}/{slot['model_path']}", 8_000_000)
        if _sha256(prediction) != reviewed["prediction_sha256"] or _sha256(model) != reviewed["endpoint_model_sha256"]:
            raise ValueError(f"AEON scale endpoint artifact digest differs: {name}")
        final = _finite(reviewed["corrected_final_daily_mean_pinball_db"])
        original = _finite(reviewed["original_3k_corrected_daily_mean_pinball_db"])
        if not (final > original > 0):
            raise ValueError("AEON scale negative comparison differs from reviewed outcome.")
        summaries[name] = {
            "family": family, "seed": 7,
            "final_pinball_db": final, "original_3k_pinball_db": original,
            "relative_loss_change": final / original - 1,
            "supervised_updates": reviewed["supervised_updates"],
            "ssl_pretrain_updates": reviewed.get("ssl_pretrain_updates", 0),
            "slot_sha256": reviewed["slot_sha256"],
            "prediction_sha256": reviewed["prediction_sha256"],
            "endpoint_model_sha256": reviewed["endpoint_model_sha256"],
        }
    provenance = {name: _sha256(path) for name, path in files.items()}
    return {
        "study_id": _STUDY,
        "status": "INDEPENDENTLY_REVIEWED_30K_STAGE1_NEGATIVE",
        "classification": _CLASSIFICATION,
        "assessment_partition": "validation", "final_evaluation": False,
        "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
        "validation_rows": 1219,
        "eligible_rows_per_horizon": review["eligible_validation_rows_by_horizon"],
        "eligible_dates_per_horizon": review["eligible_validation_dates_by_horizon"],
        "eligible_date_sha256_by_horizon": review["eligible_date_sha256_by_horizon"],
        "slots": summaries,
        "stage_2": review["stage_2"],
        "fifty_thousand_updates": review["fifty_thousand_updates"],
        "selection": "FIXED_FINAL_ENDPOINT_ONLY",
        "outcome_review_sha256": _REVIEW_SHA256,
        "manifest_sha256": review["manifest_sha256"],
        "config_sha256": review["config_sha256"],
        "protocol_sha256": config["protocol_sha256"],
        "cohort_sha256": review["cohort_sha256"],
        "provenance_sha256": provenance,
    }


def reviewed_expanded_provenance(root: Path, output: Path) -> dict[str, Path]:
    """Return reviewed pooled-TRAIN records and final binaries only."""
    return {
        "outcome-review.json": _bounded(root, f"orchestration/reviews/{_EXPANDED_REVIEW_NAME}", 128_000),
        "config.json": _bounded(root, _EXPANDED_CONFIG, 128_000),
        "protocol.md": _bounded(root, _EXPANDED_PROTOCOL, 128_000),
        "cohort.json": _bounded(output, "cohort.json", 128_000),
        "manifest.json": _bounded(output, "manifest.json", 128_000),
        **{f"{slot}/slot.json": _bounded(output, f"{slot}/slot.json", 256_000) for slot in _SLOTS},
        **{
            f"{slot}/validation-predictions.npz": _bounded(
                output, f"{slot}/validation-predictions.npz", 8_000_000
            ) for slot in _SLOTS
        },
        **{
            f"{slot}/checkpoint-supervised-{step}.pt": _bounded(
                output, f"{slot}/checkpoint-supervised-{step}.pt", 8_000_000
            ) for slot, step in (("direct_seed7", 3000), ("ema_jepa_seed7", 1500))
        },
    }


def load_reviewed_expanded(root: Path, output: Path) -> dict[str, Any]:
    """Bind pooled-TRAIN development to its distinct reviewed exact-byte outcome."""
    files = reviewed_expanded_provenance(root, output)
    if _sha256(files["outcome-review.json"]) != _EXPANDED_REVIEW_SHA256:
        raise ValueError("AEON expanded outcome review digest differs from pinned approval.")
    review = _json(files["outcome-review.json"])
    if (
        _sha256(files["manifest.json"]) != review.get("manifest_sha256")
        or _sha256(files["cohort.json"]) != review.get("cohort_report_sha256")
    ):
        raise ValueError("AEON expanded manifest or cohort digest differs from review.")
    config = _json(files["config.json"])
    manifest = _json(files["manifest.json"])
    cohort = _json(files["cohort.json"])
    support_hashes = ["5d26e8b41caa05ad7189fc1a402b977ff368a7c13b9ab4fecb1694f2f6c10ae1"] * 3
    counts = cohort.get("counts", {})
    if (
        review.get("study_id") != _EXPANDED_STUDY
        or review.get("classification") != _EXPANDED_CLASSIFICATION
        or review.get("status") != "APPROVED_AEON_EXPANDED_3K_OUTCOME_POST_HOC_DEVELOPMENT_ONLY"
        or review.get("verdict") != "APPROVE_ARTIFACTS_AND_CORRECTED_METRICS_NO_CAUSAL_DATA_VOLUME_OR_EXTERNAL_CLAIM"
        or review.get("reviewer_session") != "/root/scale_reviewer"
        or review.get("calibration_access") != "PROHIBITED"
        or review.get("test_access") != "PROHIBITED"
        or review.get("manifest_status") != "COMPLETED_POST_HOC_TRAIN_VALIDATION_PENDING_INDEPENDENT_OUTCOME_REVIEW"
        or review.get("joint_train_windows") != 13472
        or review.get("prior_train_windows") != 8507
        or review.get("current_train_windows") != 4965
        or review.get("validation_windows") != 1219
        or review.get("eligible_validation_rows_by_horizon") != [1194, 1192, 1189]
        or review.get("eligible_validation_dates_by_horizon") != [50, 50, 50]
        or review.get("eligible_date_sha256_by_horizon") != support_hashes
        or _sha256(files["config.json"]) != review.get("config_sha256")
        or _sha256(files["protocol.md"]) != review.get("protocol_sha256")
        or config.get("study_id") != _EXPANDED_STUDY
        or config.get("classification") != _EXPANDED_CLASSIFICATION
        or config.get("protocol_sha256") != review.get("protocol_sha256")
        or config.get("prior_source", {}).get("archive_sha256") != cohort.get("prior_archive_sha256")
        or config.get("current_source", {}).get("archive_sha256") != cohort.get("current_archive_sha256")
        or config.get("fit_partitions") != ["prior_reclassified_train", "original_train"]
        or config.get("assessment_partition") != "original_validation"
        or config.get("calibration_access") != "PROHIBITED"
        or config.get("test_access") != "PROHIBITED"
        or config.get("sampling", {}).get("source_policy") != "natural_pooled_window_proportions"
        or cohort.get("study_id") != _EXPANDED_STUDY
        or cohort.get("classification") != _EXPANDED_CLASSIFICATION
        or cohort.get("cohort_sha256") != review.get("cohort_sha256")
        or cohort.get("config_sha256") != review.get("config_sha256")
        or cohort.get("code_sha256") != review.get("code_sha256")
        or cohort.get("protocol_sha256") != review.get("protocol_sha256")
        or cohort.get("calibration_access") != "PROHIBITED"
        or cohort.get("test_access") != "PROHIBITED"
        or any(counts.get(source) != review.get(source) for source in (
            "prior_train_windows", "current_train_windows", "joint_train_windows"
        ))
        or counts.get("validation_windows") != review.get("validation_windows")
        or counts.get("validation_source_dates_with_18_scored_anchors_by_horizon") != [50, 50, 50]
        or manifest.get("study_id") != _EXPANDED_STUDY
        or manifest.get("status") != review.get("manifest_status")
        or manifest.get("config_sha256") != review.get("config_sha256")
        or manifest.get("code_sha256") != review.get("code_sha256")
        or manifest.get("cohort_sha256") != review.get("cohort_sha256")
        or manifest.get("cohort_report_sha256") != review.get("cohort_report_sha256")
        or manifest.get("train_review_sha256") != review.get("prefit_review_sha256")
        or manifest.get("eligible_source_date_sha256_by_horizon") != support_hashes
        or set(manifest.get("slots", {})) != set(_SLOTS)
    ):
        raise ValueError("AEON expanded review, cohort, protocol or manifest differs.")
    summaries = {}
    for name in _SLOTS:
        reviewed = review["slots"][name]
        slot_path = files[f"{name}/slot.json"]
        if _sha256(slot_path) != reviewed["slot_sha256"]:
            raise ValueError(f"AEON expanded slot digest differs: {name}")
        slot = _json(slot_path)
        entry = manifest["slots"][name]
        family = name.removesuffix("_seed7")
        step = 3000 if family == "direct" else 1500
        metrics = slot.get("protocol_validation_metrics", {})
        checks = slot.get("validation_checks", [])
        checkpoints = slot.get("checkpoints", [])
        if (
            entry != {
                "status": "DONE", "run_id": name,
                "slot_sha256": reviewed["slot_sha256"],
                "primary_daily_mean_pinball_db": reviewed["corrected_final_daily_mean_pinball_db"],
            }
            or slot.get("study_id") != _EXPANDED_STUDY
            or slot.get("cohort_sha256") != review.get("cohort_sha256")
            or slot.get("run_id") != name or slot.get("family") != family or slot.get("seed") != 7
            or slot.get("selection") != "FINAL_ENDPOINT_ONLY"
            or slot.get("supervised_updates") != step
            or slot.get("pretrain_updates") != (0 if family == "direct" else 1500)
            or slot.get("prediction_path") != "validation-predictions.npz"
            or slot.get("model_path") != f"checkpoint-supervised-{step}.pt"
            or slot.get("prediction_sha256") != reviewed["prediction_sha256"]
            or slot.get("model_sha256") != reviewed["endpoint_checkpoint_sha256"]
            or metrics.get("issued_rows") != review["validation_windows"]
            or metrics.get("eligible_scored_rows_per_horizon") != review["eligible_validation_rows_by_horizon"]
            or metrics.get("eligible_days_per_horizon") != review["eligible_validation_dates_by_horizon"]
            or slot.get("eligible_source_date_sha256_by_horizon") != support_hashes
            or _finite(metrics.get("primary_daily_mean_pinball_db"))
            != _finite(reviewed["corrected_final_daily_mean_pinball_db"])
            or _finite(slot.get("primary_daily_mean_pinball_db"))
            != _finite(reviewed["corrected_final_daily_mean_pinball_db"])
            or slot.get("sampled_supervised_rows_by_source") != reviewed["sampled_supervised_rows_by_source"]
            or slot.get("scaler_joint_train_only") != review["joint_train_scaler"]
            or not isinstance(checks, list) or len(checks) != step // 500
            or checks[-1].get("step") != step
            or checks[-1].get("sha256") != reviewed["prediction_sha256"]
            or checks[-1].get("eligible_source_date_sha256_by_horizon") != support_hashes
            or _finite(checks[-1].get("protocol_primary_daily_mean_pinball_db"))
            != _finite(reviewed["corrected_final_daily_mean_pinball_db"])
            or not isinstance(checkpoints, list)
            or not any(item.get("phase") == "supervised" and item.get("step") == step
                       and item.get("sha256") == reviewed["endpoint_checkpoint_sha256"] for item in checkpoints)
            or (family == "ema_jepa" and (
                slot.get("sampled_pretrain_rows_by_source") != reviewed["sampled_pretrain_rows_by_source"]
            ))
        ):
            raise ValueError(f"AEON expanded reviewed slot differs: {name}")
        prediction = _bounded(output, f"{name}/{slot['prediction_path']}", 8_000_000)
        model = _bounded(output, f"{name}/{slot['model_path']}", 8_000_000)
        if _sha256(prediction) != reviewed["prediction_sha256"] or _sha256(model) != reviewed["endpoint_checkpoint_sha256"]:
            raise ValueError(f"AEON expanded endpoint artifact digest differs: {name}")
        final = _finite(reviewed["corrected_final_daily_mean_pinball_db"])
        original = _finite(reviewed["original_only_3k_corrected_daily_mean_pinball_db"])
        if not final > 0 or not original > 0:
            raise ValueError("AEON expanded reviewed score is not positive.")
        summaries[name] = {
            "family": family, "seed": 7,
            "final_pinball_db": final,
            "original_only_3k_pinball_db": original,
            "relative_loss_reduction": 1 - final / original,
            "supervised_updates": step,
            "ssl_pretrain_updates": slot["pretrain_updates"],
            "sampled_supervised_rows_by_source": reviewed["sampled_supervised_rows_by_source"],
            "slot_sha256": reviewed["slot_sha256"],
            "prediction_sha256": reviewed["prediction_sha256"],
            "endpoint_model_sha256": reviewed["endpoint_checkpoint_sha256"],
        }
    return {
        "study_id": _EXPANDED_STUDY,
        "status": "INDEPENDENTLY_REVIEWED_EXPANDED_TRAIN_3K_DEVELOPMENT",
        "classification": _EXPANDED_CLASSIFICATION,
        "assessment_partition": "original_validation", "final_evaluation": False,
        "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
        "joint_train_windows": review["joint_train_windows"],
        "prior_train_windows": review["prior_train_windows"],
        "current_train_windows": review["current_train_windows"],
        "validation_rows": review["validation_windows"],
        "eligible_rows_per_horizon": review["eligible_validation_rows_by_horizon"],
        "eligible_dates_per_horizon": review["eligible_validation_dates_by_horizon"],
        "eligible_date_sha256_by_horizon": support_hashes,
        "sampling": "NATURAL_POOLED_SOURCE_PROPORTIONS",
        "slots": summaries,
        "outcome_review_sha256": _EXPANDED_REVIEW_SHA256,
        "manifest_sha256": review["manifest_sha256"],
        "cohort_report_sha256": review["cohort_report_sha256"],
        "cohort_sha256": review["cohort_sha256"],
        "config_sha256": review["config_sha256"],
        "protocol_sha256": review["protocol_sha256"],
        "source_archive_sha256": {
            "prior": cohort["prior_archive_sha256"],
            "current": cohort["current_archive_sha256"],
        },
        "provenance_sha256": {name: _sha256(path) for name, path in files.items()},
    }
