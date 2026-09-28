"""Separate, reviewed AEON long-schedule TRAIN/validation development study.

This module reuses the original campaign's hash-bound TRAIN/validation reader,
model, optimizer and metric code. It cannot open CAL or TEST outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.training import aeon_campaign
from marine_echo.training.aeon_corpus import AEON_SOURCE_SHA256
from marine_echo.training.aeon_development import _save_predictions, _sha256, _verify_and_score
from marine_echo.training.aeon_rescore import _date_hashes


_STUDY = "aeon3_geb_2024_hourly_sv_scale_30k_development_v1"


def validate_config(config: dict[str, Any]) -> None:
    """Reject any quiet change to source, scope, schedule or resource ceiling."""
    n = config.get("neural", {})
    p = config.get("device_policy", {})
    rule = config.get("stage_2_rule", {})
    fixed = {
        "study_id": _STUDY,
        "parent_study_id": "aeon3_geb_2024_hourly_sv_v1",
        "classification": "POST_HOC_TRAIN_VALIDATION_DEVELOPMENT_NOT_FINAL_EVALUATION",
        "protocol_path": "docs/adr/0013-aeon-scaling-development.md",
        "source_sha256": AEON_SOURCE_SHA256,
        "fit_partition": "train",
        "assessment_partition": "validation",
        "calibration_access": "PROHIBITED",
        "test_access": "PROHIBITED",
        "families": ["direct", "ema_jepa"],
        "seeds_stage_1": [7],
        "seeds_stage_2": [13, 23],
        "horizon_interval_steps": [1, 3, 6],
        "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        "primary_metric": "daily_mean_pinball_db_equal_quantiles_horizons_source_dates",
        "fifty_thousand_update_extension": "NOT_AUTHORIZED_BY_THIS_CONFIG",
    }
    neural = {
        "encoder_width": 128, "encoder_layers": 3, "batch_size": 64,
        "total_optimizer_updates_per_run": 30000,
        "direct_supervised_updates": 30000,
        "ssl_pretrain_updates": 15000,
        "ssl_supervised_updates": 15000,
        "checkpoint_every_updates": 2500,
        "checkpoint_selection": "final_endpoint_only_intermediate_checks_diagnostic",
        "learning_rate": 0.0003, "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0, "ema_teacher_momentum": 0.996,
        "ema_sigreg_weight": 0.03,
        "pretrain_view": "first_18_vs_last_6_of_24_past_source_products",
        "normalizer": "train_only_observed_values_and_targets",
    }
    policy = {
        "preferred": "cuda", "minimum_free_gpu_bytes": 2 * 1024**3,
        "fallback": "cpu", "peak_process_rss_limit_bytes": 22 * 1024**3,
        "peak_gpu_reserved_limit_bytes": 10 * 1024**3,
        "one_training_process": True,
    }
    stage_rule = {
        "minimum_relative_improvement_vs_same_family_original_3k_seed7": 0.01,
        "if_either_family_qualifies_run_both_families_remaining_seeds": True,
    }
    protocol_path = Path(__file__).resolve().parents[3] / "docs/adr/0013-aeon-scaling-development.md"
    if (
        any(config.get(key) != value for key, value in fixed.items())
        or config.get("protocol_sha256") != _sha256(protocol_path)
        or n != neural or p != policy or rule != stage_rule
        or config.get("conventional") != {"max_threads": 4}
    ):
        raise ValueError("AEON scale contract differs from the bounded development plan.")


def stage_slots(stage: int) -> tuple[aeon_campaign.CampaignSlot, ...]:
    if stage not in (1, 2):
        raise ValueError("AEON scale stage must be 1 or 2.")
    seeds = (7,) if stage == 1 else (13, 23)
    return tuple(
        aeon_campaign.CampaignSlot(f"{family}_seed{seed}", family, seed)
        for seed in seeds for family in ("direct", "ema_jepa")
    )


def validate_validation_checks(
    checks: list[dict[str, Any]], *, supervised_updates: int
) -> None:
    every = 2500
    expected = list(range(every, supervised_updates + 1, every))
    if [item.get("step") for item in checks] != expected or expected[-1] != supervised_updates:
        raise ValueError("AEON scale validation checks lack a scheduled endpoint.")
    scores = [item.get("primary_daily_mean_pinball_db") for item in checks]
    if any(not isinstance(value, (float, int)) or not np.isfinite(value) for value in scores):
        raise ValueError("AEON scale validation primary score is invalid.")
    # All scheduled checkpoints are retained; only the final endpoint is selected.


def authorize_stage_two(old: dict[str, float], new: dict[str, dict[str, Any]]) -> bool:
    """Fixed 1% protocol-valid endpoint gain; run both families if either qualifies."""
    if set(old) != {"direct", "ema_jepa"} or set(new) != set(old):
        raise ValueError("AEON scale stage-2 comparison lacks a family.")
    qualifying = []
    for family in ("direct", "ema_jepa"):
        prior = old[family]
        current = new[family]["protocol_validation_metrics"]["primary_daily_mean_pinball_db"]
        if not (np.isfinite(prior) and prior > 0 and np.isfinite(current)):
            raise ValueError("AEON scale stage-2 score is invalid.")
        qualifying.append(current <= prior * 0.99)
    return any(qualifying)


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__), Path(daily_pinball.__code__.co_filename),
        Path(_date_hashes.__code__.co_filename),
    ):
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    digest.update(aeon_campaign._campaign_code_sha256().encode("ascii"))
    return digest.hexdigest()


def _baseline_scores(
    root: Path, rescore_path: Path, rescore_review_path: Path, review: dict[str, Any]
) -> tuple[dict[str, float], list[str]]:
    manifest_path = root / "manifest.json"
    if _sha256(manifest_path) != review.get("baseline_manifest_sha256"):
        raise ValueError("Original 3k campaign manifest differs from scale review.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("study_id") != "aeon3_geb_2024_hourly_sv_v1" or manifest.get("status") != "COMPLETED_TRAIN_VALIDATION_CAMPAIGN_NOT_FINAL_EVALUATION":
        raise ValueError("Original 3k campaign is incomplete or from another study.")
    if (
        _sha256(rescore_path) != review.get("baseline_rescore_sha256")
        or _sha256(rescore_review_path) != review.get("baseline_rescore_outcome_review_sha256")
    ):
        raise ValueError("Corrected 3k baseline evidence differs from scale review.")
    corrected = json.loads(rescore_path.read_text(encoding="utf-8"))
    outcome_review = json.loads(rescore_review_path.read_text(encoding="utf-8"))
    if (
        corrected.get("status") != "PROTOCOL_VALIDATION_RESCORE_PENDING_INDEPENDENT_SELECTION_REVIEW"
        or corrected.get("campaign_manifest_sha256") != _sha256(manifest_path)
        or outcome_review.get("status") != "APPROVED_AEON_VALIDATION_RESCORE_OUTCOME_NO_SELECTION"
        or outcome_review.get("artifact_sha256", {}).get("rescore_report") != _sha256(rescore_path)
    ):
        raise ValueError("Corrected 3k baseline lacks its independent outcome review.")
    scores = {}
    for family in ("direct", "ema_jepa"):
        run_id = f"{family}_seed7"
        entry = manifest["slots"][run_id]
        aeon_campaign._verify_done_slot(root, entry)
        original = json.loads((root / run_id / "slot.json").read_text(encoding="utf-8"))
        if corrected["slots"][run_id]["prediction_sha256"] != original["prediction_sha256"]:
            raise ValueError("Corrected 3k score differs from original prediction artifact.")
        score = corrected["slots"][run_id]["protocol_validation_metrics"]["primary_daily_mean_pinball_db"]
        scores[family] = float(score)
    return scores, corrected["eligible_source_date_sha256_by_horizon"]


def _execute_neural(
    slot: aeon_campaign.CampaignSlot, fit: list, assess: list,
    config: dict[str, Any], stage: Path, device: str,
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    final_prediction, final_checkpoint, details = aeon_campaign._neural_prediction(
        slot, fit, assess, config, stage, device
    )
    updates = config["neural"]["direct_supervised_updates" if slot.family == "direct" else "ssl_supervised_updates"]
    raw_checks = details.pop("validation_checks")
    validate_validation_checks(raw_checks, supervised_updates=updates)
    checks = []
    reference_hashes = None
    for item in raw_checks:
        path = stage / item["path"]
        if _sha256(path) != item["sha256"]:
            raise ValueError("AEON scale scheduled validation artifact digest differs.")
        score, date_hashes = _protocol_score_prediction(path, assess)
        if reference_hashes is None:
            reference_hashes = date_hashes
        elif date_hashes != reference_hashes:
            raise ValueError("AEON scale validation date support changed between checkpoints.")
        checks.append({
            "step": item["step"], "path": item["path"], "sha256": item["sha256"],
            "protocol_primary_daily_mean_pinball_db": score["primary_daily_mean_pinball_db"],
            "raw_all_scored_day_primary_non_protocol_diagnostic": item["primary_daily_mean_pinball_db"],
        })
    with np.load(stage / checks[-1]["path"], allow_pickle=False) as stored:
        if not np.array_equal(final_prediction, stored["quantiles_db"]):
            raise ValueError("AEON scale final forecast differs from final saved validation check.")
    matching = [item for item in details["checkpoints"] if item["phase"] == "supervised" and item["step"] == updates]
    if len(matching) != 1 or final_checkpoint != stage / matching[0]["path"]:
        raise ValueError("AEON scale final checkpoint is absent or ambiguous.")
    details.update(
        selection="FINAL_ENDPOINT_ONLY_INTERMEDIATE_CHECKS_DIAGNOSTIC",
        protocol_validation_checks=checks,
        eligible_source_date_sha256_by_horizon=reference_hashes,
        final_endpoint_step=updates,
    )
    return final_prediction, final_checkpoint, details


def _protocol_score_prediction(path: Path, assess: list) -> tuple[dict[str, Any], list[str]]:
    # The old scorer validates saved identities but includes sub-18-anchor days;
    # its metric is discarded. The corrected evaluator supplies every decision score.
    _verify_and_score(path, assess)
    with np.load(path, allow_pickle=False) as stored:
        score = daily_pinball(
            stored["truth_db"], stored["quantiles_db"],
            stored["target_mask"], stored["target_source_timestamps"],
        )
    return score, _date_hashes(score)


def _save_scale_slot_result(
    stage: Path, slot: aeon_campaign.CampaignSlot, assess: list,
    prediction: np.ndarray, model: Path, details: dict[str, Any],
) -> dict[str, Any]:
    prediction_path = stage / "validation-predictions.npz"
    _save_predictions(prediction_path, assess, prediction)
    score, date_hashes = _protocol_score_prediction(prediction_path, assess)
    if date_hashes != details["eligible_source_date_sha256_by_horizon"]:
        raise ValueError("AEON scale final prediction uses a different eligible-date cohort.")
    if score["primary_daily_mean_pinball_db"] != details["protocol_validation_checks"][-1]["protocol_primary_daily_mean_pinball_db"]:
        raise ValueError("AEON scale final protocol metric differs from scheduled endpoint.")
    result = {
        "run_id": slot.run_id, "family": slot.family, "seed": slot.seed,
        "prediction_path": prediction_path.name, "prediction_sha256": _sha256(prediction_path),
        "model_path": model.name, "model_sha256": _sha256(model),
        "protocol_validation_metrics": score,
        "primary_daily_mean_pinball_db": score["primary_daily_mean_pinball_db"],
        "eligible_source_date_sha256_by_horizon": date_hashes,
        **details,
    }
    (stage / "slot.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _verify_scale_done_slot(output: Path, entry: dict[str, Any]) -> dict[str, Any]:
    aeon_campaign._verify_done_slot(output, entry)
    directory = output / entry["run_id"]
    result = json.loads((directory / "slot.json").read_text(encoding="utf-8"))
    for item in result["protocol_validation_checks"]:
        relative = Path(item["path"])
        if relative.is_absolute() or len(relative.parts) != 1 or _sha256(directory / relative) != item["sha256"]:
            raise ValueError("AEON scale saved protocol validation artifact differs.")
    return result


def run_scale(
    archive: Path, split_review: Path, scale_review: Path, config_path: Path,
    baseline_output: Path, baseline_rescore: Path, baseline_rescore_review: Path,
    output: Path, *, stage: int,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    config_sha = _sha256(config_path)
    fit, assess, cohort_sha, split_sha = aeon_campaign.load_cohort(archive, split_review)
    review = json.loads(scale_review.read_text(encoding="utf-8"))
    if any(review.get(key) != value for key, value in {
        "status": "APPROVED_AEON_SCALE_TRAIN_VALIDATION_ONLY",
        "config_sha256": config_sha,
        "code_sha256": _code_sha256(),
        "cohort_sha256": cohort_sha,
        "protocol_sha256": config["protocol_sha256"],
        "source_sha256": AEON_SOURCE_SHA256,
        "split_review_sha256": split_sha,
        "test_access": "PROHIBITED",
        "calibration_access": "PROHIBITED",
    }.items()):
        raise ValueError("AEON scale lacks exact independent prefit approval.")
    baseline, baseline_dates = _baseline_scores(
        baseline_output, baseline_rescore, baseline_rescore_review, review
    )
    device, _ = aeon_campaign._resolve_device(config)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    identity = {
        "study_id": _STUDY, "config_sha256": config_sha,
        "code_sha256": _code_sha256(), "cohort_sha256": cohort_sha,
        "review_sha256": _sha256(scale_review), "baseline_manifest_sha256": review["baseline_manifest_sha256"],
        "baseline_rescore_sha256": review["baseline_rescore_sha256"],
        "device": device,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("AEON scale resume identity differs.")
    else:
        manifest = {**identity, "status": "STAGE_1_PENDING", "slots": {}}
        aeon_campaign._atomic_json(manifest_path, manifest)
    slots = manifest["slots"]
    if stage == 2:
        if any(slots.get(slot.run_id, {}).get("status") != "DONE" for slot in stage_slots(1)):
            raise ValueError("AEON scale stage 1 is incomplete.")
        first = {}
        for slot in stage_slots(1):
            first[slot.family] = _verify_scale_done_slot(output, slots[slot.run_id])
        if not authorize_stage_two(baseline, first):
            manifest["status"] = "STAGE_2_NOT_AUTHORIZED_BY_PREDECLARED_RULE"
            aeon_campaign._atomic_json(manifest_path, manifest)
            return manifest
    for slot in stage_slots(stage):
        prior = slots.get(slot.run_id)
        if prior is not None and prior.get("status") == "DONE":
            _verify_scale_done_slot(output, prior)
            continue
        if (output / slot.run_id).exists():
            raise ValueError("AEON scale slot exists without a completed ledger entry.")
        temporary = Path(tempfile.mkdtemp(prefix=slot.run_id + ".stage.", dir=output))
        try:
            prediction, model, details = _execute_neural(slot, fit, assess, config, temporary, device)
            if details["eligible_source_date_sha256_by_horizon"] != baseline_dates:
                raise ValueError("AEON scale eligible-date support differs from corrected 3k campaign.")
            _save_scale_slot_result(temporary, slot, assess, prediction, model, details)
            os.replace(temporary, output / slot.run_id)
            slots[slot.run_id] = {
                "status": "DONE", "run_id": slot.run_id,
                "slot_sha256": _sha256(output / slot.run_id / "slot.json"),
            }
            aeon_campaign._atomic_json(manifest_path, manifest)
        except BaseException:
            if temporary.exists() and temporary.parent == output:
                shutil.rmtree(temporary)
            slots[slot.run_id] = {"status": "FAILED", "run_id": slot.run_id}
            aeon_campaign._atomic_json(manifest_path, manifest)
            raise
    manifest["status"] = "STAGE_1_COMPLETE_PENDING_RULE" if stage == 1 else "STAGE_2_COMPLETE_DEVELOPMENT_ONLY"
    aeon_campaign._atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "archive", "split-review", "scale-review", "config", "baseline-output",
        "baseline-rescore", "baseline-rescore-review", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    result = run_scale(
        args.archive, args.split_review, args.scale_review, args.config,
        args.baseline_output, args.baseline_rescore, args.baseline_rescore_review,
        args.output, stage=args.stage,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
