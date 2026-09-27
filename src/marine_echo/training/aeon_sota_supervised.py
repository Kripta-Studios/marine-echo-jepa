"""One fixed post-hoc LightGBM AEON TRAIN/validation quantile challenger."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import lightgbm as lgb
import numpy as np
import psutil  # type: ignore[import-untyped]

from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.training.aeon_campaign import (
    QUANTILES,
    _campaign_code_sha256,
    _config_gate,
    _past_features,
    load_cohort,
)
from marine_echo.training.aeon_development import _save_predictions, _sha256
from marine_echo.training.aeon_rescore import (
    _artifact,
    _date_hashes,
    _load_prediction,
    _row_digest,
    _same_rows,
)
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _features(rows: list[AeonHourlyWindow]) -> np.ndarray:
    """Expose each permitted past value and mask alongside frozen B3 summaries."""
    if not rows:
        raise ValueError("AEON supervised challenger needs nonempty past windows.")
    values = np.stack([row.context_db for row in rows])
    mask = np.stack([row.context_mask for row in rows])
    if values.shape[1:] != (24, 4) or mask.shape != values.shape:
        raise ValueError("AEON supervised past geometry differs.")
    if not np.isfinite(values[mask]).all() or not mask[:, :, 0].all():
        raise ValueError("AEON supervised observed past values are invalid.")
    clean = np.where(mask, values, 0.0)
    features = np.concatenate(
        (_past_features(rows), clean.reshape(len(rows), -1), mask.reshape(len(rows), -1)), axis=1
    )
    if features.shape != (len(rows), 210) or not np.isfinite(features).all():
        raise ValueError("AEON supervised feature vector differs from the fixed recipe.")
    return features


def _recipe_gate(recipe: dict[str, Any]) -> None:
    expected_tree = {
        "implementation": "lightgbm", "version": "4.6.0", "heads": 15,
        "objective": "quantile", "n_estimators": 256, "max_depth": 4,
        "num_leaves": 15, "min_child_samples": 80, "learning_rate": 0.03,
        "colsample_bytree": 0.8, "reg_lambda": 2.0, "random_state": 7,
        "n_jobs": 4, "checkpoint_selection": "single_fixed_endpoint_no_validation_early_stopping",
    }
    if (
        recipe.get("schema_version") != "1.0"
        or recipe.get("study_id") != "aeon3_geb_2024_hourly_sv_v1"
        or recipe.get("phase") != "post_hoc_train_validation_sota_supervised_development"
        or recipe.get("status") != "PROPOSED_FOR_INDEPENDENT_PREFIT_REVIEW"
        or recipe.get("classification") != "POST_HOC_DEVELOPMENT_NOT_FINAL_EVALUATION"
        or recipe.get("fit_partition") != "train"
        or recipe.get("assessment_partition") != "validation"
        or recipe.get("calibration_access") != "PROHIBITED_IN_THIS_PHASE"
        or recipe.get("test_access") != "PROHIBITED_IN_THIS_PHASE"
        or recipe.get("horizon_interval_steps") != [1, 3, 6]
        or recipe.get("quantiles") != QUANTILES.tolist()
        or recipe.get("feature_dimensions") != 210
        or recipe.get("features")
        != "core_18_past_summaries_plus_all_96_past_values_missing_set_zero_plus_96_observed_masks"
        or recipe.get("tree") != expected_tree
        or recipe.get("selection") != "NONE_UNTIL_INDEPENDENT_RESULT_REVIEW"
        or recipe.get("resource_limits")
        != {"peak_process_rss_bytes": 22 * 1024**3,
            "peak_gpu_reserved_bytes": 10 * 1024**3, "one_training_process": True}
        or lgb.__version__ != "4.6.0"
    ):
        raise ValueError("AEON fixed post-hoc supervised recipe differs.")


def _fit_quantiles(
    x_train: np.ndarray,
    y_train: np.ndarray,
    observed: np.ndarray,
    x_assess: np.ndarray,
    tree: dict[str, Any],
) -> tuple[np.ndarray, list[list[lgb.LGBMRegressor]]]:
    if (
        x_train.ndim != 2 or x_train.shape[1] != 210
        or x_assess.ndim != 2 or x_assess.shape[1] != 210
        or y_train.shape != (len(x_train), 3)
        or observed.shape != y_train.shape
        or not np.isfinite(x_train).all() or not np.isfinite(x_assess).all()
        or not np.isfinite(y_train[observed]).all()
    ):
        raise ValueError("AEON supervised TRAIN/validation arrays differ.")
    predictions = np.empty((len(x_assess), 3, 5), dtype=np.float64)
    models: list[list[lgb.LGBMRegressor]] = []
    for horizon in range(3):
        valid = observed[:, horizon]
        if valid.sum() < tree["min_child_samples"] * 2:
            raise ValueError("AEON supervised TRAIN lacks enough observed labels.")
        heads = []
        for index, quantile in enumerate(QUANTILES):
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=float(quantile),
                n_estimators=tree["n_estimators"], max_depth=tree["max_depth"],
                num_leaves=tree["num_leaves"], min_child_samples=tree["min_child_samples"],
                learning_rate=tree["learning_rate"],
                colsample_bytree=tree["colsample_bytree"],
                reg_lambda=tree["reg_lambda"], random_state=tree["random_state"],
                n_jobs=tree["n_jobs"], verbosity=-1,
            )
            model.fit(x_train[valid], y_train[valid, horizon])
            predictions[:, horizon, index] = model.predict(x_assess)
            heads.append(model)
        models.append(heads)
    predictions.sort(axis=-1)
    return predictions, models


def _verified_b3_score(
    truth: np.ndarray,
    forecast: np.ndarray,
    observed: np.ndarray,
    source_times: np.ndarray,
    rescore: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the immutable raw-only reference with the unchanged evaluator."""
    if rescore.get("evaluation_code_sha256") != _sha256(Path(daily_pinball.__code__.co_filename)):
        raise ValueError("AEON supervised evaluator differs from the reviewed rescore.")
    score = daily_pinball(truth, forecast, observed, source_times)
    if score != rescore["slots"]["hist_gradient_boosting"]["protocol_validation_metrics"]:
        raise ValueError("AEON supervised B3 score differs from the reviewed rescore.")
    return score


def run_supervised(
    archive: Path,
    split_review: Path,
    core_config_path: Path,
    campaign_output: Path,
    rescore_report: Path,
    rescore_outcome_review_path: Path,
    recipe_path: Path,
    prefit_review_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Fit a single TRAIN-only recipe after exact distinct review; never read CAL/TEST."""
    recipe_sha = _sha256(recipe_path)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    _recipe_gate(recipe)
    core_sha = _sha256(core_config_path)
    core = json.loads(core_config_path.read_text(encoding="utf-8"))
    _config_gate(core)
    campaign_output = campaign_output.resolve(strict=True)
    manifest_path = campaign_output / "manifest.json"
    manifest_sha = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rescore_sha = _sha256(rescore_report)
    rescore = json.loads(rescore_report.read_text(encoding="utf-8"))
    rescore_outcome_sha = _sha256(rescore_outcome_review_path)
    rescore_outcome = json.loads(rescore_outcome_review_path.read_text(encoding="utf-8"))
    split_sha = _sha256(split_review)
    review_sha = _sha256(prefit_review_path)
    review = json.loads(prefit_review_path.read_text(encoding="utf-8"))
    code_sha = _sha256(Path(__file__))
    if (
        recipe.get("core_campaign_config_sha256") != core_sha
        or recipe.get("source_sha256") != core.get("source_sha256")
        or recipe.get("protocol_sha256") != core.get("protocol_sha256")
        or manifest.get("status") != "COMPLETED_TRAIN_VALIDATION_CAMPAIGN_NOT_FINAL_EVALUATION"
        or manifest.get("config_sha256") != core_sha
        or manifest.get("code_sha256") != _campaign_code_sha256()
        or rescore.get("campaign_manifest_sha256") != manifest_sha
        or rescore.get("status") != "PROTOCOL_VALIDATION_RESCORE_PENDING_INDEPENDENT_SELECTION_REVIEW"
        or recipe.get("rescore_outcome_review_sha256") != rescore_outcome_sha
        or rescore_outcome.get("status")
        != "APPROVED_AEON_VALIDATION_RESCORE_OUTCOME_NO_SELECTION"
        or rescore_outcome.get("artifact_sha256", {}).get("rescore_report") != rescore_sha
        or review.get("status") != "APPROVED_AEON_POST_HOC_SUPERVISED_PREFIT"
        or review.get("recipe_sha256") != recipe_sha
        or review.get("runner_sha256") != code_sha
        or review.get("campaign_manifest_sha256") != manifest_sha
        or review.get("rescore_sha256") != rescore_sha
        or review.get("rescore_outcome_review_sha256") != rescore_outcome_sha
        or review.get("split_review_sha256") != split_sha
        or review.get("cohort_sha256") != recipe.get("cohort_sha256")
        or review.get("test_access") != "PROHIBITED"
    ):
        raise ValueError("AEON supervised challenger lacks exact independent prefit approval.")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("AEON supervised output already exists; never refit implicitly.")
    fit, assess, cohort_sha, _ = load_cohort(archive, split_review)
    if cohort_sha != recipe["cohort_sha256"] or cohort_sha != manifest["cohort_sha256"]:
        raise ValueError("AEON supervised cohort differs from the frozen source/split.")
    raw_dir = campaign_output / "hist_gradient_boosting"
    raw_record = json.loads(_artifact(
        raw_dir, "slot.json", manifest["slots"]["hist_gradient_boosting"]["slot_sha256"]
    ).read_text(encoding="utf-8"))
    raw_path = _artifact(raw_dir, raw_record["prediction_path"], raw_record["prediction_sha256"])
    raw_rows, raw_forecast = _load_prediction(raw_path)
    check = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    check.close()
    check_path = Path(check.name)
    try:
        _save_predictions(check_path, assess, raw_forecast)
        cohort_rows, _ = _load_prediction(check_path)
    finally:
        check_path.unlink(missing_ok=True)
    if (
        not _same_rows(raw_rows, cohort_rows)
        or rescore["slots"]["hist_gradient_boosting"]["prediction_sha256"]
        != raw_record["prediction_sha256"]
        or _date_hashes(rescore["slots"]["hist_gradient_boosting"]["protocol_validation_metrics"])
        != rescore["eligible_source_date_sha256_by_horizon"]
    ):
        raise ValueError("AEON supervised comparison rows differ from reviewed B3.")
    truth = np.stack([row.target_db for row in assess])
    target_mask = np.stack([row.target_mask for row in assess])
    target_times = np.stack([row.target_source_timestamps for row in assess])
    _verified_b3_score(truth, raw_forecast, target_mask, target_times, rescore)
    start = time.perf_counter()
    process = psutil.Process()
    x_train, x_assess = _features(fit), _features(assess)
    y_train = np.stack([row.target_db for row in fit])
    observed = np.stack([row.target_mask for row in fit])
    forecast, models = _fit_quantiles(x_train, y_train, observed, x_assess, recipe["tree"])
    if process.memory_info().rss >= recipe["resource_limits"]["peak_process_rss_bytes"]:
        raise MemoryError("AEON supervised challenger reached process RAM limit.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    model_path = stage / "model.joblib"
    joblib.dump({"family": "lightgbm_full_past_quantile", "models": models,
                 "recipe_sha256": recipe_sha}, model_path)
    prediction_path = stage / "validation-predictions.npz"
    _save_predictions(prediction_path, assess, forecast)
    saved_rows, saved_forecast = _load_prediction(prediction_path)
    if not _same_rows(saved_rows, raw_rows) or not np.array_equal(saved_forecast, forecast):
        raise ValueError("AEON supervised saved validation predictions differ.")
    score = daily_pinball(truth, forecast, target_mask, target_times)
    if _date_hashes(score) != rescore[
        "eligible_source_date_sha256_by_horizon"
    ]:
        raise ValueError("AEON supervised eligible validation dates differ from core.")
    report = {
        "status": "POST_HOC_SUPERVISED_VALIDATION_COMPLETED_PENDING_INDEPENDENT_REVIEW",
        "classification": "POST_HOC_DEVELOPMENT_NOT_FINAL_EVALUATION",
        "sota_claim": "NOT_ESTABLISHED",
        "test_access": "PROHIBITED",
        "source_archive_sha256": recipe["source_sha256"], "cohort_sha256": cohort_sha,
        "campaign_manifest_sha256": manifest_sha, "rescore_sha256": rescore_sha,
        "rescore_outcome_review_sha256": rescore_outcome_sha,
        "recipe_sha256": recipe_sha, "runner_sha256": code_sha,
        "prefit_review_sha256": review_sha, "validation_row_sha256": _row_digest(saved_rows),
        "train_rows": len(fit), "validation_issued_rows": len(assess),
        "model_path": model_path.name, "model_sha256": _sha256(model_path),
        "prediction_path": prediction_path.name,
        "prediction_sha256": _sha256(prediction_path),
        "fit_seconds": time.perf_counter() - start,
        "process_rss_bytes_after_fit": process.memory_info().rss,
        "peak_process_rss_bytes": "NOT_MEASURED",
        "resume_policy": "NO_MID_HEAD_RESUME_ONE_SERIAL_FIT_ATTEMPT",
        "gpu_peak_reserved_bytes": 0,
        "protocol_validation_metrics": score,
        "selection": "NONE_PENDING_INDEPENDENT_REVIEW",
    }
    (stage / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(stage, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "split-review", "core-config", "campaign-output",
                 "rescore-report", "rescore-outcome-review", "recipe", "prefit-review", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = run_supervised(
        args.archive, args.split_review, args.core_config, args.campaign_output,
        args.rescore_report, args.rescore_outcome_review, args.recipe,
        args.prefit_review, args.output,
    )
    print(json.dumps({key: report[key] for key in (
        "status", "recipe_sha256", "runner_sha256", "prediction_sha256", "model_sha256"
    )}, indent=2))


if __name__ == "__main__":
    main()
