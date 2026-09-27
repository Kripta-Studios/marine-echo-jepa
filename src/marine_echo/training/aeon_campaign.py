"""Serial, restart-safe AEON TRAIN/validation campaign; no CAL or TEST access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import joblib  # type: ignore[import-untyped]
import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from torch import nn
from torch.nn import functional as F

from marine_echo.models.aeon_ssl import AeonDirect, AeonTemporalSSL
from marine_echo.training.aeon_corpus import AEON_ADR_SHA256, AEON_SOURCE_SHA256, AeonDevelopmentReader
from marine_echo.training.aeon_development import (
    _context_tensors,
    _normalizer,
    _save_predictions,
    _sha256,
    _tensors,
    _validate_rows,
    _verify_and_score,
)
from marine_echo.training.aeon_windows import AeonHourlyWindow, AeonWindowPlan, iter_aeon_windows


QUANTILES = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64)
CONVENTIONAL = (
    "persistence", "seasonal_24_source_intervals", "ridge", "hist_gradient_boosting"
)
NEURAL = ("direct", "ema_jepa", "shared_sigreg")
CONTROLS = (
    "random_encoder_ema",
    "random_encoder_shared_sigreg",
    "temporally_shuffled_pretrain_target_ema",
    "temporally_shuffled_pretrain_target_shared_sigreg",
)


@dataclass(frozen=True)
class CampaignSlot:
    run_id: str
    family: str
    seed: int | None


def campaign_slots(config: dict[str, Any]) -> tuple[CampaignSlot, ...]:
    if tuple(config.get("families", [])) != CONVENTIONAL + NEURAL:
        raise ValueError("AEON campaign family order differs from the finite core contract.")
    if config.get("neural", {}).get("seeds") != [7, 13, 23]:
        raise ValueError("AEON campaign requires three declared neural seeds.")
    slots = [CampaignSlot(name, name, None) for name in CONVENTIONAL]
    for family in NEURAL:
        slots.extend(CampaignSlot(f"{family}_seed{seed}", family, seed) for seed in (7, 13, 23))
    slots.extend(
        CampaignSlot(f"{family}_seed7", family, 7) for family in CONTROLS
    )
    return tuple(slots)


def _campaign_code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        Path(__file__).with_name("aeon_corpus.py"),
        Path(__file__).with_name("aeon_windows.py"),
        Path(__file__).with_name("aeon_development.py"),
        root / "models/aeon_ssl.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _review_contract() -> dict[str, Any]:
    return {
        "data_kind": "REAL",
        "archive_sha256": AEON_SOURCE_SHA256,
        "window_plan": json.loads(json.dumps(asdict(AeonWindowPlan()))),
        "development_fit": {
            "start": "2024-03-06T00:00:00",
            "end_exclusive": "2024-10-08T00:00:00",
        },
        "development_assessment": {
            "start": "2024-10-08T00:00:00",
            "end_exclusive": "2024-12-01T00:00:00",
        },
    }


def load_cohort(
    archive: Path, split_review: Path
) -> tuple[list[AeonHourlyWindow], list[AeonHourlyWindow], str, str]:
    review_sha256 = _sha256(split_review)
    reader = AeonDevelopmentReader(
        archive, review_path=split_review, review_sha256=review_sha256
    )
    plan = AeonWindowPlan()
    fit = list(
        iter_aeon_windows(
            reader.iter_partition("train"),
            plan=plan,
            partition="train",
            partition_start="2024-03-06",
            partition_end_exclusive="2024-10-08",
        )
    )
    assess = list(
        iter_aeon_windows(
            reader.iter_partition("validation"),
            plan=plan,
            partition="validation",
            partition_start="2024-10-08",
            partition_end_exclusive="2024-12-01",
        )
    )
    cohort_sha256 = _validate_rows(fit, assess, plan, _review_contract())
    return fit, assess, cohort_sha256, review_sha256


def _config_gate(config: dict[str, Any]) -> None:
    if (
        config.get("status") != "PROPOSED_FOR_INDEPENDENT_PREFIT_REVIEW"
        or config.get("study_id") != "aeon3_geb_2024_hourly_sv_v1"
        or config.get("phase") != "train_validation_finite_core_campaign"
        or config.get("protocol_sha256") != AEON_ADR_SHA256
        or config.get("source_sha256") != AEON_SOURCE_SHA256
        or config.get("calibration_access") != "PROHIBITED_IN_THIS_PHASE"
        or config.get("test_access") != "PROHIBITED_IN_THIS_PHASE"
        or config.get("horizon_interval_steps") != [1, 3, 6]
        or config.get("quantiles") != QUANTILES.tolist()
        or config.get("fit_partition") != "train"
        or config.get("assessment_partition") != "validation"
    ):
        raise ValueError("AEON campaign config differs from the reviewed source/split contract.")
    campaign_slots(config)
    conventional = config.get("conventional", {})
    neural = config.get("neural", {})
    controls = config.get("controls", {})
    resource = config.get("device_policy", {})
    if (
        conventional.get("ridge_alpha") != 10.0
        or conventional.get("hist_gradient_boosting_max_iter") != 128
        or conventional.get("hist_gradient_boosting_loss") != "quantile"
        or conventional.get("hist_gradient_boosting_heads") != 15
        or conventional.get("hist_gradient_boosting_max_depth") != 4
        or conventional.get("hist_gradient_boosting_learning_rate") != 0.05
        or conventional.get("max_threads") != 4
        or neural.get("encoder_width") != 128
        or neural.get("encoder_layers") != 3
        or neural.get("batch_size") != 64
        or neural.get("total_optimizer_updates_per_run") != 3000
        or neural.get("direct_supervised_updates") != 3000
        or neural.get("ssl_pretrain_updates") != 1500
        or neural.get("ssl_supervised_updates") != 1500
        or neural.get("checkpoint_every_updates") != 500
        or neural.get("checkpoint_selection") != "final_endpoint_only"
        or neural.get("pretrain_view") != "first_18_vs_last_6_of_24_past_source_products"
        or neural.get("learning_rate") != 3e-4
        or neural.get("weight_decay") != 1e-4
        or neural.get("gradient_clip_norm") != 1.0
        or neural.get("ema_teacher_momentum") != 0.996
        or neural.get("ema_sigreg_weight") != 0.03
        or neural.get("shared_sigreg_weight") != 0.04
        or controls.get("seed") != 7
        or controls.get("modes") != ["ema", "shared_sigreg"]
        or controls.get("random_encoder_pretrain_updates") != 0
        or controls.get("random_encoder_supervised_updates") != 1500
        or controls.get("shuffled_pretrain_updates") != 1500
        or controls.get("shuffled_supervised_updates") != 1500
        or controls.get("random_encoder_has_unequal_total_learned_update_budget") is not True
        or resource.get("peak_process_rss_limit_bytes") != 22 * 1024**3
        or resource.get("peak_gpu_reserved_limit_bytes") != 10 * 1024**3
        or resource.get("one_training_process") is not True
    ):
        raise ValueError("AEON finite campaign budget or model geometry differs.")


def _resolve_device(config: dict[str, Any]) -> tuple[str, int | None]:
    policy = config["device_policy"]
    if policy.get("preferred") != "cuda" or policy.get("fallback") != "cpu":
        raise ValueError("AEON campaign has an unknown local device policy.")
    if not torch.cuda.is_available():
        return "cpu", None
    free_bytes, _ = torch.cuda.mem_get_info()
    minimum = policy.get("minimum_free_gpu_bytes")
    if not isinstance(minimum, int) or minimum < 2 * 1024**3:
        raise ValueError("AEON GPU free-memory floor differs from reviewed config.")
    return ("cuda" if free_bytes >= minimum else "cpu"), int(free_bytes)


def _past_features(rows: list[AeonHourlyWindow]) -> np.ndarray:
    features = []
    for row in rows:
        values = row.context_db
        mask = row.context_mask
        parts: list[float] = []
        for channel in range(4):
            present = values[mask[:, channel], channel]
            parts.extend(
                (
                    float(present[-1]) if len(present) else 0.0,
                    float(present.mean()) if len(present) else 0.0,
                    float(present.std()) if len(present) else 0.0,
                    float(mask[:, channel].mean()),
                )
            )
        parts.extend((float(values[-1, 0]), float(values[0, 0])))
        features.append(parts)
    return np.asarray(features, dtype=np.float64)


def _residual_quantiles(
    base_fit: np.ndarray, fit: list[AeonHourlyWindow], base_assess: np.ndarray
) -> np.ndarray:
    truth = np.stack([row.target_db for row in fit])
    mask = np.stack([row.target_mask for row in fit])
    residuals = []
    for horizon in range(3):
        valid = mask[:, horizon]
        if not valid.any():
            raise ValueError("AEON campaign TRAIN lacks a horizon label.")
        residuals.append(np.quantile(truth[valid, horizon] - base_fit[valid, horizon], QUANTILES))
    return base_assess[:, :, None] + np.asarray(residuals)[None, :, :]


def _conventional_prediction(
    slot: CampaignSlot,
    fit: list[AeonHourlyWindow],
    assess: list[AeonHourlyWindow],
    config: dict[str, Any],
    stage: Path,
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    model_path = stage / "model.joblib"
    if slot.family in ("persistence", "seasonal_24_source_intervals"):
        indices = (-1, -1, -1) if slot.family == "persistence" else (0, 2, 5)
        fit_base = np.stack(
            [[row.context_db[index, 0] for index in indices] for row in fit]
        )
        assess_base = np.stack(
            [[row.context_db[index, 0] for index in indices] for row in assess]
        )
        prediction = _residual_quantiles(fit_base, fit, assess_base)
        joblib.dump(
            {
                "family": slot.family,
                "context_indices_by_horizon": indices,
                "train_residual_quantiles": (prediction[0] - assess_base[0, :, None]).tolist(),
            },
            model_path,
        )
    elif slot.family == "hist_gradient_boosting":
        x_fit, x_assess = _past_features(fit), _past_features(assess)
        truth = np.stack([row.target_db for row in fit])
        masks = np.stack([row.target_mask for row in fit])
        prediction = np.zeros((len(assess), 3, 5))
        models = []
        conventional = config["conventional"]
        with threadpool_limits(limits=conventional["max_threads"]):
            for horizon in range(3):
                valid = masks[:, horizon]
                if not valid.any():
                    raise ValueError("AEON HGB TRAIN lacks a horizon label.")
                horizon_models = []
                for index, quantile in enumerate(QUANTILES):
                    model = HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=float(quantile),
                        max_iter=conventional["hist_gradient_boosting_max_iter"],
                        max_depth=conventional["hist_gradient_boosting_max_depth"],
                        learning_rate=conventional["hist_gradient_boosting_learning_rate"],
                        random_state=conventional["random_state"],
                    )
                    model.fit(x_fit[valid], truth[valid, horizon])
                    prediction[:, horizon, index] = model.predict(x_assess)
                    horizon_models.append(model)
                models.append(horizon_models)
        prediction.sort(axis=-1)
        joblib.dump({"family": slot.family, "models": models}, model_path)
    else:
        x_fit, x_assess = _past_features(fit), _past_features(assess)
        truth = np.stack([row.target_db for row in fit])
        masks = np.stack([row.target_mask for row in fit])
        fit_base = np.zeros((len(fit), 3))
        assess_base = np.zeros((len(assess), 3))
        models = []
        conventional = config["conventional"]
        with threadpool_limits(limits=conventional["max_threads"]):
            for horizon in range(3):
                valid = masks[:, horizon]
                if not valid.any():
                    raise ValueError("AEON conventional fit lacks horizon labels.")
                if slot.family == "ridge":
                    model = make_pipeline(
                        StandardScaler(), Ridge(alpha=conventional["ridge_alpha"])
                    )
                else:
                    raise ValueError("Unknown AEON conventional family.")
                model.fit(x_fit[valid], truth[valid, horizon])
                fit_base[:, horizon] = model.predict(x_fit)
                assess_base[:, horizon] = model.predict(x_assess)
                models.append(model)
        prediction = _residual_quantiles(fit_base, fit, assess_base)
        joblib.dump(
            {
                "family": slot.family,
                "models": models,
                "train_residual_quantiles": (prediction[0] - assess_base[0, :, None]).tolist(),
            },
            model_path,
        )
    peak_rss = max(peak_rss, process.memory_info().rss)
    if peak_rss >= config["device_policy"]["peak_process_rss_limit_bytes"]:
        raise MemoryError("AEON conventional slot reached the process RAM limit.")
    return prediction, model_path, {
        "fit_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "gpu_peak_reserved_bytes": None,
    }


def _primary(metrics: dict[str, Any]) -> float:
    scores = [entry["daily_mean_pinball_db"] for entry in metrics["horizons"]]
    if any(score is None or not np.isfinite(score) for score in scores):
        raise ValueError("AEON campaign validation has an unscored horizon.")
    return float(np.mean(scores))


def _save_slot_result(
    stage: Path,
    slot: CampaignSlot,
    assess: list[AeonHourlyWindow],
    prediction: np.ndarray,
    model_artifact: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    prediction_path = stage / "validation-predictions.npz"
    _save_predictions(prediction_path, assess, prediction)
    metrics = _verify_and_score(prediction_path, assess)
    result = {
        "run_id": slot.run_id,
        "family": slot.family,
        "seed": slot.seed,
        "prediction_path": prediction_path.name,
        "prediction_sha256": _sha256(prediction_path),
        "model_path": model_artifact.name,
        "model_sha256": _sha256(model_artifact),
        "metrics": metrics,
        "primary_daily_mean_pinball_db": _primary(metrics),
        **metadata,
    }
    (stage / "slot.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _verify_done_slot(root: Path, entry: dict[str, Any]) -> None:
    directory = root / entry["run_id"]
    result_path = directory / "slot.json"
    if not result_path.is_file() or _sha256(result_path) != entry["slot_sha256"]:
        raise ValueError("AEON completed campaign slot manifest differs.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for key, digest_key in (("prediction_path", "prediction_sha256"), ("model_path", "model_sha256")):
        relative = Path(result[key])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("AEON campaign slot artifact path escapes its output.")
        if _sha256(directory / relative) != result[digest_key]:
            raise ValueError("AEON completed campaign artifact digest differs.")
    for item in result.get("checkpoints", []) + result.get("validation_checks", []):
        relative = Path(item["path"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("AEON campaign checkpoint path escapes its output.")
        if _sha256(directory / relative) != item["sha256"]:
            raise ValueError("AEON completed campaign checkpoint digest differs.")


def execute_campaign_slots(
    fit: list[AeonHourlyWindow],
    assess: list[AeonHourlyWindow],
    output: Path,
    *,
    config: dict[str, Any],
    config_sha256: str,
    cohort_sha256: str,
    review_sha256: str,
    device: str,
    executor: Callable[[CampaignSlot, list[AeonHourlyWindow], list[AeonHourlyWindow], dict[str, Any], Path, str], tuple[np.ndarray, Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Publish one validated slot at a time, safely resumable by file hashes."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    slots = campaign_slots(config)
    manifest_path = output / "manifest.json"
    identity = {
        "status": "TRAIN_VALIDATION_CAMPAIGN_IN_PROGRESS",
        "study_id": config["study_id"],
        "config_sha256": config_sha256,
        "cohort_sha256": cohort_sha256,
        "review_sha256": review_sha256,
        "code_sha256": _campaign_code_sha256(),
        "device": device,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in identity.items() if key != "status"):
            raise ValueError("AEON campaign resume identity differs.")
        entries = manifest.get("slots")
        if not isinstance(entries, dict):
            raise ValueError("AEON campaign manifest slot ledger is malformed.")
    else:
        entries = {}
        manifest = {**identity, "slots": entries}
        _atomic_json(manifest_path, manifest)
    for slot in slots:
        prior = entries.get(slot.run_id)
        if prior is not None and prior.get("status") == "DONE":
            _verify_done_slot(output, prior)
            continue
        if (output / slot.run_id).exists():
            raise ValueError("AEON campaign output exists without a completed ledger entry.")
        stage = Path(tempfile.mkdtemp(prefix=slot.run_id + ".stage.", dir=output))
        try:
            prediction, model_path, details = executor(slot, fit, assess, config, stage, device)
            result = _save_slot_result(stage, slot, assess, prediction, model_path, details)
            os.replace(stage, output / slot.run_id)
            entries[slot.run_id] = {
                "status": "DONE",
                "run_id": slot.run_id,
                "slot_sha256": _sha256(output / slot.run_id / "slot.json"),
                "primary_daily_mean_pinball_db": result["primary_daily_mean_pinball_db"],
            }
            _atomic_json(manifest_path, manifest)
        except BaseException:
            if stage.exists() and stage.parent == output:
                shutil.rmtree(stage)
            entries[slot.run_id] = {"status": "FAILED", "run_id": slot.run_id}
            _atomic_json(manifest_path, manifest)
            raise
    manifest["status"] = "COMPLETED_TRAIN_VALIDATION_CAMPAIGN_NOT_FINAL_EVALUATION"
    _atomic_json(manifest_path, manifest)
    return manifest


def run_campaign(
    archive: Path,
    split_review: Path,
    campaign_review: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Real campaign gate; not callable without distinct reviewed code/config/cohort."""
    config_sha256 = _sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _config_gate(config)
    fit, assess, cohort_sha256, split_review_sha256 = load_cohort(archive, split_review)
    campaign_review_sha256 = _sha256(campaign_review)
    approval = json.loads(campaign_review.read_text(encoding="utf-8"))
    if (
        approval.get("status") != "APPROVED_AEON_TRAIN_VALIDATION_CAMPAIGN"
        or approval.get("protocol_sha256") != AEON_ADR_SHA256
        or approval.get("source_sha256") != AEON_SOURCE_SHA256
        or approval.get("split_review_sha256") != split_review_sha256
        or approval.get("config_sha256") != config_sha256
        or approval.get("cohort_sha256") != cohort_sha256
        or approval.get("code_sha256") != _campaign_code_sha256()
        or approval.get("test_access") != "PROHIBITED"
    ):
        raise ValueError("AEON campaign lacks an exact independent prefit approval.")
    device, _ = _resolve_device(config)
    return execute_campaign_slots(
        fit,
        assess,
        output,
        config=config,
        config_sha256=config_sha256,
        cohort_sha256=cohort_sha256,
        review_sha256=campaign_review_sha256,
        device=device,
        executor=_execute_slot,
    )


def _execute_slot(
    slot: CampaignSlot,
    fit: list[AeonHourlyWindow],
    assess: list[AeonHourlyWindow],
    config: dict[str, Any],
    stage: Path,
    device: str,
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    if slot.family in CONVENTIONAL:
        return _conventional_prediction(slot, fit, assess, config, stage)
    return _neural_prediction(slot, fit, assess, config, stage, device)


def _is_random_control(family: str) -> bool:
    return family.startswith("random_encoder_")


def _is_shuffled_control(family: str) -> bool:
    return family.startswith("temporally_shuffled_pretrain_target_")


def _shuffled_pretrain_loss(
    model: AeonTemporalSSL, values: torch.Tensor, mask: torch.Tensor, *, shift: int
) -> torch.Tensor:
    """Mode-faithful false-pair objective on disjoint TRAIN histories."""
    if not 1 <= shift < len(values):
        raise ValueError("Shuffled TRAIN target pairing needs a nonzero batch rotation.")
    predicted = model.predictor(model.context_view(values, mask))
    target = model.teacher_view(values, mask)
    if model.mode == "ema":
        target = target.detach()
    target = target.roll(shifts=shift, dims=0)
    regularized = predicted if model.mode == "ema" else target
    return F.smooth_l1_loss(predicted, target) + model.regularizer_weight * model.regularizer(
        regularized
    )


def _representation_diagnostics(
    model: AeonTemporalSSL,
    fit: list[AeonHourlyWindow],
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: str,
) -> dict[str, Any]:
    """Deterministic TRAIN-only final-pretrain collapse and scale evidence."""
    if len(fit) < 2 or len(values) != len(fit) or mask.shape != values.shape:
        raise ValueError("AEON TRAIN representation diagnostic cohort is invalid.")
    selected = np.linspace(0, len(fit) - 1, min(512, len(fit)), dtype=int)
    if len(set(selected.tolist())) != len(selected):
        raise ValueError("AEON diagnostic source row selection repeated an issue.")
    target_parts = []
    predictor_parts = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(selected), 64):
            batch = selected[first : first + 64]
            past = values[batch].to(device)
            past_mask = mask[batch].to(device)
            target_parts.append(model.teacher_view(past, past_mask).cpu().numpy())
            predictor_parts.append(
                model.predictor(model.context_view(past, past_mask)).cpu().numpy()
            )
    target = np.concatenate(target_parts).astype(np.float64)
    predicted = np.concatenate(predictor_parts).astype(np.float64)
    if target.shape != predicted.shape or not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError("AEON TRAIN representation diagnostic vectors are invalid.")
    target_variance = target.var(axis=0, ddof=1)
    predictor_variance = predicted.var(axis=0, ddof=1)
    centered = target - target.mean(axis=0)
    covariance = centered.T @ centered / (len(target) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    trace = float(eigenvalues.sum())
    if trace <= 1e-12:
        effective_rank = 0.0
    else:
        mass = eigenvalues[eigenvalues > 0] / trace
        effective_rank = float(np.exp(-(mass * np.log(mass)).sum()))
    target_rms = float(np.sqrt(np.square(target).mean()))
    predictor_rms = float(np.sqrt(np.square(predicted).mean()))
    variance_ratio = (
        float(predictor_variance.mean() / target_variance.mean())
        if float(target_variance.mean()) > 1e-12
        else None
    )
    scale_ratio = predictor_rms / target_rms if target_rms > 1e-12 else None
    return {
        "selection": "chronological_evenly_spaced_train_issue_rows_v1",
        "source_partition": "train",
        "source_archive_sha256": fit[0].source_archive_sha256,
        "row_count": len(selected),
        "row_ids": [fit[index].row_id for index in selected],
        "cutoff_interval_ids": [fit[index].cutoff_interval_id for index in selected],
        "target_dimension_variance_min": float(target_variance.min()),
        "target_dimension_variance_median": float(np.median(target_variance)),
        "target_dimension_variance_mean": float(target_variance.mean()),
        "target_effective_rank": effective_rank,
        "target_effective_rank_fraction": effective_rank / target.shape[1],
        "target_exact_zero_variance": trace <= 1e-12,
        "predictor_dimension_variance_min": float(predictor_variance.min()),
        "predictor_dimension_variance_mean": float(predictor_variance.mean()),
        "predictor_rms": predictor_rms,
        "target_rms": target_rms,
        "prediction_target_rms_ratio": scale_ratio,
        "predictor_target_variance_ratio": variance_ratio,
    }


def _neural_prediction(
    slot: CampaignSlot,
    fit: list[AeonHourlyWindow],
    assess: list[AeonHourlyWindow],
    config: dict[str, Any],
    stage: Path,
    device: str,
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    if slot.seed is None:
        raise ValueError("AEON neural slot requires a declared seed.")
    neural = config["neural"]
    seed = slot.seed
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config["conventional"]["max_threads"])
    try:
        if slot.family == "direct":
            model: nn.Module = AeonDirect(
                width=neural["encoder_width"], layers=neural["encoder_layers"]
            )
        else:
            mode: Literal["ema", "shared_sigreg"] = (
                "shared_sigreg" if slot.family.endswith("shared_sigreg") else "ema"
            )
            weight = (
                neural["shared_sigreg_weight"]
                if mode == "shared_sigreg"
                else neural["ema_sigreg_weight"]
            )
            model = AeonTemporalSSL(
                mode=mode,
                width=neural["encoder_width"],
                layers=neural["encoder_layers"],
                regularizer_weight=weight,
            )
        model = model.to(device)
        if _is_random_control(slot.family):
            assert isinstance(model, AeonTemporalSSL)
            model.encoder.requires_grad_(False)
            model.predictor.requires_grad_(False)
        scaler = _normalizer(fit)
        x_fit, m_fit, y_fit, y_mask = _tensors(fit, scaler)
        x_assess, m_assess = _context_tensors(assess, scaler)
        candidate = np.flatnonzero(y_mask.any(dim=1).numpy())
        if not len(candidate):
            raise ValueError("AEON neural TRAIN cohort lacks observed labels.")
        process = psutil.Process()
        peak_rss = process.memory_info().rss
        if peak_rss >= config["device_policy"]["peak_process_rss_limit_bytes"]:
            raise MemoryError("AEON campaign starts above the process RAM limit.")
        checkpoints: list[dict[str, Any]] = []
        validation_checks: list[dict[str, Any]] = []
        pretrain_seconds = 0.0
        supervised_seconds = 0.0
        representation_diagnostics: dict[str, Any] | None = None

        def check_resources() -> None:
            nonlocal peak_rss
            peak_rss = max(peak_rss, process.memory_info().rss)
            if peak_rss >= config["device_policy"]["peak_process_rss_limit_bytes"]:
                raise MemoryError("AEON campaign reached its 22 GiB process RAM limit.")
            if (
                device == "cuda"
                and torch.cuda.max_memory_reserved()
                >= config["device_policy"]["peak_gpu_reserved_limit_bytes"]
            ):
                raise MemoryError("AEON campaign reached its 10 GiB GPU reserve limit.")

        def save_checkpoint(phase: str, step: int, optimizer: torch.optim.Optimizer) -> Path:
            path = stage / f"checkpoint-{phase}-{step}.pt"
            torch.save(
                {
                    "phase": phase,
                    "step": step,
                    "family": slot.family,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_fit_only": scaler,
                    "source_sha256": AEON_SOURCE_SHA256,
                    "protocol_sha256": AEON_ADR_SHA256,
                },
                path,
            )
            checkpoints.append(
                {"phase": phase, "step": step, "path": path.name, "sha256": _sha256(path)}
            )
            return path

        def forecast() -> np.ndarray:
            model.eval()
            predicted = []
            with torch.no_grad():
                for start in range(0, len(assess), neural["batch_size"]):
                    part = slice(start, start + neural["batch_size"])
                    output = model(x_assess[part].to(device), m_assess[part].to(device))
                    predicted.append(output.cpu().numpy() * scaler[3] + scaler[2])
            return np.concatenate(predicted)

        if slot.family in ("ema_jepa", "shared_sigreg") or _is_shuffled_control(slot.family):
            assert isinstance(model, AeonTemporalSSL)
            pretrain_optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=neural["learning_rate"],
                weight_decay=neural["weight_decay"],
            )
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            residues: dict[int, np.ndarray] = {}
            if _is_shuffled_control(slot.family):
                interval_ids = np.asarray([row.cutoff_interval_id for row in fit])
                for residue in range(24):
                    indices = np.flatnonzero(interval_ids % 24 == residue)
                    if len(indices) >= neural["batch_size"]:
                        residues[residue] = indices
                if not residues:
                    raise ValueError("AEON shuffled control lacks separated TRAIN windows.")
            started = time.perf_counter()
            for step in range(1, neural["ssl_pretrain_updates"] + 1):
                model.train()
                if residues:
                    pool = residues[sorted(residues)[step % len(residues)]]
                    selected = rng.choice(pool, size=neural["batch_size"], replace=False)
                else:
                    selected = rng.choice(
                        len(fit), size=neural["batch_size"], replace=len(fit) < neural["batch_size"]
                    )
                values = x_fit[selected].to(device)
                mask = m_fit[selected].to(device)
                if _is_shuffled_control(slot.family):
                    shift = int(rng.integers(1, len(selected)))
                    loss = _shuffled_pretrain_loss(model, values, mask, shift=shift)
                else:
                    loss = model.pretrain_loss(values, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("AEON SSL pretraining loss is non-finite.")
                pretrain_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    neural["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
                pretrain_optimizer.step()
                if model.teacher is not None:
                    model.update_teacher(momentum=neural["ema_teacher_momentum"])
                check_resources()
                if step % neural["checkpoint_every_updates"] == 0:
                    save_checkpoint("pretrain", step, pretrain_optimizer)
            pretrain_seconds = time.perf_counter() - started
            representation_diagnostics = _representation_diagnostics(
                model, fit, x_fit, m_fit, device=device
            )
            check_resources()

        if _is_random_control(slot.family):
            supervised_updates = config["controls"]["random_encoder_supervised_updates"]
        elif slot.family == "direct":
            supervised_updates = neural["direct_supervised_updates"]
        else:
            supervised_updates = neural["ssl_supervised_updates"]
        supervised_optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=neural["learning_rate"],
            weight_decay=neural["weight_decay"],
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        quantiles = torch.as_tensor(QUANTILES, dtype=torch.float32, device=device)
        started = time.perf_counter()
        final_forecast: np.ndarray | None = None
        final_checkpoint: Path | None = None
        for step in range(1, supervised_updates + 1):
            model.train()
            selected = rng.choice(
                candidate,
                size=neural["batch_size"],
                replace=len(candidate) < neural["batch_size"],
            )
            predicted = model(x_fit[selected].to(device), m_fit[selected].to(device))
            truth = y_fit[selected].to(device)
            valid = y_mask[selected].to(device)
            error = truth[:, :, None] - predicted
            loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)[valid].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("AEON supervised campaign loss is non-finite.")
            supervised_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                neural["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            supervised_optimizer.step()
            check_resources()
            if step % neural["checkpoint_every_updates"] == 0:
                checkpoint = save_checkpoint("supervised", step, supervised_optimizer)
                prediction = forecast()
                validation_path = stage / f"validation-at-supervised-{step}.npz"
                _save_predictions(validation_path, assess, prediction)
                metrics = _verify_and_score(validation_path, assess)
                validation_checks.append(
                    {
                        "step": step,
                        "path": validation_path.name,
                        "sha256": _sha256(validation_path),
                        "primary_daily_mean_pinball_db": _primary(metrics),
                    }
                )
                if step == supervised_updates:
                    final_forecast = prediction
                    final_checkpoint = checkpoint
        supervised_seconds = time.perf_counter() - started
        if final_forecast is None or final_checkpoint is None:
            raise ValueError("AEON neural run lacks a final supervised endpoint.")
        metadata = {
            "pretrain_updates": 0
            if slot.family == "direct" or _is_random_control(slot.family)
            else neural["ssl_pretrain_updates"],
            "supervised_updates": supervised_updates,
            "pretrain_seconds": pretrain_seconds,
            "supervised_seconds": supervised_seconds,
            "checkpoints": checkpoints,
            "validation_checks": validation_checks,
            "selection": "FINAL_ENDPOINT_ONLY",
            "random_encoder_unequal_total_learned_update_budget": _is_random_control(slot.family),
            "peak_process_rss_bytes": peak_rss,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "scaler_fit_only": scaler,
            "final_pretrain_train_representation_diagnostics": representation_diagnostics,
            "device": device,
        }
        return final_forecast, final_checkpoint, metadata
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-review", type=Path, required=True)
    parser.add_argument("--campaign-review", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_campaign(
        arguments.archive,
        arguments.split_review,
        arguments.campaign_review,
        arguments.config,
        arguments.output,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
