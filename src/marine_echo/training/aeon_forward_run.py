"""Finite, independently gated forward-EMA TRAIN/validation development campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from torch import nn

from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.models.aeon_forward_ssl import AeonForwardSSL
from marine_echo.training.aeon_campaign import _campaign_code_sha256, load_cohort
from marine_echo.training.aeon_corpus import AEON_ADR_SHA256, AEON_SOURCE_SHA256
from marine_echo.training.aeon_development import (
    _context_tensors, _normalizer, _save_predictions, _sha256, _tensors,
)
from marine_echo.training.aeon_forward import ForwardPairs, load_train_forward_cohort, pair_inventory
from marine_echo.training.aeon_rescore import _artifact, _load_prediction
from marine_echo.training.aeon_windows import AeonHourlyWindow


FORWARD_SLOTS = (
    "forward_ema_seed7",
    "forward_ema_seed13",
    "forward_ema_seed23",
    "random_encoder_seed7",
    "temporally_shuffled_future_target_seed7",
)


def _shuffled_pair_indices(
    cutoff_interval_ids: np.ndarray, rng: np.random.Generator, batch_size: int
) -> np.ndarray:
    """Select distinct TRAIN windows all at least 24 source intervals apart."""
    if cutoff_interval_ids.ndim != 1 or batch_size < 2:
        raise ValueError("Forward shuffled target needs one-dimensional TRAIN cutoff IDs.")
    pools = [
        np.flatnonzero(cutoff_interval_ids % 24 == residue)
        for residue in range(24)
    ]
    eligible = [pool for pool in pools if len(pool) >= batch_size]
    if not eligible:
        raise ValueError("Forward shuffled target lacks separated TRAIN windows.")
    selected = rng.choice(eligible[int(rng.integers(len(eligible)))], size=batch_size,
                          replace=False)
    if np.min(np.diff(np.sort(cutoff_interval_ids[selected]))) < 24:
        raise ValueError("Forward shuffled target pairing was not interval-separated.")
    return selected


def _branch_diagnostics(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Forward TRAIN embedding branch is invalid.")
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / (len(values) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    trace = float(eigenvalues.sum())
    if trace <= 1e-12:
        rank = 0.0
    else:
        mass = eigenvalues[eigenvalues > 0] / trace
        rank = float(np.exp(-(mass * np.log(mass)).sum()))
    variance = values.var(axis=0, ddof=1)
    off_diagonal = covariance - np.diag(np.diag(covariance))
    return {
        "dimension_variance_min": float(variance.min()),
        "dimension_variance_median": float(np.median(variance)),
        "dimension_variance_mean": float(variance.mean()),
        "effective_rank": rank,
        "effective_rank_fraction": rank / values.shape[1],
        "offdiagonal_covariance_mean_square": float(np.square(off_diagonal).mean()),
        "rms": float(np.sqrt(np.square(values).mean())),
        "exact_zero_variance": trace <= 1e-12,
    }


def _forward_representation_diagnostics(
    model: AeonForwardSSL,
    past: torch.Tensor,
    past_mask: torch.Tensor,
    future: torch.Tensor,
    future_mask: torch.Tensor,
    *,
    row_ids: list[str],
    cutoff_interval_ids: np.ndarray,
    source_sha256: str,
) -> dict[str, Any]:
    """Four-branch rank/scale/covariance on the same fixed 512 TRAIN pairs."""
    if (
        len(past) < 2
        or len(past) != len(future)
        or len(past) != len(row_ids)
        or len(cutoff_interval_ids) != len(row_ids)
        or past.shape[1:] != (24, 4)
        or future.shape[1:] != (6, 4)
        or past_mask.shape != past.shape
        or future_mask.shape != future.shape
    ):
        raise ValueError("Forward representation diagnostics need matching TRAIN pairs.")
    selected = np.linspace(0, len(row_ids) - 1, min(512, len(row_ids)), dtype=int)
    if len(set(selected.tolist())) != len(selected):
        raise ValueError("Forward diagnostic repeated a TRAIN pair.")
    branches: dict[str, list[np.ndarray]] = {
        name: [] for name in ("online_context", "online_future_target", "ema_future_teacher", "predictor")
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), 64):
            batch = selected[start : start + 64]
            target_values, target_mask = model.target_view(future[batch], future_mask[batch])
            context = model.encoder(past[batch], past_mask[batch])
            embeddings = {
                "online_context": context,
                "online_future_target": model.encoder(target_values, target_mask),
                "ema_future_teacher": model.teacher(target_values, target_mask),
                "predictor": model.predictor(context),
            }
            for name, value in embeddings.items():
                branches[name].append(value.cpu().numpy().astype(np.float64))
    diagnostics = {
        name: _branch_diagnostics(np.concatenate(parts)) for name, parts in branches.items()
    }
    target_rms = diagnostics["ema_future_teacher"]["rms"]
    predictor_rms = diagnostics["predictor"]["rms"]
    return {
        "selection": "chronological_evenly_spaced_train_forward_pairs_v1",
        "source_partition": "train",
        "source_archive_sha256": source_sha256,
        "row_count": len(selected),
        "row_ids": [row_ids[index] for index in selected],
        "cutoff_interval_ids": cutoff_interval_ids[selected].astype(int).tolist(),
        "branches": diagnostics,
        "target_effective_rank": diagnostics["ema_future_teacher"]["effective_rank"],
        "target_dimension_variance_mean": diagnostics["ema_future_teacher"]["dimension_variance_mean"],
        "prediction_target_rms_ratio": predictor_rms / target_rms if target_rms > 1e-12 else None,
    }


def _code_sha256() -> str:
    """Bind the runner, objective, pairer and frozen shared acoustic contracts."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__), Path(__file__).with_name("aeon_forward.py"),
        root / "models/aeon_forward_ssl.py",
        root / "models/aeon_ssl.py",
        Path(__file__).with_name("aeon_corpus.py"),
        Path(__file__).with_name("aeon_windows.py"),
        Path(__file__).with_name("aeon_development.py"),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


_QUANTILES = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float32)


def _config_gate(config: dict[str, Any]) -> None:
    model = config.get("model", {})
    controls = config.get("controls", {})
    device = config.get("device_policy", {})
    if (
        config.get("study_id") != "aeon3_geb_2024_hourly_sv_v1"
        or config.get("source_sha256") != AEON_SOURCE_SHA256
        or config.get("protocol_sha256") != AEON_ADR_SHA256
        or config.get("phase") != "post_hoc_train_validation_forward_ema_jepa_development"
        or config.get("status") != "PROPOSED_FOR_INDEPENDENT_PREFIT_REVIEW"
        or config.get("classification") != "POST_HOC_DEVELOPMENT_NOT_FINAL_EVALUATION"
        or config.get("fit_partition") != "train"
        or config.get("assessment_partition") != "validation"
        or config.get("calibration_access") != "PROHIBITED_IN_THIS_PHASE"
        or config.get("test_access") != "PROHIBITED_IN_THIS_PHASE"
        or config.get("slots") != list(FORWARD_SLOTS)
        or config.get("seeds") != [7, 13, 23]
        or config.get("pretrain_pair_rows") != 4923
        or config.get("pretrain_context") != "24_prior_source_interval_ids_all_38khz_observed_other_channels_masked"
        or config.get("pretrain_future_target") != "six_exact_following_source_interval_ids_all_within_train_all_38khz_observed_other_channels_masked"
        or config.get("teacher_input") != "future_six_source_products_only_zero_padded_into_24x4_encoder_input"
        or config.get("teacher_gradient") != "detached_ema_teacher"
        or config.get("online_future_gradient") != "variance_and_covariance_regularizers_only"
        or config.get("horizon_interval_steps") != [1, 3, 6]
        or config.get("quantiles") != [0.05, 0.25, 0.5, 0.75, 0.95]
        or model.get("encoder_width") != 128
        or model.get("encoder_layers") != 3
        or model.get("batch_size") != 64
        or model.get("pretrain_updates") != 1500
        or model.get("supervised_updates") != 1500
        or model.get("total_main_updates") != 3000
        or model.get("checkpoint_every_updates") != 500
        or model.get("checkpoint_selection") != "final_endpoint_only"
        or model.get("optimizer") != "AdamW"
        or model.get("learning_rate") != 3e-4
        or model.get("weight_decay") != 1e-4
        or model.get("gradient_clip_norm") != 1.0
        or model.get("ema_teacher_momentum") != 0.996
        or model.get("prediction_loss_weight") != 25.0
        or model.get("pretrain_loss_formula") != "25*smooth_l1_predictor_vs_detached_ema_future+25*mean_variance_penalty_online_context_and_online_future+1*mean_covariance_penalty_online_context_and_online_future"
        or model.get("variance_floor_std") != 1.0
        or model.get("variance_epsilon") != 1e-4
        or model.get("variance_weight_mean_two_online_views") != 25.0
        or model.get("covariance_weight_mean_two_online_views") != 1.0
        or model.get("regularized_views") != ["online_context", "online_future_target"]
        or model.get("variance_formula") != "mean_relu_gamma_minus_sqrt_unbiased_batch_variance_plus_epsilon"
        or model.get("covariance_formula") != "sum_squared_off_diagonal_centered_sample_covariance_divided_by_embedding_width"
        or model.get("train_representation_diagnostics", {}).get("views") != ["online_context", "online_future_target", "ema_future_teacher", "predictor"]
        or controls.get("seed") != 7
        or controls.get("random_encoder_has_unequal_total_learned_update_budget") is not True
        or device.get("preferred") != "cuda"
        or device.get("fallback") != "cpu"
        or not isinstance(device.get("minimum_free_gpu_bytes"), int)
        or device["minimum_free_gpu_bytes"] < 2 * 1024**3
        or device.get("peak_process_rss_limit_bytes") != 22 * 1024**3
        or device.get("peak_gpu_reserved_limit_bytes") != 10 * 1024**3
        or device.get("one_training_process") is not True
    ):
        raise ValueError("AEON forward configuration differs from finite reviewed contract.")


def _resolve_device(config: dict[str, Any]) -> str:
    policy = config["device_policy"]
    if policy.get("preferred") != "cuda" or policy.get("fallback") != "cpu":
        raise ValueError("AEON forward device policy differs from reviewed config.")
    if not torch.cuda.is_available():
        return "cpu"
    free_bytes, _ = torch.cuda.mem_get_info()
    return "cuda" if free_bytes >= policy["minimum_free_gpu_bytes"] else "cpu"


def _future_tensors(
    pairs: ForwardPairs, scaler: tuple[float, float, float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.where(pairs.future_mask, (pairs.future_db - scaler[0]) / scaler[1], 0.0)
    if not np.isfinite(values).all():
        raise ValueError("AEON forward TRAIN future tensor is non-finite.")
    return torch.as_tensor(values, dtype=torch.float32), torch.as_tensor(
        pairs.future_mask, dtype=torch.bool
    )


def _train_slot(
    slot_id: str,
    fit: list[AeonHourlyWindow],
    assess: list[AeonHourlyWindow],
    pairs: ForwardPairs,
    config: dict[str, Any],
    stage: Path,
    device: str,
    *,
    config_sha256: str,
) -> dict[str, Any]:
    if slot_id not in FORWARD_SLOTS:
        raise ValueError("Unknown AEON forward slot.")
    model_config = config["model"]
    seed = int(slot_id.rsplit("seed", 1)[1])
    is_random = slot_id == "random_encoder_seed7"
    is_shuffled = slot_id == "temporally_shuffled_future_target_seed7"
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = AeonForwardSSL(
        width=model_config["encoder_width"], layers=model_config["encoder_layers"],
        variance_floor=model_config["variance_floor_std"],
        variance_epsilon=model_config["variance_epsilon"],
        similarity_weight=model_config["prediction_loss_weight"],
        variance_weight=model_config["variance_weight_mean_two_online_views"],
        covariance_weight=model_config["covariance_weight_mean_two_online_views"],
    ).to(device)
    if is_random:
        model.encoder.requires_grad_(False)
        model.teacher.requires_grad_(False)
        model.predictor.requires_grad_(False)
    scaler = _normalizer(fit)
    fit_x, fit_mask, fit_y, fit_y_mask = _tensors(fit, scaler)
    assess_x, assess_mask = _context_tensors(assess, scaler)
    future, future_mask = _future_tensors(pairs, scaler)
    paired_past = fit_x[pairs.fit_indices]
    paired_mask = fit_mask[pairs.fit_indices]
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    start_time = time.perf_counter()
    checkpoints: list[dict[str, Any]] = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)

    def check_resources() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        if peak_rss >= config["device_policy"]["peak_process_rss_limit_bytes"]:
            raise MemoryError("AEON forward reached 22 GiB process RAM limit.")
        if device == "cuda" and torch.cuda.max_memory_reserved() >= config["device_policy"]["peak_gpu_reserved_limit_bytes"]:
            raise MemoryError("AEON forward reached 10 GiB GPU reserved limit.")

    def checkpoint(phase: str, step: int, optimizer: torch.optim.Optimizer) -> Path:
        path = stage / f"checkpoint-{phase}-{step}.pt"
        torch.save({
            "phase": phase, "step": step, "slot_id": slot_id, "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_fit_only": scaler, "source_sha256": config["source_sha256"],
            "protocol_sha256": config["protocol_sha256"],
            "pair_id_and_interval_sha256": config["pair_id_and_interval_sha256"],
            "config_sha256": config_sha256,
        }, path)
        checkpoints.append({"phase": phase, "step": step, "path": path.name,
                            "sha256": _sha256(path)})
        return path

    try:
        check_resources()
        if not is_random:
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=model_config["learning_rate"], weight_decay=model_config["weight_decay"],
            )
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            for step in range(1, model_config["pretrain_updates"] + 1):
                model.train()
                if is_shuffled:
                    selected = _shuffled_pair_indices(
                        pairs.cutoff_interval_ids, rng, model_config["batch_size"]
                    )
                    shift = int(rng.integers(1, len(selected)))
                    false_targets = np.roll(selected, shift)
                else:
                    selected = rng.choice(
                        len(pairs.row_ids), size=model_config["batch_size"], replace=False
                    )
                    false_targets = selected
                loss = model.pretrain_loss(
                    paired_past[selected].to(device), paired_mask[selected].to(device),
                    future[false_targets].to(device), future_mask[false_targets].to(device),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("AEON forward pretrain loss is non-finite.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    model_config["gradient_clip_norm"], error_if_nonfinite=True,
                )
                optimizer.step()
                model.update_teacher(momentum=model_config["ema_teacher_momentum"])
                check_resources()
                if step % model_config["checkpoint_every_updates"] == 0:
                    checkpoint("pretrain", step, optimizer)
        diagnostics = _forward_representation_diagnostics(
            model.to("cpu"), paired_past, paired_mask, future, future_mask,
            row_ids=list(pairs.row_ids), cutoff_interval_ids=pairs.cutoff_interval_ids,
            source_sha256=pairs.source_archive_sha256,
        )
        model.to(device)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=model_config["learning_rate"], weight_decay=model_config["weight_decay"],
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        labeled = np.flatnonzero(fit_y_mask.any(dim=1).numpy())
        if not len(labeled):
            raise ValueError("AEON forward supervised TRAIN has no labels.")
        quantiles = _QUANTILES.to(device)
        final_checkpoint: Path | None = None
        for step in range(1, model_config["supervised_updates"] + 1):
            model.train()
            selected = rng.choice(
                labeled, size=model_config["batch_size"], replace=len(labeled) < model_config["batch_size"]
            )
            predicted = model(fit_x[selected].to(device), fit_mask[selected].to(device))
            truth = fit_y[selected].to(device)
            valid = fit_y_mask[selected].to(device)
            residual = truth[:, :, None] - predicted
            loss = torch.maximum(quantiles * residual, (quantiles - 1) * residual)[valid].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("AEON forward supervised loss is non-finite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                model_config["gradient_clip_norm"], error_if_nonfinite=True,
            )
            optimizer.step()
            check_resources()
            if step % model_config["checkpoint_every_updates"] == 0:
                final_checkpoint = checkpoint("supervised", step, optimizer)
        if final_checkpoint is None:
            raise ValueError("AEON forward lacks final supervised checkpoint.")
        model.eval()
        predictions = []
        with torch.no_grad():
            for first in range(0, len(assess), model_config["batch_size"]):
                part = slice(first, first + model_config["batch_size"])
                predicted = model(assess_x[part].to(device), assess_mask[part].to(device))
                predictions.append(predicted.cpu().numpy() * scaler[3] + scaler[2])
        forecast = np.concatenate(predictions)
        prediction_path = stage / "validation-predictions.npz"
        _save_predictions(prediction_path, assess, forecast)
        saved_rows, saved_forecast = _load_prediction(prediction_path)
        if not np.array_equal(saved_forecast, forecast):
            raise ValueError("AEON forward saved validation forecast differs.")
        metrics = daily_pinball(
            saved_rows["truth_db"], forecast, saved_rows["target_mask"],
            saved_rows["target_source_timestamps"],
        )
        result = {
            "slot_id": slot_id, "seed": seed, "source_partition": "train",
            "assessment_partition": "validation", "classification": "DEVELOPMENT_NOT_FINAL_EVALUATION",
            "pretrain_updates": 0 if is_random else model_config["pretrain_updates"],
            "supervised_updates": model_config["supervised_updates"],
            "random_unequal_total_learned_update_budget": is_random,
            "shuffled_false_future_pairs": is_shuffled,
            "checkpoint_selection": "FINAL_ENDPOINT_ONLY",
            "final_checkpoint_path": final_checkpoint.name,
            "final_checkpoint_sha256": _sha256(final_checkpoint),
            "prediction_path": prediction_path.name,
            "prediction_sha256": _sha256(prediction_path),
            "train_representation_diagnostics": diagnostics,
            "protocol_validation_metrics": metrics,
            "checkpoints": checkpoints,
            "peak_process_rss_bytes": peak_rss,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "fit_seconds": time.perf_counter() - start_time,
            "device": device,
        }
        (stage / "slot.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        torch.set_num_threads(previous_threads)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _verify_done(directory: Path, entry: dict[str, Any]) -> None:
    slot_path = _artifact(directory, "slot.json", entry.get("slot_sha256"))
    slot = json.loads(slot_path.read_text(encoding="utf-8"))
    if slot.get("slot_id") != entry.get("slot_id"):
        raise ValueError("AEON forward completed slot identity differs.")
    _artifact(directory, slot.get("prediction_path"), slot.get("prediction_sha256"))
    _artifact(directory, slot.get("final_checkpoint_path"), slot.get("final_checkpoint_sha256"))
    for checkpoint in slot.get("checkpoints", []):
        _artifact(directory, checkpoint.get("path"), checkpoint.get("sha256"))


def run_forward_campaign(
    archive: Path,
    split_review: Path,
    core_config_path: Path,
    core_campaign_output: Path,
    core_rescore_path: Path,
    pair_inventory_path: Path,
    config_path: Path,
    prefit_review_path: Path,
    output: Path,
) -> dict[str, Any]:
    """No fitting without exact independent code/config/cohort/pair prefit approval."""
    config_sha256 = _sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _config_gate(config)
    core_config_sha256 = _sha256(core_config_path)
    manifest_path = core_campaign_output / "manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rescore_sha256 = _sha256(core_rescore_path)
    rescore = json.loads(core_rescore_path.read_text(encoding="utf-8"))
    inventory_sha256 = _sha256(pair_inventory_path)
    inventory = json.loads(pair_inventory_path.read_text(encoding="utf-8"))
    split_sha256 = _sha256(split_review)
    review_sha256 = _sha256(prefit_review_path)
    review = json.loads(prefit_review_path.read_text(encoding="utf-8"))
    code_sha256 = _code_sha256()
    if (
        config.get("core_campaign_config_sha256") != core_config_sha256
        or config.get("core_campaign_manifest_sha256") != manifest_sha256
        or config.get("core_validation_rescore_sha256") != rescore_sha256
        or config.get("pair_inventory_sha256") != inventory_sha256
        or config.get("split_review_sha256") != split_sha256
        or manifest.get("status") != "COMPLETED_TRAIN_VALIDATION_CAMPAIGN_NOT_FINAL_EVALUATION"
        or manifest.get("code_sha256") != _campaign_code_sha256()
        or rescore.get("campaign_manifest_sha256") != manifest_sha256
        or rescore.get("status") != "PROTOCOL_VALIDATION_RESCORE_PENDING_INDEPENDENT_SELECTION_REVIEW"
        or review.get("status") != "APPROVED_AEON_FORWARD_PREFIT"
        or review.get("code_sha256") != code_sha256
        or review.get("config_sha256") != config_sha256
        or review.get("split_review_sha256") != split_sha256
        or review.get("pair_inventory_sha256") != inventory_sha256
        or review.get("cohort_sha256") != config.get("cohort_sha256")
        or review.get("test_access") != "PROHIBITED"
    ):
        raise ValueError("AEON forward lacks exact independent prefit review.")
    fit, assess, cohort_sha256, actual_split_sha256 = load_cohort(archive, split_review)
    pair_fit, pairs, _ = load_train_forward_cohort(archive, split_review)
    if (
        cohort_sha256 != config["cohort_sha256"]
        or actual_split_sha256 != split_sha256
        or [row.row_id for row in fit] != [row.row_id for row in pair_fit]
    ):
        raise ValueError("AEON forward TRAIN/validation cohort differs from reviewed core.")
    actual_inventory = pair_inventory(fit, pairs)
    for key in (
        "pair_id_and_interval_sha256", "pair_target_values_and_mask_sha256", "six_future_pair_rows"
    ):
        if actual_inventory[key] != inventory[key]:
            raise ValueError("AEON forward pair inventory differs from approved TRAIN rows.")
    if (
        inventory["six_future_pair_rows"] != config["pretrain_pair_rows"]
        or inventory["pair_id_and_interval_sha256"] != config["pair_id_and_interval_sha256"]
        or inventory["pair_target_values_and_mask_sha256"] != config["pair_target_values_and_mask_sha256"]
    ):
        raise ValueError("AEON forward pair support differs from frozen config.")
    reference = {}
    for seed in (7, 13, 23):
        slot_id = f"direct_seed{seed}"
        entry = manifest["slots"][slot_id]
        directory = core_campaign_output / slot_id
        slot = json.loads(_artifact(
            directory, "slot.json", entry["slot_sha256"]
        ).read_text(encoding="utf-8"))
        prediction_path = _artifact(directory, slot["prediction_path"], slot["prediction_sha256"])
        rows, _ = _load_prediction(prediction_path)
        if (
            rescore["slots"][slot_id]["prediction_sha256"] != slot["prediction_sha256"]
            or not np.array_equal(rows["row_ids"], [row.row_id for row in assess])
            or not np.array_equal(rows["target_mask"], np.stack([row.target_mask for row in assess]))
        ):
            raise ValueError("AEON forward direct reference differs in validation support.")
        reference[slot_id] = {
            "prediction_sha256": slot["prediction_sha256"],
            "protocol_validation_metrics": rescore["slots"][slot_id]["protocol_validation_metrics"],
        }
    device = _resolve_device(config)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    identity = {
        "device": device,
        "status": "TRAIN_VALIDATION_FORWARD_IN_PROGRESS",
        "classification": "POST_HOC_DEVELOPMENT_NOT_FINAL_EVALUATION",
        "test_access": "PROHIBITED",
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "review_sha256": review_sha256,
        "cohort_sha256": cohort_sha256,
        "pair_inventory_sha256": inventory_sha256,
        "core_campaign_manifest_sha256": manifest_sha256,
        "core_validation_rescore_sha256": rescore_sha256,
        "direct_reference": reference,
    }
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(current.get(key) != value for key, value in identity.items() if key != "status"):
            raise ValueError("AEON forward restart identity differs.")
        entries = current.get("slots")
        if not isinstance(entries, dict):
            raise ValueError("AEON forward restart manifest is malformed.")
    else:
        entries = {}
        current = {**identity, "slots": entries}
        _atomic_json(manifest_path, current)
    for slot_id in FORWARD_SLOTS:
        prior = entries.get(slot_id)
        if prior is not None and prior.get("status") == "DONE":
            _verify_done(output / slot_id, prior)
            continue
        if (output / slot_id).exists():
            raise ValueError("AEON forward output exists without a DONE ledger entry.")
        stage = Path(tempfile.mkdtemp(prefix=slot_id + ".stage.", dir=output))
        try:
            result = _train_slot(
                slot_id, fit, assess, pairs, config, stage, device,
                config_sha256=config_sha256,
            )
            os.replace(stage, output / slot_id)
            entries[slot_id] = {
                "status": "DONE", "slot_id": slot_id,
                "slot_sha256": _sha256(output / slot_id / "slot.json"),
                "prediction_sha256": result["prediction_sha256"],
            }
            _atomic_json(manifest_path, current)
        except BaseException:
            if stage.exists() and stage.resolve().is_relative_to(output.resolve()):
                shutil.rmtree(stage)
            entries[slot_id] = {"status": "FAILED", "slot_id": slot_id}
            _atomic_json(manifest_path, current)
            raise
    current["status"] = "COMPLETED_FORWARD_TRAIN_VALIDATION_DEVELOPMENT_NOT_FINAL_EVALUATION"
    _atomic_json(manifest_path, current)
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "archive", "split-review", "core-config", "core-campaign-output",
        "core-rescore", "pair-inventory", "config", "prefit-review", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_forward_campaign(
        arguments.archive, arguments.split_review, arguments.core_config,
        arguments.core_campaign_output, arguments.core_rescore, arguments.pair_inventory,
        arguments.config, arguments.prefit_review, arguments.output,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


