"""Finite TRAIN-only EMA/SIGReg pretraining, probes, controls and frozen hybrids."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil
import torch
from numpy.typing import NDArray

from marine_echo.models.compact import ModelConfig
from marine_echo.models.v2_development import (
    JointJEPAForecaster,
    JointPrediction,
    joint_supervised_loss,
    past_features,
)
from marine_echo.models.v2_hybrid import NativeHybridRidge
from marine_echo.training.loop import save_checkpoint
from marine_echo.training.v2_executor import (
    V2_PROTOCOL_SHA256,
    _Scaler,
    _sha256,
    _tensors,
    _valid_hash,
    _validate_rows,
    _verify_predictions,
    _write_predictions,
)
from marine_echo.training.v2_stream import HourlyWindow

Family = Literal["ema_jepa", "shared_sigreg"]
Control = Literal["random_encoder", "shuffled_future"]


def _future(rows: list[HourlyWindow], scaler: _Scaler) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.stack([row.future_train_db for row in rows])
    mask = np.stack([row.future_train_mask for row in rows])
    if values.shape != (len(rows), 3, 4, 4, 64) or mask.shape != values.shape:
        raise ValueError("JEPA TRAIN future must have fixed observed hour bins and masks.")
    if not np.isfinite(values[mask]).all():
        raise ValueError("Observed JEPA future has non-finite values.")
    normalized = np.where(mask, (values - scaler.context_mean) / scaler.context_std, 0)
    return torch.as_tensor(normalized, dtype=torch.float32), torch.as_tensor(mask, dtype=torch.bool)


def _separated_shuffle(rows: list[HourlyWindow]) -> NDArray[np.int64]:
    """Choose a fixed cyclic TRAIN permutation with >=24 hours between paired cutoffs."""
    times = np.array([row.cutoff for row in rows], dtype="datetime64[ns]")
    count = len(rows)
    for offset in range(1, count):
        mapping = (np.arange(count) + offset) % count
        if np.all(np.abs(times - times[mapping]) >= np.timedelta64(24, "h")):
            return mapping.astype(np.int64)
    raise ValueError("No TRAIN-only future shuffle satisfies the 24-hour separation.")


def _representation_features(
    model: JointJEPAForecaster,
    rows: list[HourlyWindow],
    tensors: Any,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    features = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            part = slice(start, start + batch_size)
            encoded, valid = model.base.encoder(
                tensors.context[part].to(device), tensors.mask[part].to(device)
            )
            current = torch.where(valid[..., None], encoded, 0).sum(dim=1) / valid.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            predicted = model.base.predictor(encoded, valid).mean(dim=2).flatten(start_dim=1)
            features.append(torch.cat((current, predicted), dim=1).cpu().numpy())
    raw = np.stack([past_features(row) for row in rows])
    return np.concatenate((raw, np.concatenate(features)), axis=1).astype(np.float64)


def _campaign_gate(
    *,
    fixture_only: bool,
    review_path: Path | None,
    review_sha256: str | None,
    protocol_sha256: str,
    family: Family,
    control: Control | None,
    seed: int,
    pretrain_updates: int,
    probe_updates: int,
    source_hashes: tuple[str, ...],
) -> None:
    if fixture_only:
        return
    if review_path is None or review_sha256 is None or not _valid_hash(review_sha256):
        raise ValueError("Independent v2 representation campaign review is required.")
    if _sha256(review_path) != review_sha256:
        raise ValueError("Representation campaign review digest differs.")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("status") != "APPROVED_V2_TRAIN_REPRESENTATION"
        or review.get("candidate_id") != "v2_candidate2"
        or review.get("protocol_sha256") != protocol_sha256
        or review.get("family") != family
        or review.get("control") != control
        or review.get("seed") != seed
        or review.get("pretrain_updates") != pretrain_updates
        or review.get("probe_updates") != probe_updates
        or not set(source_hashes).issubset(set(review.get("source_sha256", [])))
    ):
        raise ValueError("Representation campaign review does not authorize these inputs.")


def execute_representation(
    fit: list[HourlyWindow],
    assess: list[HourlyWindow],
    output: Path,
    *,
    family: Family,
    protocol_sha256: str,
    control: Control | None = None,
    fixture_only: bool = False,
    review_path: Path | None = None,
    review_sha256: str | None = None,
    model_config: ModelConfig | None = None,
    seed: int = 7,
    pretrain_updates: int = 128,
    probe_updates: int = 128,
    batch_size: int = 16,
    device: str = "cpu",
    representation_regularizer_weight: float = 0.03,
    sigreg_weight: float = 0.04,
) -> dict[str, Any]:
    """Execute one declared family/seed/control; never inspect protected partitions."""
    if family not in ("ema_jepa", "shared_sigreg") or control not in (
        None,
        "random_encoder",
        "shuffled_future",
    ):
        raise ValueError("Unknown JEPA family or control.")
    if (
        seed not in (7, 13, 23)
        or (control is not None and seed != 7)
        or not 1 <= pretrain_updates <= 3000
        or not 1 <= probe_updates <= 3000
        or not 1 <= batch_size <= 16
        or device not in ("cpu", "cuda")
        or not _valid_hash(protocol_sha256)
        or (family == "ema_jepa" and representation_regularizer_weight not in (0.03, 0.1))
        or (family == "shared_sigreg" and sigreg_weight not in (0.04, 0.1))
    ):
        raise ValueError("Representation config exceeds the frozen finite grid.")
    if not fixture_only and protocol_sha256 != V2_PROTOCOL_SHA256:
        raise ValueError("Real representation requires the reviewed v2 protocol digest.")
    if not fixture_only and batch_size != 16:
        raise ValueError("Real representation requires batch size 16.")
    if not fixture_only and model_config is not None and model_config != ModelConfig():
        raise ValueError("Real representation requires the default compact model configuration.")
    input_digest, source_hashes = _validate_rows(fit, assess, fixture_only=fixture_only)
    _campaign_gate(
        fixture_only=fixture_only,
        review_path=review_path,
        review_sha256=review_sha256,
        protocol_sha256=protocol_sha256,
        family=family,
        control=control,
        seed=seed,
        pretrain_updates=pretrain_updates,
        probe_updates=probe_updates,
        source_hashes=source_hashes,
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Representation run output already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        torch.manual_seed(seed)
        process = psutil.Process()
        peak_rss = process.memory_info().rss
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        scaler = _Scaler.fit(fit)
        train = _tensors(fit, scaler)
        assessment = _tensors(assess, scaler)
        future, future_mask = _future(fit, scaler)
        shuffle = _separated_shuffle(fit) if control == "shuffled_future" else np.arange(len(fit))
        model = JointJEPAForecaster(
            model_config or ModelConfig(),
            mode="ema" if family == "ema_jepa" else "shared_sigreg",
            sigreg_weight=sigreg_weight,
            ema_regularizer_weight=representation_regularizer_weight,
        ).to(device)
        pretrain_checkpoint: str | None = None
        started = time.perf_counter()
        actual_pretrain = 0 if control == "random_encoder" else pretrain_updates
        if actual_pretrain:
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=3e-4,
                weight_decay=1e-4,
                betas=(0.9, 0.95),
            )
            candidates = np.flatnonzero(future_mask[shuffle].any(dim=(1, 2, 3, 4)).numpy())
            if not len(candidates):
                raise ValueError("No observed TRAIN future tokens for JEPA pretraining.")
            model.train()
            for step in range(actual_pretrain):
                rng = np.random.default_rng(np.random.SeedSequence([seed, step, 1]))
                chosen = rng.choice(
                    candidates, size=batch_size, replace=len(candidates) < batch_size
                )
                targets = shuffle[chosen]
                for group in optimizer.param_groups:
                    group["lr"] = 3e-4 * min(1.0, (step + 1) / 150)
                optimizer.zero_grad(set_to_none=True)
                loss = model.pretrain_objective(
                    train.context[chosen].to(device),
                    train.mask[chosen].to(device),
                    future[targets].to(device),
                    future_mask[targets].to(device),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("JEPA TRAIN pretraining loss is non-finite.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if family == "ema_jepa":
                    model.base.update_teacher(momentum=0.99)
                peak_rss = max(peak_rss, process.memory_info().rss)
                if peak_rss >= 22 * 1024**3:
                    raise MemoryError("JEPA process RAM reached the 22 GiB limit.")
                if device == "cuda" and torch.cuda.max_memory_reserved() >= 10 * 1024**3:
                    raise MemoryError("JEPA GPU reserve reached the 10 GiB target.")
            checkpoint_path = staging / f"{family}-pretrain-{actual_pretrain}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step=actual_pretrain,
                protocol_sha256=protocol_sha256,
            )
            pretrain_checkpoint = str(output / checkpoint_path.name)
        model.freeze_encoder()
        probe_optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=3e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.95),
        )
        candidates = np.flatnonzero((train.target_mask | train.fraction_mask).any(dim=1).numpy())
        if not len(candidates):
            raise ValueError("No observed TRAIN index or fraction for the JEPA probe.")
        model.train()
        for step in range(probe_updates):
            rng = np.random.default_rng(np.random.SeedSequence([seed, step, 2]))
            chosen = rng.choice(candidates, size=batch_size, replace=len(candidates) < batch_size)
            for group in probe_optimizer.param_groups:
                group["lr"] = 3e-4 * min(1.0, (step + 1) / 150)
            probe_optimizer.zero_grad(set_to_none=True)
            prediction = model(
                train.context[chosen].to(device),
                train.mask[chosen].to(device),
                train.aux[chosen].to(device),
            )
            loss = joint_supervised_loss(
                prediction,
                train.target[chosen].to(device),
                train.target_mask[chosen].to(device),
                train.fraction[chosen].to(device),
                train.fraction_mask[chosen].to(device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("JEPA TRAIN probe loss is non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            probe_optimizer.step()
            peak_rss = max(peak_rss, process.memory_info().rss)
            if peak_rss >= 22 * 1024**3:
                raise MemoryError("JEPA process RAM reached the 22 GiB limit.")
            if device == "cuda" and torch.cuda.max_memory_reserved() >= 10 * 1024**3:
                raise MemoryError("JEPA GPU reserve reached the 10 GiB target.")
        probe_path = staging / f"{family}-probe-{probe_updates}.pt"
        save_checkpoint(
            probe_path,
            model,
            probe_optimizer,
            step=probe_updates,
            protocol_sha256=protocol_sha256,
        )
        elapsed = time.perf_counter() - started
        model.eval()
        quantiles = []
        fractions = []
        with torch.no_grad():
            for start in range(0, len(assess), batch_size):
                part = slice(start, start + batch_size)
                prediction = model(
                    assessment.context[part].to(device),
                    assessment.mask[part].to(device),
                    assessment.aux[part].to(device),
                )
                quantiles.append(
                    prediction.quantiles.cpu().numpy() * scaler.target_std + scaler.target_mean
                )
                fractions.append(prediction.detection_fraction.cpu().numpy())
        representation = JointPrediction(
            quantiles_db=np.concatenate(quantiles),
            detection_fraction=np.concatenate(fractions),
        )
        representation_path = staging / "representation-assessment-predictions.npz"
        _write_predictions(representation_path, assess, representation)
        result: dict[str, Any] = {
            "status": "COMPLETED_SYNTHETIC_FIXTURE"
            if fixture_only
            else "COMPLETED_TRAIN_DEVELOPMENT",
            "family": family,
            "control": control,
            "seed": seed,
            "protocol_sha256": protocol_sha256,
            "input_sha256": input_digest,
            "source_sha256": source_hashes,
            "review_sha256": review_sha256 if not fixture_only else None,
            "model_config": asdict(model_config or ModelConfig()),
            "pretrain_updates": actual_pretrain,
            "probe_updates": probe_updates,
            "pretrain_checkpoint": pretrain_checkpoint,
            "probe_checkpoint": str(output / probe_path.name),
            "probe_checkpoint_sha256": _sha256(probe_path),
            "train_scaler": scaler.__dict__,
            "train_seconds": elapsed,
            "device": device,
            "peak_process_rss_bytes": peak_rss,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()
            if device == "cuda"
            else None,
            "representation": {
                "predictions": str(output / representation_path.name),
                "prediction_sha256": _sha256(representation_path),
                "metrics": _verify_predictions(representation_path, assess),
            },
        }
        if pretrain_checkpoint is not None:
            result["pretrain_checkpoint_sha256"] = _sha256(staging / Path(pretrain_checkpoint).name)
        if control is None:
            xtrain = _representation_features(
                model, fit, train, batch_size=batch_size, device=device
            )
            xassess = _representation_features(
                model, assess, assessment, batch_size=batch_size, device=device
            )
            hybrid = NativeHybridRidge().fit(
                xtrain,
                np.stack([row.target_db for row in fit]),
                np.stack([row.target_mask for row in fit]),
                np.stack([row.target_detection_fraction for row in fit]),
                np.stack([row.target_detection_mask for row in fit]),
                partition="train",
            )
            hybrid_prediction = hybrid.predict(xassess)
            hybrid_path = staging / "hybrid-assessment-predictions.npz"
            hybrid_head = staging / "hybrid-head.npz"
            hybrid.save(hybrid_head)
            _write_predictions(hybrid_path, assess, hybrid_prediction)
            result["hybrid"] = {
                "predictions": str(output / hybrid_path.name),
                "prediction_sha256": _sha256(hybrid_path),
                "head": str(output / hybrid_head.name),
                "head_sha256": _sha256(hybrid_head),
                "metrics": _verify_predictions(hybrid_path, assess),
            }
        (staging / "run.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return result
    except BaseException:
        if staging.exists() and staging.resolve().parent == output.parent.resolve():
            shutil.rmtree(staging)
        raise
