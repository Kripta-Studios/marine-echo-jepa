"""Finite v2 TRAIN-development ridge/direct execution with immutable row artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from marine_echo.models.compact import ModelConfig
from marine_echo.models.v2_development import (
    QUANTILES,
    DevelopmentRidge,
    JointDirectForecaster,
    JointPrediction,
    joint_supervised_loss,
)
from marine_echo.training.loop import load_checkpoint, save_checkpoint
from marine_echo.training.v2_stream import HourlyWindow

V2_PROTOCOL_SHA256 = "0270b9fc88b2511dc931126fe980a5cd6cc411f88a3579964f4cf4501d325029"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _valid_hash(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_rows(
    fit: list[HourlyWindow], assess: list[HourlyWindow], *, fixture_only: bool
) -> tuple[str, tuple[str, ...]]:
    if not fit or not assess or any(row.partition != "train" for row in fit + assess):
        raise ValueError("Only nonempty TRAIN development and assessment rows are allowed.")
    for rows in (fit, assess):
        if len({row.row_id for row in rows}) != len(rows):
            raise ValueError("Development rows contain duplicate identities.")
        if any(left.cutoff >= right.cutoff for left, right in pairwise(rows)):
            raise ValueError("Development rows must be chronological.")
    if fit[-1].cutoff + np.timedelta64(6, "h") > assess[0].cutoff - np.timedelta64(24, "h"):
        raise ValueError("Fit and assessment raw support would overlap.")
    if not fixture_only:
        fit_start, fit_end = np.datetime64("2020-02-17"), np.datetime64("2020-04-01")
        assess_start, assess_end = np.datetime64("2020-04-01"), np.datetime64("2020-04-15")
        if (
            fit[0].cutoff - np.timedelta64(24, "h") < fit_start
            or fit[-1].cutoff + np.timedelta64(6, "h") > fit_end
            or assess[0].cutoff - np.timedelta64(24, "h") < assess_start
            or assess[-1].cutoff + np.timedelta64(6, "h") > assess_end
        ):
            raise ValueError("Rows exceed the predeclared TRAIN development calendar.")
        if any(row.context_index_db is None or not row.past_source_sha256 for row in fit + assess):
            raise ValueError("Reviewed native rows need exact past index and past provenance.")
    source_hashes = tuple(sorted({digest for row in fit + assess for digest in row.source_sha256}))
    if not source_hashes or not all(_valid_hash(digest) for digest in source_hashes):
        raise ValueError("Every development row needs registered real source hashes.")
    digest = hashlib.sha256()
    for row in fit + assess:
        digest.update(row.row_id.encode())
        digest.update(np.datetime64(row.cutoff, "ns").tobytes())
        for field in (
            row.context,
            row.context_mask,
            row.context_acquisition_fraction,
            row.context_detection_fraction,
            row.target_interval_start,
            row.target_interval_end,
            row.target_db,
            row.target_mask,
            row.target_detection_fraction,
            row.target_detection_mask,
        ):
            digest.update(np.ascontiguousarray(field).tobytes())
        if row.context_index_db is not None:
            digest.update(np.ascontiguousarray(row.context_index_db).tobytes())
        digest.update(";".join(row.source_sha256).encode())
        digest.update(";".join(row.past_source_sha256).encode())
        digest.update(";".join(row.target_source_sha256).encode())
    return digest.hexdigest(), source_hashes


def _review_gate(
    *,
    fixture_only: bool,
    review_path: Path | None,
    review_sha256: str | None,
    protocol_sha256: str,
    source_hashes: tuple[str, ...],
) -> None:
    if fixture_only:
        return
    if review_path is None or review_sha256 is None or not _valid_hash(review_sha256):
        raise ValueError("Reviewed v2 TRAIN development approval and digest are required.")
    if _sha256(review_path) != review_sha256:
        raise ValueError("V2 development review digest differs.")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("status") != "APPROVED_V2_TRAIN_DEVELOPMENT"
        or review.get("candidate_id") != "v2_candidate2"
        or review.get("protocol_sha256") != protocol_sha256
        or not set(source_hashes).issubset(set(review.get("source_sha256", [])))
    ):
        raise ValueError("V2 development review does not authorize these inputs.")


@dataclass(frozen=True)
class _Scaler:
    context_mean: float
    context_std: float
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, rows: list[HourlyWindow]) -> _Scaler:
        if any(row.partition != "train" for row in rows):
            raise ValueError("Normalizers fit TRAIN rows only.")
        contexts = np.concatenate([row.context[row.context_mask] for row in rows])
        targets = np.concatenate([row.target_db[row.target_mask] for row in rows])
        if (
            not len(contexts)
            or not len(targets)
            or not np.isfinite(contexts).all()
            or not np.isfinite(targets).all()
        ):
            raise ValueError("TRAIN context and index targets must contain finite observations.")
        return cls(
            context_mean=float(contexts.mean()),
            context_std=max(float(contexts.std()), 1e-6),
            target_mean=float(targets.mean()),
            target_std=max(float(targets.std()), 1e-6),
        )


@dataclass(frozen=True)
class _Tensors:
    context: torch.Tensor
    mask: torch.Tensor
    aux: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    fraction: torch.Tensor
    fraction_mask: torch.Tensor


def _tensors(rows: list[HourlyWindow], scaler: _Scaler) -> _Tensors:
    context = np.stack([row.context for row in rows])
    mask = np.stack([row.context_mask for row in rows])
    normalized = np.where(mask, (context - scaler.context_mean) / scaler.context_std, 0)
    auxiliary = np.stack(
        [
            np.stack(
                (
                    row.context_acquisition_fraction,
                    row.context_detection_fraction,
                    row.context_age_minutes / 1440,
                    np.isfinite(row.context_detection_fraction).astype(float),
                ),
                axis=-1,
            )
            for row in rows
        ]
    )
    target = (np.stack([row.target_db for row in rows]) - scaler.target_mean) / scaler.target_std
    return _Tensors(
        context=torch.as_tensor(normalized, dtype=torch.float32),
        mask=torch.as_tensor(mask, dtype=torch.bool),
        aux=torch.as_tensor(np.nan_to_num(auxiliary), dtype=torch.float32),
        target=torch.as_tensor(target, dtype=torch.float32),
        target_mask=torch.as_tensor(np.stack([row.target_mask for row in rows]), dtype=torch.bool),
        fraction=torch.as_tensor(
            np.stack([row.target_detection_fraction for row in rows]), dtype=torch.float32
        ),
        fraction_mask=torch.as_tensor(
            np.stack([row.target_detection_mask for row in rows]), dtype=torch.bool
        ),
    )


def _score(rows: list[HourlyWindow], prediction: JointPrediction) -> dict[str, Any]:
    truth = np.stack([row.target_db for row in rows])
    mask = np.stack([row.target_mask for row in rows])
    fraction = np.stack([row.target_detection_fraction for row in rows])
    fraction_mask = np.stack([row.target_detection_mask for row in rows])
    result: dict[str, Any] = {"issued_rows": len(rows), "horizons": []}
    for horizon in range(3):
        valid = mask[:, horizon]
        fraction_valid = fraction_mask[:, horizon]
        q = prediction.quantiles_db[valid, horizon]
        observed = truth[valid, horizon]
        error = observed[:, None] - q
        pinball = np.maximum(np.asarray(QUANTILES) * error, (np.asarray(QUANTILES) - 1) * error)
        days = np.array([row.cutoff for row in rows]).astype("datetime64[D]")[valid]
        day_losses = [pinball[days == day].mean() for day in np.unique(days)]
        result["horizons"].append(
            {
                "index_rows": int(valid.sum()),
                "index_days": len(np.unique(days)),
                "daily_mean_pinball_db": float(np.mean(day_losses)) if day_losses else None,
                "fraction_rows": int(fraction_valid.sum()),
                "fraction_mae": float(
                    np.abs(
                        prediction.detection_fraction[fraction_valid, horizon]
                        - fraction[fraction_valid, horizon]
                    ).mean()
                )
                if fraction_valid.any()
                else None,
                "fraction_mse": float(
                    np.square(
                        prediction.detection_fraction[fraction_valid, horizon]
                        - fraction[fraction_valid, horizon]
                    ).mean()
                )
                if fraction_valid.any()
                else None,
            }
        )
    return result


def _write_predictions(path: Path, rows: list[HourlyWindow], prediction: JointPrediction) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            row_ids=np.array([row.row_id for row in rows]),
            cutoffs=np.array([row.cutoff for row in rows]),
            past_source_sha256=np.array([",".join(row.past_source_sha256) for row in rows]),
            target_source_sha256=np.array([",".join(row.target_source_sha256) for row in rows]),
            target_interval_start=np.stack([row.target_interval_start for row in rows]),
            target_interval_end=np.stack([row.target_interval_end for row in rows]),
            truth_db=np.stack([row.target_db for row in rows]),
            target_mask=np.stack([row.target_mask for row in rows]),
            truth_detection_fraction=np.stack([row.target_detection_fraction for row in rows]),
            detection_mask=np.stack([row.target_detection_mask for row in rows]),
            target_acquisition_fraction=np.stack([row.target_acquisition_fraction for row in rows]),
            quantiles_db=prediction.quantiles_db,
            detection_fraction=prediction.detection_fraction,
        )


def _verify_predictions(path: Path, rows: list[HourlyWindow]) -> dict[str, Any]:
    """Reconstruct all reported scores from immutable saved prediction rows."""
    with np.load(path, allow_pickle=False) as saved:
        expected = {
            "row_ids",
            "cutoffs",
            "past_source_sha256",
            "target_source_sha256",
            "target_interval_start",
            "target_interval_end",
            "truth_db",
            "target_mask",
            "truth_detection_fraction",
            "detection_mask",
            "target_acquisition_fraction",
            "quantiles_db",
            "detection_fraction",
        }
        if set(saved.files) != expected:
            raise ValueError("Saved development prediction fields differ.")
        checks = {
            "row_ids": np.array([row.row_id for row in rows]),
            "cutoffs": np.array([row.cutoff for row in rows]),
            "past_source_sha256": np.array([",".join(row.past_source_sha256) for row in rows]),
            "target_source_sha256": np.array([",".join(row.target_source_sha256) for row in rows]),
            "target_interval_start": np.stack([row.target_interval_start for row in rows]),
            "target_interval_end": np.stack([row.target_interval_end for row in rows]),
            "truth_db": np.stack([row.target_db for row in rows]),
            "target_mask": np.stack([row.target_mask for row in rows]),
            "truth_detection_fraction": np.stack([row.target_detection_fraction for row in rows]),
            "detection_mask": np.stack([row.target_detection_mask for row in rows]),
            "target_acquisition_fraction": np.stack(
                [row.target_acquisition_fraction for row in rows]
            ),
        }
        if any(
            not np.array_equal(
                saved[key],
                value,
                equal_nan=saved[key].dtype.kind in "fc",
            )
            for key, value in checks.items()
        ):
            raise ValueError(
                "Saved prediction cohort, truth or provenance differs from issued rows."
            )
        quantiles = saved["quantiles_db"]
        fraction = saved["detection_fraction"]
        if (
            quantiles.shape != (len(rows), 3, 5)
            or fraction.shape != (len(rows), 3)
            or not np.isfinite(quantiles).all()
            or not np.isfinite(fraction).all()
            or (np.diff(quantiles, axis=-1) < 0).any()
            or ((fraction < 0) | (fraction > 1)).any()
        ):
            raise ValueError("Saved development predictions are invalid or incomplete.")
        prediction = JointPrediction(quantiles_db=quantiles, detection_fraction=fraction)
        return _score(rows, prediction)


def _train_direct(
    fit: list[HourlyWindow],
    assess: list[HourlyWindow],
    output: Path,
    *,
    protocol_sha256: str,
    model_config: ModelConfig,
    updates: int,
    batch_size: int,
    device: str,
) -> tuple[JointPrediction, str, _Scaler, dict[str, Any]]:
    torch.manual_seed(7)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    scaler = _Scaler.fit(fit)
    train = _tensors(fit, scaler)
    assessment = _tensors(assess, scaler)
    model = JointDirectForecaster(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.95))
    candidates = np.flatnonzero((train.target_mask | train.fraction_mask).any(dim=1).numpy())
    if not len(candidates):
        raise ValueError("No labelled TRAIN update rows exist.")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    def update_range(
        current_model: JointDirectForecaster,
        current_optimizer: torch.optim.Optimizer,
        first: int,
        last: int,
    ) -> None:
        nonlocal peak_rss
        current_model.train()
        for step in range(first, last):
            rng = np.random.default_rng(np.random.SeedSequence([7, step]))
            selected = rng.choice(candidates, size=batch_size, replace=len(candidates) < batch_size)
            rate = 3e-4 * min(1.0, (step + 1) / 16)
            for group in current_optimizer.param_groups:
                group["lr"] = rate
            current_optimizer.zero_grad(set_to_none=True)
            prediction = current_model(
                train.context[selected].to(device),
                train.mask[selected].to(device),
                train.aux[selected].to(device),
            )
            loss = joint_supervised_loss(
                prediction,
                train.target[selected].to(device),
                train.target_mask[selected].to(device),
                train.fraction[selected].to(device),
                train.fraction_mask[selected].to(device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Direct TRAIN update loss is non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0, error_if_nonfinite=True)
            current_optimizer.step()
            peak_rss = max(peak_rss, process.memory_info().rss)
            if peak_rss >= 22 * 1024**3:
                raise MemoryError("Direct TRAIN process RAM reached the 22 GiB limit.")
            if device == "cuda" and torch.cuda.max_memory_reserved() >= 10 * 1024**3:
                raise MemoryError("Direct TRAIN GPU reserve reached the 10 GiB target.")

    started = time.perf_counter()
    midpoint = updates // 2 if updates > 1 else 1
    update_range(model, optimizer, 0, midpoint)
    midway_path = output / f"checkpoint-{midpoint}.pt"
    save_checkpoint(midway_path, model, optimizer, step=midpoint, protocol_sha256=protocol_sha256)
    update_range(model, optimizer, midpoint, updates)
    checkpoint = output / f"checkpoint-{updates}.pt"
    if checkpoint != midway_path:
        save_checkpoint(checkpoint, model, optimizer, step=updates, protocol_sha256=protocol_sha256)
    elapsed = time.perf_counter() - started
    verification_started = time.perf_counter()
    resumed = JointDirectForecaster(model_config).to(device)
    resumed_optimizer = torch.optim.AdamW(
        resumed.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.95)
    )
    if (
        load_checkpoint(midway_path, resumed, resumed_optimizer, protocol_sha256=protocol_sha256)
        != midpoint
    ):
        raise ValueError("Direct midpoint checkpoint has the wrong update count.")
    update_range(resumed, resumed_optimizer, midpoint, updates)
    for key, value in model.state_dict().items():
        if not torch.allclose(value, resumed.state_dict()[key], rtol=1e-6, atol=1e-7):
            raise ValueError(f"Direct checkpoint resume diverged in {key}.")
    verification_elapsed = time.perf_counter() - verification_started
    model.eval()
    quantiles = []
    fractions = []
    with torch.no_grad():
        for start in range(0, len(assess), batch_size):
            part = slice(start, start + batch_size)
            forecast = model(
                assessment.context[part].to(device),
                assessment.mask[part].to(device),
                assessment.aux[part].to(device),
            )
            quantiles.append(
                forecast.quantiles.cpu().numpy() * scaler.target_std + scaler.target_mean
            )
            fractions.append(forecast.detection_fraction.cpu().numpy())
    result = JointPrediction(
        quantiles_db=np.concatenate(quantiles), detection_fraction=np.concatenate(fractions)
    )
    resources = {
        "train_seconds": elapsed,
        "resume_verification_seconds": verification_elapsed,
        "resume_equivalent": True,
        "resume_verified_updates": updates - midpoint,
        "midpoint_checkpoint": str(midway_path),
        "peak_process_rss_bytes": peak_rss,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
    }
    return result, str(checkpoint), scaler, resources


def execute_development(
    fit: list[HourlyWindow],
    assess: list[HourlyWindow],
    output: Path,
    *,
    protocol_sha256: str,
    fixture_only: bool = False,
    review_path: Path | None = None,
    review_sha256: str | None = None,
    model_config: ModelConfig | None = None,
    updates: int = 128,
    batch_size: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit the declared first ridge/direct slice; no validation selection or test access."""
    if not _valid_hash(protocol_sha256):
        raise ValueError("A frozen protocol SHA-256 is required.")
    if not fixture_only and protocol_sha256 != V2_PROTOCOL_SHA256:
        raise ValueError("Real v2 development requires the reviewed frozen protocol digest.")
    if updates != 128 and not fixture_only:
        raise ValueError("Real v2 development uses exactly 128 direct updates.")
    if not fixture_only and batch_size != 16:
        raise ValueError("Real v2 development requires batch size 16.")
    if not fixture_only and model_config is not None and model_config != ModelConfig():
        raise ValueError("Real v2 development requires the default compact model configuration.")
    if not 1 <= updates <= 128 or not 1 <= batch_size <= 16 or device not in ("cpu", "cuda"):
        raise ValueError("Invalid bounded development execution parameters.")
    input_digest, source_hashes = _validate_rows(fit, assess, fixture_only=fixture_only)
    _review_gate(
        fixture_only=fixture_only,
        review_path=review_path,
        review_sha256=review_sha256,
        protocol_sha256=protocol_sha256,
        source_hashes=source_hashes,
    )
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Development run output already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        ridge = DevelopmentRidge(alpha=1.0).fit(fit)
        ridge_prediction = ridge.predict(assess)
        direct_prediction, checkpoint, scaler, resources = _train_direct(
            fit,
            assess,
            staging,
            protocol_sha256=protocol_sha256,
            model_config=model_config or ModelConfig(),
            updates=updates,
            batch_size=batch_size,
            device=device,
        )
        status = "COMPLETED_SYNTHETIC_FIXTURE" if fixture_only else "COMPLETED_TRAIN_DEVELOPMENT"
        result: dict[str, Any] = {
            "status": status,
            "protocol_sha256": protocol_sha256,
            "input_sha256": input_digest,
            "source_sha256": source_hashes,
            "review_sha256": review_sha256 if not fixture_only else None,
            "fit_rows": len(fit),
            "assessment_rows": len(assess),
            "model_config": asdict(model_config or ModelConfig()),
            "ridge": {},
            "direct": {},
        }
        for family, prediction in (("ridge", ridge_prediction), ("direct", direct_prediction)):
            path = staging / f"{family}-assessment-predictions.npz"
            _write_predictions(path, assess, prediction)
            result[family] = {
                "predictions": str(output / path.name),
                "prediction_sha256": _sha256(path),
                "metrics": _verify_predictions(path, assess),
            }
        resources["midpoint_checkpoint"] = str(output / Path(resources["midpoint_checkpoint"]).name)
        result["direct"].update(
            {
                "updates": updates,
                "checkpoint": str(output / Path(checkpoint).name),
                "checkpoint_sha256": _sha256(Path(checkpoint)),
                "train_scaler": scaler.__dict__,
                "device": device,
                **resources,
            }
        )
        (staging / "run.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return result
    except BaseException:
        if staging.exists() and staging.resolve().parent == output.parent.resolve():
            shutil.rmtree(staging)
        raise
