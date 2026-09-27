"""Real ridge and direct-neural TRAIN development for raw AZFP response codes."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import sklearn
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from marine_echo.models.compact import DirectForecaster, ModelConfig
from marine_echo.training.raw_response_cohort import (
    FROZEN_RUN_CONFIG,
    SUPPORT_SHA256,
    RawCohort,
    file_sha256,
    load_raw_cohorts,
)

QUANTILES = np.array(FROZEN_RUN_CONFIG["quantiles"], dtype=np.float64)
MODEL_CONFIG = ModelConfig(
    width=FROZEN_RUN_CONFIG["direct_width"],
    layers=FROZEN_RUN_CONFIG["direct_layers"],
    heads=FROZEN_RUN_CONFIG["direct_heads"],
)
UPDATES = FROZEN_RUN_CONFIG["updates"]
BATCH = FROZEN_RUN_CONFIG["batch_size"]
SEED = FROZEN_RUN_CONFIG["seed"]
MIDPOINT = FROZEN_RUN_CONFIG["checkpoints"][0]


def _features(cohort: RawCohort) -> np.ndarray:
    """Fixed past-only, mask-aware summaries for the conventional comparator."""
    values = np.where(cohort.context_mask, cohort.context, np.nan)
    parts = []
    for start in (0, 72, 88, 92):
        subset = values[:, start:]
        total = np.nansum(subset, axis=(1, 3), dtype=np.float64)
        count = np.isfinite(subset).sum(axis=(1, 3))
        parts.append(np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0))
        parts.append(count / (subset.shape[1] * 64))
    parts.append(cohort.context_mask[:, -4:].mean(axis=(1, 3)))
    return np.concatenate(parts, axis=1)


def _score(cohort: RawCohort, prediction: np.ndarray) -> dict:
    if prediction.shape != (len(cohort.cutoffs), 3, 5) or not np.isfinite(prediction).all():
        raise ValueError("Every issued row needs finite five-quantile predictions")
    if np.any(np.diff(prediction, axis=-1) < -1e-7):
        raise ValueError("Forecast quantiles cross")
    result: dict[str, Any] = {"issued_rows": len(cohort.cutoffs), "horizons": {}}
    for index, horizon in enumerate((1, 3, 6)):
        mask = cohort.target_mask[:, index]
        truth = cohort.targets[mask, index]
        forecast = prediction[mask, index]
        if not len(truth) or not np.isfinite(truth).all():
            raise ValueError("Reviewed cohort has no finite score truth")
        error = truth[:, None] - forecast
        loss = np.maximum(QUANTILES * error, (QUANTILES - 1) * error).mean(axis=1)
        days = cohort.target_times[mask, index].astype("datetime64[D]")
        daily = [float(loss[days == day].mean()) for day in np.unique(days)]
        result["horizons"][str(horizon)] = {
            "eligible_rows": int(mask.sum()),
            "target_days": len(daily),
            "issued_but_unscored": int((~mask).sum()),
            "daily_mean_pinball_code": float(np.mean(daily)),
            "mae_code_median": float(np.abs(truth - forecast[:, 2]).mean()),
            "empirical_90pct_interval_coverage": float(
                ((truth >= forecast[:, 0]) & (truth <= forecast[:, 4])).mean()
            ),
        }
    return result


def _write_predictions(path: Path, cohort: RawCohort, prediction: np.ndarray) -> dict:
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            study_id=np.array("raw_response_development_v1"),
            quantity=np.array("complete_positive_azfp_backscatter_r_code_mean"),
            cutoffs=cohort.cutoffs,
            target_times=cohort.target_times,
            targets_code=cohort.targets,
            target_mask=cohort.target_mask,
            forecast_quantiles_code=prediction,
            quantile_levels=QUANTILES,
            reviewed_rows_sha256=np.array(cohort.row_sha256),
        )
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(
            saved["targets_code"], cohort.targets, equal_nan=True
        ) or not np.array_equal(saved["target_mask"], cohort.target_mask):
            raise ValueError("Saved predictions differ from reviewed cohort")
        verified = _score(cohort, saved["forecast_quantiles_code"])
    return {"path": str(path), "sha256": file_sha256(path), "metrics": verified}


def _ridge(fit: RawCohort, assess: RawCohort) -> np.ndarray:
    xfit = _features(fit)
    xassess = _features(assess)
    result = np.empty((len(assess.cutoffs), 3, 5), dtype=np.float64)
    for horizon in range(3):
        mask = fit.target_mask[:, horizon]
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        model.fit(xfit[mask], fit.targets[mask, horizon])
        residual = fit.targets[mask, horizon] - model.predict(xfit[mask])
        result[:, horizon] = (
            model.predict(xassess)[:, None] + np.quantile(residual, QUANTILES)[None, :]
        )
    return result


def _checkpoint_identity(fit: RawCohort, assess: RawCohort, root: Path, normalizer: dict) -> dict:
    return {
        "study_id": "raw_response_development_v1",
        "index_sha256": fit.index_sha256,
        "support_sha256": SUPPORT_SHA256,
        "fit_rows_sha256": fit.row_sha256,
        "assessment_rows_sha256": assess.row_sha256,
        "run_config": FROZEN_RUN_CONFIG,
        "environment": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "normalizer": normalizer,
        "code_sha256": {
            "windows": file_sha256(root / "src/marine_echo/training/raw_response_windows.py"),
            "cohort": file_sha256(root / "src/marine_echo/training/raw_response_cohort.py"),
            "executor": file_sha256(root / "src/marine_echo/training/raw_response_development.py"),
            "runner": file_sha256(root / "tools/v2_run_raw_development.py"),
            "model": file_sha256(root / "src/marine_echo/models/compact.py"),
        },
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    identity: dict,
) -> None:
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(
                {
                    "identity": identity,
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    identity: dict,
    device: str,
) -> None:
    saved = torch.load(path, map_location=device, weights_only=True)
    if (
        set(saved) != {"identity", "step", "model", "optimizer"}
        or saved["step"] != step
        or saved["identity"] != identity
    ):
        raise ValueError("Checkpoint schema, step or frozen provenance differs")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])


def _checked_batch_label_count(mask: torch.Tensor) -> int:
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("An all-unlabelled direct update batch is forbidden")
    return count


def _validate_run_request(root: Path, output: Path, device: str) -> None:
    expected = (root / FROZEN_RUN_CONFIG["output_path"]).resolve()
    if output.resolve() != expected or device != FROZEN_RUN_CONFIG["device"]:
        raise ValueError("Only the frozen raw-response output and device are permitted")


def _direct(
    fit: RawCohort, assess: RawCohort, output: Path, root: Path, *, device: str, resume: bool
) -> tuple[np.ndarray, dict]:
    if device == "cuda":
        if (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != FROZEN_RUN_CONFIG["cublas_workspace_config"]
        ):
            raise RuntimeError("Frozen deterministic cuBLAS workspace configuration required")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if (
            not torch.are_deterministic_algorithms_enabled()
            or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
        ):
            raise RuntimeError("Deterministic CUDA mode was not established")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    observed = fit.context[fit.context_mask]
    labelled = fit.targets[fit.target_mask]
    if (
        not len(observed)
        or not len(labelled)
        or not np.isfinite(observed).all()
        or not np.isfinite(labelled).all()
    ):
        raise ValueError("Finite fit-only normalizer support required")
    xmean, xstd = (
        float(observed.mean(dtype=np.float64)),
        max(float(observed.std(dtype=np.float64)), 1e-6),
    )
    ymean, ystd = (
        float(labelled.mean(dtype=np.float64)),
        max(float(labelled.std(dtype=np.float64)), 1e-6),
    )
    context = np.where(fit.context_mask, (fit.context - xmean) / xstd, 0).astype(np.float32)
    assessment = np.where(assess.context_mask, (assess.context - xmean) / xstd, 0).astype(
        np.float32
    )
    targets = np.where(fit.target_mask, (fit.targets - ymean) / ystd, 0).astype(np.float32)
    train_x = torch.from_numpy(context)
    train_mask = torch.from_numpy(fit.context_mask)
    train_y = torch.from_numpy(targets)
    train_y_mask = torch.from_numpy(fit.target_mask)
    model = DirectForecaster(MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_RUN_CONFIG["learning_rate"],
        weight_decay=FROZEN_RUN_CONFIG["weight_decay"],
        betas=tuple(FROZEN_RUN_CONFIG["adamw_betas"]),
        eps=FROZEN_RUN_CONFIG["adamw_eps"],
    )
    identity = _checkpoint_identity(
        fit,
        assess,
        root,
        {"context_mean": xmean, "context_std": xstd, "target_mean": ymean, "target_std": ystd},
    )
    midpoint_path = output / f"direct-checkpoint-{MIDPOINT}.pt"
    checkpoint = output / f"direct-checkpoint-{UPDATES}.pt"
    resumed_step = 0
    if resume:
        resumed_step = UPDATES if checkpoint.exists() else MIDPOINT
        _load_checkpoint(
            checkpoint if resumed_step == UPDATES else midpoint_path,
            model,
            optimizer,
            step=resumed_step,
            identity=identity,
            device=device,
        )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    level = torch.tensor(QUANTILES, dtype=torch.float32, device=device)

    def train_range(
        current: DirectForecaster, current_optimizer: torch.optim.Optimizer, start: int, stop: int
    ) -> None:
        nonlocal peak_rss
        current.train()
        for step in range(start, stop):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, step]))
            selected = rng.choice(len(fit.cutoffs), BATCH, replace=False)
            current_optimizer.param_groups[0]["lr"] = FROZEN_RUN_CONFIG["learning_rate"] * min(
                1.0, (step + 1) / FROZEN_RUN_CONFIG["warmup_updates"]
            )
            current_optimizer.zero_grad(set_to_none=True)
            forecast = current(
                train_x[selected].to(device), train_mask[selected].to(device)
            ).quantiles
            truth = train_y[selected].to(device)[:, :, None]
            mask = train_y_mask[selected].to(device)[:, :, None]
            labelled_count = _checked_batch_label_count(mask)
            error = truth - forecast
            pinball = torch.maximum(level * error, (level - 1) * error)
            loss = torch.where(mask, pinball, 0).sum() / labelled_count / len(QUANTILES)
            if not torch.isfinite(loss):
                raise FloatingPointError("Direct raw-response loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                current.parameters(),
                FROZEN_RUN_CONFIG["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            current_optimizer.step()
            peak_rss = max(peak_rss, process.memory_info().rss)
            if peak_rss >= 22 * 1024**3:
                raise MemoryError("Direct development process reached 22 GiB RAM")
            if device == "cuda" and torch.cuda.max_memory_reserved() >= 10 * 1024**3:
                raise MemoryError("Direct development process reached 10 GiB GPU reserve")

    if not resume:
        train_range(model, optimizer, 0, MIDPOINT)
        _save_checkpoint(midpoint_path, model, optimizer, step=MIDPOINT, identity=identity)
    if resumed_step != UPDATES:
        train_range(model, optimizer, MIDPOINT, UPDATES)
    elapsed = time.perf_counter() - started
    if resumed_step != UPDATES:
        _save_checkpoint(checkpoint, model, optimizer, step=UPDATES, identity=identity)
    resume_verification_started = time.perf_counter()
    if resume:
        torch.manual_seed(SEED)
    resumed = DirectForecaster(MODEL_CONFIG).to(device)
    resumed_optimizer = torch.optim.AdamW(
        resumed.parameters(),
        lr=FROZEN_RUN_CONFIG["learning_rate"],
        weight_decay=FROZEN_RUN_CONFIG["weight_decay"],
        betas=tuple(FROZEN_RUN_CONFIG["adamw_betas"]),
        eps=FROZEN_RUN_CONFIG["adamw_eps"],
    )
    if resume:
        train_range(resumed, resumed_optimizer, 0, UPDATES)
    else:
        _load_checkpoint(
            midpoint_path,
            resumed,
            resumed_optimizer,
            step=MIDPOINT,
            identity=identity,
            device=device,
        )
        train_range(resumed, resumed_optimizer, MIDPOINT, UPDATES)
    resume_max_abs_difference = max(
        float(torch.max(torch.abs(value - resumed.state_dict()[key])).item())
        for key, value in model.state_dict().items()
    )
    resume_equivalent = all(
        torch.allclose(value, resumed.state_dict()[key], rtol=1e-6, atol=1e-7)
        for key, value in model.state_dict().items()
    )
    if not resume_equivalent:
        raise ValueError("Midpoint continuation diverged from independent reference")
    resume_verification_seconds = time.perf_counter() - resume_verification_started
    restored = DirectForecaster(MODEL_CONFIG).to(device)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(),
        lr=FROZEN_RUN_CONFIG["learning_rate"],
        weight_decay=FROZEN_RUN_CONFIG["weight_decay"],
        betas=tuple(FROZEN_RUN_CONFIG["adamw_betas"]),
        eps=FROZEN_RUN_CONFIG["adamw_eps"],
    )
    _load_checkpoint(
        checkpoint, restored, restored_optimizer, step=UPDATES, identity=identity, device=device
    )
    restored.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(assess.cutoffs), BATCH):
            part = slice(start, start + BATCH)
            p = restored(
                torch.from_numpy(assessment[part]).to(device),
                torch.from_numpy(assess.context_mask[part]).to(device),
            ).quantiles
            predictions.append(p.cpu().numpy().astype(np.float64) * ystd + ymean)
    resource = {
        "updates": UPDATES,
        "batch_size": BATCH,
        "seed": SEED,
        "seconds_train": elapsed,
        "peak_process_rss_bytes": peak_rss,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "midpoint_checkpoint_sha256": file_sha256(midpoint_path),
        "resumed_from_interruption": resume,
        "resumed_from_step": resumed_step,
        "resume_equivalent": resume_equivalent,
        "resume_max_abs_weight_difference": resume_max_abs_difference,
        "resume_verification_seconds": resume_verification_seconds,
        "checkpoint_identity": identity,
    }
    return np.concatenate(predictions), resource


def execute(
    root: Path, output: Path, review_path: Path, *, device: str = "cuda", resume: bool = False
) -> dict:
    """Run the two declared engineering developments; never consume calibrated slots."""
    _validate_run_request(root, output, device)
    fit, assess = load_raw_cohorts(root, review_path=review_path)
    if resume:
        if not output.is_dir() or (output / "result.json").exists():
            raise ValueError("Only an incomplete development run may be resumed")
        baseline_path = output / "ridge-assessment-predictions.npz"
        with np.load(baseline_path, allow_pickle=False) as saved:
            if (
                not np.array_equal(saved["cutoffs"], assess.cutoffs)
                or not np.array_equal(saved["target_mask"], assess.target_mask)
                or not np.array_equal(saved["targets_code"], assess.targets, equal_nan=True)
                or saved["reviewed_rows_sha256"].item() != assess.row_sha256
                or not np.allclose(
                    saved["forecast_quantiles_code"], _ridge(fit, assess), rtol=1e-12, atol=1e-12
                )
            ):
                raise ValueError("Incomplete-run ridge provenance differs")
            baseline = {
                "path": str(baseline_path),
                "sha256": file_sha256(baseline_path),
                "metrics": _score(assess, saved["forecast_quantiles_code"]),
            }
    else:
        output.mkdir(parents=True, exist_ok=False)
        baseline = _write_predictions(
            output / "ridge-assessment-predictions.npz", assess, _ridge(fit, assess)
        )
    direct_prediction, resources = _direct(fit, assess, output, root, device=device, resume=resume)
    direct_path = output / "direct-assessment-predictions.npz"
    if resume and direct_path.exists():
        with np.load(direct_path, allow_pickle=False) as saved:
            if (
                not np.array_equal(saved["cutoffs"], assess.cutoffs)
                or not np.array_equal(saved["target_mask"], assess.target_mask)
                or not np.array_equal(saved["targets_code"], assess.targets, equal_nan=True)
                or saved["reviewed_rows_sha256"].item() != assess.row_sha256
                or not np.allclose(
                    saved["forecast_quantiles_code"], direct_prediction, rtol=1e-6, atol=1e-6
                )
            ):
                raise ValueError("Incomplete-run direct prediction provenance differs")
            direct = {
                "path": str(direct_path),
                "sha256": file_sha256(direct_path),
                "metrics": _score(assess, saved["forecast_quantiles_code"]),
            }
    else:
        direct = _write_predictions(direct_path, assess, direct_prediction)
    result = {
        "study_id": "raw_response_development_v1",
        "run_id": FROZEN_RUN_CONFIG["run_id"],
        "data_kind": "REAL",
        "quantity": "complete_positive_azfp_backscatter_r_code_mean",
        "scientific_scope": "engineering TRAIN development only; not calibrated Sv or final evaluation",
        "review_sha256": file_sha256(review_path),
        "fit_rows_sha256": fit.row_sha256,
        "assessment_rows_sha256": assess.row_sha256,
        "index_sha256": fit.index_sha256,
        "support_sha256": SUPPORT_SHA256,
        "run_config": FROZEN_RUN_CONFIG,
        "device": device,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "models": {"ridge": baseline, "direct_neural": direct},
        "direct_resources": resources,
        "calibrated_core_slots_consumed": 0,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
