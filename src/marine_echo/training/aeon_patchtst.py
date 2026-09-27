"""Bounded post-hoc PatchTST-style TRAIN/validation challenger; no CAL/TEST access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.models.aeon_patchtst import AeonMaskedPatchTransformer
from marine_echo.training.aeon_campaign import load_cohort
from marine_echo.training.aeon_corpus import AEON_SOURCE_SHA256
from marine_echo.training.aeon_development import (
    _context_tensors,
    _normalizer,
    _save_predictions,
    _tensors,
)
from marine_echo.training.aeon_windows import AeonHourlyWindow

_EXPECTED_COHORT = "5e475f6af798025f187c70ae638710d2b4362f688db2ad43d60ec436dda3e25d"
_EXPECTED_SPLIT_REVIEW = "b8031e2e113e39630c066cee65b4371f69b229ee905a72d832e53e6f505f297b"
_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _code_sha256() -> str:
    files = (
        Path(__file__), Path(__file__).resolve().parents[1] / "models/aeon_patchtst.py",
        Path(__file__).with_name("aeon_campaign.py"),
        Path(__file__).with_name("aeon_corpus.py"),
        Path(__file__).with_name("aeon_windows.py"),
        Path(__file__).with_name("aeon_development.py"),
        Path(__file__).resolve().parents[1] / "evaluation/aeon.py",
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _gate(config: dict[str, Any], cohort_sha: str, review_sha: str) -> None:
    if (
        config.get("schema_version") != "1.0"
        or config.get("study_id") != "aeon3_geb_2024_hourly_sv_v1"
        or config.get("classification") != "POST_HOC_TRAIN_VALIDATION_EXPLORATORY_NOT_SELECTION_OR_FINAL_EVALUATION"
        or config.get("source_archive_sha256") != AEON_SOURCE_SHA256
        or config.get("cohort_sha256") != _EXPECTED_COHORT
        or cohort_sha != _EXPECTED_COHORT
        or config.get("split_review_sha256") != _EXPECTED_SPLIT_REVIEW
        or review_sha != _EXPECTED_SPLIT_REVIEW
        or config.get("fit_partition") != "train"
        or config.get("assessment_partition") != "validation"
        or config.get("calibration_access") != "PROHIBITED"
        or config.get("test_access") != "PROHIBITED"
        or config.get("input") != "24x4_past_full_depth_Sv_mean_plus_observed_mask"
        or config.get("model") != "masked_channel_independent_patch_transformer_with_persistence_skip"
        or config.get("paper") != "https://openreview.net/forum?id=Jbdc0vTOcol"
        or config.get("patch_hours") != 4
        or config.get("patch_stride_hours") != 4
        or config.get("width") != 128
        or config.get("layers") != 3
        or config.get("heads") != 4
        or config.get("dropout") != 0.1
        or config.get("horizons") != [1, 3, 6]
        or config.get("quantiles") != list(_QUANTILES)
        or config.get("seeds") != [7, 13, 23]
        or config.get("first_seed") != 7
        or config.get("max_updates") != 3000
        or config.get("batch_size") != 64
        or config.get("learning_rate") != 3e-4
        or config.get("weight_decay") != 1e-4
        or config.get("gradient_clip_norm") != 1.0
        or config.get("max_threads") != 4
        or config.get("checkpoint_selection") != "final_endpoint_only"
        or config.get("first_seed_promising_if_at_most_direct_ensemble_times") != 1.05
        or config.get("direct_ensemble_reviewed_validation_pinball_db") != 0.6390617418
        or config.get("metric") != "daily_mean_pinball_db_equal_quantiles_horizons_eligible_source_dates_min18"
        or config.get("resource", {}).get("max_process_rss_bytes") != 22 * 1024**3
        or config.get("resource", {}).get("max_gpu_reserved_bytes") != 10 * 1024**3
        or config.get("resource", {}).get("minimum_free_gpu_bytes") != 2 * 1024**3
        or config.get("resource", {}).get("one_training_process") is not True
    ):
        raise ValueError("Exploratory PatchTST contract differs from prospectively fixed config")


def _device(config: dict[str, Any]) -> tuple[str, int | None]:
    if not torch.cuda.is_available():
        return "cpu", None
    free, _ = torch.cuda.mem_get_info()
    minimum = config["resource"].get("minimum_free_gpu_bytes")
    if minimum != 2 * 1024**3:
        raise ValueError("GPU free-memory gate differs")
    return ("cuda" if free >= minimum else "cpu"), int(free)


def _prediction(
    model: AeonMaskedPatchTransformer, rows: list[AeonHourlyWindow],
    scaler: tuple[float, float, float, float], device: str,
) -> np.ndarray:
    values, mask = _context_tensors(rows, scaler)
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            forecast = model(values[start:start + 64].to(device), mask[start:start + 64].to(device))
            pieces.append(forecast.cpu().numpy() * scaler[3] + scaler[2])
    return np.concatenate(pieces).astype(np.float64)


def _score(rows: list[AeonHourlyWindow], prediction: np.ndarray) -> dict[str, object]:
    return daily_pinball(
        np.stack([row.target_db for row in rows]), prediction,
        np.stack([row.target_mask for row in rows]),
        np.stack([row.target_source_timestamps for row in rows]),
    )


def _fit_seed(
    seed: int, fit: list[AeonHourlyWindow], assess: list[AeonHourlyWindow],
    config: dict[str, Any], output: Path, device: str, lineage: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Existing challenger seed output cannot be overwritten")
    output.mkdir(parents=True)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(config["max_threads"])
    scaler = _normalizer(fit)
    x, mask, target, target_mask = _tensors(fit, scaler)
    candidate = np.flatnonzero(target_mask.any(dim=1).numpy())
    if not len(candidate):
        raise ValueError("TRAIN has no labeled rows")
    model = AeonMaskedPatchTransformer(
        width=config["width"], layers=config["layers"], heads=config["heads"],
        dropout=config["dropout"], baseline_scale=scaler[1] / scaler[3],
        baseline_shift=(scaler[0] - scaler[2]) / scaler[3],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    quantiles = torch.tensor(_QUANTILES, dtype=torch.float32, device=device)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    final_loss = float("nan")
    for step in range(1, config["max_updates"] + 1):
        model.train()
        selected = rng.choice(candidate, size=config["batch_size"], replace=False)
        predicted = model(x[selected].to(device), mask[selected].to(device))
        observed = target_mask[selected].to(device)
        error = target[selected].to(device)[:, :, None] - predicted
        loss = torch.maximum(quantiles * error, (quantiles - 1) * error)[observed].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("PatchTST TRAIN loss nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 100 == 0 or step == 1:
            peak_rss = max(peak_rss, process.memory_info().rss)
            if peak_rss >= config["resource"]["max_process_rss_bytes"]:
                raise MemoryError("PatchTST exceeded 22 GiB process RSS")
            if device == "cuda" and torch.cuda.max_memory_reserved() >= config["resource"]["max_gpu_reserved_bytes"]:
                raise MemoryError("PatchTST exceeded 10 GiB GPU reserve")
            if step % 500 == 0:
                print(json.dumps({"seed": seed, "update": step, "train_loss_normalized": final_loss}), flush=True)
    checkpoint = output / "checkpoint-final-3000.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "seed": seed, "updates": config["max_updates"], "scaler_fit_only": scaler,
        **lineage,
    }, checkpoint)
    prediction = _prediction(model, assess, scaler, device)
    prediction_path = output / "validation-predictions.npz"
    _save_predictions(prediction_path, assess, prediction)
    metrics = _score(assess, prediction)
    result = {
        **lineage, "seed": seed, "device": device, "updates": config["max_updates"],
        "fit_rows": len(fit), "validation_rows": len(assess),
        "train_loss_normalized_at_final": final_loss,
        "fit_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "checkpoint_sha256": _sha256(checkpoint),
        "prediction_sha256": _sha256(prediction_path),
        "metrics": metrics,
        "classification": config["classification"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def run_challenger(
    archive: Path, split_review: Path, config_path: Path, output: Path,
) -> dict[str, Any]:
    """Run seed 7, then seeds 13/23 only if fixed five-percent promise gate passes."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha = _sha256(config_path)
    fit, assess, cohort_sha, review_sha = load_cohort(archive, split_review)
    _gate(config, cohort_sha, review_sha)
    if output.exists():
        raise FileExistsError("Existing challenger output cannot be overwritten")
    device, free_gpu_bytes = _device(config)
    output.mkdir(parents=True)
    lineage = {
        "source_archive_sha256": AEON_SOURCE_SHA256,
        "cohort_sha256": cohort_sha, "split_review_sha256": review_sha,
        "config_sha256": config_sha, "code_sha256": _code_sha256(),
    }
    report: dict[str, Any] = {
        **lineage, "device": device, "initial_free_gpu_bytes": free_gpu_bytes,
        "classification": config["classification"], "seeds": {},
    }
    for seed in config["seeds"]:
        if seed != 7 and report.get("first_seed_promising") is not True:
            break
        result = _fit_seed(seed, fit, assess, config, output / f"seed{seed}", device, lineage)
        report["seeds"][str(seed)] = {
            "primary_daily_mean_pinball_db": result["metrics"]["primary_daily_mean_pinball_db"],
            "result_sha256": _sha256(output / f"seed{seed}/result.json"),
        }
        if seed == 7:
            threshold = (
                config["direct_ensemble_reviewed_validation_pinball_db"]
                * config["first_seed_promising_if_at_most_direct_ensemble_times"]
            )
            report["first_seed_promising_threshold_db"] = threshold
            report["first_seed_promising"] = result["metrics"]["primary_daily_mean_pinball_db"] <= threshold
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-review", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_challenger(args.archive, args.split_review, args.config, args.output), indent=2))
