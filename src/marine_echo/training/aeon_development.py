"""Finite AEON development baseline and direct neural execution.

This module cannot select a source, target, split or held-out result. A distinct
review artifact must bind those inputs before any real development run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from torch import nn

from marine_echo.training.aeon_corpus import AEON_ADR_SHA256, AEON_SOURCE_SHA256
from marine_echo.training.aeon_windows import AeonHourlyWindow, AeonWindowPlan


_QUANTILES = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64)
AEON_DEVELOPMENT_CONFIG_SHA256 = "8dbd58cc3641862b5a206e2b75b12ea5bbcf1b3949bee699abcf321e2282439e"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("aeon_corpus.py", "aeon_windows.py", "aeon_development.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_rows(
    fit: list[AeonHourlyWindow],
    assess: list[AeonHourlyWindow],
    plan: AeonWindowPlan,
    review: dict[str, Any],
) -> str:
    if not fit or not assess or plan != AeonWindowPlan():
        raise ValueError("AEON development needs nonempty 24-hour fit and assessment rows.")
    source = review.get("archive_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise ValueError("AEON review source digest is missing.")
    if review.get("window_plan") != json.loads(json.dumps(asdict(plan))):
        raise ValueError("AEON row plan differs from reviewed prospective target.")
    if any(
        row.source_archive_sha256 != source
        or row.context_db.shape != (24, 4)
        or row.context_mask.shape != row.context_db.shape
        or row.target_db.shape != (3,)
        or row.target_mask.shape != (3,)
        or row.context_interval_ids.shape != (24,)
        or row.target_interval_ids.shape != (3,)
        for row in fit + assess
    ):
        raise ValueError("AEON rows differ in source, geometry or fixed schedule.")
    for name, rows in (("development_fit", fit), ("development_assessment", assess)):
        bounds = review.get(name)
        if not isinstance(bounds, dict):
            raise ValueError("AEON reviewed development split is missing.")
        start = np.datetime64(bounds["start"], "us")
        end = np.datetime64(bounds["end_exclusive"], "us")
        if np.isnat(start) or np.isnat(end) or end <= start:
            raise ValueError("AEON reviewed development split is invalid.")
        expected_partition = ("train" if name == "development_fit" else "validation") if review["data_kind"] == "REAL" else name
        if any(row.partition != expected_partition for row in rows):
            raise ValueError("AEON row partition differs from review.")
        ids = [row.row_id for row in rows]
        cutoffs = [row.cutoff_interval_id for row in rows]
        if len(set(ids)) != len(ids) or any(a >= b for a, b in zip(cutoffs, cutoffs[1:])):
            raise ValueError("AEON development rows are duplicated or nonchronological.")
        for row in rows:
            if (
                row.context_source_timestamps[0] < start
                or row.cutoff_source_timestamp + np.timedelta64(6, "h") >= end
                or not np.array_equal(
                    row.target_interval_ids,
                    row.cutoff_interval_id + np.asarray(plan.horizons_hours),
                )
                or not np.array_equal(row.context_mask, np.isfinite(row.context_db))
                or not np.array_equal(row.target_mask, np.isfinite(row.target_db))
                or not row.context_mask[:, 0].all()
            ):
                raise ValueError("AEON row support exceeds its reviewed split or masks differ.")
    if (
        fit[-1].cutoff_interval_id + 6 >= assess[0].context_interval_ids[0]
        or review["development_fit"]["end_exclusive"]
        > review["development_assessment"]["start"]
    ):
        raise ValueError("AEON fit and assessment raw observation support overlaps.")
    digest = hashlib.sha256()
    for row in fit + assess:
        digest.update(row.row_id.encode("ascii"))
        digest.update(row.partition.encode("ascii"))
        for array in (
            row.context_db,
            row.context_mask,
            row.context_interval_ids,
            row.context_source_timestamps,
            row.target_interval_ids,
            row.target_source_timestamps,
            row.target_db,
            row.target_mask,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _baseline(
    fit: list[AeonHourlyWindow], assess: list[AeonHourlyWindow], plan: AeonWindowPlan
) -> np.ndarray:
    last_fit = np.asarray([row.context_db[-1, 0] for row in fit])
    residuals = np.stack([row.target_db - last for row, last in zip(fit, last_fit)])
    masks = np.stack([row.target_mask for row in fit])
    quantile_residuals = []
    for horizon in range(3):
        valid = masks[:, horizon]
        if not valid.any():
            raise ValueError("Each AEON forecast horizon needs fit labels.")
        quantile_residuals.append(np.quantile(residuals[valid, horizon], _QUANTILES))
    last_assess = np.asarray([row.context_db[-1, 0] for row in assess])
    return last_assess[:, None, None] + np.asarray(quantile_residuals)[None, :, :]


class _DirectNet(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(24 * layers * 2, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 15),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((values.flatten(1), mask.float().flatten(1)), dim=1)
        return self.layers(combined).reshape(-1, 3, 5)


def _normalizer(rows: list[AeonHourlyWindow]) -> tuple[float, float, float, float]:
    contexts = np.concatenate([row.context_db[row.context_mask] for row in rows])
    targets = np.concatenate([row.target_db[row.target_mask] for row in rows])
    if not len(contexts) or not len(targets):
        raise ValueError("AEON fit lacks observed contexts or targets.")
    return (
        float(contexts.mean()),
        max(float(contexts.std()), 1e-6),
        float(targets.mean()),
        max(float(targets.std()), 1e-6),
    )


def _context_tensors(
    rows: list[AeonHourlyWindow], scaler: tuple[float, float, float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    context = np.stack([row.context_db for row in rows])
    mask = np.stack([row.context_mask for row in rows])
    normalized = np.where(mask, (context - scaler[0]) / scaler[1], 0.0)
    return (
        torch.as_tensor(normalized, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.bool),
    )


def _tensors(
    rows: list[AeonHourlyWindow], scaler: tuple[float, float, float, float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context, mask = _context_tensors(rows, scaler)
    targets = np.stack([row.target_db for row in rows])
    target_mask = np.stack([row.target_mask for row in rows])
    normalized_target = np.where(target_mask, (targets - scaler[2]) / scaler[3], 0.0)
    return (
        context,
        mask,
        torch.as_tensor(normalized_target, dtype=torch.float32),
        torch.as_tensor(target_mask, dtype=torch.bool),
    )


def _direct(
    fit: list[AeonHourlyWindow],
    assess: list[AeonHourlyWindow],
    staging: Path,
    *,
    input_sha256: str,
    review_sha256: str,
    updates: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    scaler = _normalizer(fit)
    train_x, train_mask, train_y, train_y_mask = _tensors(fit, scaler)
    assess_x, assess_mask = _context_tensors(assess, scaler)
    torch.manual_seed(seed)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = _DirectNet(train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    candidates = np.flatnonzero(train_y_mask.any(dim=1).numpy())
    if not len(candidates):
        raise ValueError("No labelled AEON fit rows exist.")
    generator = np.random.default_rng(seed)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    quantiles = torch.as_tensor(_QUANTILES, dtype=torch.float32, device=device)
    started = time.perf_counter()
    model.train()
    for _ in range(updates):
        selected = generator.choice(candidates, size=batch_size, replace=len(candidates) < batch_size)
        predicted = torch.sort(
            model(train_x[selected].to(device), train_mask[selected].to(device)), dim=-1
        ).values
        truth = train_y[selected].to(device)
        valid = train_y_mask[selected].to(device)
        error = truth[:, :, None] - predicted
        pinball = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
        loss = pinball[valid].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("AEON direct update has non-finite loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if peak_rss >= 22 * 1024**3:
            raise MemoryError("AEON development RAM reached the 22 GiB limit.")
        if device == "cuda" and torch.cuda.max_memory_reserved() >= 10 * 1024**3:
            raise MemoryError("AEON development GPU reserve reached the 10 GiB limit.")
    train_seconds = time.perf_counter() - started
    checkpoint = staging / "direct-checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler": scaler,
            "input_sha256": input_sha256,
            "review_sha256": review_sha256,
            "updates": updates,
            "seed": seed,
            "layers": train_x.shape[-1],
        },
        checkpoint,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for first in range(0, len(assess), max(batch_size, 1)):
            part = slice(first, first + batch_size)
            normalized = model(assess_x[part].to(device), assess_mask[part].to(device))
            predictions.append(normalized.cpu().numpy() * scaler[3] + scaler[2])
    forecast = np.sort(np.concatenate(predictions), axis=-1)
    resources = {
        "train_seconds": train_seconds,
        "peak_process_rss_bytes": peak_rss,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "scaler_fit_only": scaler,
    }
    return forecast, checkpoint, resources


def _save_predictions(path: Path, rows: list[AeonHourlyWindow], forecast: np.ndarray) -> None:
    if forecast.shape != (len(rows), 3, 5) or not np.isfinite(forecast).all():
        raise ValueError("AEON forecast shape or values are invalid.")
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            row_ids=np.asarray([row.row_id for row in rows]),
            cutoff_source_timestamps=np.asarray(
                [row.cutoff_source_timestamp for row in rows], dtype="datetime64[us]"
            ),
            target_source_timestamps=np.stack([row.target_source_timestamps for row in rows]),
            target_interval_ids=np.stack([row.target_interval_ids for row in rows]),
            truth_db=np.stack([row.target_db for row in rows]),
            target_mask=np.stack([row.target_mask for row in rows]),
            target_qc_status=np.asarray([row.target_qc_status for row in rows]),
            past_members=np.asarray([";".join(row.past_members) for row in rows]),
            target_members=np.asarray([";".join(row.target_members) for row in rows]),
            quantiles_db=forecast,
        )


def _verify_and_score(path: Path, rows: list[AeonHourlyWindow]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        expected = {
            "row_ids": np.asarray([row.row_id for row in rows]),
            "cutoff_source_timestamps": np.asarray(
                [row.cutoff_source_timestamp for row in rows], dtype="datetime64[us]"
            ),
            "target_source_timestamps": np.stack([row.target_source_timestamps for row in rows]),
            "target_interval_ids": np.stack([row.target_interval_ids for row in rows]),
            "truth_db": np.stack([row.target_db for row in rows]),
            "target_mask": np.stack([row.target_mask for row in rows]),
            "target_qc_status": np.asarray([row.target_qc_status for row in rows]),
            "past_members": np.asarray([";".join(row.past_members) for row in rows]),
            "target_members": np.asarray([";".join(row.target_members) for row in rows]),
        }
        if set(data.files) != set(expected) | {"quantiles_db"}:
            raise ValueError("AEON saved prediction schema differs.")
        for key, value in expected.items():
            if not np.array_equal(data[key], value, equal_nan=value.dtype.kind in "fM"):
                raise ValueError(f"AEON saved {key} differs from cohort.")
        prediction = data["quantiles_db"]
        if (
            prediction.shape != (len(rows), 3, 5)
            or not np.isfinite(prediction).all()
            or (np.diff(prediction, axis=-1) < 0).any()
        ):
            raise ValueError("AEON saved quantiles are invalid.")
        truth = data["truth_db"]
        mask = data["target_mask"]
        dates = data["target_source_timestamps"].astype("datetime64[D]")
        scores = []
        for horizon in range(3):
            valid = mask[:, horizon]
            error = truth[valid, horizon, None] - prediction[valid, horizon]
            pinball = np.maximum(_QUANTILES * error, (_QUANTILES - 1) * error).mean(axis=1)
            daily = [
                float(pinball[dates[valid, horizon] == day].mean())
                for day in np.unique(dates[valid, horizon])
            ]
            scores.append(
                {
                    "horizon_hours": (1, 3, 6)[horizon],
                    "scored_rows": int(valid.sum()),
                    "scored_days": len(daily),
                    "daily_mean_pinball_db": float(np.mean(daily)) if daily else None,
                    "median_mae_db": float(
                        np.abs(truth[valid, horizon] - prediction[valid, horizon, 2]).mean()
                    )
                    if valid.any()
                    else None,
                }
            )
        return {"issued_rows": len(rows), "horizons": scores}


def execute_aeon_development(
    fit: list[AeonHourlyWindow],
    assess: list[AeonHourlyWindow],
    output: Path,
    *,
    plan: AeonWindowPlan,
    review_path: Path,
    review_sha256: str,
    updates: int = 128,
    batch_size: int = 16,
    device: str = "cpu",
    fixture_only: bool = False,
) -> dict[str, Any]:
    """Run one reviewed development slice; never read or score held-out rows."""
    if not 1 <= updates <= 128 or not 1 <= batch_size <= 16 or device not in ("cpu", "cuda"):
        raise ValueError("AEON development execution exceeds the bounded budget.")
    review_path = review_path.resolve(strict=True)
    if _sha256(review_path) != review_sha256:
        raise ValueError("AEON development review digest differs.")
    source_review = json.loads(review_path.read_text(encoding="utf-8"))
    if fixture_only:
        if (
            source_review.get("status") != "APPROVED_AEON_DEVELOPMENT"
            or source_review.get("source_time_basis") != "SOURCE_REPORTED_UNSPECIFIED"
            or source_review.get("data_kind") != "SYNTHETIC_FIXTURE"
            or source_review.get("direct_updates") != updates
            or source_review.get("direct_batch_size") != batch_size
            or not isinstance(source_review.get("direct_seed"), int)
        ):
            raise ValueError("Synthetic AEON development fixture is malformed.")
        review = source_review
    else:
        root = Path(__file__).resolve().parents[3]
        config_path = root / "configs/aeon_development.json"
        if _sha256(config_path) != AEON_DEVELOPMENT_CONFIG_SHA256:
            raise ValueError("AEON first-development configuration digest differs.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            source_review.get("disposition") != "APPROVE_TRAIN_VALIDATION_DEVELOPMENT_ONLY"
            or source_review.get("reviewed_adr_sha256") != AEON_ADR_SHA256
            or source_review.get("source_archive_sha256") != AEON_SOURCE_SHA256
            or config.get("protocol_sha256") != AEON_ADR_SHA256
            or config.get("source_sha256") != AEON_SOURCE_SHA256
            or config.get("test_access") != "PROHIBITED_IN_THIS_PHASE"
            or config.get("direct_neural", {}).get("updates") != updates
            or config.get("direct_neural", {}).get("batch_size") != batch_size
            or config.get("direct_neural", {}).get("seed") != 7
            or config.get("direct_neural", {}).get("device") != device
            or updates != 128
            or batch_size != 16
            or device != "cpu"
        ):
            raise ValueError("Real AEON development differs from independent prefit approval.")
        adr_path = root / "docs/adr/0008-aeon-hourly-sv-study.md"
        if _sha256(adr_path) != AEON_ADR_SHA256:
            raise ValueError("AEON frozen ADR digest differs from independent review.")
        review = {
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
            "direct_seed": 7,
        }
    input_sha256 = _validate_rows(fit, assess, plan, review)
    output = output.resolve()
    if output.exists():
        raise FileExistsError("AEON development output already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        baseline = _baseline(fit, assess, plan)
        direct, checkpoint, resources = _direct(
            fit,
            assess,
            staging,
            input_sha256=input_sha256,
            review_sha256=review_sha256,
            updates=updates,
            batch_size=batch_size,
            seed=review["direct_seed"],
            device=device,
        )
        result: dict[str, Any] = {
            "status": "COMPLETED_SYNTHETIC_FIXTURE"
            if review["data_kind"] == "SYNTHETIC_FIXTURE"
            else "COMPLETED_AEON_DEVELOPMENT",
            "data_kind": review["data_kind"],
            "review_sha256": review_sha256,
            "source_archive_sha256": review["archive_sha256"],
            "first_development_config_sha256": AEON_DEVELOPMENT_CONFIG_SHA256
            if not fixture_only
            else None,
            "input_sha256": input_sha256,
            "code_sha256": _code_sha256(),
            "fit_rows": len(fit),
            "assessment_rows": len(assess),
            "window_plan": json.loads(json.dumps(asdict(plan))),
            "baseline": {},
            "direct": {},
        }
        for name, prediction in (("baseline", baseline), ("direct", direct)):
            path = staging / f"{name}-assessment-predictions.npz"
            _save_predictions(path, assess, prediction)
            result[name] = {
                "predictions": str(output / path.name),
                "prediction_sha256": _sha256(path),
                "metrics": _verify_and_score(path, assess),
            }
        result["direct"].update(
            {
                "checkpoint": str(output / checkpoint.name),
                "checkpoint_sha256": _sha256(checkpoint),
                "updates": updates,
                "batch_size": batch_size,
                "seed": review["direct_seed"],
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
