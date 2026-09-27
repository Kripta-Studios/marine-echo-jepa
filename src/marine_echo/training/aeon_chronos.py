"""Frozen, resumable Chronos-2 baseline for AEON TRAIN/validation development.

The real-data entry point is intentionally unable to open calibration or test
partitions. Chronos-2 receives the four past source-product channels as native
multivariate targets. Unobserved past channel values stay NaN so the official
pipeline constructs its observation mask; no imputation is fitted or applied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import psutil
from numpy.typing import NDArray

from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.training.aeon_windows import AeonHourlyWindow

MODEL_REPO_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
PACKAGE_VERSION = "2.3.2"
PACKAGE_WHEEL_SHA256 = "0f0d9a1972f252d6cf584b9fa749bd1b389b3c1cf05a889c4130cb10652c2117"
PACKAGE_SDIST_SHA256 = "910b0891310b74598a937bb9fe8447ea16393bcc2e1ad8d328c1f8099f909201"
SOURCE_ARCHIVE_SHA256 = "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde"
CHRONOS_QUANTILES: NDArray[np.float64] = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64)
HORIZON_INDICES: NDArray[np.int64] = np.asarray([0, 2, 5], dtype=np.int64)
_SNAPSHOT_FILES = {".gitattributes", "README.md", "config.json", "model.safetensors"}


class Chronos2Like(Protocol):
    """Narrow official inference API used by the campaign executor."""

    def predict_quantiles(
        self, inputs: NDArray[np.float64], **kwargs: object
    ) -> tuple[Sequence[object], Sequence[object]]: ...


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        Path(__file__).with_name("aeon_corpus.py"),
        Path(__file__).with_name("aeon_windows.py"),
        root / "evaluation/aeon.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_gate(config: dict[str, Any]) -> None:
    model = config.get("model", {})
    input_config = config.get("input", {})
    resource = config
    if (
        config.get("schema_version") != "1.0"
        or config.get("study_id") != "aeon3_geb_2024_hourly_sv_v1"
        or config.get("phase") != "train_validation_frozen_chronos2_zero_shot"
        or config.get("status") != "PROPOSED_FOR_INDEPENDENT_PREFIT_REVIEW"
        or config.get("source_sha256") != SOURCE_ARCHIVE_SHA256
        or config.get("calibration_access") != "PROHIBITED_IN_THIS_PHASE"
        or config.get("test_access") != "PROHIBITED_IN_THIS_PHASE"
        or model.get("repo_id") != MODEL_REPO_ID
        or model.get("revision") != MODEL_REVISION
        or model.get("license") != "Apache-2.0"
        or model.get("package") != "chronos-forecasting"
        or model.get("package_version") != PACKAGE_VERSION
        or model.get("package_wheel_sha256") != PACKAGE_WHEEL_SHA256
        or model.get("package_sdist_sha256") != PACKAGE_SDIST_SHA256
        or set(model.get("snapshot_files_sha256", {})) != _SNAPSHOT_FILES
        or input_config.get("layout") != "multivariate_4x24"
        or input_config.get("frequencies_hz") != [38000, 125000, 200000, 455000]
        or input_config.get("missing_values") != "nan_native_observation_mask"
        or input_config.get("target_variate_index") != 0
        or config.get("prediction_length") != 6
        or config.get("horizon_zero_based_indices") != [0, 2, 5]
        or config.get("quantiles") != CHRONOS_QUANTILES.tolist()
        or config.get("quantile_monotonicity") != "sort_each_horizon_five_outputs"
        or config.get("cross_learning") is not False
        or config.get("context_length") != 24
        or resource.get("peak_process_rss_limit_bytes") != 22 * 1024**3
        or resource.get("peak_gpu_reserved_limit_bytes") != 10 * 1024**3
        or resource.get("max_gpu_seconds") != 7200
    ):
        raise ValueError("Chronos-2 config differs from the frozen bounded baseline contract.")
    rows_per_shard = config.get("rows_per_resume_shard")
    pipeline_batch = config.get("pipeline_series_batch_size")
    if (
        not isinstance(rows_per_shard, int)
        or not 1 <= rows_per_shard <= 64
        or not isinstance(pipeline_batch, int)
        or pipeline_batch != rows_per_shard * 4
    ):
        raise ValueError("Chronos-2 resumable batch geometry is invalid.")


def _verify_review(review: dict[str, Any], config_sha256: str, code_sha256: str) -> None:
    if (
        review.get("disposition") != "APPROVE_CHRONOS2_TRAIN_VALIDATION_ZERO_SHOT_ONLY"
        or review.get("config_sha256") != config_sha256
        or review.get("implementation_code_sha256") != code_sha256
        or review.get("model_revision") != MODEL_REVISION
        or review.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or review.get("calibration_access") != "PROHIBITED"
        or review.get("test_access") != "PROHIBITED"
    ):
        raise ValueError("Chronos-2 execution lacks the exact independent prefit approval.")


def _verify_snapshot(snapshot: Path, expected: dict[str, str]) -> dict[str, str]:
    snapshot = snapshot.resolve(strict=True)
    present = {path.name for path in snapshot.iterdir() if path.is_file()}
    if present != _SNAPSHOT_FILES:
        raise ValueError("Chronos-2 snapshot file set differs from the frozen model revision.")
    actual = {name: _sha256(snapshot / name) for name in sorted(_SNAPSHOT_FILES)}
    if actual != expected:
        raise ValueError("Chronos-2 snapshot digest differs from the frozen model revision.")
    return actual


def _row_sha256(rows: Sequence[AeonHourlyWindow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.row_id.encode("ascii"))
        digest.update(np.ascontiguousarray(row.context_interval_ids).tobytes())
        digest.update(np.ascontiguousarray(row.target_interval_ids).tobytes())
        digest.update(np.ascontiguousarray(row.target_source_timestamps).tobytes())
        digest.update(np.ascontiguousarray(row.target_db).tobytes())
        digest.update(np.ascontiguousarray(row.target_mask).tobytes())
    return digest.hexdigest()


def _save_predictions(
    path: Path, rows: Sequence[AeonHourlyWindow], forecast: NDArray[np.float64]
) -> None:
    """Persist the existing AEON row schema without importing the training stack."""
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


def prepare_multivariate_context(rows: Sequence[AeonHourlyWindow]) -> NDArray[np.float64]:
    """Return official Chronos-2 (batch, variate, history) input with NaN missingness."""
    if not rows:
        raise ValueError("Chronos-2 needs at least one validation issuance.")
    for row in rows:
        if (
            row.partition != "validation"
            or row.context_db.shape != (24, 4)
            or row.context_mask.shape != (24, 4)
            or row.context_mask.dtype.kind != "b"
            or not row.context_mask[:, 0].all()
            or not np.isfinite(row.context_db[row.context_mask]).all()
            or not np.array_equal(np.diff(row.context_interval_ids), np.ones(23, dtype=np.int64))
            or row.context_interval_ids[-1] != row.cutoff_interval_id
        ):
            raise ValueError("Chronos-2 validation rows differ from the fixed issuance contract.")
    values = np.stack([row.context_db for row in rows]).astype(np.float64, copy=True)
    mask = np.stack([row.context_mask for row in rows])
    values[~mask] = np.nan
    return values.transpose(0, 2, 1)


def _as_numpy(value: object) -> NDArray[np.float64]:
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    return np.asarray(candidate, dtype=np.float64)


def _predict_batch(
    pipeline: Chronos2Like, context: NDArray[np.float64], config: dict[str, Any]
) -> NDArray[np.float64]:
    quantiles, _ = pipeline.predict_quantiles(
        inputs=context,
        prediction_length=6,
        quantile_levels=CHRONOS_QUANTILES.tolist(),
        batch_size=config["pipeline_series_batch_size"],
        context_length=24,
        cross_learning=False,
        limit_prediction_length=False,
    )
    if len(quantiles) != len(context):
        raise ValueError("Chronos-2 returned a different number of forecast rows.")
    result: NDArray[np.float64] = np.empty((len(context), 3, 5), dtype=np.float64)
    for index, item in enumerate(quantiles):
        values = _as_numpy(item)
        if values.shape != (4, 6, 5) or not np.isfinite(values).all():
            raise ValueError("Chronos-2 returned invalid multivariate quantile geometry.")
        result[index] = values[0, HORIZON_INDICES]
    result.sort(axis=-1)
    return result


def _resource_usage() -> tuple[int, int | None]:
    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    torch = sys.modules.get("torch")
    gpu_reserved = None
    if torch is not None and torch.cuda.is_available():
        gpu_reserved = int(torch.cuda.max_memory_reserved())
    return rss, gpu_reserved


def _load_official_pipeline(snapshot: Path, *, device: str) -> Chronos2Like:
    try:
        installed_version = importlib.metadata.version("chronos-forecasting")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "chronos-forecasting==2.3.2 is required for real Chronos-2 inference."
        ) from error
    if installed_version != PACKAGE_VERSION:
        raise ValueError("The installed Chronos inference package is not the frozen version.")
    try:
        chronos = importlib.import_module("chronos")
        pipeline_class = chronos.Chronos2Pipeline
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "chronos-forecasting==2.3.2 is required for real Chronos-2 inference."
        ) from error
    return pipeline_class.from_pretrained(
        str(snapshot.resolve(strict=True)), device_map=device, local_files_only=True
    )


def execute_zero_shot(
    rows: Sequence[AeonHourlyWindow],
    *,
    output: Path,
    config_path: Path,
    review_path: Path,
    model_snapshot: Path,
    cohort_sha256: str,
    pipeline: Chronos2Like | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Forecast immutable validation rows with restart-safe, hash-bound shards."""
    if device not in ("cuda", "cpu") or len(cohort_sha256) != 64:
        raise ValueError("Chronos-2 device or cohort digest is invalid.")
    config_path = config_path.resolve(strict=True)
    config_sha256 = _sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _config_gate(config)
    code_sha256 = _code_sha256()
    review_path = review_path.resolve(strict=True)
    review_sha256 = _sha256(review_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    _verify_review(review, config_sha256, code_sha256)
    model_files = _verify_snapshot(model_snapshot, config["model"]["snapshot_files_sha256"])
    context = prepare_multivariate_context(rows)
    row_sha256 = _row_sha256(rows)
    binding = {
        "config_sha256": config_sha256,
        "review_sha256": review_sha256,
        "model_revision": MODEL_REVISION,
        "model_snapshot_files_sha256": model_files,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "cohort_sha256": cohort_sha256,
        "validation_row_sha256": row_sha256,
        "code_sha256": code_sha256,
        "issued_rows": len(rows),
        "device": device,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shards = output / "shards"
    shards.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("binding") != binding:
            raise ValueError("Chronos-2 resume binding differs from the existing run.")
        if manifest.get("status", "").startswith("COMPLETED"):
            prediction_path = output / "validation-predictions.npz"
            if _sha256(prediction_path) != manifest.get("prediction_sha256"):
                raise ValueError("Completed Chronos-2 prediction artifact digest differs.")
            return manifest
    else:
        manifest = {
            "status": "IN_PROGRESS",
            "classification": "DEVELOPMENT_NOT_FINAL_EVALUATION",
            "fit_behavior": "FROZEN_ZERO_SHOT_NO_TRAINING",
            "calibration_access": "PROHIBITED",
            "test_access": "PROHIBITED",
            "binding": binding,
            "completed_shards": {},
        }
        _atomic_json(manifest_path, manifest)
    completed = manifest.get("completed_shards")
    if not isinstance(completed, dict):
        raise TypeError("Chronos-2 resume manifest has invalid shard state.")
    rows_per_shard = config["rows_per_resume_shard"]
    started = time.perf_counter()
    peak_rss = 0
    peak_gpu = 0
    for batch_index, start in enumerate(range(0, len(rows), rows_per_shard)):
        stop = min(start + rows_per_shard, len(rows))
        name = f"batch-{batch_index:06d}.npz"
        shard = shards / name
        expected_ids = np.asarray([row.row_id for row in rows[start:stop]])
        if name in completed:
            if not shard.is_file() or _sha256(shard) != completed[name]:
                raise ValueError("Chronos-2 completed resume shard digest differs.")
            with np.load(shard, allow_pickle=False) as saved:
                if not np.array_equal(saved["row_ids"], expected_ids):
                    raise ValueError("Chronos-2 completed resume shard rows differ.")
            continue
        if shard.exists():
            with np.load(shard, allow_pickle=False) as saved:
                if (
                    set(saved.files) != {"row_ids", "quantiles_db"}
                    or not np.array_equal(saved["row_ids"], expected_ids)
                    or saved["quantiles_db"].shape != (stop - start, 3, 5)
                ):
                    raise ValueError("Chronos-2 orphan resume shard is invalid.")
            completed[name] = _sha256(shard)
            _atomic_json(manifest_path, manifest)
            continue
        if pipeline is None:
            pipeline = _load_official_pipeline(model_snapshot, device=device)
        prediction = _predict_batch(pipeline, context[start:stop], config)
        temporary = shard.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, row_ids=expected_ids, quantiles_db=prediction)
        os.replace(temporary, shard)
        completed[name] = _sha256(shard)
        rss, gpu = _resource_usage()
        peak_rss = max(peak_rss, rss)
        peak_gpu = max(peak_gpu, gpu or 0)
        if rss > config["peak_process_rss_limit_bytes"] or (
            gpu is not None and gpu > config["peak_gpu_reserved_limit_bytes"]
        ):
            raise MemoryError("Chronos-2 inference exceeded the frozen local resource cap.")
        if device == "cuda" and time.perf_counter() - started > config["max_gpu_seconds"]:
            raise TimeoutError("Chronos-2 inference exceeded the frozen two-GPU-hour cap.")
        manifest["peak_process_tree_rss_bytes"] = peak_rss
        manifest["peak_gpu_reserved_bytes"] = peak_gpu if gpu is not None else None
        _atomic_json(manifest_path, manifest)
    forecasts = []
    for name in completed:
        shard = shards / name
        if _sha256(shard) != completed[name]:
            raise ValueError("Chronos-2 resume shard changed before assembly.")
        with np.load(shard, allow_pickle=False) as saved:
            forecasts.append(saved["quantiles_db"].copy())
    quantiles_db = np.concatenate(forecasts)
    if quantiles_db.shape != (len(rows), 3, 5):
        raise ValueError("Chronos-2 assembled forecast geometry differs from validation rows.")
    truth = np.stack([row.target_db for row in rows])
    observed = np.stack([row.target_mask for row in rows])
    source_times = np.stack([row.target_source_timestamps for row in rows])
    metrics = daily_pinball(truth, quantiles_db, observed, source_times)
    prediction_path = output / "validation-predictions.npz"
    if prediction_path.exists():
        with np.load(prediction_path, allow_pickle=False) as saved:
            if not np.array_equal(
                saved["row_ids"], np.asarray([row.row_id for row in rows])
            ) or not np.array_equal(saved["quantiles_db"], quantiles_db):
                raise FileExistsError(
                    "Incomplete Chronos-2 run has a different final prediction file."
                )
    else:
        temporary_prediction = output / "validation-predictions.tmp.npz"
        _save_predictions(temporary_prediction, rows, quantiles_db)
        os.replace(temporary_prediction, prediction_path)
    manifest.update(
        {
            "status": "COMPLETED_TRAIN_VALIDATION_ZERO_SHOT_NOT_FINAL_EVALUATION",
            "prediction_path": prediction_path.name,
            "prediction_sha256": _sha256(prediction_path),
            "metrics": metrics,
            "elapsed_inference_seconds": time.perf_counter() - started,
            "model_input_decision": (
                "Four native target variates with NaN missingness; evaluate only 38 kHz "
                "at source steps 1, 3 and 6."
            ),
        }
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def run_chronos(
    archive: Path,
    split_review: Path,
    chronos_review: Path,
    config: Path,
    model_snapshot: Path,
    output: Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Load only reviewed TRAIN/validation partitions and forecast exact validation rows."""
    # The campaign module imports the optional CUDA training stack. Keep that
    # dependency out of pure executor/tests and load it only for the real CLI.
    from marine_echo.training.aeon_campaign import load_cohort

    _, validation, cohort_sha256, _ = load_cohort(archive, split_review)
    return execute_zero_shot(
        validation,
        output=output,
        config_path=config,
        review_path=chronos_review,
        model_snapshot=model_snapshot,
        cohort_sha256=cohort_sha256,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-review", type=Path, required=True)
    parser.add_argument("--chronos-review", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    arguments = parser.parse_args()
    result = run_chronos(
        arguments.archive,
        arguments.split_review,
        arguments.chronos_review,
        arguments.config,
        arguments.model_snapshot,
        arguments.output,
        device=arguments.device,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
