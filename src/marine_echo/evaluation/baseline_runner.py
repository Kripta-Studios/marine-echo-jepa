"""Run the four fixed conventional baselines on bounded synthetic canonical rows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from marine_echo.evaluation import metrics as metrics_module
from marine_echo.evaluation.metrics import daily_metrics
from marine_echo.models import baselines as baselines_module
from marine_echo.models.baselines import (
    DailySeasonalBaseline,
    PersistenceBaseline,
    RidgeQuantileBaseline,
    TreeQuantileBaseline,
)
from marine_echo.training import data_adapter as adapter_module
from marine_echo.training.data_adapter import CanonicalWindowAdapter, TrainOnlyScaler, WindowBatch

FAMILIES = ("persistence", "seasonal", "ridge", "hist_gradient_boosting")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _input_digest(train: WindowBatch, validation: WindowBatch, tree_max_iter: int) -> str:
    digest = hashlib.sha256()
    for batch in (train, validation):
        digest.update(
            _json_bytes(
                {
                    "partition": batch.partition,
                    "row_ids": batch.row_ids,
                    "provenance": batch.provenance,
                }
            )
        )
        for array in (
            batch.context_db,
            batch.context_mask,
            batch.target_db,
            batch.target_support,
            batch.cutoffs,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(_json_bytes({"tree_max_iter": tree_max_iter}))
    return digest.hexdigest()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".stage.", dir=path.parent)
    staging = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(document))
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError("Existing baseline artifact cannot be overwritten.")
        os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _prediction_artifact(
    path: Path, *, batch: WindowBatch, quantiles: np.ndarray, eligible: np.ndarray
) -> None:
    if path.exists():
        raise FileExistsError("Existing predictions cannot be overwritten.")
    descriptor, name = tempfile.mkstemp(
        prefix=path.name + ".stage.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    staging = Path(name)
    try:
        np.savez_compressed(
            staging,
            row_ids=np.array(batch.row_ids, dtype="U64"),
            cutoffs=batch.cutoffs,
            truth_db=batch.target_db,
            quantiles=quantiles,
            eligible=eligible,
            target_support=batch.target_support,
        )
        with staging.open("rb+") as stream:
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError("Existing predictions cannot be overwritten.")
        os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _verify_resume(
    saved: dict[str, Any],
    prediction_path: Path,
    *,
    family: str,
    provenance: dict[str, Any],
    train: WindowBatch,
    validation: WindowBatch,
) -> None:
    if (
        set(saved)
        != {
            "schema_version",
            "status",
            "family",
            "partition",
            "metrics",
            "provenance",
            "prediction_sha256",
            "train_rows",
            "validation_rows",
        }
        or saved.get("schema_version") != "1.0"
        or saved.get("status") != "COMPLETED_SYNTHETIC_FIXTURE"
        or saved.get("family") != family
        or saved.get("partition") != "validation"
        or saved.get("train_rows") != len(train.row_ids)
        or saved.get("validation_rows") != len(validation.row_ids)
        or saved.get("provenance") != provenance
    ):
        raise ValueError("Baseline resume manifest schema or provenance differs.")
    if _sha256(prediction_path) != saved.get("prediction_sha256"):
        raise ValueError("Baseline prediction checksum differs from saved run.")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        if set(loaded.files) != {
            "row_ids",
            "cutoffs",
            "truth_db",
            "quantiles",
            "eligible",
            "target_support",
        }:
            raise ValueError("Baseline prediction artifact fields differ from the run contract.")
        quantiles = loaded["quantiles"]
        eligible = loaded["eligible"]
        if (
            quantiles.shape != (len(validation.row_ids), 3, 5)
            or quantiles.dtype.kind != "f"
            or eligible.shape != (len(validation.row_ids), 3)
            or eligible.dtype.kind != "b"
            or loaded["truth_db"].dtype.kind != "f"
            or loaded["target_support"].dtype.kind != "f"
            or loaded["cutoffs"].dtype.kind != "M"
            or loaded["row_ids"].dtype.kind != "U"
            or not np.array_equal(loaded["row_ids"], np.array(validation.row_ids))
            or not np.array_equal(loaded["cutoffs"], validation.cutoffs)
            or not np.array_equal(loaded["truth_db"], validation.target_db)
            or not np.array_equal(loaded["target_support"], validation.target_support)
            or not np.array_equal(eligible, np.isfinite(quantiles).all(axis=-1))
            or not np.all(np.isnan(quantiles[~eligible]))
            or not np.all(np.diff(quantiles[eligible], axis=-1) >= 0)
        ):
            raise ValueError("Baseline prediction artifact does not match validation rows.")
        metrics = daily_metrics(validation.target_db, quantiles, validation.cutoffs)
    if saved.get("metrics") != metrics:
        raise ValueError("Baseline resume metrics differ from saved predictions.")


def run_baselines(
    adapter: CanonicalWindowAdapter,
    output: Path,
    *,
    max_windows: int,
    tree_max_iter: int = 150,
) -> dict[str, dict[str, Any]]:
    """Execute B0-B3 with train-only fit and immutable validation artifacts.

    This backend is deliberately fixture-only; it cannot authorize a real benchmark.
    """
    if tree_max_iter < 1 or tree_max_iter > 150:
        raise ValueError("Tree fitting requires one bounded declared configuration.")
    if adapter.store._manifest()["promotion_status"] != "NONPROMOTABLE_ENGINEERING_FIXTURE":
        raise ValueError("Real R0/R1 promotion is not implemented in this executor.")
    train = adapter.materialize("train", max_windows=max_windows)
    validation = adapter.materialize("validation", max_windows=max_windows)
    scaler = TrainOnlyScaler().fit(train)
    assert scaler.mean is not None and scaler.scale is not None
    scaler_sha = hashlib.sha256(
        np.ascontiguousarray(scaler.mean).tobytes() + np.ascontiguousarray(scaler.scale).tobytes()
    ).hexdigest()
    input_sha = _input_digest(train, validation, tree_max_iter)
    provenance = {
        **validation.provenance,
        "input_sha256": input_sha,
        "scaler_partition": scaler.provenance["partition"],
        "train_scaler_sha256": scaler_sha,
        "tree_max_iter": tree_max_iter,
        "runner_sha256": _sha256(Path(__file__)),
        "baseline_source_sha256": _sha256(Path(baselines_module.__file__ or "")),
        "metrics_source_sha256": _sha256(Path(metrics_module.__file__ or "")),
        "adapter_source_sha256": _sha256(Path(adapter_module.__file__ or "")),
    }
    output.mkdir(parents=True, exist_ok=True)
    models: dict[
        str,
        PersistenceBaseline | DailySeasonalBaseline | RidgeQuantileBaseline | TreeQuantileBaseline,
    ] = {
        "persistence": PersistenceBaseline(),
        "seasonal": DailySeasonalBaseline(),
        "ridge": RidgeQuantileBaseline(),
        "hist_gradient_boosting": TreeQuantileBaseline(max_iter=tree_max_iter),
    }
    results: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        family_dir = output / family
        manifest_path = family_dir / "run.json"
        prediction_path = family_dir / "validation_predictions.npz"
        if family_dir.exists() and (not manifest_path.is_file() or not prediction_path.is_file()):
            raise FileExistsError("Incomplete or unregistered baseline artifact exists.")
        if manifest_path.is_file():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            _verify_resume(
                saved,
                prediction_path,
                family=family,
                provenance=provenance,
                train=train,
                validation=validation,
            )
            results[family] = {**saved, "disposition": "SKIPPED_VERIFIED"}
            continue
        family_dir.mkdir(parents=False, exist_ok=False)
        model = models[family]
        model.fit(
            train.context_db, train.context_mask, train.target_db, partition="train", units="sv_db"
        )
        forecast = model.predict(validation.context_db, validation.context_mask, units="sv_db")
        metrics = daily_metrics(validation.target_db, forecast.quantiles, validation.cutoffs)
        _prediction_artifact(
            prediction_path,
            batch=validation,
            quantiles=forecast.quantiles,
            eligible=forecast.eligible,
        )
        document = {
            "schema_version": "1.0",
            "status": "COMPLETED_SYNTHETIC_FIXTURE",
            "family": family,
            "partition": "validation",
            "metrics": metrics,
            "provenance": provenance,
            "prediction_sha256": _sha256(prediction_path),
            "train_rows": len(train.row_ids),
            "validation_rows": len(validation.row_ids),
        }
        _atomic_json(manifest_path, document)
        results[family] = {**document, "disposition": "WRITTEN"}
    return results
