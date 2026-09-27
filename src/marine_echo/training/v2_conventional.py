"""Finite native conventional TRAIN development execution and row artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import psutil

from marine_echo.models.v2_conventional import ConventionalFamily, NativeConventional
from marine_echo.training.v2_executor import (
    V2_PROTOCOL_SHA256,
    _review_gate,
    _sha256,
    _valid_hash,
    _validate_rows,
    _verify_predictions,
    _write_predictions,
)
from marine_echo.training.v2_stream import HourlyWindow


def execute_conventional(
    fit: list[HourlyWindow],
    assess: list[HourlyWindow],
    output: Path,
    *,
    family: ConventionalFamily,
    protocol_sha256: str,
    fixture_only: bool = False,
    review_path: Path | None = None,
    review_sha256: str | None = None,
    tree_max_iter: int = 150,
) -> dict[str, Any]:
    """Fit one frozen TRAIN-only conventional family, then save every issued row."""
    if not _valid_hash(protocol_sha256):
        raise ValueError("A frozen protocol digest is required.")
    if not fixture_only and protocol_sha256 != V2_PROTOCOL_SHA256:
        raise ValueError("Real conventional runs require the reviewed v2 protocol digest.")
    if not fixture_only and tree_max_iter != 150:
        raise ValueError("Real histogram boosting uses the frozen 150-iteration budget.")
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
        raise FileExistsError("Conventional output already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        process = psutil.Process()
        peak_rss = process.memory_info().rss
        started = time.perf_counter()
        model = NativeConventional(family, tree_max_iter=tree_max_iter).fit(fit)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if peak_rss >= 22 * 1024**3:
            raise MemoryError("Conventional fit process RAM reached the 22 GiB limit.")
        prediction = model.predict(assess)
        elapsed = time.perf_counter() - started
        model_path = staging / "model.joblib"
        joblib.dump(model, model_path)
        rows_path = staging / "assessment-predictions.npz"
        _write_predictions(rows_path, assess, prediction)
        result: dict[str, Any] = {
            "status": "COMPLETED_SYNTHETIC_FIXTURE"
            if fixture_only
            else "COMPLETED_TRAIN_DEVELOPMENT",
            "family": family,
            "protocol_sha256": protocol_sha256,
            "input_sha256": input_digest,
            "source_sha256": source_hashes,
            "review_sha256": review_sha256 if not fixture_only else None,
            "tree_max_iter": tree_max_iter if family == "hist_gradient_boosting" else None,
            "fallback_count": model.fallback_count,
            "fit_rows": len(fit),
            "assessment_rows": len(assess),
            "fit_seconds": elapsed,
            "peak_process_rss_bytes": max(peak_rss, process.memory_info().rss),
            "model": str(output / model_path.name),
            "model_sha256": _sha256(model_path),
            "predictions": str(output / rows_path.name),
            "prediction_sha256": _sha256(rows_path),
            "metrics": _verify_predictions(rows_path, assess),
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
