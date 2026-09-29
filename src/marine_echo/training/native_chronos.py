"""Pinned local Chronos-2 zero-shot comparator on native acoustic context support."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from marine_echo.evaluation.native_product import QUANTILES, native_scores
from marine_echo.training.aeon_chronos import _load_official_pipeline, _verify_snapshot
from marine_echo.training.native_references import digest, load


def contexts(corpus: dict, history: int) -> np.ndarray:
    if history not in (24, 96):
        raise ValueError("Chronos receives the declared matched history.")
    values = corpus["x"][:, -history:].astype(np.float64, copy=True)
    values[~corpus["observed"][:, -history:]] = np.nan
    return values.transpose(0, 2, 1)


def native_inputs(corpus: dict, history: int) -> list[dict]:
    """Public pretrained API: three past acoustic covariates plus known queries.

    Measurement descriptors are known properties of the requested product,
    rather than future acoustic observations or fabricated intervention actions.
    """
    values = contexts(corpus, history)
    inputs = []
    for index, row in enumerate(values):
        past = {
            f"acoustic_{frequency}_hz": row[channel]
            for channel, frequency in enumerate((38000, 125000, 200000, 455000))
            if channel > 0
        }
        future = {}
        for column, name in enumerate(
            (
                "frequency",
                "interval_duration",
                "geometry_kind",
                "upper_bound",
                "lower_bound",
                "orientation_unknown_code",
                "processing_known",
                "instrument_known",
                "clock_known",
            )
        ):
            value = corpus["metadata"][index, 0, column]
            past[f"query_{name}"] = np.full(history, value, dtype=np.float32)
            future[f"query_{name}"] = np.full(6, value, dtype=np.float32)
        past["relative_source_offset"] = np.arange(-history + 1, 1, dtype=np.float32)
        future["relative_source_offset"] = np.arange(1, 7, dtype=np.float32)
        inputs.append({"target": row[0:1], "past_covariates": past, "future_covariates": future})
    return inputs


def predict_rows(pipeline: object, values: np.ndarray | list[dict], history: int) -> np.ndarray:
    quantiles, _ = pipeline.predict_quantiles(
        inputs=values,
        prediction_length=6,
        quantile_levels=QUANTILES.tolist(),
        batch_size=64,
        context_length=history,
        cross_learning=False,
        limit_prediction_length=False,
    )
    if len(quantiles) != len(values):
        raise ValueError("Chronos returned different issuance support.")
    output = np.empty((len(values), 3, 5), dtype=np.float64)
    for index, item in enumerate(quantiles):
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        item = np.asarray(item)
        expected_variates = 1 if isinstance(values, list) else 4
        if item.shape != (expected_variates, 6, 5):
            raise ValueError("Chronos variate/horizon/quantile shape differs.")
        output[index] = np.sort(item[0, [0, 2, 5]], axis=-1)
    return output


def run(
    dev_path: Path,
    output: Path,
    review_path: Path,
    snapshot: Path,
    history: int = 96,
    device: str = "cuda",
) -> dict:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_PREFIT" or "chronos2" not in review.get(
        "allowed_methods", []
    ):
        raise ValueError("Chronos native comparator needs exact independent approval.")
    if not review.get("reviewer_session_id") or review.get("reviewer_session_id") == review.get(
        "implementer_session_id"
    ):
        raise ValueError("Chronos requires distinct reviewer identity.")
    required = [
        dev_path,
        Path(__file__),
        Path(__file__).with_name("aeon_chronos.py"),
        Path(__file__).with_name("native_references.py"),
        Path(__file__).parents[1] / "evaluation/native_product.py",
    ]
    for path in required:
        if review.get("bindings", {}).get(str(path.resolve())) != digest(path):
            raise ValueError(f"Chronos native review binding differs: {path}")
    expected = review.get("chronos_snapshot_sha256")
    if not expected:
        raise ValueError("Exact Chronos snapshot bindings required.")
    model_files = _verify_snapshot(snapshot, expected)
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("The bounded Chronos slot requires verified CUDA.")
    corpus = load(dev_path, "development")
    context = native_inputs(corpus, history)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    identities = {
        "dev_sha256": digest(dev_path),
        "history": history,
        "snapshot_sha256": model_files,
        "review_sha256": digest(review_path),
        "executor_sha256": digest(Path(__file__)),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identities"] != identities:
            raise ValueError("Chronos resume source/data/history/model/review identity differs.")
    else:
        if any(output.iterdir()):
            raise ValueError("Preserve unbound output files; use a new Chronos attempt directory.")
        manifest = {"identities": identities, "elapsed_seconds": 0.0, "shards": {}}
    previous_elapsed = manifest["elapsed_seconds"]
    started = time.perf_counter()

    def save_manifest() -> None:
        manifest["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    save_manifest()
    pipeline = _load_official_pipeline(snapshot, device=device)
    predictions = []
    for start in range(0, len(context), 16):
        shard = output / f"shard-{start:06d}.npz"
        stop = min(start + 16, len(context))
        if shard.exists():
            if manifest["shards"].get(shard.name) != digest(shard):
                raise ValueError("Chronos resume shard digest is unbound or changed.")
            with np.load(shard, allow_pickle=False) as archive:
                if not np.array_equal(archive["row_id"], corpus["row_id"][start:stop]) or str(
                    archive["dev_sha256"]
                ) != digest(dev_path):
                    raise ValueError("Chronos resume shard identity differs.")
                if (
                    int(archive["history"]) != history
                    or json.loads(str(archive["snapshot_sha256"])) != model_files
                ):
                    raise ValueError("Chronos resume shard model/history differs.")
                prediction = archive["predictions"]
                if prediction.shape != (stop - start, 3, 5):
                    raise ValueError("Chronos resume shard shape differs.")
        else:
            prediction = predict_rows(pipeline, context[start:stop], history)
            np.savez_compressed(
                shard,
                predictions=prediction,
                row_id=corpus["row_id"][start:stop],
                dev_sha256=digest(dev_path),
                history=np.asarray(history),
                snapshot_sha256=np.asarray(json.dumps(model_files, sort_keys=True)),
            )
            manifest["shards"][shard.name] = digest(shard)
        predictions.append(prediction)
        save_manifest()
        if (
            torch.cuda.max_memory_reserved() >= 10 * 1024**3
            or psutil.Process().memory_info().rss >= 22 * 1024**3
        ):
            raise MemoryError("Chronos local resource ceiling exceeded.")
        if manifest["elapsed_seconds"] >= 2 * 3600:
            raise TimeoutError("Chronos fixed two GPU-hour slot exhausted.")
    predictions = np.concatenate(predictions)
    np.savez_compressed(
        output / "predictions.npz",
        predictions=predictions,
        targets=corpus["y"],
        observed=corpus["y_observed"],
        target_dates=corpus["target_dates"],
        deployment=corpus["deployment"],
        row_id=corpus["row_id"],
        query=corpus["query"],
    )
    result = {
        "status": "COMPLETED_ZERO_SHOT_DEVELOPMENT",
        "history": history,
        "weights_fitted_in_this_study": False,
        "cross_learning": False,
        "permitted_information": "same four past channels/masks; native query metadata supplied as known numeric covariates; no future acoustic covariates",
        "metadata_limitation": "Providing numeric measurement covariates does not establish pretrained sensor semantics or cross-geometry comparability",
        "pretraining_exposure": "public pretrained model ancestry; no assertion of comprehensive acoustic-source exclusion",
        "snapshot_files": model_files,
        "dev_sha256": digest(dev_path),
        "elapsed_seconds": manifest["elapsed_seconds"],
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "metrics": native_scores(
            predictions,
            corpus["y"],
            corpus["y_observed"],
            corpus["target_dates"],
            corpus["deployment"],
        ),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dev", "output", "review", "snapshot"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--history", type=int, choices=(24, 96), default=96)
    args = parser.parse_args()
    result = run(args.dev, args.output, args.review, args.snapshot, args.history)
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_pinball_db": result["metrics"]["primary_pinball_db"],
            }
        )
    )


if __name__ == "__main__":
    main()
