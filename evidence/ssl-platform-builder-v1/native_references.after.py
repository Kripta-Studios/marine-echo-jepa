"""Matched native-product persistence, seasonal and LightGBM reference executors."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psutil

from marine_echo.evaluation.native_product import QUANTILES, native_scores


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_booster_text(booster, path: Path) -> None:
    """Save complete model text through Python's Unicode-aware exclusive IO.

    Use only a reviewed new run path. Permission/existence errors propagate;
    there is no native filename fallback, relocation, or write retry.
    """
    text = booster.model_to_string()
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def load_booster_text(path: Path):
    """Replay complete model text without passing a filename to the C++ API."""
    import lightgbm as lgb

    with Path(path).open("r", encoding="utf-8") as stream:
        text = stream.read()
    return lgb.Booster(model_str=text)


def feature_matrix(corpus: dict, history: int) -> np.ndarray:
    """Only declared context/masks/native metadata enter reference features."""
    x = corpus["x"][:, -history:].copy()
    observed = corpus["observed"][:, -history:]
    x[~observed] = 0
    parts = [
        x.reshape(len(x), -1),
        observed.reshape(len(x), -1),
        corpus["metadata"].reshape(len(x), -1),
    ]
    for length in sorted({4, 24, history}):
        values, mask = x[:, -length:], observed[:, -length:]
        count = mask.sum(axis=1)
        mean = values.sum(axis=1) / np.maximum(count, 1)
        variance = ((values - mean[:, None]) ** 2 * mask).sum(axis=1) / np.maximum(count, 1)
        parts.extend([mean, np.sqrt(variance), count / length])
    return np.concatenate(parts, axis=1).astype(np.float32)


def load(path: Path, role: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data.get("corpus_role")) != role:
        raise ValueError("Reference corpus role differs.")
    return data


def run(
    train_path: Path,
    dev_path: Path,
    output: Path,
    review_path: Path,
    method: str,
    history: int = 96,
    seed: int = 7,
) -> dict:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_PREFIT" or method not in review.get("allowed_methods", []):
        raise ValueError("Reference requires exact distinct prefit approval.")
    if review.get("reviewer_session_id") == review.get("implementer_session_id") or not review.get(
        "reviewer_session_id"
    ):
        raise ValueError("Reference requires distinct review session.")
    for path in (
        train_path,
        dev_path,
        Path(__file__),
        Path(__file__).parents[1] / "evaluation/native_product.py",
    ):
        if review.get("bindings", {}).get(str(path.resolve())) != digest(path):
            raise ValueError(f"Reference review hash differs: {path}")
    started = time.perf_counter()
    train, dev = load(train_path, "train"), load(dev_path, "development")
    if set(train["deployment"]) & set(dev["deployment"]):
        raise ValueError("References require whole deployment separation.")
    if history not in (24, 96):
        raise ValueError("Unregistered reference history.")
    output.mkdir(parents=True, exist_ok=False)
    models = []
    if method in ("persistence", "seasonal24"):
        if method == "persistence":
            values = np.repeat(dev["x"][:, -1, 0, None], 3, axis=1)
        else:
            # Last context index is cutoff t; t+h-24 is index -25+h.
            values = dev["x"][:, -25 + np.asarray([1, 3, 6]), 0]
        predictions = np.repeat(values[..., None], 5, axis=-1)
    elif method == "lightgbm":
        import lightgbm as lgb

        tx, dx = feature_matrix(train, history), feature_matrix(dev, history)
        predictions = np.zeros((len(dx), 3, 5))
        for horizon in range(3):
            eligible = train["y_observed"][:, horizon]
            for q_index, quantile in enumerate(QUANTILES):
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=float(quantile),
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=64,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=4,
                    verbosity=-1,
                )
                model.fit(tx[eligible], train["y"][eligible, horizon])
                predictions[:, horizon, q_index] = model.predict(dx)
                path = output / f"h{(1, 3, 6)[horizon]}_q{quantile:.2f}.txt"
                save_booster_text(model.booster_, path)
                models.append({"path": path.name, "sha256": digest(path)})
        predictions.sort(axis=-1)
    else:
        raise ValueError("Unknown native reference method.")
    metrics = native_scores(
        predictions, dev["y"], dev["y_observed"], dev["target_dates"], dev["deployment"]
    )
    np.savez_compressed(
        output / "predictions.npz",
        predictions=predictions,
        targets=dev["y"],
        observed=dev["y_observed"],
        target_dates=dev["target_dates"],
        deployment=dev["deployment"],
        row_id=dev["row_id"],
        query=dev["query"],
    )
    result = {
        "status": "COMPLETED_DEVELOPMENT_REFERENCE",
        "method": method,
        "history": history,
        "seed": seed,
        "train_sha256": digest(train_path),
        "dev_sha256": digest(dev_path),
        "review_sha256": digest(review_path),
        "metrics": metrics,
        "models": models,
        "elapsed_seconds": time.perf_counter() - started,
        "rss_gib_at_completion": psutil.Process().memory_info().rss / 1024**3,
        "gpu_hours": 0.0,
        "tuning": "one_fixed_development_recipe_no_test_access",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("train", "dev", "output", "review"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    parser.add_argument(
        "--method", choices=("persistence", "seasonal24", "lightgbm"), required=True
    )
    parser.add_argument("--history", type=int, choices=(24, 96), default=96)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run(
        args.train, args.dev, args.output, args.review, args.method, args.history, args.seed
    )
    print(
        json.dumps(
            {
                "method": result["method"],
                "primary_pinball_db": result["metrics"]["primary_pinball_db"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
