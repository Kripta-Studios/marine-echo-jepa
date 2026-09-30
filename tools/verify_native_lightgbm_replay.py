"""Verify real persisted native boosters on target-free DEV context prefixes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from marine_echo.inference.native_acoustic import load_context, validate_context
from marine_echo.models.native_temporal import QUANTILES
from marine_echo.training.native_references import feature_matrix, load_booster_text

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "outputs/native_acoustic_ssl_v1/lightgbm_seed7_h96_text"
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if (
        result.get("status") != "COMPLETED_DEVELOPMENT_REFERENCE"
        or result.get("method") != "lightgbm"
    ):
        raise ValueError("Require the actual completed native tree comparator.")
    expected_paths = {f"h{h}_q{q:.2f}.txt" for h in (1, 3, 6) for q in QUANTILES}
    models = result["models"]
    if len(models) != 15 or {model["path"] for model in models} != expected_paths:
        raise ValueError("Exactly15 distinct bound native horizon/quantile boosters required.")
    identities = {model["path"]: model["sha256"] for model in models}
    for name, expected in identities.items():
        if digest(folder / name) != expected:
            raise ValueError("Persisted booster identity differs from the actual fit.")
    data_path = ROOT / "data/processed/native_ssl_v1/development.npz"
    if digest(data_path) != result["dev_sha256"]:
        raise ValueError("Development cohort identity changed.")
    data = {key: array[:64] for key, array in load_context(data_path).items()}
    validate_context(data, 96)
    features = feature_matrix(data, 96)
    predictions = np.empty((64, 3, 5), np.float64)
    for h_index, horizon in enumerate((1, 3, 6)):
        for q_index, quantile in enumerate(QUANTILES):
            booster = load_booster_text(folder / f"h{horizon}_q{quantile:.2f}.txt")
            predictions[:, h_index, q_index] = booster.predict(features, num_threads=4)
    predictions.sort(axis=-1)
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        expected = saved["predictions"][:64]
    np.testing.assert_array_equal(predictions, expected)
    proof = {
        "status": "PASSED_ACTUAL_PERSISTED_LIGHTGBM_CPU_REPLAY",
        "result_sha256": digest(folder / "result.json"),
        "boosters_verified": 15,
        "input_keys": sorted(data),
        "target_or_future_acoustic_inputs": [],
        "development_prefixes": 64,
        "prediction_shape": list(predictions.shape),
        "maximum_absolute_replay_error_db": 0.0,
        "model_transport": "reviewed_UTF8_text_to_model_str",
        "native_bounds_m": np.unique(
            data["query"][:, :, 3:5].reshape(-1, 2) * 250, axis=0
        ).tolist(),
        "fitted_weights_or_optimizer_updates": 0,
        "test_access": "NOT_RUN",
    }
    with (ROOT / "evidence/ssl-research-v1/lightgbm-context-only-cpu-replay.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
