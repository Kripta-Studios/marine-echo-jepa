"""Verify actual pretrained CPU encoding/replay using target-free DEV prefixes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from marine_echo.inference.native_acoustic import NativeAcousticPredictor, load_context

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    torch.set_num_threads(4)
    run = ROOT / "outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0"
    data = {
        key: array[:64]
        for key, array in load_context(
            ROOT / "data/processed/native_ssl_v1/development.npz"
        ).items()
    }
    predictor = NativeAcousticPredictor(run / "inference.pt", device="cpu")
    result = predictor.encode_and_forecast(data)
    with np.load(run / "predictions.npz", allow_pickle=False) as saved:
        predictions = saved["predictions"][:64]
    np.testing.assert_allclose(result["quantile_forecasts_db"], predictions, rtol=1e-5, atol=2e-5)
    latent = result["embeddings"]
    assert latent.shape == (64, 64) and np.isfinite(latent).all()
    assert np.std(latent, axis=0).max() > 0
    report = {
        "status": "PASSED_ACTUAL_PRETRAINED_CPU_REPLAY",
        "development_prefixes": 64,
        "input_keys": sorted(data),
        "target_or_future_inputs": [],
        "embedding_shape": list(latent.shape),
        "maximum_absolute_replay_error_db": float(
            np.abs(result["quantile_forecasts_db"] - predictions).max()
        ),
        "selected_encoder_sha256": digest(run / "selected_encoder.pt"),
        "inference_sha256": digest(run / "inference.pt"),
        "query_native_bounds_m": np.unique(
            data["query"][:, :, 3:5].reshape(-1, 2) * 250, axis=0
        ).tolist(),
        "device": "cpu",
        "optimizer_updates": 0,
        "test_access": "NOT_RUN",
        "representation_transfer_advantage": "NOT_ESTABLISHED",
    }
    output = ROOT / "evidence/ssl-research-v1/shared-pretrained-cpu-replay.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
