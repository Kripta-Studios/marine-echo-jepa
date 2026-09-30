"""Verify the real selected SSL encoder without a head or query input."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from marine_echo.inference.native_acoustic import NativeAcousticPredictor, load_context
from marine_echo.inference.native_encoder import NativeAcousticEncoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    folder = ROOT / "outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0"
    data = {
        key: value[:64]
        for key, value in load_context(
            ROOT / "data/processed/native_ssl_v1/development.npz"
        ).items()
    }
    encoder = NativeAcousticEncoder(folder / "selected_encoder.pt")
    latent = encoder.encode(data["x"], data["observed"], data["metadata"])
    predictor = NativeAcousticPredictor(folder / "inference.pt")
    expected = predictor.encode(data)
    np.testing.assert_array_equal(latent, expected)
    assert encoder.training_kind == "ssl_pretrained"
    assert not any(parameter.requires_grad for parameter in encoder.encoder.parameters())
    report = {
        "status": "PASSED_ACTUAL_ENCODER_ONLY_CPU_REPLAY",
        "embedding_shape": list(latent.shape),
        "input_keys": ["x", "observed", "metadata"],
        "query_head_target_future_inputs": [],
        "maximum_absolute_replay_error": 0.0,
        "native_bounds_m": np.unique(
            data["metadata"][:, :, 3:5].reshape(-1, 2) * 250, axis=0
        ).tolist(),
        "method": encoder.method,
        "training_kind": encoder.training_kind,
        "optimizer_updates": 0,
        "test_access": "NOT_RUN",
    }
    with (ROOT / "evidence/ssl-research-v1/shared-encoder-only-cpu-replay.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
