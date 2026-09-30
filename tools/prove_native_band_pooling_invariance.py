"""SYNTHETIC_CORRECTNESS_ONLY frequency-conditioning mechanism check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from marine_echo.inference.native_encoder import NativeAcousticEncoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    weights = ROOT / "outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0/selected_encoder.pt"
    encoder = NativeAcousticEncoder(weights)
    mean = np.asarray(encoder.scalers.channel_mean, np.float64)
    std = np.asarray(encoder.scalers.channel_std, np.float64)
    wave = np.sin(np.arange(96, dtype=np.float64) * 0.17)[:, None] * 0.3
    first = mean[None] + wave * std[None]
    second = first.copy()
    second[:, 0] += 0.7 * std[0]
    second[:, 1] -= 0.7 * std[1]
    values = np.stack([first, second])
    observed = np.ones(values.shape, dtype=np.bool_)
    metadata = np.zeros((2, 4, 10), np.float32)
    metadata[:, :, 0] = np.asarray([38000, 125000, 200000, 455000]) / 455000
    metadata[:, :, 1:3] = 1
    metadata[:, :, 4] = 200 / 250
    features = encoder.encode(values, observed, metadata)
    maximum = float(np.abs(features[0] - features[1]).max())
    np.testing.assert_allclose(features[0], features[1], atol=1e-5, rtol=1e-5)
    proof = {
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "status": "REPRODUCED_LINEAR_CHANNEL_POOLING_NON_IDENTIFIABILITY",
        "public_acoustic_inputs": "NONE; two generated sine contexts, not publisher observations",
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "native_geometry_m": [0, 200],
        "forecast_primary_frequency_hz": 38000,
        "primary_change_db": float(0.7 * std[0]),
        "secondary_compensating_change_db": float(-0.7 * std[1]),
        "metadata_or_mask_changes": False,
        "maximum_absolute_embedding_difference": maximum,
        "mechanism": "Shared linear projection followed by fixed observed-count channel averaging retains only a weighted channel sum before nonlinear temporal processing; metadata addition cannot recover lost frequency-value associations",
        "interpretation": "Structural information limitation, not a proven cause of the development score gap",
        "optimizer_or_backward_calls": 0,
        "real_corpus_parsing": "NOT_RUN",
        "test_access": "NOT_RUN",
    }
    with (ROOT / "evidence/ssl-research-v1/band-pooling-invariance.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
