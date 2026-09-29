"""Reusable target-free inference for native integrated-product checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from marine_echo.models.native_temporal import HORIZONS, QUANTILES
from marine_echo.training.native_ssl import (
    Config,
    Scalers,
    initialize_model,
    model_dimensions,
    tensor_batch,
)

INPUT_KEYS = ("x", "observed", "metadata", "query")


def load_context(path):
    """Read observed prefixes and issued queries only, never assessment arrays."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in INPUT_KEYS}


def validate_context(data, history):
    x, mask, meta, query = (data[key] for key in INPUT_KEYS)
    if x.ndim != 3 or x.shape[1:] != (history, 4) or mask.shape != x.shape:
        raise ValueError("Expected the checkpoint's complete declared history [N,H,4].")
    if mask.dtype != np.bool_ or not len(x) or not np.isfinite(x[mask]).all():
        raise ValueError("Observed masks must be Boolean and observed acoustic values finite.")
    if not mask[:, :, 0].all():
        raise ValueError(
            "The declared primary channel issuance requires a complete observed prefix."
        )
    if meta.shape != (len(x), 4, 10) or query.shape != (len(x), 3, 10):
        raise ValueError("Expected encoded native metadata [N,4,10] and queries [N,3,10].")
    if not np.isfinite(meta).all() or not np.isfinite(query).all():
        raise ValueError("Native query and metadata must be finite.")
    if not np.allclose(query[:, :, :9], meta[:, :1, :9], atol=1e-6, rtol=0):
        raise ValueError("This product forecasts the observed primary channel's native geometry.")
    if not np.allclose(query[:, :, -1], np.asarray(HORIZONS)[None], atol=1e-6, rtol=0):
        raise ValueError("Only the declared later1/3/6 interval queries are supported.")
    if not np.allclose(meta[:, :, -1], 0, atol=1e-6, rtol=0):
        raise ValueError("Context metadata uses issuance-relative offset zero.")
    if not np.allclose(meta[:, :, 0], np.asarray([38000, 125000, 200000, 455000])[None] / 455000):
        raise ValueError("Channel order is38/125/200/455kHz.")
    if (meta[:, :, 3] < 0).any() or (meta[:, :, 4] <= meta[:, :, 3]).any():
        raise ValueError("Each native integration interval requires increasing physical bounds.")
    if not np.allclose(meta[:, :, 2], 1):
        raise ValueError("This checkpoint requires integrated products, not depth profiles.")


class NativeAcousticPredictor:
    """Load safe weights once; expose raw-dB context encode and forecast methods."""

    def __init__(self, weights, *, device="cpu"):
        artifact = torch.load(weights, weights_only=True, map_location="cpu")
        if artifact.get("kind") != "native_ssl_weights_only_inference_v1":
            raise ValueError("Unsupported native inference artifact.")
        self.config = Config(**artifact["config"])
        self.scalers = Scalers.from_dict(artifact["scalers"])
        self.device = device
        self.model = initialize_model(
            self.config.seed, **model_dimensions(self.config), method=self.config.method
        ).to(device)
        self.model.load_state_dict(artifact["model"], strict=True)
        self.model.eval()
        self.model.freeze_encoder()

    @torch.inference_mode()
    def encode_and_forecast(self, data):
        validate_context(data, self.config.history)
        encoded, forecasts = [], []
        for start in range(0, len(data["x"]), self.config.batch_size):
            indices = np.arange(start, min(len(data["x"]), start + self.config.batch_size))
            batch = tensor_batch(
                data, indices, self.scalers, self.config.history, self.device, include_targets=False
            )
            latent = self.model.encoder.encode(batch["x"], batch["observed"], batch["metadata"])
            prediction = self.model.readout(latent, batch["query"]).sort(-1).values.cpu().numpy()
            encoded.append(latent.cpu().numpy())
            forecasts.append(
                prediction * self.scalers.target_std[None, :, None]
                + self.scalers.target_mean[None, :, None]
            )
        return {
            "embeddings": np.concatenate(encoded),
            "quantile_forecasts_db": np.concatenate(forecasts),
        }

    def encode(self, data):
        return self.encode_and_forecast(data)["embeddings"]

    def forecast(self, data):
        return self.encode_and_forecast(data)["quantile_forecasts_db"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Inference output already exists; preserve earlier artifacts.")
    data = load_context(args.context)
    predictor = NativeAcousticPredictor(args.weights, device=args.device)
    result = predictor.encode_and_forecast(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream,
            **result,
            quantiles=np.asarray(QUANTILES),
            horizons=np.asarray(HORIZONS),
            query_native_bounds_m=data["query"][:, :, 3:5] * 250,
            query_frequency_hz=data["query"][:, :, 0] * 455000,
            query_interval_seconds=data["query"][:, :, 1] * 3600,
        )


if __name__ == "__main__":
    main()
