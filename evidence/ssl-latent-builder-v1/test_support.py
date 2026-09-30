"""SYNTHETIC_CORRECTNESS_ONLY real Torch codecs; no optimizer or public data."""

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import torch

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))

from marine_echo.models.native_band_temporal import ARCHITECTURE, NativeBandTemporalModel
from marine_echo.models.native_temporal import CFNativeModel, NativeTemporalModel
from marine_echo.training.native_band_ssl import Config as BandConfig
from marine_echo.training.native_ssl import Config


def load_api():
    spec = importlib.util.spec_from_file_location(
        "_native_latent_builder", BUILDER / "src/marine_echo/inference/native_latent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(method="shared_ssl", band=False):
    config_type = BandConfig if band else Config
    config = config_type(
        method=method,
        width=8,
        latent=4,
        blocks=1,
        heads=2,
        batch_size=2,
        cf_width=8,
        cf_latent=6,
        cf_blocks=1,
    )
    # Fixture construction only; deliberately no training initializer/cache.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1929)
        if method == "cf_jepa":
            model = CFNativeModel(8, 6, 1, 2)
            with torch.no_grad():
                model.online.input_fc.bias.add_(0.8)
                model.online.blocks[0]["bn1"].running_mean.fill_(0.3)
                for i, predictor in enumerate(model.predictors):
                    predictor.bias.fill_(0.15 * (i + 1))
        else:
            model = (NativeBandTemporalModel if band else NativeTemporalModel)(8, 4, 1, 2)
    model.eval().requires_grad_(False)
    artifact = {
        "kind": "native_band_ssl_weights_only_inference_v1"
        if band
        else "native_ssl_weights_only_inference_v1",
        "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "config": config.to_dict(),
        "scalers": {
            "channel_mean": [-65.0, -60.0, -55.0, -50.0],
            "channel_std": [2.0, 3.0, 4.0, 5.0],
            "target_mean": [-62.0, -63.0, -64.0],
            "target_std": [4.0, 5.0, 6.0],
        },
        "selected_pretrain_step": 2 if method != "random_frozen" else 0,
        "bindings": {"Z:/INACCESSIBLE_ANCESTRY/source.py": "a" * 64},
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "correctness_smoke": True,
    }
    if band:
        artifact["architecture"] = ARCHITECTURE
    return artifact, model


def codec(artifact):
    stream = io.BytesIO()
    torch.save(artifact, stream)
    stream.seek(0)
    return stream


def inputs(n=3, lower=230.0, upper=0.0):
    x = np.linspace(-75, -48, n * 96 * 4, dtype=np.float32).reshape(n, 96, 4)
    observed = np.ones_like(x, dtype=bool)
    observed[:, 2::3, 1:] = False
    metadata = np.zeros((n, 4, 10), dtype=np.float32)
    metadata[..., 0] = np.array([38000, 125000, 200000, 455000]) / 455000
    metadata[..., 1:3] = 1
    metadata[..., 3] = upper / 250
    metadata[..., 4] = lower / 250
    query = np.repeat(metadata[:, :1], 3, axis=1)
    query[..., 9] = [1, 3, 6]
    return x, observed, metadata, query


def tensors(artifact, arrays):
    x, observed, metadata, query = arrays
    mean = np.asarray(artifact["scalers"]["channel_mean"], dtype=np.float32)
    std = np.asarray(artifact["scalers"]["channel_std"], dtype=np.float32)
    clean = np.where(observed, (np.where(observed, x, 0) - mean) / std, 0)
    return tuple(torch.from_numpy(a) for a in (clean, observed, metadata, query))
