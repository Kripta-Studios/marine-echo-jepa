"""Actual Unicode disk relocation of synthetic seed13/23 encoder and latent weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from replication_test_support import artifact, synthetic_inputs

from marine_echo.inference.native_band_replication_acoustic import (
    NativeBandAcousticEncoder,
    NativeBandAcousticPredictor,
)
from marine_echo.inference.native_band_replication_latent import NativeLatentPredictor

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "Unicode-replication-Álvaro-root-01")
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(HERE):
        raise ValueError("Root-owned synthetic evidence directory required")
    output.mkdir(exist_ok=False)
    x, observed, metadata, query = synthetic_inputs(n=5)
    results = []
    for seed in (13, 23):
        original = output / f"seed{seed}-original"
        relocated = output / f"seed{seed}-relocated"
        original.mkdir()
        relocated.mkdir()
        full, _ = artifact(seed=seed)
        selected, _ = artifact(seed=seed, selected=True)
        for name, payload in (("inference.pt", full), ("selected_encoder.pt", selected)):
            torch.save(payload, original / name)
            with (relocated / name).open("xb") as stream:
                stream.write((original / name).read_bytes())
            assert digest(original / name) == digest(relocated / name)
        rng = torch.get_rng_state().clone()
        forecaster = NativeBandAcousticPredictor(relocated / "inference.pt")
        encoder = NativeBandAcousticEncoder(relocated / "selected_encoder.pt")
        latent = NativeLatentPredictor(relocated / "inference.pt")
        assert torch.equal(rng, torch.get_rng_state())
        states = {
            "forecast": {k: v.clone() for k, v in forecaster.model.state_dict().items()},
            "encoder": {k: v.clone() for k, v in encoder.encoder.state_dict().items()},
        }
        expected = NativeBandAcousticPredictor(original / "inference.pt")
        np.testing.assert_array_equal(
            forecaster.forecast(x, observed, metadata, query),
            expected.forecast(x, observed, metadata, query),
        )
        np.testing.assert_array_equal(
            encoder.encode(x, observed, metadata), forecaster.encode(x, observed, metadata)
        )
        np.testing.assert_array_equal(
            latent.encode(x, observed, metadata), forecaster.encode(x, observed, metadata)
        )
        predicted = latent.predict_latents(x, observed, metadata, query)
        expected_latent = NativeLatentPredictor(original / "inference.pt")
        np.testing.assert_array_equal(
            predicted, expected_latent.predict_latents(x, observed, metadata, query)
        )
        assert predicted.shape == (5, 3, 8) and np.isfinite(predicted).all()
        assert np.all(query[..., 4] == np.float32(230 / 250))
        assert torch.equal(rng, torch.get_rng_state())
        assert all(
            torch.equal(v, forecaster.model.state_dict()[k]) for k, v in states["forecast"].items()
        )
        assert all(
            torch.equal(v, encoder.encoder.state_dict()[k]) for k, v in states["encoder"].items()
        )
        results.append(
            {
                "seed": seed,
                "encoder_shape": [5, 8],
                "latent_shape": list(predicted.shape),
                "forecast_shape": [5, 3, 5],
                "native_primary_bounds_m": [0, 230],
                "cpu_relocated_replay": "BITIDENTICAL",
                "rng_and_frozen_state": "UNCHANGED",
                "artifact_sha256": {
                    name: digest(relocated / name)
                    for name in ("inference.pt", "selected_encoder.pt")
                },
            }
        )
    report = {
        "status": "COMPLETED",
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "seeds": results,
        "optimizer_updates": 0,
        "public_weights_or_arrays_decoded": False,
        "gpu_execution": False,
        "fitting": False,
        "durable_replay": "PASSED",
    }
    with (output / "completion.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
