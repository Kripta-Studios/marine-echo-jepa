"""Replay a completed native run on context-only DEV prefixes, without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from marine_echo.inference.native_acoustic import NativeAcousticPredictor, load_context
from marine_echo.inference.native_encoder import NativeAcousticEncoder

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve every earlier replay receipt.")
    report = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "COMPLETED"
        or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
    ):
        raise ValueError("Require a completed actual TRAIN/DEV fit.")
    hashes = {name: digest(args.run / (name + ".pt")) for name in ("inference", "selected_encoder")}
    for name in ("inference", "selected_encoder"):
        expected_hash = report.get(name + "_sha256")
        if name == "inference" and expected_hash is None:
            raise ValueError("Completed inference identity is absent.")
        if expected_hash is not None and hashes[name] != expected_hash:
            raise ValueError("Completed artifact bytes differ from the actual fit report.")
    torch.set_num_threads(4)
    data = {
        key: array[:64]
        for key, array in load_context(
            ROOT / "data/processed/native_ssl_v1/development.npz"
        ).items()
    }
    predictor = NativeAcousticPredictor(args.run / "inference.pt", device="cpu")
    replay = predictor.encode_and_forecast(data)
    with np.load(args.run / "predictions.npz", allow_pickle=False) as saved:
        expected = saved["predictions"][:64]
    np.testing.assert_allclose(replay["quantile_forecasts_db"], expected, rtol=1e-5, atol=2e-5)
    latent = replay["embeddings"]
    assert np.isfinite(latent).all() and latent.shape[0] == 64
    selected = torch.load(args.run / "selected_encoder.pt", map_location="cpu", weights_only=True)
    encoder_replay = {"status": "UNSUPPORTED_SUPERVISED_DOWNSTREAM_ARTIFACT_EXPLICIT"}
    if selected.get("kind") == "native_ssl_selected_encoder_v1":
        encoder = NativeAcousticEncoder(args.run / "selected_encoder.pt")
        features = encoder.encode(data["x"], data["observed"], data["metadata"])
        np.testing.assert_array_equal(features, latent)
        encoder_replay = {
            "status": "BIT_IDENTICAL",
            "input_keys": ["x", "observed", "metadata"],
            "training_kind": encoder.training_kind,
            "maximum_absolute_error": 0.0,
        }
    result = {
        "status": "PASSED_ACTUAL_CONTEXT_ONLY_CPU_REPLAY",
        "run": str(args.run.resolve()),
        "run_sha256": digest(args.run / "run.json"),
        "artifact_sha256": hashes,
        "selected_encoder_hash_in_run_report": report.get("selected_encoder_sha256"),
        "method": report["config"]["method"],
        "mode": report.get("mode", "screen"),
        "development_prefixes": 64,
        "forecast_input_keys": sorted(data),
        "target_or_future_acoustic_inputs": [],
        "embedding_shape": list(latent.shape),
        "encoder_only_replay": encoder_replay,
        "maximum_absolute_forecast_replay_error_db": float(
            np.abs(replay["quantile_forecasts_db"] - expected).max()
        ),
        "native_bounds_m": np.unique(
            data["metadata"][:, :, 3:5].reshape(-1, 2) * 250, axis=0
        ).tolist(),
        "optimizer_updates": 0,
        "test_access": "NOT_RUN",
        "representation_transfer_advantage": "NOT_ESTABLISHED",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
