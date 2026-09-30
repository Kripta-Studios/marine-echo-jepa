"""Admitted immutable SSL weights on 64 DEV contexts; no fitting or final data."""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = "01a0ef27-876b-7692-917e-3975afc6893d"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_path, output = args.manifest.resolve(), args.output.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if (review.get("status") != "APPROVED_LATENT_CPU_REPLAY"
            or review.get("reviewer_session_id") != REVIEWER
            or review.get("manifest_sha256") != digest(manifest_path)
            or review.get("approved_output") != str(output)
            or manifest.get("kind") != "native_latent_cpu_replay_manifest_v1"
            or manifest.get("role") != "development"
            or manifest.get("context_rows") != 64
            or manifest.get("fitting") is not False
            or manifest.get("final_numeric_access") is not False):
        raise ValueError("Exact independent context-only CPU admission required")
    if not output.is_relative_to(ROOT / "evidence") or output.exists():
        raise ValueError("Fresh root-owned evidence directory required")
    required = {str(Path(__file__).resolve()), str(manifest_path)}
    data_path = (ROOT / "data/processed/native_ssl_v1/development.npz").resolve()
    data_report_path = data_path.with_suffix(".json")
    required.update([str(data_path), str(data_report_path)])
    for run in manifest["runs"]:
        folder = Path(run["directory"]).resolve()
        if not folder.is_relative_to(ROOT / "outputs/native_acoustic_ssl_v1"):
            raise ValueError("Only original local research weights supported")
        required.update(str(folder / name) for name in ("inference.pt", "selected_encoder.pt", "run.json"))
    if not required.issubset(review.get("bindings", {})):
        raise ValueError("Incomplete exact execution/data/checkpoint admission")
    for name, expected in review["bindings"].items():
        if digest(name) != expected:
            raise ValueError(f"Reviewed bytes differ: {name}")
    data_report = json.loads(data_report_path.read_text(encoding="utf-8"))
    if data_report.get("role") != "development" or data_report.get("npz_sha256") != digest(data_path):
        raise ValueError("Real original DEV corpus identity required")
    # API imports only after the complete source/data/weight gate passes.
    from marine_echo.inference.native_encoder import NativeAcousticEncoder
    from marine_echo.inference.native_latent import NativeLatentPredictor

    torch.set_num_threads(4)
    with np.load(data_path, allow_pickle=False) as data:
        arrays = tuple(np.array(data[name][:64], copy=True) for name in ("x", "observed", "metadata", "query"))
    output.mkdir()
    records = []
    for run in manifest["runs"]:
        folder = Path(run["directory"])
        report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if (report.get("status") != "COMPLETED"
                or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
                or report["config"]["method"] != run["method"]
                or report["config"]["seed"] != run["seed"]
                or report.get("inference_sha256") != digest(folder / "inference.pt")):
            raise ValueError("Original completed fit identity changed")
        copy_path = output / (folder.name + ".pt")
        shutil.copyfile(folder / "inference.pt", copy_path)
        if digest(copy_path) != digest(folder / "inference.pt"):
            raise ValueError("Relocated payload differs")
        rng = torch.get_rng_state().clone()
        numpy_rng, python_rng = np.random.get_state(), random.getstate()
        adapter = NativeLatentPredictor(copy_path)
        embedding = adapter.encode(*arrays[:3])
        assert torch.equal(torch.get_rng_state(), rng)
        selected = NativeAcousticEncoder(folder / "selected_encoder.pt")
        np.testing.assert_array_equal(embedding, selected.encode(*arrays[:3]))
        artifact = torch.load(copy_path, weights_only=True, map_location="cpu")
        for name, tensor in adapter._model.state_dict().items():
            assert torch.equal(tensor, artifact["model"][name])
        before = {name: value.clone() for name, value in adapter._model.state_dict().items()}
        mean = np.asarray(artifact["scalers"]["channel_mean"], dtype=np.float32)
        std = np.asarray(artifact["scalers"]["channel_std"], dtype=np.float32)
        x, mask, metadata, query = arrays
        normalized = np.where(mask, (np.where(mask, x, 0) - mean) / std, 0)
        tensors = tuple(torch.from_numpy(value) for value in (normalized, mask, metadata, query))
        with torch.inference_mode():
            if run["method"] == "cf_jepa":
                latent = adapter.predict_cf_zones(*arrays[:3])
                sequence = adapter._model.online.sequence(*tensors[:3])
                expected = torch.stack([
                    F.linear(sequence, artifact["model"][f"predictors.{i}.weight"],
                             artifact["model"][f"predictors.{i}.bias"]) for i in range(3)
                ], dim=2).numpy()
                # Saved EMA and ONLINE are distinct scientific branches.
                branch_distance = float((sequence - adapter._model.encoder.sequence(*tensors[:3])).abs().max())
                assert branch_distance > 0
            else:
                latent = adapter.predict_latents(*arrays)
                expected = adapter._model.predictor(
                    adapter._model.encoder.encode(*tensors[:3]), tensors[3]
                ).numpy()
                branch_distance = None
        np.testing.assert_array_equal(latent, expected)
        for name, tensor in before.items():
            assert torch.equal(tensor, adapter._model.state_dict()[name])
        assert torch.equal(torch.get_rng_state(), rng)
        after = np.random.get_state()
        assert after[0] == numpy_rng[0] and after[2:] == numpy_rng[2:]
        np.testing.assert_array_equal(after[1], numpy_rng[1])
        assert random.getstate() == python_rng
        assert np.isfinite(embedding).all() and np.isfinite(latent).all()
        records.append({
            "run": folder.name, "method": run["method"], "seed": run["seed"],
            "artifact_sha256": digest(copy_path), "embedding_shape": list(embedding.shape),
            "latent_shape": list(latent.shape), "branch_distance": branch_distance,
            "selected_encoder_cpu_replay": "BITIDENTICAL",
            "saved_forward_head_cpu_replay": "BITIDENTICAL",
            "semantics": adapter.objective_metadata,
        })
    receipt = {
        "status": "PASSED_ACTUAL_IMMUTABLE_SSL_CPU_REPLAY", "records": records,
        "device": "cpu", "context_rows": 64, "optimizer_updates": 0,
        "input_keys": ["x", "observed", "metadata", "query"],
        "future_or_target_acoustic_inputs": [], "role": "development",
        "native_bounds_m": np.unique(arrays[2][:, :, 3:5].reshape(-1, 2) * 250, axis=0).tolist(),
        "test_access": "NOT_RUN", "representation_value": "NOT_ESTABLISHED",
        "review_sha256": digest(args.review), "manifest_sha256": digest(manifest_path),
    }
    with (output / "completion.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
