"""Actual relocated Torch ZIP replay, strictly SYNTHETIC_CORRECTNESS_ONLY."""

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    spec = importlib.util.spec_from_file_location(
        "_durable_latent_support", Path(__file__).with_name("test_support.py")
    )
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    api = support.load_api()
    torch.set_num_threads(1)
    output = Path(__file__).parent / "Unicode-native230-latent-Álvaro-root-01"
    output.mkdir(exist_ok=False)
    records = []
    for method, band in [("shared_ssl", False), ("cf_jepa", False), ("shared_ssl", True)]:
        artifact, _ = support.fixture(method, band)
        arrays = support.inputs()
        memory = api.NativeLatentPredictor(support.codec(artifact))
        expected = memory.encode(*arrays[:3])
        args = arrays[:3] if method == "cf_jepa" else arrays
        prediction_method = "predict_cf_zones" if method == "cf_jepa" else "predict_latents"
        expected_prediction = getattr(memory, prediction_method)(*args)
        path = output / (f"{'band_' if band else ''}{method}.pt")
        with path.open("xb") as stream:
            torch.save(artifact, stream)
        torch_rng = torch.get_rng_state().clone()
        numpy_rng = np.random.get_state()
        python_rng = random.getstate()
        default_dtype = torch.get_default_dtype()
        relocated = api.NativeLatentPredictor(path)
        np.testing.assert_array_equal(relocated.encode(*arrays[:3]), expected)
        np.testing.assert_array_equal(
            getattr(relocated, prediction_method)(*args), expected_prediction
        )
        assert torch.equal(torch.get_rng_state(), torch_rng)
        after = np.random.get_state()
        assert after[0] == numpy_rng[0] and after[2:] == numpy_rng[2:]
        np.testing.assert_array_equal(after[1], numpy_rng[1])
        assert random.getstate() == python_rng
        assert torch.get_default_dtype() == default_dtype
        assert relocated.source_bindings == artifact["bindings"]
        assert arrays[3][0, 0, 4] == np.float32(230 / 250)
        for key, tensor in artifact["model"].items():
            assert torch.equal(relocated._model.state_dict()[key], tensor)
        records.append({
            "method": method, "band": band,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "encoding_shape": list(expected.shape),
            "prediction_shape": list(expected_prediction.shape),
            "replay": "CPU_BITIDENTICAL",
        })
    receipt = {
        "status": "PASS", "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "native_lower_m": 230, "device": "cpu", "fitting": False,
        "public_data_access": False, "records": records,
    }
    with (output / "completion.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
