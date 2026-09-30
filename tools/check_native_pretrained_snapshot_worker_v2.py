"""Isolated CPU package correctness on synthetic inputs; no scientific metrics."""

import hashlib
import json
import sys
from pathlib import Path


def main():
    bundle = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    sys.path.insert(0, str(bundle / "src"))
    import numpy as np
    import torch

    from marine_echo.inference.native_encoder import NativeAcousticEncoder
    from marine_echo.inference.native_latent import NativeLatentPredictor

    torch.set_num_threads(2)
    state = torch.get_rng_state().clone()
    x = np.linspace(-88, -62, 2 * 96 * 4, dtype=np.float32).reshape(2, 96, 4)
    observed = np.ones(x.shape, dtype=bool)
    metadata = np.zeros((2, 4, 10), dtype=np.float32)
    metadata[:, :, 0] = np.array([38000, 125000, 200000, 455000]) / 455000
    metadata[:, :, 1:3] = 1
    metadata[:, :, 4] = 230 / 250
    query = np.repeat(metadata[:, :1], 3, axis=1)
    query[:, :, 9] = [1, 3, 6]
    records = []
    for item in manifest["models"]:
        encoder = NativeAcousticEncoder(bundle / item["selected_encoder"], device="cpu")
        predictor = NativeLatentPredictor(bundle / item["latent_inference"], device="cpu")
        embedding = encoder.encode(x, observed, metadata)
        if not np.array_equal(embedding, predictor.encode(x, observed, metadata)):
            raise ValueError("Copied selected encoder and inference backbone differ")
        predicted = predictor.predict_cf_zones(x, observed, metadata) if item["method"] == "cf_jepa" else predictor.predict_latents(x, observed, metadata, query)
        if not np.isfinite(embedding).all() or not np.isfinite(predicted).all():
            raise ValueError("Nonfinite copied CPU inference")
        if item["method"] != "cf_jepa":
            bad = query.copy()
            bad[:, :, 4] = 200 / 250
            try:
                predictor.predict_latents(x, observed, metadata, bad)
            except ValueError:
                pass
            else:
                raise ValueError("Native230 query was relabelled200")
        records.append({"id": item["id"], "embedding_shape": list(embedding.shape), "latent_shape": list(predicted.shape),
                        "embedding_sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
                        "selected_and_latent_encoder_cpu_replay": "BITIDENTICAL"})
        del encoder, predictor
    for name, module in sys.modules.items():
        if name == "marine_echo" or name.startswith("marine_echo."):
            filename = getattr(module, "__file__", None)
            paths = [Path(filename)] if filename else [Path(p) for p in getattr(module, "__path__", [])]
            if not paths or any(not path.resolve().is_relative_to(bundle / "src") for path in paths):
                raise ValueError("Inference imported outside copied source closure")
    if torch.cuda.is_initialized() or not torch.equal(state, torch.get_rng_state()):
        raise ValueError("Load-only CPU correctness changed RNG or initialized CUDA")
    result = {"status": "COPIED_MODEL_ONLY_PACKAGE_ISOLATED_CPU_CORRECTNESS_PASSED",
              "evidence_kind": "SYNTHETIC_INPUT_CORRECTNESS_ONLY_REAL_PRETRAINED_WEIGHTS",
              "records": records, "all_imports_from_copied_source": True, "native230_shared_query_guard": "PASSED",
              "fitting": False, "public_numerical_corpus_decoded": False, "scientific_performance_assessment": False,
              "cuda_initialized": False, "rng_unchanged": True, "independent_package_review": "NOT_RUN"}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "models": len(records)}))


if __name__ == "__main__":
    main()
