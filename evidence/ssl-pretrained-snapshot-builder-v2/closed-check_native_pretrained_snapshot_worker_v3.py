"""Isolated copied-source CPU QA on synthetic contexts, never scientific metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_KIND = "native_ssl_weights_only_inference_v1"
BAND_KIND = "native_band_ssl_weights_only_inference_v1"
REPLICATION_KIND = "native_band_replication_ssl_weights_only_inference_v2"
SPECS = (
    ("shared_ssl_seed7_h96_cuda0", "shared_ssl", 7, LEGACY_KIND),
    ("cf_jepa_seed7_h96_deterministic", "cf_jepa", 7, LEGACY_KIND),
    ("cf_jepa_seed13_h96_replication", "cf_jepa", 13, LEGACY_KIND),
    ("cf_jepa_seed23_h96_replication", "cf_jepa", 23, LEGACY_KIND),
    ("band_shared_ssl_seed7_h96_reviewed", "shared_ssl", 7, BAND_KIND),
    ("band_shared_ssl_seed13_h96_replication_v2", "shared_ssl", 13, REPLICATION_KIND),
    ("band_shared_ssl_seed23_h96_replication_v2", "shared_ssl", 23, REPLICATION_KIND),
)


def regular(path):
    path = Path(os.path.abspath(path))
    for parent in (path, *path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Unsafe copied artifact identity")
    if not path.is_file():
        raise ValueError("Regular copied artifact required")
    return path


def digest(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_models(manifest):
    models = manifest.get("models")
    if (
        manifest.get("kind") != "native_pretrained_model_snapshot_v2"
        or manifest.get("status") != "INCOMPLETE_RESEARCH_SNAPSHOT"
        or not isinstance(models, list)
        or len(models) != 7
    ):
        raise ValueError("Exactly seven fixed incomplete-snapshot pretrained models required")
    identities = [
        (m.get("id"), m.get("method"), m.get("seed"), m.get("inference_kind")) for m in models
    ]
    if len(set(identities)) != 7 or set(identities) != set(SPECS):
        raise ValueError("Fixed method/seed/native artifact kinds required")
    for item in models:
        selected = {
            LEGACY_KIND: "native_ssl_selected_encoder_v1",
            BAND_KIND: "native_band_ssl_selected_encoder_v1",
            REPLICATION_KIND: "native_band_replication_ssl_selected_encoder_v2",
        }[item["inference_kind"]]
        if (
            item.get("selected_kind") != selected
            or item.get("selected_encoder") != f"models/{item['id']}/selected_encoder.pt"
            or item.get("latent_inference") != f"models/{item['id']}/inference.pt"
            or item.get("scalers") != f"models/{item['id']}/scalers.json"
        ):
            raise ValueError("Exact typed relative copied weights/scaler paths required")
    return models


def validate_bundle(bundle):
    bundle = Path(os.path.abspath(bundle))
    manifest = json.loads(regular(bundle / "manifest.json").read_bytes())
    validate_models(manifest)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Exact copied payload hashes required")
    for name, expected in files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or digest(bundle / relative) != expected
        ):
            raise ValueError("Copied payload hash/path differs")
    for item in manifest["models"]:
        if any(
            item[key] not in files for key in ("selected_encoder", "latent_inference", "scalers")
        ):
            raise ValueError("Required native model payload missing from manifest")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(files) | {"manifest.json", "MODEL_CARD.md"}:
        raise ValueError("Unexpected or missing copied files")
    return manifest


def route(item):
    routes = {LEGACY_KIND: "legacy", BAND_KIND: "band_v1", REPLICATION_KIND: "band_v2"}
    if item.get("inference_kind") not in routes:
        raise ValueError("Unknown native inference artifact kind")
    return routes[item["inference_kind"]]


def load_pair(bundle, item):
    selected, inference = bundle / item["selected_encoder"], bundle / item["latent_inference"]
    branch = route(item)
    if branch == "legacy":
        from marine_echo.inference.native_encoder import NativeAcousticEncoder
        from marine_echo.inference.native_latent import NativeLatentPredictor

        return NativeAcousticEncoder(selected, device="cpu"), NativeLatentPredictor(
            inference, device="cpu"
        )
    from marine_echo.inference.native_band_replication_encoder import NativeBandReplicationEncoder

    if branch == "band_v1":
        from marine_echo.inference.native_latent import NativeLatentPredictor
    else:
        from marine_echo.inference.native_band_replication_latent import NativeLatentPredictor
    return NativeBandReplicationEncoder(selected, device="cpu"), NativeLatentPredictor(
        inference, device="cpu"
    )


def check_outputs(item, encoder, predictor, np, x, observed, metadata, query):
    if predictor.artifact_kind != item["inference_kind"] or (
        item["inference_kind"] != LEGACY_KIND and encoder.artifact_kind != item["selected_kind"]
    ):
        raise ValueError("Loaded actual native kinds differ from fixed declaration")
    for model in (encoder, predictor):
        config = model.config_metadata
        if (config.get("method"), config.get("seed"), config.get("history")) != (
            item["method"],
            item["seed"],
            96,
        ):
            raise ValueError("Loaded actual method/seed/history differs")
    embedding = encoder.encode(x, observed, metadata)
    if not np.array_equal(embedding, predictor.encode(x, observed, metadata)):
        raise ValueError("Copied selected encoder and inference backbone differ")
    predicted = (
        predictor.predict_cf_zones(x, observed, metadata)
        if item["method"] == "cf_jepa"
        else predictor.predict_latents(x, observed, metadata, query)
    )
    if (
        not np.isfinite(embedding).all()
        or not np.isfinite(predicted).all()
        or len(embedding) != len(x)
    ):
        raise ValueError("Nonfinite or incompatible copied CPU inference")
    if item["method"] == "cf_jepa":
        if (
            predicted.ndim != 4
            or predicted.shape[0] != len(x)
            or predicted.shape[2] != 3
            or predicted.shape[-1] != embedding.shape[-1]
        ):
            raise ValueError("CF online ordinal sequence-zone axes differ")
    else:
        if predicted.shape != (len(x), 3, embedding.shape[-1]):
            raise ValueError("Shared native latent axes differ")
        bad = query.copy()
        bad[:, :, 4] = 200 / 250
        try:
            predictor.predict_latents(x, observed, metadata, bad)
        except ValueError:
            pass
        else:
            raise ValueError("Native230 query was relabelled200")
    return {
        "id": item["id"],
        "inference_kind": predictor.artifact_kind,
        "embedding_shape": list(embedding.shape),
        "latent_shape": list(predicted.shape),
        "embedding_sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
        "selected_and_latent_encoder_cpu_replay": "BITIDENTICAL",
        "latent_semantics": item["latent_semantics"],
        "native230_query_guard": "NOT_APPLICABLE_TO_CF_ORDINAL_ZONES"
        if item["method"] == "cf_jepa"
        else "PASSED",
    }


def check_namespace(bundle, modules):
    for name, module in modules.items():
        if name == "marine_echo" or name.startswith("marine_echo."):
            filename = getattr(module, "__file__", None)
            paths = ([Path(filename)] if filename else []) + [
                Path(p) for p in getattr(module, "__path__", [])
            ]
            if not paths or any(
                not path.resolve().is_relative_to(bundle / "src") for path in paths
            ):
                raise ValueError("Inference file or namespace __path__ escaped copied source")


def run(bundle, output):
    bundle = Path(os.path.abspath(bundle))
    output = Path(os.path.abspath(output))
    if (
        output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
        or output.is_relative_to(bundle)
    ):
        raise FileExistsError("Fresh external QA completion required")
    manifest = validate_bundle(bundle)
    if any(name == "marine_echo" or name.startswith("marine_echo.") for name in sys.modules):
        raise ValueError("Scientific package was imported before isolated copied-source admission")
    sys.path.insert(0, str(bundle / "src"))
    import numpy as np
    import torch

    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only QA must not initialize CUDA")
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
        encoder, predictor = load_pair(bundle, item)
        if (
            encoder.scaler_metadata != json.loads((bundle / item["scalers"]).read_bytes())
            or predictor.scaler_metadata != encoder.scaler_metadata
        ):
            raise ValueError(
                "Actual copied tensor/scaler metadata differs from original TRAIN JSON"
            )
        records.append(check_outputs(item, encoder, predictor, np, x, observed, metadata, query))
        del encoder, predictor
    check_namespace(bundle, sys.modules)
    if torch.cuda.is_initialized() or not torch.equal(state, torch.get_rng_state()):
        raise ValueError("Load-only CPU correctness changed RNG or initialized CUDA")
    result = {
        "status": "COPIED_SEVEN_MODEL_PACKAGE_ISOLATED_CPU_CORRECTNESS_PASSED",
        "evidence_kind": "SYNTHETIC_INPUT_CORRECTNESS_ONLY_REAL_PRETRAINED_WEIGHTS",
        "records": records,
        "manifest_sha256": digest(bundle / "manifest.json"),
        "all_imports_from_copied_source": True,
        "fitting": False,
        "public_numerical_corpus_decoded": False,
        "scientific_performance_assessment": False,
        "cuda_initialized": False,
        "rng_unchanged": True,
        "independent_package_review": "NOT_RUN",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    if ROOT.name != "marine-echo-jepa":
        raise RuntimeError("Actual copied pretrained weights QA is ROOT-only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.bundle, args.output)
    print(json.dumps({"status": result["status"], "models": len(result["records"])}))


if __name__ == "__main__":
    main()
