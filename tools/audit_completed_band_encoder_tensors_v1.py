"""Inspect completed local encoder tensors; no corpus, forecasts, fitting or GPU."""

import hashlib
import json
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    base = ROOT / "outputs/native_acoustic_ssl_v1"
    rng = torch.get_rng_state().clone()
    findings, bindings = [], {}
    for method, modes in (("shared_ssl", ("frozen_readout", "full_finetune")),
                          ("masked_ssl", ("frozen_readout", "full_finetune")),
                          ("permuted_ssl", ("frozen_readout",))):
        parent = base / f"band_{method}_seed7_h96_reviewed"
        parent_report = json.loads((parent / "run.json").read_bytes())
        parent_selected = parent / "selected_encoder.pt"
        original = torch.load(parent_selected, weights_only=True, map_location="cpu")
        if original.get("kind") != "native_band_ssl_selected_encoder_v1" or parent_report.get("status") != "COMPLETED":
            raise ValueError("Actual completed SSL selected parent required")
        for mode in modes:
            folder = base / f"band_{method}_{mode}_seed7_h96"
            report = json.loads((folder / "run.json").read_bytes())
            selected_path, inference_path = folder / "selected_encoder.pt", folder / "inference.pt"
            if (report.get("status") != "COMPLETED" or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
                    or report.get("mode") != mode or report["config"]["method"] != method
                    or report.get("inference_sha256") != digest(inference_path)
                    or report.get("selected_encoder_sha256") != digest(selected_path)
                    or report["supervised_ancestry"]["ancestor_encoder_sha256"] != digest(parent_selected)
                    or report["supervised_ancestry"]["ancestor_run_sha256"] != digest(parent / "run.json")):
                raise ValueError("Exact completed child and parent lineage required")
            selected = torch.load(selected_path, weights_only=True, map_location="cpu")
            inference = torch.load(inference_path, weights_only=True, map_location="cpu")
            if set(selected["encoder"]) != set(original["encoder"]):
                raise ValueError("Encoder tensor identities differ")
            changed, replay_mismatches = [], []
            for key, initial in original["encoder"].items():
                tensor = selected["encoder"][key]
                if not isinstance(tensor, torch.Tensor) or tensor.dtype != initial.dtype or tensor.shape != initial.shape:
                    raise ValueError("Encoder tensor shape/dtype changed")
                if not torch.isfinite(tensor).all():
                    raise ValueError("Nonfinite completed encoder")
                if not torch.equal(initial, tensor):
                    changed.append(key)
                if not torch.equal(tensor, inference["model"]["encoder." + key]):
                    replay_mismatches.append(key)
            if (replay_mismatches or selected["scalers"] != original["scalers"]
                    or inference["scalers"] != original["scalers"]
                    or (mode == "frozen_readout" and (changed or digest(selected_path) != digest(parent_selected)))
                    or (mode == "full_finetune" and (not changed or selected.get("kind") != "native_band_downstream_supervised_encoder_v1"))):
                raise ValueError("Completed tensors contradict frozen/fine-tuned contract")
            findings.append({"id": folder.name, "encoder_tensor_count": len(original["encoder"]),
                             "changed_encoder_tensors": len(changed), "changed_tensor_names": changed,
                             "selected_matches_inference_encoder": True, "parent_train_scalers_unchanged": True,
                             "frozen_parent_artifact_byte_identical": digest(selected_path) == digest(parent_selected)})
            for path in (parent / "run.json", parent_selected, folder / "run.json", selected_path, inference_path):
                bindings[str(path)] = digest(path)
            del selected, inference
        del original
    if not torch.equal(rng, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise RuntimeError("Artifact audit changed RNG or initialized CUDA")
    receipt = {"status": "FIVE_COMPLETED_BAND_ENCODER_TENSOR_CONTRACTS_VERIFIED",
               "device": "cpu", "weights_only": True, "fitting": False, "corpus_decoded": False,
               "forecast_or_performance_assessment": False, "rng_unchanged": True,
               "cuda_initialized": False, "process_rss_bytes_at_close": psutil.Process().memory_info().rss,
               "findings": findings, "bindings": bindings, "independent_review": "NOT_RUN"}
    if receipt["process_rss_bytes_at_close"] >= 22 * 1024**3:
        raise MemoryError("CPU artifact audit reaches 22 GiB")
    destination = ROOT / "evidence/ssl-research-v1/band-five-encoder-tensor-audit-v1.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "endpoints": len(findings),
                      "changed_counts": [item["changed_encoder_tensors"] for item in findings]}))


if __name__ == "__main__":
    main()
