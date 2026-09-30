"""Reserve all 47 completed endpoints without selecting seeds or decoding values."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    target = ROOT / "orchestration/native_development_comparison_v4.json"
    admission = ROOT / "orchestration/native_development_comparison_admission_v4.json"
    if target.exists() or admission.exists():
        raise FileExistsError("Preserve previous comparison reservations")
    original = ROOT / "orchestration/native_development_comparison_v3.json"
    previous = ROOT / "evidence/ssl-research-v1/development-comparison-v3-review-final.json"
    manifest = json.loads(original.read_bytes())
    review = json.loads(previous.read_bytes())
    bindings = dict(review["bindings"])
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError(f"Reviewed original comparison bytes changed: {path}")
    endpoints = []
    for method in ("shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen"):
        endpoints.append((f"band_{method}_short_probe", f"band_{method}_seed7_h96_reviewed", method, 7, None))
    endpoints.append(("band_direct_end_to_end", "band_direct_end_to_end_seed7_h96_reviewed", "direct", 7, "direct_end_to_end"))
    for method, modes in (("shared_ssl", ("frozen_readout", "full_finetune")),
                          ("masked_ssl", ("frozen_readout", "full_finetune")),
                          ("permuted_ssl", ("frozen_readout",)),
                          ("random_frozen", ("frozen_readout",))):
        for mode in modes:
            name = f"band_{method}_{mode}"
            directory = f"{name}_seed7_h96" + ("_native_retry01" if method == "random_frozen" else "")
            endpoints.append((name, directory, method, 7, mode))
    for seed in (13, 23):
        endpoints.extend([
            (f"band_shared_ssl_short_probe_seed{seed}", f"band_shared_ssl_seed{seed}_h96_replication_v2", "shared_ssl", seed, None),
            (f"band_direct_end_to_end_seed{seed}", f"band_direct_end_to_end_seed{seed}_h96_replication_v2", "direct", seed, "direct_end_to_end"),
        ])
        for mode in ("frozen_readout", "full_finetune"):
            name = f"band_shared_ssl_{mode}_seed{seed}"
            endpoints.append((name, f"band_shared_ssl_{mode}_seed{seed}_h96_replication_v3", "shared_ssl", seed, mode))
    if len(endpoints) != 19 or len(manifest["methods"]) != 28:
        raise ValueError("Exactly 28 original and 19 fixed Band endpoints required")
    for name, directory, method, seed, mode in endpoints:
        folder = ROOT / "outputs/native_acoustic_ssl_v1" / directory
        report = json.loads((folder / "run.json").read_bytes())
        if (report.get("status") != "COMPLETED"
                or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
                or report.get("architecture") != "nonlinear_frequency_conditioned_v1"
                or report["config"]["method"] != method or report["config"]["seed"] != seed
                or (mode is not None and report.get("mode") != mode)
                or report.get("inference_sha256") != digest(folder / "inference.pt")):
            raise ValueError(f"Actual completed exact endpoint required: {name}")
        prediction = folder / "predictions.npz"
        if name in manifest["methods"]:
            raise ValueError("Duplicate comparison identity")
        manifest["methods"][name] = str(prediction)
        for path in (prediction, *[folder / leaf for leaf in
                                  ("run.json", "membership.json", "inference.pt", "selected_encoder.pt", "scalers.json")]):
            bindings[str(path)] = digest(path)
    manifest["interpretation"] = (
        "All 47 fixed completed endpoints, including seeds7/13/23 and controls; no best-seed selection. "
        "One development deployment; CF versus shared architecture/head differences remain confounded. "
        "Runner development scores are not independent evidence. Final transfer and SOTA NOT_ESTABLISHED."
    )
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for path in (target, original, previous, Path(__file__).resolve()):
        bindings[str(path)] = digest(path)
    proposal = {
        "status": "PROPOSED_COMPARISON_RECONSTRUCTION_NOT_APPROVAL",
        "manifest_sha256": digest(target), "bindings": bindings,
        "allowed_roles": ["development"], "evidence_kind": "REVIEWED_SAVED_PREDICTIONS",
        "manifest_path": str(target), "methods": 47,
        "output": str(ROOT / "evidence/ssl-research-v1/development-comparison-v4.json"),
        "fitting": False, "final_numeric_access": False,
        "retained_failed_attempts": "Original CF13 and Band random native failures preserved and charged; exact reviewed retries only",
    }
    with admission.open("x", encoding="utf-8") as stream:
        json.dump(proposal, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": proposal["status"], "methods": 47, "bindings": len(bindings), "numeric_decoding": False}))


if __name__ == "__main__":
    main()
