"""Retain all three replication seeds in a fixed 28-endpoint DEV comparison."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    manifest_path = ROOT / "orchestration/native_development_comparison_v3.json"
    proposal_path = ROOT / "orchestration/native_development_comparison_admission_v3.json"
    if manifest_path.exists() or proposal_path.exists():
        raise FileExistsError("Preserve earlier comparison reservations")
    original = ROOT / "orchestration/native_development_comparison_v2.json"
    manifest = json.loads(original.read_text(encoding="utf-8"))
    previous = ROOT / "evidence/ssl-research-v1/development-comparison-v2-review-final.json"
    review = json.loads(previous.read_text(encoding="utf-8"))
    bindings = dict(review["bindings"])
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError(f"Original closed comparison binding changed: {path}")
    base = ROOT / "outputs/native_acoustic_ssl_v1"
    for seed in (13, 23):
        extra = {
            f"cf_short_probe_seed{seed}": (f"cf_jepa_seed{seed}_h96_replication", "cf_jepa", None),
            f"direct_end_to_end_seed{seed}": (f"direct_direct_end_to_end_seed{seed}_h96_replication", "direct", "direct_end_to_end"),
            f"cf_jepa_frozen_readout_seed{seed}": (f"cf_jepa_frozen_readout_seed{seed}_h96" + ("_runtime_retry01" if seed == 13 else ""), "cf_jepa", "frozen_readout"),
            f"cf_jepa_full_finetune_seed{seed}": (f"cf_jepa_full_finetune_seed{seed}_h96", "cf_jepa", "full_finetune"),
        }
        for name, (directory, method, mode) in extra.items():
            folder = base / directory
            report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            if (report.get("status") != "COMPLETED"
                    or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
                    or report["config"]["method"] != method
                    or report["config"]["seed"] != seed
                    or (mode is not None and report.get("mode") != mode)
                    or report.get("inference_sha256") != digest(folder / "inference.pt")):
                raise ValueError("Only actual closed exact replications are eligible")
            prediction = folder / "predictions.npz"
            manifest["methods"][name] = str(prediction)
            for path in [prediction, *[folder / p for p in (
                "run.json", "membership.json", "inference.pt", "selected_encoder.pt")]]:
                bindings[str(path)] = digest(path)
    if len(manifest["methods"]) != 28:
        raise ValueError("Exactly20 original and8 replicated endpoints required")
    manifest["interpretation"] = (
        "DEV reconstruction with every seed7/13/23 replication retained; no best-seed selection; "
        "CF128 versus shared64 and different heads are cross-architecture confounded; "
        "single development deployment; final-test and JEPA-value claims NOT_ESTABLISHED"
    )
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    proof_paths = [manifest_path, Path(__file__).resolve(), previous,
                   ROOT / "evidence/ssl-research-v1/cf-strong-three-seed-completion-v1.json",
                   ROOT / "evidence/ssl-research-v1/cf-replication-completion-v1.json",
                   ROOT / "evidence/ssl-research-v1/direct-replication-completion-v1.json"]
    for path in proof_paths:
        bindings[str(path)] = digest(path)
    proposal = {
        "status": "PROPOSED_COMPARISON_RECONSTRUCTION_NOT_APPROVAL",
        "manifest_sha256": digest(manifest_path), "bindings": bindings,
        "allowed_roles": ["development"], "evidence_kind": "REVIEWED_SAVED_PREDICTIONS",
        "manifest_path": str(manifest_path), "methods": 28,
        "output": str(ROOT / "evidence/ssl-research-v1/development-comparison-v3.json"),
        "fitting": False, "final_numeric_access": False,
        "retained_failed_attempt": "CF13 frozen original native0xc000070a,164.14s charged; exact-recipe one fresh-output retry only",
    }
    with proposal_path.open("x", encoding="utf-8") as stream:
        json.dump(proposal, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": proposal["status"], "methods": 28, "bindings": len(bindings),
                      "numeric_decoding": False}))


if __name__ == "__main__":
    main()
