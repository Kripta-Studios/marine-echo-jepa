"""Reserve all 47 completed endpoints without selecting seeds or decoding values."""

import hashlib
import json
from pathlib import Path

from native_train_scaler_reservation_v1 import reserve_train_scalers
from prepare_completed_native_inventory_v6 import (
    REFERENCES,
    declared_kind,
    dependencies,
    document,
    fresh,
    idle_ledger,
    regular,
)

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root):
    root = Path(root).absolute()
    before, _ = idle_ledger(root)
    target = fresh(
        root / "orchestration/native_development_comparison_v5.json", root, "orchestration"
    )
    admission = fresh(
        root / "orchestration/native_development_comparison_admission_v5.json",
        root,
        "orchestration",
    )
    if target.exists() or admission.exists():
        raise FileExistsError("Preserve previous comparison reservations")
    original = root / "orchestration/native_development_comparison_v3.json"
    previous = root / "evidence/ssl-research-v1/development-comparison-v3-review-final.json"
    manifest = document(original)
    review = document(previous)
    bindings = dict(review["bindings"])
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError(f"Reviewed original comparison bytes changed: {path}")
    scaler_reservations = {}
    endpoints = []
    for method in ("shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen"):
        endpoints.append(
            (f"band_{method}_short_probe", f"band_{method}_seed7_h96_reviewed", method, 7, None)
        )
    endpoints.append(
        (
            "band_direct_end_to_end",
            "band_direct_end_to_end_seed7_h96_reviewed",
            "direct",
            7,
            "direct_end_to_end",
        )
    )
    for method, modes in (
        ("shared_ssl", ("frozen_readout", "full_finetune")),
        ("masked_ssl", ("frozen_readout", "full_finetune")),
        ("permuted_ssl", ("frozen_readout",)),
        ("random_frozen", ("frozen_readout",)),
    ):
        for mode in modes:
            name = f"band_{method}_{mode}"
            directory = f"{name}_seed7_h96" + (
                "_native_retry01" if method == "random_frozen" else ""
            )
            endpoints.append((name, directory, method, 7, mode))
    for seed in (13, 23):
        endpoints.extend(
            [
                (
                    f"band_shared_ssl_short_probe_seed{seed}",
                    f"band_shared_ssl_seed{seed}_h96_replication_v2",
                    "shared_ssl",
                    seed,
                    None,
                ),
                (
                    f"band_direct_end_to_end_seed{seed}",
                    f"band_direct_end_to_end_seed{seed}_h96_replication_v2"
                    + ("_ownership_retry01" if seed == 13 else ""),
                    "direct",
                    seed,
                    "direct_end_to_end",
                ),
            ]
        )
        for mode in ("frozen_readout", "full_finetune"):
            name = f"band_shared_ssl_{mode}_seed{seed}"
            endpoints.append(
                (
                    name,
                    f"band_shared_ssl_{mode}_seed{seed}_h96_replication_v3",
                    "shared_ssl",
                    seed,
                    mode,
                )
            )
    if len(endpoints) != 19 or len(manifest["methods"]) != 28:
        raise ValueError("Exactly 28 original and 19 fixed Band endpoints required")
    if (
        manifest.get("role") != "development"
        or not set(REFERENCES) <= manifest["methods"].keys()
        or len(set(manifest["methods"].values())) != 28
    ):
        raise ValueError("Original28 development identities and four named references required")
    for name, directory, method, seed, mode in endpoints:
        folder = root / "outputs/native_acoustic_ssl_v1" / directory
        report = document(folder / "run.json")
        if (
            report.get("status") != "COMPLETED"
            or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
            or report.get("architecture") != "nonlinear_frequency_conditioned_v1"
            or report["config"]["method"] != method
            or report["config"]["seed"] != seed
            or (mode is not None and report.get("mode") != mode)
            or report.get("inference_sha256") != digest(folder / "inference.pt")
        ):
            raise ValueError(f"Actual completed exact endpoint required: {name}")
        prediction = folder / "predictions.npz"
        if name in manifest["methods"]:
            raise ValueError("Duplicate comparison identity")
        manifest["methods"][name] = str(prediction)
        kind = declared_kind(root, report)
        scaler_reservations[name] = reserve_train_scalers(
            root,
            folder,
            report,
            kind,
            method=method,
            seed=seed,
            mode=mode if mode is not None else "core_frozen_readout",
        )
        for path in (
            prediction,
            *[
                folder / leaf
                for leaf in ("run.json", "membership.json", "inference.pt", "selected_encoder.pt")
            ],
            Path(scaler_reservations[name]["scalers_path"]),
        ):
            bindings[str(path)] = digest(path)
    manifest["interpretation"] = (
        "All 47 fixed completed endpoints, including seeds7/13/23 and controls; no best-seed selection. "
        "One development deployment; CF versus shared architecture/head differences remain confounded. "
        "Runner development scores are not independent evidence. Final transfer and SOTA NOT_ESTABLISHED."
    )
    if len(manifest["methods"]) != 47:
        raise ValueError("All47 fixed methods required; no incomplete endpoint substitution")
    for path in dependencies(root):
        bindings[str(path)] = digest(path)
    after, _ = idle_ledger(root)
    if after != before:
        raise ValueError("Live scientific ledger changed during metadata reservation")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for path in (target, original, previous, Path(__file__).resolve()):
        bindings[str(path)] = digest(path)
    proposal = {
        "status": "PROPOSED_COMPARISON_RECONSTRUCTION_NOT_APPROVAL",
        "manifest_sha256": digest(target),
        "bindings": bindings,
        "allowed_roles": ["development"],
        "evidence_kind": "REVIEWED_SAVED_PREDICTIONS",
        "manifest_path": str(target),
        "methods": 47,
        "output": str(root / "evidence/ssl-research-v1/development-comparison-v5.json"),
        "fitting": False,
        "final_numeric_access": False,
        "retained_failed_attempts": "CF13 and Band random native failures and Band direct13 pre-update ownership refusal preserved and charged; exact reviewed retries only",
        "train_scaler_reservations": scaler_reservations,
    }
    with admission.open("x", encoding="utf-8") as stream:
        json.dump(proposal, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return proposal


def main():
    if ROOT.name != "marine-echo-jepa":
        raise RuntimeError("Production metadata preparation is ROOT-only")
    proposal = prepare(ROOT)
    print(
        json.dumps(
            {
                "status": proposal["status"],
                "methods": 47,
                "bindings": len(proposal["bindings"]),
                "numeric_decoding": False,
            }
        )
    )


if __name__ == "__main__":
    main()
