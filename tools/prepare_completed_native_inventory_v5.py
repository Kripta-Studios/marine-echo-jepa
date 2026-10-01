"""Reserve every currently completed neural endpoint, including five new real jobs."""

import copy
import hashlib
import json
from pathlib import Path

from prepare_completed_native_inventory_v4 import idle_ledger

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    before, ledger = idle_ledger(ROOT)
    basis = ROOT / "orchestration/native_completed_inventory_v3b.json"
    manifest = copy.deepcopy(json.loads(basis.read_bytes()))
    if len(manifest["endpoints"]) != 35:
        raise ValueError("Exact previously audited fixed35 basis required")
    for path, expected in manifest["bindings"].items():
        if digest(path) != expected:
            raise ValueError(f"Audited basis input changed: {path}")
    proof = ROOT / "evidence/ssl-research-v1/native-completed-inventory-owned-v3b/resources.json"
    previous = json.loads(proof.read_bytes())
    if previous.get("manifest_sha256") != digest(basis) or previous["resources"].get("exit_code") != 0 or previous["resources"].get("owned_tree_cleanup_verified") is not True:
        raise ValueError("Actual successful fixed35 owned audit required")
    additions = (
        ("band_shared_ssl_short_probe_seed23", "band_shared_ssl_seed23_h96_replication_v2", "native_band_replication_ssl_weights_only_inference_v2"),
        ("band_direct_end_to_end_seed23", "band_direct_end_to_end_seed23_h96_replication_v2", "native_band_replication_ssl_weights_only_inference_v2"),
        ("band_shared_ssl_frozen_readout_seed13", "band_shared_ssl_frozen_readout_seed13_h96_replication_v3", "native_band_replication_ssl_weights_only_inference_v2"),
        ("band_shared_ssl_full_finetune_seed13", "band_shared_ssl_full_finetune_seed13_h96_replication_v3", "native_band_replication_ssl_weights_only_inference_v2"),
        ("band_random_frozen_frozen_readout", "band_random_frozen_frozen_readout_seed7_h96_native_retry01", "native_band_ssl_weights_only_inference_v1"),
    )
    base = ROOT / "outputs/native_acoustic_ssl_v1"
    for name, directory, kind in additions:
        folder = base / directory
        report = json.loads((folder / "run.json").read_bytes())
        if report.get("status") != "COMPLETED" or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT" or name in manifest["endpoints"]:
            raise ValueError("Every explicitly named new endpoint must actually be complete")
        reviews = {(ROOT / run["review"]).resolve() for run in ledger["runs"]
                   if run.get("review") and run.get("output") and Path(run["output"]).resolve() == folder.resolve()}
        reviews = {path for path in reviews if digest(path) == report["review_sha256"]}
        configs = [Path(path) for path in report["bindings"] if Path(path).is_relative_to(ROOT / "configs")
                   and Path(path).suffix == ".json" and json.loads(Path(path).read_bytes()) == report["config"]]
        if len(reviews) != 1 or len(configs) != 1:
            raise ValueError("Exact original new completed review and config required")
        review_path = next(iter(reviews))
        review = json.loads(review_path.read_bytes())
        for group in (report["bindings"], review["bindings"]):
            for path, expected in group.items():
                if digest(path) != expected:
                    raise ValueError("The new jobs used unchanged currently bound scientific sources")
                manifest["bindings"][path] = expected
        core = report.get("core_config", report["config"])
        scalers = folder / "scalers.json"
        if not scalers.exists():
            # The decoder compares both embedded scaler states to these exact TRAIN bytes.
            scalers = base / "shared_ssl_seed7_h96_cuda0/scalers.json"
        manifest["endpoints"][name] = {
            "directory": str(folder), "kind": kind, "method": core["method"], "seed": core["seed"],
            "mode": report.get("mode", "core_frozen_readout"), "parent": None,
            "config_path": str(configs[0]), "review_path": str(review_path), "scalers_path": str(scalers),
        }
        for path in (review_path, configs[0], scalers, *(folder / leaf for leaf in ("run.json", "membership.json", "selected_encoder.pt", "inference.pt"))):
            manifest["bindings"][str(path)] = digest(path)
    by_hash = {digest(Path(entry["directory"]) / "run.json"): name for name, entry in manifest["endpoints"].items()}
    if len(by_hash) != 40:
        raise ValueError("Exactly40 unique actual completed neural endpoints required")
    for name, entry in manifest["endpoints"].items():
        report = json.loads((Path(entry["directory"]) / "run.json").read_bytes())
        ancestry = report.get("supervised_ancestry")
        if ancestry is not None and ancestry.get("ancestor_run_sha256") is not None:
            parent = by_hash.get(ancestry["ancestor_run_sha256"])
            if parent is None:
                raise ValueError("Every fitted ancestor must be explicitly reserved")
            entry["parent"] = parent
    folder = ROOT / "evidence/ssl-research-v1/native-completed40-reservation-v5"
    target = ROOT / "orchestration/native_completed_inventory_v5.json"
    output = ROOT / "evidence/native-completed-inventory-v5"
    if folder.exists() or target.exists() or output.exists():
        raise FileExistsError("Preserve every earlier reservation and artifact")
    after, _ = idle_ledger(ROOT)
    if after != before:
        raise ValueError("Scientific ledger changed during metadata capture")
    folder.mkdir()
    snapshot = folder / "closed-ledger.json"
    with snapshot.open("xb") as stream:
        stream.write(before)
    manifest["output_path"] = str(output)
    manifest["bindings"].update({str(path): digest(path) for path in (basis, proof, snapshot, Path(__file__).resolve())})
    if idle_ledger(ROOT)[0] != before:
        raise ValueError("Ledger changed after capture; partial files grant no authority")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "ALL40_ACTUAL_COMPLETED_NEURAL_ENDPOINTS_RESERVED_NOT_FINAL_FREEZE", "models": 40,
                      "missing_planned_neural_endpoints": 3, "reference_inventory": "NOT_AUDITED", "fitting": False}))


if __name__ == "__main__":
    main()
