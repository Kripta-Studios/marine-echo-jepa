"""Bounded JSON ancestry packet after reviewer transport timeouts; no arrays."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    proposal = load(ROOT / "orchestration/native_latent_cpu_replay_admission_v1.json")
    manifest = load(ROOT / "orchestration/native_latent_cpu_replay_v1.json")
    split_path = ROOT / "configs/native_ssl_split_v1.json"
    split = load(split_path)
    allowed = {s["deployment"]: s["archive_sha256"] for s in split["sources"] if s["role"] == "train"}
    records = []
    for run in manifest["runs"]:
        folder = Path(run["directory"])
        report, membership = load(folder / "run.json"), load(folder / "membership.json")
        rows = list(zip(membership["train_deployments"], membership["train_row_ids"], strict=True))
        if (len(rows) != 18593 or len(set(rows)) != 18593
                or any(allowed.get(dep) != archive for dep, archive in zip(
                    membership["train_deployments"], membership["train_archive_sha256"], strict=True))
                or report["membership_sha256"] != digest(folder / "membership.json")):
            raise ValueError("Actual complete TRAIN parent membership differs")
        paths = {name: {"path": str(folder / name), "sha256": digest(folder / name)}
                 for name in ("run.json", "membership.json", "scalers.json", "inference.pt", "selected_encoder.pt")}
        scalers = load(folder / "scalers.json")
        records.append({
            "directory": str(folder), "status": report["status"],
            "evidence_kind": report["evidence_kind"], "config": report["config"],
            "selected_pretrain_step": report["selected_pretrain_step"],
            "membership_rows": len(rows), "unique_membership_rows": len(set(rows)),
            "deployment_counts": dict(Counter(membership["train_deployments"])),
            "all_archive_pairs_match_original_train_split": True,
            "original_report_bindings": report["bindings"], "scalers_json": scalers,
            "artifacts": paths,
        })
        for entry in paths.values():
            proposal["bindings"][entry["path"]] = entry["sha256"]
    log = ROOT / "evidence/ssl-latent-builder-v1/root-durable-fixture-v1.log"
    packet = {
        "status": "ROOT_DERIVED_METADATA_PACKET_NOT_INDEPENDENT_APPROVAL",
        "records": records, "original_split_sha256": digest(split_path),
        "durable_synthetic_log": log.read_text(encoding="utf-8"),
        "durable_synthetic_log_sha256": digest(log),
        "fresh_output_exists": Path(proposal["approved_output"]).exists(),
        "public_numeric_decoding": False, "checkpoint_tensor_decoding": False,
        "derivation_source_sha256": digest(Path(__file__).resolve()),
    }
    path = ROOT / "evidence/ssl-research-v1/native-latent-replay-review-packet-v2.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(packet, stream, indent=2)
        stream.write("\n")
    for item in [path, Path(__file__).resolve(),
                 ROOT / "evidence/ssl-research-v1/native-latent-cpu-replay-review-final.json"]:
        proposal["bindings"][str(item)] = digest(item)
    with (ROOT / "orchestration/native_latent_cpu_replay_admission_v2.json").open("x", encoding="utf-8") as stream:
        json.dump(proposal, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": packet["status"], "records": len(records),
                      "bindings": len(proposal["bindings"]), "packet_bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
