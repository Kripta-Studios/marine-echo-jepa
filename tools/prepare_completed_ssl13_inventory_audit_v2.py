"""Bind one completed real SSL parent for a CPU metadata-only ancestry audit."""

import hashlib
import json
from pathlib import Path

from marine_echo.evaluation.native_ancestry_inventory import required_sources

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "outputs/native_acoustic_ssl_v1/band_shared_ssl_seed13_h96_replication_v2"
    review_path = ROOT / "evidence/ssl-research-v1/band_shared_ssl_seed13_h96_replication_v2-prefit-review-final.json"
    config_path = ROOT / "configs/native_band_replication_v2/native_band_shared_ssl_seed13_v2.json"
    review = json.loads(review_path.read_bytes())
    report = json.loads((folder / "run.json").read_bytes())
    if report.get("status") != "COMPLETED" or report.get("review_sha256") != digest(review_path):
        raise ValueError("Exact actual completed parent and distinct original prefit required")
    output = ROOT / "evidence/native-completed-ssl13-inventory-v2"
    destination = ROOT / "orchestration/native_completed_ssl13_inventory_audit_v2.json"
    if output.exists() or destination.exists():
        raise FileExistsError("Preserve audit history")
    train = {"input_path": str(ROOT / "data/processed/native_ssl_v1/train.npz"),
             "report_path": str(ROOT / "data/processed/native_ssl_v1/train.json"),
             "identity_cohort_path": str(ROOT / "orchestration/native_assessment_seed7_metadata_v1/train_cohort.json")}
    paths = [*required_sources(), ROOT / "configs/native_ssl_split_v1.json", config_path, review_path,
             *(Path(name) for name in train.values()),
             *(folder / name for name in ("run.json", "membership.json", "selected_encoder.pt", "inference.pt", "scalers.json"))]
    bindings = dict(review["bindings"])
    bindings.update({str(path): digest(path) for path in paths})
    for path, sha in bindings.items():
        if digest(path) != sha:
            raise ValueError("Historical prefit/source binding changed")
    manifest = {"kind": "native_research_inventory_manifest_v1", "purpose": "LOCAL_METADATA_DERIVATION_ONLY",
                "execution": "ROOT_LOCAL_COMPLETED_METADATA_AUDIT", "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT",
                "device": "cpu", "owner_session_id": "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10",
                "output_path": str(output), "split_path": str(ROOT / "configs/native_ssl_split_v1.json"),
                "train": train, "references": {}, "bindings": bindings,
                "endpoints": {"band_shared_ssl_seed13": {
                    "directory": str(folder), "kind": "native_band_replication_ssl_weights_only_inference_v2",
                    "method": "shared_ssl", "seed": 13, "mode": "core_frozen_readout", "parent": None,
                    "config_path": str(config_path), "review_path": str(review_path)}}}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "CPU_COMPLETED_PARENT_METADATA_AUDIT_BOUND", "bindings": len(bindings), "numerical_corpus_decoded": False}))


if __name__ == "__main__":
    main()
