"""Preserve rejected V2 transport and prepare identical V3 transport for repaired proposals."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rejected = ROOT / "evidence/ssl-research-v1/band-fixed-replication-prefit-compact-final-v2.json"
    if json.loads(rejected.read_bytes()).get("status") != "REQUEST_CHANGES":
        raise ValueError("Closed rejected proposal review required")
    source_names = ("prepare_native_band_replication_references_v2.py",
                    "materialize_native_band_replication_reviews_v2.py",
                    "run_reviewed_native_band_replications_v2.py")
    changes = {}
    for name in source_names:
        source = ROOT / "tools" / name
        target = ROOT / "tools" / name.replace("_v2.py", "_v3.py")
        original = source.read_text(encoding="utf-8")
        replacements = {
            "native_band_replication_admission_v2.json": "native_band_replication_admission_v3.json",
            "native_band_replication_references_v2.json": "native_band_replication_references_v3.json",
            "prepare_native_band_replication_references_v2.py": "prepare_native_band_replication_references_v3.py",
            "materialize_native_band_replication_reviews_v2.py": "materialize_native_band_replication_reviews_v3.py",
            "run_reviewed_native_band_replications_v2.py": "run_reviewed_native_band_replications_v3.py",
            "band-fixed-replication-prefit-compact-final-v2.json": "band-fixed-replication-prefit-compact-final-v3.json",
            "band-fixed-replication-prefit-exit-witness-v2.json": "band-fixed-replication-prefit-exit-witness-v3.json",
            "band-fixed-replication-review-extraction-v2.json": "band-fixed-replication-review-extraction-v3.json",
        }
        revised = original
        for before, after in replacements.items():
            revised = revised.replace(before, after)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(revised)
        changes[str(source)] = {"original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                "new_path": str(target), "new_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                "only_literal_metadata_paths_changed": True}
    with (ROOT / "evidence/ssl-research-v1/band-replication-transport-versioning-v3.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "SOURCE_TRANSPORT_VERSIONED_NOT_APPROVAL", "changes": changes,
                   "scientific_source_and_arguments_changed": False}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "SOURCE_TRANSPORT_VERSIONED_NOT_APPROVAL", "new_files": len(changes)}))


if __name__ == "__main__":
    main()
