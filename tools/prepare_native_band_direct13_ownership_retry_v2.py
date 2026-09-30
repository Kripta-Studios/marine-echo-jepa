"""Correct only prospective wrapper authority after a distinct metadata rejection."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    original_path = ROOT / "orchestration/native_band_direct13_ownership_retry_admission_v1.json"
    original = json.loads(original_path.read_bytes())
    rejected_path = Path(original["approval"]["runtime_arguments"]["review"])
    rejected = json.loads(rejected_path.read_bytes())
    if rejected.get("status") != "REQUEST_CHANGES":
        raise ValueError("Actual distinct metadata rejection required")
    revised = copy.deepcopy(original)
    leaf = revised["approval"]
    leaf["required_entrypoint"] = str(ROOT / "tools/execute_native_band_replication_job.py")
    review_path = rejected_path.with_name(rejected_path.stem + "-v2.json")
    leaf["runtime_arguments"].update(review=str(review_path), **{"trainer-review": str(review_path)})
    leaf["bindings"].update({str(path): digest(path) for path in (original_path, rejected_path, Path(__file__).resolve())})
    revised.update(metadata_correction="required_entrypoint now names the actual approved V2 wrapper; recipe and fit arguments unchanged",
                   previous_rejection_sha256=digest(rejected_path))
    for name, expected in original["approval"]["bindings"].items():
        if digest(name) != expected:
            raise ValueError("Original reviewed scientific source changed")
    for key in ("output", "receipt", "review"):
        if Path(leaf["runtime_arguments"][key]).exists():
            raise FileExistsError("Absent prospective destinations required")
    target = ROOT / "orchestration/native_band_direct13_ownership_retry_admission_v2.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(revised, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "CORRECTED_METADATA_PROPOSAL_NOT_APPROVED_OR_FITTED", "required_entrypoint": leaf["required_entrypoint"]}))


if __name__ == "__main__":
    main()
