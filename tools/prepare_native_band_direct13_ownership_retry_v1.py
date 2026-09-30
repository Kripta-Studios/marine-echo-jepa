"""Propose one fresh-output retry after an actual pre-update GPU ownership refusal."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    original_path = folder / "band_direct_end_to_end_seed13_h96_replication_v2-prefit-review-final.json"
    original = json.loads(original_path.read_bytes())
    failed = folder / "band_direct_end_to_end_seed13_h96_replication_v2-attempt-01/attempt.json"
    failure = json.loads(failed.read_bytes())
    closeout = folder / "band-fixed-replication-partial-closeout-v3.json"
    if (original.get("status") != "APPROVED_DOWNSTREAM_PREFIT"
            or failure.get("resources_full_attempt", {}).get("exit_code") != 1
            or failure.get("requires_reconciliation") is not False
            or json.loads(closeout.read_bytes()).get("status") != "ONE_REAL_REPLICATION_COMPLETED_DIRECT13_OWNERSHIP_GUARD_STOPPED"):
        raise ValueError("Actual original approval and closed pre-update refusal required")
    revised = copy.deepcopy(original)
    revised.update(status="PROPOSED_OWNERSHIP_RETRY_NOT_APPROVAL",
                   approval_scope="PROPOSED: one unchanged recipe after ownership guard refusal; distinct review required")
    revised.pop("reviewer_session_id", None)
    revised.pop("referential_review", None)
    identifier = "band_direct_end_to_end_seed13_h96_replication_v2_ownership_retry01"
    runtime = revised["runtime_arguments"]
    review_path = folder / (identifier + "-prefit-review-final.json")
    runtime.update(output=str(ROOT / "outputs/native_acoustic_ssl_v1" / identifier),
                   receipt=str(folder / (identifier + "-attempt-01")), review=str(review_path))
    runtime["trainer-review"] = str(review_path)
    for name, sha in original["bindings"].items():
        if digest(name) != sha:
            raise ValueError("Original reviewed recipe/source binding changed")
    for path in (original_path, failed, closeout, Path(__file__).resolve(),
                 failed.parent / "console.log", ROOT / "outputs/native_acoustic_ssl_v1/band_direct_end_to_end_seed13_h96_replication_v2/gpu-ownership.json"):
        revised["bindings"][str(path)] = digest(path)
    for key in ("output", "receipt", "review"):
        if Path(runtime[key]).exists():
            raise FileExistsError("Preserve all earlier attempts and destinations")
    if any(runtime.get(key) is not None for key in ("resume", "encoder", "ancestor-review", "ancestor-config")):
        raise ValueError("Fresh scratch supervised retry only")
    target = ROOT / "orchestration/native_band_direct13_ownership_retry_admission_v1.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump({"status": "PROPOSED_NOT_APPROVED_OR_FITTED", "id": identifier,
                   "approval": revised, "original_review_sha256": digest(original_path),
                   "same_scientific_recipe": True, "retry_number": 1,
                   "failed_full_owned_seconds_retained": failure["resources_full_attempt"]["elapsed_full_attempt_seconds"],
                   "launch_requires": "Foreign VLC GPU runtime absent, exact unchanged guard passes, closed ownership and remaining12/96 budget"}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PROPOSED_NOT_APPROVED_OR_FITTED", "id": identifier}))


if __name__ == "__main__":
    main()
