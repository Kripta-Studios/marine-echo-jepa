"""Propose one exact-recipe fresh-output control retry; grant no approval or fit."""

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
    original_path = folder / "band_random_frozen_frozen_readout_seed7_h96-prefit-review-final.json"
    original = json.loads(original_path.read_bytes())
    failed = folder / "band_random_frozen_frozen_readout_seed7_h96-attempt-01/attempt.json"
    failure = json.loads(failed.read_bytes())
    reconciliation = folder / "band-random-failed-owner-reconciliation-v1.json"
    if (original.get("status") != "APPROVED_DOWNSTREAM_PREFIT"
            or failure.get("resources_full_attempt", {}).get("exit_code") != 3221227274
            or json.loads(reconciliation.read_bytes()).get("status") != "EXACT_DEAD_BAND_OWNER_RECONCILED_FAILURE_CHARGES_PRESERVED"):
        raise ValueError("Actual original approval, closed failure and exact reconciliation required")
    revised = copy.deepcopy(original)
    revised.update(status="PROPOSED_NATIVE_FAILURE_RETRY_NOT_APPROVAL",
                   approval_scope="PROPOSED: one fresh-output unchanged-recipe control retry; distinct review required")
    revised.pop("reviewer_session_id", None)
    revised.pop("referential_review", None)
    identifier = "band_random_frozen_frozen_readout_seed7_h96_native_retry01"
    runtime = revised["runtime_arguments"]
    review_path = folder / (identifier + "-prefit-review-final.json")
    runtime.update(output=str(ROOT / "outputs/native_acoustic_ssl_v1" / identifier),
                   receipt=str(folder / (identifier + "-attempt-01")), review=str(review_path))
    runtime["trainer-review"] = str(review_path)
    for name, sha in original["bindings"].items():
        if digest(name) != sha:
            raise ValueError("Original reviewed recipe/source/parent binding changed")
    for path in (original_path, failed, reconciliation, Path(__file__).resolve(),
                 folder / "band-downstream-queue-exit-witness-v1.json",
                 folder / "band-random-native-event1000-v1.json"):
        revised["bindings"][str(path)] = digest(path)
    check = copy.deepcopy(revised)
    for field in ("status", "approval_scope", "reviewer_session_id", "referential_review", "bindings"):
        check.pop(field, None)
    expected = copy.deepcopy(original)
    for field in ("status", "approval_scope", "reviewer_session_id", "referential_review", "bindings"):
        expected.pop(field, None)
    for key in ("output", "receipt", "review", "trainer-review"):
        check["runtime_arguments"][key] = expected["runtime_arguments"][key]
    if check != expected or runtime.get("resume") is not None:
        raise ValueError("Only authority metadata, evidence bindings and fresh destinations may change")
    for key in ("output", "receipt", "review"):
        if Path(runtime[key]).exists():
            raise FileExistsError("Exact fresh prospective paths required")
    target = ROOT / "orchestration/native_band_random_strong_retry_admission_v1.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump({"status": "PROPOSED_NOT_APPROVED_OR_FITTED", "id": identifier,
                   "approval": revised, "original_review_sha256": digest(original_path),
                   "same_scientific_recipe": True, "retry_number": 1,
                   "failed_full_owned_seconds_retained": failure["resources_full_attempt"]["elapsed_full_attempt_seconds"]}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PROPOSED_NOT_APPROVED_OR_FITTED", "id": identifier}))


if __name__ == "__main__":
    main()
