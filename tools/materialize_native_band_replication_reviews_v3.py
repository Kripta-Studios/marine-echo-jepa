"""Materialize four reviewer-authorized fixed replication references without fitting."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = "01a0ef27-876b-7692-917e-3975afc6893d"
AUTHOR = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
COORDINATOR = "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def approved_payloads(review, proposal, proposal_path, proposal_sha):
    if (review.get("status") != "APPROVED_BAND_FIXED_REPLICATION_REFERENCES"
            or review.get("reviewer_session_id") != REVIEWER
            or review.get("implementer_session_id") != AUTHOR
            or review.get("root_coordinator_session_id") != COORDINATOR
            or review.get("unresolved_defects") != []
            or review.get("scientific_fields_unchanged") is not True
            or review.get("proposal_path") != str(proposal_path)
            or review.get("proposal_sha256") != proposal_sha
            or review.get("proof_bindings", {}).get(str(proposal_path)) != proposal_sha):
        raise ValueError("Exact distinct fixed replication approval required")
    jobs, refs = proposal.get("jobs", []), review.get("approvals", [])
    indexed = {ref.get("job_id"): ref for ref in refs}
    if len(jobs) != 4 or len(refs) != 4 or len(indexed) != 4 or set(indexed) != {job["id"] for job in jobs}:
        raise ValueError("Exactly four unique replication references required")
    payloads, pairs = {}, set()
    for job in jobs:
        original, ref = job["approval"], indexed[job["id"]]
        config, runtime = original["approved_config"], original["runtime_arguments"]
        method, seed = config["method"], config["seed"]
        pairs.add((method, seed))
        status = "APPROVED_PREFIT" if job["kind"] == "ssl" else "APPROVED_DOWNSTREAM_PREFIT"
        if (ref.get("status") != status or ref.get("method") != method
                or type(ref.get("seed")) is not int or ref["seed"] != seed
                or ref.get("approved_config_sha256") != canonical_digest(config)
                or ref.get("runtime_arguments_sha256") != canonical_digest(runtime)
                or not isinstance(ref.get("approval_scope"), str) or not ref["approval_scope"].strip()
                or original.get("allowed_seeds") != [seed]
                or original.get("allowed_methods") != [method]
                or set(original.get("allowed_roles", [])) != {"train", "development"}
                or original.get("band_budget_status") != "ROOT_RESOLVED"
                or runtime.get("kind") != job["kind"]
                or any(runtime.get(key) is not None for key in ("encoder", "ancestor-review", "ancestor-config", "resume"))
                or any(review.get("bindings", {}).get(path) != sha for path, sha in original["bindings"].items())):
            raise ValueError("Exact replication science/runtime/bindings required")
        approved = copy.deepcopy(original)
        approved.update(status=status, reviewer_session_id=REVIEWER, approval_scope=ref["approval_scope"])
        payloads[job["id"]] = approved
    if pairs != {("shared_ssl", 13), ("direct", 13), ("shared_ssl", 23), ("direct", 23)}:
        raise ValueError("Only fixed shared/direct seeds13/23 required")
    return payloads


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    review_path = folder / "band-fixed-replication-prefit-compact-final-v3.json"
    proposal_path = ROOT / "orchestration/native_band_replication_admission_v3.json"
    review, proposal = (json.loads(path.read_bytes()) for path in (review_path, proposal_path))
    witness = json.loads((folder / "band-fixed-replication-prefit-exit-witness-v3.json").read_bytes())
    if witness.get("actual_cli_exit_code") != 0 or witness.get("review_sha256") != digest(review_path):
        raise ValueError("Actual independently closed review required")
    required = {str(ROOT / path) for path in (
        "tools/materialize_native_band_replication_reviews_v3.py",
        "tools/prepare_native_band_replication_references_v3.py",
        "orchestration/native_band_replication_references_v3.json",
        "tools/run_reviewed_native_band_replications_v3.py")}
    if not required.issubset(review.get("proof_bindings", {})):
        raise ValueError("Exact reviewed transport source closure required")
    for group in ("bindings", "proof_bindings"):
        for name, sha in review.get(group, {}).items():
            if digest(name) != sha:
                raise ValueError(f"Reviewed binding changed: {name}")
    payloads = approved_payloads(review, proposal, proposal_path, digest(proposal_path))
    receipt_path = folder / "band-fixed-replication-review-extraction-v3.json"
    if receipt_path.exists():
        raise FileExistsError("Preserve earlier approvals")
    for job in proposal["jobs"]:
        runtime = job["approval"]["runtime_arguments"]
        destination = Path(job["review_path"]).resolve()
        if (destination.parent != folder or runtime["review"] != str(destination)
                or runtime["trainer-review"] != str(destination) or destination.exists()
                or Path(runtime["output"]).exists() or Path(runtime["receipt"]).exists()
                or not Path(runtime["output"]).is_relative_to(ROOT / "outputs")
                or not Path(runtime["receipt"]).is_relative_to(ROOT / "evidence")):
            raise ValueError("Original absent destinations required")
    derived = []
    for job in proposal["jobs"]:
        approved = payloads[job["id"]]
        approved["referential_review"] = {"path": str(review_path), "sha256": digest(review_path),
                                          "proposal_sha256": digest(proposal_path), "job_id": job["id"]}
        path = Path(job["review_path"])
        with path.open("x", encoding="utf-8") as stream:
            json.dump(approved, stream, indent=2, sort_keys=True)
            stream.write("\n")
        derived.append({"id": job["id"], "path": str(path), "sha256": digest(path)})
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "FOUR_EXACT_FIXED_REPLICATION_APPROVALS_MATERIALIZED", "derived": derived,
                   "parent_review_path": str(review_path), "parent_review_sha256": digest(review_path),
                   "scientific_fields_unchanged": True, "actual_exit_witness": witness}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "FOUR_EXACT_FIXED_REPLICATION_APPROVALS_MATERIALIZED", "jobs": 4}))


if __name__ == "__main__":
    main()
