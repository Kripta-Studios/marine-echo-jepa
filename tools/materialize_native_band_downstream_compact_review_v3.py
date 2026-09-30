"""Serialize only six explicitly approved immutable proposal references."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = "01a0ef27-876b-7692-917e-3975afc6893d"
AUTHOR = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
COORDINATOR = "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10"
POLICY = "EXACT_PROPOSAL_COPY_WITH_REVIEWER_APPROVED_STATUS_IDENTITY_AND_SCOPE_ONLY"
EXPECTED = {
    ("shared_ssl", "frozen_readout"),
    ("shared_ssl", "full_finetune"),
    ("masked_ssl", "frozen_readout"),
    ("masked_ssl", "full_finetune"),
    ("permuted_ssl", "frozen_readout"),
    ("random_frozen", "frozen_readout"),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_authority(review, proposal, proposal_path, proposal_sha, transport=None):
    required_proofs = {
        str(ROOT / path)
        for path in (
            "tools/materialize_native_band_downstream_compact_review_v3.py",
            "tools/run_reviewed_native_band_downstream_v3.py",
            "tools/prepare_native_band_downstream_canonical_references_v3.py",
            "orchestration/native_band_downstream_canonical_references_v3.json",
        )
    }
    authority = transport or review
    if not required_proofs.issubset(authority.get("proof_bindings", {})):
        raise ValueError("Exact compact materializer/controller/metadata proofs required")
    if transport is not None and (
        transport.get("status") != "APPROVED_EXACT_BAND_APPROVAL_TRANSPORT"
        or transport.get("reviewer_session_id") != REVIEWER
        or transport.get("unresolved_defects") != []
        or transport.get("scientific_scope_unchanged") is not True
    ):
        raise ValueError("Distinct transport approval required")
    if (
        review.get("status") != "APPROVED_BAND_DOWNSTREAM_PREFIT_REFERENCES"
        or review.get("reviewer_session_id") != REVIEWER
        or review.get("implementer_session_id") != AUTHOR
        or review.get("root_coordinator_session_id") != COORDINATOR
        or review.get("materialization_policy") != POLICY
        or review.get("unresolved_defects") != []
        or review.get("proposal_path") != str(proposal_path)
        or review.get("proposal_sha256") != proposal_sha
        or review.get("proof_bindings", {}).get(str(proposal_path)) != proposal_sha
    ):
        raise ValueError("Genuine exact distinct referential approval required")
    jobs = proposal.get("jobs", [])
    references = review.get("approvals", [])
    if not isinstance(references, list) or len(jobs) != 6 or len(references) != 6:
        raise ValueError("Exactly six explicit approvals required")
    indexed = {r.get("job_id"): r for r in references}
    if len(indexed) != 6 or set(indexed) != {j["id"] for j in jobs}:
        raise ValueError("Duplicate, missing or unknown approval reference")
    pairs = set()
    for job in jobs:
        original = job["approval"]
        config, runtime = original["approved_config"], original["runtime_arguments"]
        method, mode = config["method"], config["mode"]
        pairs.add((method, mode))
        reference = indexed[job["id"]]
        if (
            reference.get("status") != "APPROVED_DOWNSTREAM_PREFIT"
            or reference.get("method") != method
            or reference.get("mode") != mode
            or type(reference.get("seed")) is not int
            or reference["seed"] != 7
            or reference.get("approved_config_sha256") != canonical_digest(config)
            or reference.get("runtime_arguments_sha256") != canonical_digest(runtime)
            or not isinstance(reference.get("approval_scope"), str)
            or not reference["approval_scope"].strip()
            or original.get("allowed_seeds") != [7]
            or original.get("allowed_methods") != [method]
            or original.get("allowed_modes") != [mode]
            or set(original.get("allowed_roles", [])) != {"train", "development"}
            or original.get("band_budget_status") != "ROOT_RESOLVED"
            or original.get("architecture") != "nonlinear_frequency_conditioned_v1"
            or original.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
            or any(review["bindings"].get(p) != sha for p, sha in original["bindings"].items())
        ):
            raise ValueError("Approved config/runtime/scientific binding reference differs")
    if pairs != EXPECTED:
        raise ValueError("Exact six matched endpoint pairs required")
    return indexed


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    review_path = folder / "band-downstream-prefit-compact-review-final-v3.json"
    proposal_path = ROOT / "orchestration/native_band_downstream_admission_v1.json"
    review, proposal = (json.loads(path.read_bytes()) for path in (review_path, proposal_path))
    transport_path = folder / "band-downstream-approval-transport-review-final-v4.json"
    transport = json.loads(transport_path.read_bytes())
    if transport.get("scientific_review_sha256") != digest(review_path):
        raise ValueError("Transport must bind the actual independent scientific verdict")
    witness = json.loads(
        (folder / "band-downstream-prefit-compact-exit-witness-v3.json").read_bytes()
    )
    if witness.get("actual_cli_exit_code") != 0 or witness.get("session_id") != 63411:
        raise ValueError("Actual independent reviewer CLI closure required")
    indexed = validate_authority(review, proposal, proposal_path, digest(proposal_path), transport)
    for document in (review, transport):
        for group in ("bindings", "proof_bindings"):
            for name, expected in document.get(group, {}).items():
                if digest(name) != expected:
                    raise ValueError(f"Changed independent review binding: {name}")
    target = folder / "band-downstream-approval-extraction-v1.json"
    if target.exists():
        raise FileExistsError("Preserve earlier derived approvals")
    payloads = {}
    for job in proposal["jobs"]:
        runtime = job["approval"]["runtime_arguments"]
        destination = Path(job["review_path"]).resolve()
        if (
            destination.parent != folder
            or runtime["review"] != str(destination)
            or runtime["trainer-review"] != str(destination)
            or destination.exists()
            or Path(runtime["output"]).exists()
            or Path(runtime["receipt"]).exists()
            or not Path(runtime["output"]).is_relative_to(ROOT / "outputs")
            or not Path(runtime["receipt"]).is_relative_to(ROOT / "evidence")
        ):
            raise ValueError("Exact absent approved paths required")
        approved = copy.deepcopy(job["approval"])
        approved["status"] = "APPROVED_DOWNSTREAM_PREFIT"
        approved["reviewer_session_id"] = REVIEWER
        approved["approval_scope"] = indexed[job["id"]]["approval_scope"]
        approved["referential_review"] = {
            "path": str(review_path),
            "sha256": digest(review_path),
            "proposal_sha256": digest(proposal_path),
            "job_id": job["id"],
            "transport_review_sha256": digest(transport_path),
        }
        restored = copy.deepcopy(approved)
        for field in ("status", "reviewer_session_id", "approval_scope", "referential_review"):
            restored.pop(field, None)
        expected = copy.deepcopy(job["approval"])
        for field in ("status", "reviewer_session_id", "approval_scope", "referential_review"):
            expected.pop(field, None)
        if restored != expected:
            raise ValueError("Scientific fields changed during materialization")
        payloads[destination] = approved
    derived = {}
    for destination, approved in payloads.items():
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(approved, stream, indent=2, sort_keys=True)
            stream.write("\n")
        key = approved["approved_config"]["method"] + "/" + approved["approved_config"]["mode"]
        derived[key] = {
            "path": str(destination),
            "sha256": digest(destination),
            "binding_count": len(approved["bindings"]),
        }
    receipt = {
        "status": "SIX_EXACT_BAND_DOWNSTREAM_APPROVALS_VERIFIED_AND_EXTRACTED",
        "parent_review_sha256": digest(review_path),
        "parent_review_path": str(review_path),
        "proposal_sha256": digest(proposal_path),
        "scientific_fields_unchanged": True,
        "materialization_policy": POLICY,
        "actual_exit_witness": witness,
        "derived": derived,
        "training_execution": "NOT_RUN",
        "final_numeric_access": "NOT_RUN",
    }
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "jobs": len(derived)}))


if __name__ == "__main__":
    main()
