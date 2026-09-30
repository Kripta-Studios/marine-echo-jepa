"""Transport six exact, independently approved operational references."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = "01a0ef27-876b-7692-917e-3975afc6893d"
AUTHOR = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
COORDINATOR = "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10"
EXPECTED = [
    ("ssl", "shared_ssl", None, 23), ("downstream", "direct", "direct_end_to_end", 23),
    ("downstream", "direct", "direct_end_to_end", 13),
    ("downstream", "shared_ssl", "frozen_readout", 13),
    ("downstream", "shared_ssl", "full_finetune", 13),
    ("downstream", "random_frozen", "frozen_readout", 7),
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def approved_payloads(review, proposal, proposal_path, proposal_sha):
    if (review.get("status") != "APPROVED_BAND_DESKTOP_OPERATIONAL_REFERENCES"
            or review.get("reviewer_session_id") != REVIEWER or review.get("implementer_session_id") != AUTHOR
            or review.get("root_coordinator_session_id") != COORDINATOR or review.get("unresolved_defects") != []
            or review.get("scientific_fields_unchanged") is not True or review.get("proposal_path") != str(proposal_path)
            or review.get("proposal_sha256") != proposal_sha
            or review.get("proof_bindings", {}).get(str(proposal_path)) != proposal_sha):
        raise ValueError("Actual distinct exact operational prefit required")
    jobs, refs = proposal.get("jobs", []), review.get("approvals", [])
    indexed = {ref.get("job_id"): ref for ref in refs}
    if len(jobs) != 6 or len(refs) != 6 or len(indexed) != 6 or set(indexed) != {job["id"] for job in jobs}:
        raise ValueError("Exactly six unique approved references required")
    payloads = {}
    for job, expected in zip(jobs, EXPECTED, strict=True):
        original, ref = job["approval"], indexed[job["id"]]
        config, runtime = original["approved_config"], original["runtime_arguments"]
        kind, method, mode, seed = expected
        status = "APPROVED_PREFIT" if kind == "ssl" else "APPROVED_DOWNSTREAM_PREFIT"
        if (job["kind"] != kind or runtime.get("kind") != kind
                or (config.get("method"), config.get("mode"), config.get("seed")) != (method, mode, seed)
                or ref.get("status") != status or ref.get("method") != method or ref.get("mode") != mode
                or type(ref.get("seed")) is not int or ref["seed"] != seed
                or ref.get("approved_config_sha256") != canonical_digest(config)
                or ref.get("runtime_arguments_sha256") != canonical_digest(runtime)
                or original.get("required_entrypoint") != str(ROOT / "tools/execute_native_band_operational_job_v4.py")
                or original.get("allowed_methods") != [method] or original.get("allowed_seeds") != [seed]
                or runtime.get("resume") is not None or set(original.get("allowed_roles", [])) != {"train", "development"}
                or not isinstance(ref.get("approval_scope"), str) or not ref["approval_scope"].strip()
                or any(review.get("bindings", {}).get(name) != sha for name, sha in original["bindings"].items())):
            raise ValueError("Exact science/runtime/authority/bindings required")
        selected_parent = mode in {"frozen_readout", "full_finetune"}
        if any(bool(runtime.get(key)) != selected_parent for key in ("encoder", "ancestor-review", "ancestor-config")):
            raise ValueError("Exact selected parent or fresh scratch arguments required")
        approved = copy.deepcopy(original)
        approved.update(status=status, reviewer_session_id=REVIEWER, approval_scope=ref["approval_scope"])
        payloads[job["id"]] = approved
    return payloads


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    review_path = folder / "band-desktop-operational-prefit-compact-final-v4.json"
    proposal_path = ROOT / "orchestration/native_band_desktop_admission_v4.json"
    review, proposal = (json.loads(path.read_bytes()) for path in (review_path, proposal_path))
    witness = json.loads((folder / "band-desktop-operational-prefit-exit-witness-v4.json").read_bytes())
    if witness.get("actual_cli_exit_code") != 0 or witness.get("review_sha256") != digest(review_path):
        raise ValueError("Actual closed distinct review required")
    required = {str(ROOT / path) for path in (
        "tools/materialize_native_band_desktop_reviews_v4.py", "tools/prepare_native_band_desktop_admission_v4.py",
        "orchestration/native_band_desktop_references_v4.json")}
    if not required.issubset(review.get("proof_bindings", {})):
        raise ValueError("Exact approved transport source closure required")
    for group in ("bindings", "proof_bindings"):
        for name, sha in review.get(group, {}).items():
            if digest(name) != sha:
                raise ValueError(f"Reviewed bytes changed: {name}")
    payloads = approved_payloads(review, proposal, proposal_path, digest(proposal_path))
    receipt_path = folder / "band-desktop-operational-review-extraction-v4.json"
    if receipt_path.exists():
        raise FileExistsError("Preserve prior review extraction")
    for job in proposal["jobs"]:
        runtime, destination = job["approval"]["runtime_arguments"], Path(job["review_path"]).resolve()
        if (destination.parent != folder or runtime["review"] != str(destination) or runtime["trainer-review"] != str(destination)
                or destination.exists() or any(Path(runtime[key]).exists() for key in ("output", "receipt"))
                or not Path(runtime["output"]).is_relative_to(ROOT / "outputs")
                or not Path(runtime["receipt"]).is_relative_to(ROOT / "evidence")):
            raise ValueError("Exact bounded absent destinations required")
    derived = []
    for job in proposal["jobs"]:
        approved = payloads[job["id"]]
        approved["referential_review"] = {"path": str(review_path), "sha256": digest(review_path),
                                          "proposal_sha256": digest(proposal_path), "job_id": job["id"]}
        destination = Path(job["review_path"])
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(approved, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        derived.append({"id": job["id"], "path": str(destination), "sha256": digest(destination)})
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "SIX_EXACT_DESKTOP_OPERATIONAL_APPROVALS_MATERIALIZED", "derived": derived,
                   "parent_review_path": str(review_path), "parent_review_sha256": digest(review_path),
                   "scientific_fields_unchanged": True, "actual_exit_witness": witness}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "SIX_EXACT_DESKTOP_OPERATIONAL_APPROVALS_MATERIALIZED", "jobs": 6}))


if __name__ == "__main__":
    main()
