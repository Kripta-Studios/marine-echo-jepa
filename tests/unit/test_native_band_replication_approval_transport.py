"""Metadata-only exact-copy tests; no numerical data, tensors, RNG or fitting."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("_rep_transport", ROOT / "tools/materialize_native_band_replication_reviews_v2.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def case():
    path = ROOT / "orchestration/native_band_replication_admission_v2.json"
    proposal = json.loads(path.read_bytes())
    sha = module.digest(path)
    review = {"status": "APPROVED_BAND_FIXED_REPLICATION_REFERENCES",
              "reviewer_session_id": module.REVIEWER, "implementer_session_id": module.AUTHOR,
              "root_coordinator_session_id": module.COORDINATOR, "unresolved_defects": [],
              "scientific_fields_unchanged": True, "proposal_path": str(path), "proposal_sha256": sha,
              "proof_bindings": {str(path): sha}, "bindings": {}, "approvals": []}
    for job in proposal["jobs"]:
        original = job["approval"]
        config = original["approved_config"]
        review["bindings"].update(original["bindings"])
        review["approvals"].append({"job_id": job["id"],
            "status": "APPROVED_PREFIT" if job["kind"] == "ssl" else "APPROVED_DOWNSTREAM_PREFIT",
            "method": config["method"], "seed": config["seed"],
            "approval_scope": "SYNTHETIC_AUTHORITY_TEST_ONLY_NO_EXECUTION",
            "approved_config_sha256": module.canonical_digest(config),
            "runtime_arguments_sha256": module.canonical_digest(original["runtime_arguments"])})
    return review, proposal, path, sha


def test_exact_copy_changes_only_authorized_authority_fields():
    args = case()
    before = copy.deepcopy(args)
    payloads = module.approved_payloads(*args)
    assert args == before
    assert len(payloads) == 4
    for job in args[1]["jobs"]:
        actual, expected = copy.deepcopy(payloads[job["id"]]), copy.deepcopy(job["approval"])
        for field in ("status", "reviewer_session_id", "approval_scope"):
            actual.pop(field, None)
            expected.pop(field, None)
        assert actual == expected


@pytest.mark.parametrize("damage", ["proposal_proof", "identity", "status", "config", "runtime", "binding",
                                    "seed", "duplicate", "missing", "scope", "defect"])
def test_transport_cannot_grant_changed_or_missing_scientific_authority(damage):
    review, proposal, path, sha = case()
    if damage == "proposal_proof":
        review["proof_bindings"] = {}
    elif damage == "identity":
        review["reviewer_session_id"] = module.COORDINATOR
    elif damage == "status":
        review["status"] = "PROPOSED_NOT_APPROVAL"
    elif damage == "config":
        proposal["jobs"][0]["approval"]["approved_config"]["seed"] = 99
    elif damage == "runtime":
        proposal["jobs"][0]["approval"]["runtime_arguments"]["resume"] = "unapproved"
    elif damage == "binding":
        review["bindings"].pop(next(iter(review["bindings"])))
    elif damage == "seed":
        review["approvals"][0]["seed"] = True
    elif damage == "duplicate":
        review["approvals"][1] = review["approvals"][0]
    elif damage == "missing":
        review["approvals"].pop()
    elif damage == "scope":
        review["approvals"][0]["approval_scope"] = ""
    else:
        review["unresolved_defects"] = ["NOT_APPROVED"]
    with pytest.raises(ValueError):
        module.approved_payloads(review, proposal, path, sha)
