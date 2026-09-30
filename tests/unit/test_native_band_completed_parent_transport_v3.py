"""Authority transport checks use metadata only and grant no actual approval."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("_strong_transport", ROOT / "tools/materialize_native_band_downstream_reviews_v3.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def case():
    path = ROOT / "orchestration/native_band_replication_downstream_seed13_v3.json"
    proposal = json.loads(path.read_bytes())
    sha = module.digest(path)
    review = {"status": "APPROVED_BAND_COMPLETED_PARENT_REFERENCES",
              "reviewer_session_id": module.REVIEWER, "implementer_session_id": module.AUTHOR,
              "root_coordinator_session_id": module.COORDINATOR, "unresolved_defects": [],
              "scientific_fields_unchanged": True, "proposal_path": str(path), "proposal_sha256": sha,
              "proof_bindings": {str(path): sha}, "bindings": {}, "approvals": []}
    for job in proposal["jobs"]:
        original = job["approval"]
        config = original["approved_config"]
        review["bindings"].update(original["bindings"])
        review["approvals"].append({"job_id": job["id"], "status": "APPROVED_DOWNSTREAM_PREFIT",
            "method": config["method"], "mode": config["mode"], "seed": config["seed"],
            "approval_scope": "SYNTHETIC_AUTHORITY_TEST_ONLY_NO_EXECUTION",
            "approved_config_sha256": module.canonical_digest(config),
            "runtime_arguments_sha256": module.canonical_digest(original["runtime_arguments"])})
    return review, proposal, path, sha, 13


def test_exact_copy_preserves_science_and_inputs():
    args = case()
    before = copy.deepcopy(args)
    payloads = module.approved_payloads(*args)
    assert args == before
    assert len(payloads) == 2
    for job in args[1]["jobs"]:
        actual, expected = copy.deepcopy(payloads[job["id"]]), copy.deepcopy(job["approval"])
        for field in ("status", "reviewer_session_id", "approval_scope"):
            actual.pop(field, None)
            expected.pop(field, None)
        assert actual == expected


@pytest.mark.parametrize("damage", ["proof", "identity", "config", "parent", "resume", "mode", "seed", "binding", "duplicate", "defect"])
def test_changed_authority_or_science_rejected(damage):
    review, proposal, path, sha, seed = case()
    if damage == "proof":
        review["proof_bindings"] = {}
    elif damage == "identity":
        review["reviewer_session_id"] = module.COORDINATOR
    elif damage == "config":
        proposal["jobs"][0]["approval"]["approved_config"]["seed"] = 23
    elif damage in ("parent", "resume"):
        runtime = proposal["jobs"][0]["approval"]["runtime_arguments"]
        runtime["encoder" if damage == "parent" else "resume"] = "unapproved"
    elif damage == "mode":
        review["approvals"][0]["mode"] = "direct_end_to_end"
    elif damage == "seed":
        review["approvals"][0]["seed"] = True
    elif damage == "binding":
        review["bindings"].pop(next(iter(review["bindings"])))
    elif damage == "duplicate":
        review["approvals"][1] = review["approvals"][0]
    else:
        review["unresolved_defects"] = ["NOT_APPROVED"]
    with pytest.raises(ValueError):
        module.approved_payloads(review, proposal, path, sha, seed)
