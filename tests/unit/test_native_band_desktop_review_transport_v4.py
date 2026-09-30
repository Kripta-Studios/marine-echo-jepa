"""Metadata-only exact operational authority checks; no acoustic values decoded."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("desktop_transport", ROOT / "tools/materialize_native_band_desktop_reviews_v4.py")
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def fixture():
    path = ROOT / "orchestration/native_band_desktop_admission_v4.json"
    proposal = json.loads(path.read_bytes())
    review = {
        "status": "APPROVED_BAND_DESKTOP_OPERATIONAL_REFERENCES",
        "reviewer_session_id": transport.REVIEWER, "implementer_session_id": transport.AUTHOR,
        "root_coordinator_session_id": transport.COORDINATOR,
        "unresolved_defects": [], "scientific_fields_unchanged": True,
        "proposal_path": str(path), "proposal_sha256": transport.digest(path),
        "proof_bindings": {str(path): transport.digest(path)}, "bindings": {}, "approvals": [],
    }
    for job in proposal["jobs"]:
        approval = job["approval"]
        config = approval["approved_config"]
        review["bindings"].update(approval["bindings"])
        review["approvals"].append({
            "job_id": job["id"], "status": "APPROVED_PREFIT" if job["kind"] == "ssl" else "APPROVED_DOWNSTREAM_PREFIT",
            "method": config["method"], "mode": config.get("mode"), "seed": config["seed"],
            "approved_config_sha256": transport.canonical_digest(config),
            "runtime_arguments_sha256": transport.canonical_digest(approval["runtime_arguments"]),
            "approval_scope": "SYNTHETIC_CORRECTNESS_ONLY metadata authority fixture; no execution approval.",
        })
    return review, proposal, path, transport.digest(path)


def test_six_exact_payloads_change_only_review_authority_without_mutating_input():
    review, proposal, path, sha = fixture()
    before = copy.deepcopy(proposal)
    approved = transport.approved_payloads(review, proposal, path, sha)
    assert proposal == before
    assert list(approved) == [job["id"] for job in proposal["jobs"]]
    for job in proposal["jobs"]:
        expected = copy.deepcopy(job["approval"])
        ref = next(ref for ref in review["approvals"] if ref["job_id"] == job["id"])
        expected.update(status=ref["status"], reviewer_session_id=transport.REVIEWER, approval_scope=ref["approval_scope"])
        assert approved[job["id"]] == expected


@pytest.mark.parametrize("field,value", [
    ("status", "REQUEST_CHANGES"), ("reviewer_session_id", transport.COORDINATOR),
    ("implementer_session_id", transport.REVIEWER), ("scientific_fields_unchanged", 1),
    ("proposal_sha256", "0" * 64), ("unresolved_defects", ["open"]),
])
def test_invalid_authority_rejected(field, value):
    review, proposal, path, sha = fixture()
    review[field] = value
    with pytest.raises(ValueError):
        transport.approved_payloads(review, proposal, path, sha)


@pytest.mark.parametrize("index", range(6))
def test_every_runtime_and_config_is_bound(index):
    review, proposal, path, sha = fixture()
    proposal["jobs"][index]["approval"]["runtime_arguments"]["device"] = "cpu"
    with pytest.raises(ValueError):
        transport.approved_payloads(review, proposal, path, sha)
    review, proposal, path, sha = fixture()
    proposal["jobs"][index]["approval"]["approved_config"]["lr"] *= 2
    with pytest.raises(ValueError):
        transport.approved_payloads(review, proposal, path, sha)


@pytest.mark.parametrize("defect", ["duplicate", "missing_binding", "parent", "seed_bool", "wrong_entrypoint", "wrong_mode"])
def test_no_authority_or_parent_substitution(defect):
    review, proposal, path, sha = fixture()
    if defect == "duplicate":
        review["approvals"][-1] = copy.deepcopy(review["approvals"][0])
    elif defect == "missing_binding":
        review["bindings"].pop(next(iter(review["bindings"])))
    elif defect == "parent":
        proposal["jobs"][3]["approval"]["runtime_arguments"]["encoder"] = None
        review["approvals"][3]["runtime_arguments_sha256"] = transport.canonical_digest(proposal["jobs"][3]["approval"]["runtime_arguments"])
    elif defect == "seed_bool":
        review["approvals"][-1]["seed"] = True
    elif defect == "wrong_entrypoint":
        proposal["jobs"][0]["approval"]["required_entrypoint"] = str(ROOT / "tools/execute_native_band_job.py")
    else:
        review["approvals"][3]["mode"] = "full_finetune"
    with pytest.raises(ValueError):
        transport.approved_payloads(review, proposal, path, sha)
