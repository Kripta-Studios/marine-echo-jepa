"""No fit: compact approval references must preserve exact independent authority."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "_compact_band", ROOT / "tools/materialize_native_band_downstream_compact_review_v3.py"
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def case():
    path = ROOT / "orchestration/SYNTHETIC_CORRECTNESS_ONLY_PROPOSAL.json"
    jobs, approvals = [], []
    for index, (method, mode) in enumerate(sorted(module.EXPECTED)):
        config = {"method": method, "mode": mode, "seed": 7, "gradient_clip": 1.0}
        runtime = {"config": "SYNTHETIC_CORRECTNESS_ONLY", "output": "NO_EXECUTION"}
        original = {
            "approved_config": config,
            "runtime_arguments": runtime,
            "allowed_seeds": [7],
            "allowed_methods": [method],
            "allowed_modes": [mode],
            "allowed_roles": ["train", "development"],
            "band_budget_status": "ROOT_RESOLVED",
            "architecture": "nonlinear_frequency_conditioned_v1",
            "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT",
            "bindings": {"NO_FILE_OPENED": "a" * 64},
        }
        identifier = "SYNTHETIC_CORRECTNESS_ONLY_" + str(index)
        jobs.append({"id": identifier, "approval": original})
        approvals.append(
            {
                "job_id": identifier,
                "status": "APPROVED_DOWNSTREAM_PREFIT",
                "method": method,
                "mode": mode,
                "seed": 7,
                "approval_scope": "Exact synthetic reference only; no execution.",
                "approved_config_sha256": module.canonical_digest(config),
                "runtime_arguments_sha256": module.canonical_digest(runtime),
            }
        )
    review = {
        "status": "APPROVED_BAND_DOWNSTREAM_PREFIT_REFERENCES",
        "reviewer_session_id": module.REVIEWER,
        "implementer_session_id": module.AUTHOR,
        "root_coordinator_session_id": module.COORDINATOR,
        "materialization_policy": module.POLICY,
        "unresolved_defects": [],
        "proposal_path": str(path),
        "proposal_sha256": "b" * 64,
        "bindings": {str(path): "b" * 64, "NO_FILE_OPENED": "a" * 64},
        "approvals": approvals,
        "proof_bindings": {
            str(ROOT / p): "c" * 64
            for p in (
                "tools/materialize_native_band_downstream_compact_review_v3.py",
                "tools/run_reviewed_native_band_downstream_v3.py",
                "tools/prepare_native_band_downstream_canonical_references_v3.py",
                "orchestration/native_band_downstream_canonical_references_v3.json",
            )
        },
    }
    review["proof_bindings"][str(path)] = "b" * 64
    return review, {"jobs": jobs}, path, "b" * 64


def test_exact_six_references_have_authority_without_changing_proposal():
    args = case()
    before = copy.deepcopy(args)
    assert len(module.validate_authority(*args)) == 6
    assert args == before


def test_transport_approval_cannot_expand_scientific_authority():
    review, proposal, path, sha = case()
    transport = {
        "status": "APPROVED_EXACT_BAND_APPROVAL_TRANSPORT",
        "reviewer_session_id": module.REVIEWER,
        "unresolved_defects": [],
        "scientific_scope_unchanged": True,
        "proof_bindings": copy.deepcopy(review["proof_bindings"]),
    }
    assert len(module.validate_authority(review, proposal, path, sha, transport)) == 6
    transport["scientific_scope_unchanged"] = False
    with pytest.raises(ValueError, match="Distinct transport"):
        module.validate_authority(review, proposal, path, sha, transport)


def test_actual_closed_scientific_verdict_binds_proposal_in_proof_bindings():
    import json

    proposal_path = ROOT / "orchestration/native_band_downstream_admission_v1.json"
    review = json.loads((ROOT / "evidence/ssl-research-v1/band-downstream-prefit-compact-review-final-v3.json").read_bytes())
    proposal = json.loads(proposal_path.read_bytes())
    transport = case()[0]
    transport.update(status="APPROVED_EXACT_BAND_APPROVAL_TRANSPORT", scientific_scope_unchanged=True)
    assert str(proposal_path) not in review["bindings"]
    assert review["proof_bindings"][str(proposal_path)] == module.digest(proposal_path)
    assert len(module.validate_authority(review, proposal, proposal_path, module.digest(proposal_path), transport)) == 6
    for corrupt in (None, "d" * 64):
        changed = copy.deepcopy(review)
        if corrupt is None:
            changed["proof_bindings"].pop(str(proposal_path))
        else:
            changed["proof_bindings"][str(proposal_path)] = corrupt
        with pytest.raises(ValueError, match="Genuine exact"):
            module.validate_authority(changed, proposal, proposal_path, module.digest(proposal_path), transport)


@pytest.mark.parametrize(
    "damage",
    [
        "status",
        "reviewer",
        "hash",
        "config",
        "runtime",
        "duplicate",
        "missing",
        "binding",
        "seed",
        "mode",
        "policy",
        "defect",
        "proof",
    ],
)
def test_missing_or_altered_independent_authority_cannot_be_materialized(damage):
    review, proposal, path, sha = case()
    if damage == "status":
        review["status"] = "PROPOSED_NOT_APPROVAL"
    elif damage == "reviewer":
        review["reviewer_session_id"] = module.COORDINATOR
    elif damage == "hash":
        review["proposal_sha256"] = "d" * 64
    elif damage == "config":
        proposal["jobs"][0]["approval"]["approved_config"]["gradient_clip"] = 2.0
    elif damage == "runtime":
        review["approvals"][0]["runtime_arguments_sha256"] = "d" * 64
    elif damage == "duplicate":
        review["approvals"][1] = review["approvals"][0]
    elif damage == "missing":
        review["approvals"].pop()
    elif damage == "binding":
        review["bindings"]["NO_FILE_OPENED"] = "d" * 64
    elif damage == "seed":
        review["approvals"][0]["seed"] = True
    elif damage == "mode":
        review["approvals"][0]["mode"] = "direct_end_to_end"
    elif damage == "policy":
        review["materialization_policy"] = "ROOT_MAY_MODIFY_MODEL"
    elif damage == "defect":
        review["unresolved_defects"] = ["NOT_APPROVED"]
    else:
        review["proof_bindings"] = {}
    with pytest.raises(ValueError):
        module.validate_authority(review, proposal, path, sha)
