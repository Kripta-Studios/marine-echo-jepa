"""Synthetic suffix launch and completion guards; no processes or scientific data."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "suffix_wrapper", ROOT / "tools/execute_native_prefix_suffix_job_v1.py"
)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def case():
    output, receipt = ROOT / "outputs/SYNTHETIC_unwritten", ROOT / "evidence/SYNTHETIC_unwritten"
    cells = [{"name": "adapted", "zero_shot": False}]
    manifest = {
        "kind": "native_prefix_suffix_assessment_manifest_v1",
        "role": "adapted_suffix",
        "evidence_kind": "REVIEWED_PREFIX_SUFFIX_ASSESSMENT",
        "device": "cpu",
        "cells": cells,
        "implementer_session_id": "author",
        "coordinator_session_id": "coordinator",
    }
    review = {
        "status": "APPROVED_PREFIX_SUFFIX_ASSESSMENT_EXECUTION",
        "scope": "prefix_suffix_assessment_execution",
        "allowed_roles": ["adapted_suffix"],
        "evidence_kind": manifest["evidence_kind"],
        "allowed_cells": cells,
        "runtime": {"device": "cpu", "output": str(output)},
        "receipt_path": str(receipt),
        "implementer_session_id": "author",
        "coordinator_session_id": "coordinator",
        "reviewer_session_id": "distinct-fixture",
    }
    return manifest, review, output, receipt


def test_distinct_adaptation_scope_admitted():
    wrapper.validate_scope(*case())


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "final_test"),
        ("evidence_kind", "SYNTHETIC_CORRECTNESS_ONLY"),
        ("device", "cuda:1"),
        ("cells", []),
    ],
)
def test_real_wrapper_rejects_wrong_scope(field, value):
    args = case()
    args[0][field] = value
    with pytest.raises(ValueError):
        wrapper.validate_scope(*args)


@pytest.mark.parametrize("reviewer", [None, "author", "coordinator", *sorted(wrapper.AUTHORS)])
def test_authors_and_missing_reviewer_rejected(reviewer):
    args = case()
    args[1]["reviewer_session_id"] = reviewer
    with pytest.raises(ValueError):
        wrapper.validate_scope(*args)


def test_cpu_assessment_time_is_separate_from_prefix_fit_and_gpu():
    ledger = {
        "gpu_hours_spent_owned_scientific_jobs": 17.802355808369175,
        "cpu_prefix_hours_owned": 2,
    }
    wrapper.charge_attempt(
        ledger,
        device="cpu",
        elapsed_seconds=720,
        spent=17.802355808369175,
        band_spent=None,
        cpu_spent=3,
    )
    assert ledger == {
        "gpu_hours_spent_owned_scientific_jobs": 17.802355808369175,
        "cpu_prefix_hours_owned": 2,
        "cpu_assessment_hours_owned": 3.2,
    }


@pytest.mark.parametrize(
    "damage",
    [None, "exit", "stopped", "cleanup", "zero_shot", "missing_prediction", "changed_prediction"],
)
def test_completed_status_requires_owned_exit_and_every_saved_hash(monkeypatch, damage):
    manifest, _, output, _ = case()
    report = {
        "kind": "native_prefix_suffix_assessment_completion_v1",
        "status": "COMPLETED",
        "role": "adapted_suffix",
        "zero_shot": False,
        "device": "cpu",
        "evidence_kind": manifest["evidence_kind"],
        "results": {"adapted": {"zero_shot": False}},
        "prediction_hashes": {str(output / "adapted.npz"): "a" * 64},
    }
    resources = {"exit_code": 0, "stopped_for": None, "owned_tree_cleanup_verified": True}
    if damage == "exit":
        resources["exit_code"] = 1
    if damage == "stopped":
        resources["stopped_for"] = "PROCESS_RAM_LIMIT"
    if damage == "cleanup":
        resources["owned_tree_cleanup_verified"] = False
    if damage == "zero_shot":
        report["zero_shot"] = True
    if damage == "missing_prediction":
        report["prediction_hashes"] = {}
    original_read, original_file = Path.read_bytes, Path.is_file
    target = output / "completion.json"
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: json.dumps(report).encode() if p == target else original_read(p),
    )
    monkeypatch.setattr(Path, "is_file", lambda p: True if p == target else original_file(p))
    monkeypatch.setattr(
        wrapper.prefix_job,
        "digest",
        lambda p: ("b" if damage == "changed_prediction" else "a") * 64,
    )
    assert wrapper.verify_completion(output, manifest, resources) is (damage is None)


@pytest.mark.parametrize("completed", [False, True])
def test_closed_band_suffix_retains_readable_historical_accounting(completed):
    import native_band_budget_history_v2 as history

    ledger = {
        "gpu_limit_hours": 96,
        "gpu_hours_spent_owned_scientific_jobs": 18,
        "runs": [
            {
                "role": "adapted_suffix",
                "fitting": False,
                "status": wrapper.closed_status("cuda:0", completed),
                "budget_family": "native_band_v1",
                "resources_full_attempt": {"elapsed_full_attempt_seconds": 3600},
            }
        ],
    }
    assert history.budget_totals(ledger) == (18, 1, 11 * 3600)
