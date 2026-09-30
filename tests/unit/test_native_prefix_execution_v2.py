"""Admission guards before subprocesses, corpus decoding or model construction."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "_prefix_execution", ROOT / "tools/execute_native_prefix_job_v2.py"
)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def case(tmp_path):
    manifest = {
        "kind": "native_prefix_transfer_manifest_v1",
        "role": "prefix_transfer",
        "evidence_kind": "REVIEWED_PREFIX_TRANSFER",
        "implementer_session_id": "builder",
        "coordinator_session_id": "root",
        "suffix_numeric_path": None,
        "config": {
            "device": "cuda:0",
            "family": "cf",
            "seed": 7,
            "method": "cf_jepa",
            "mode": "frozen_readout",
            "prefix_days": 7,
            "updates": 2000,
            "cadence": 500,
        },
    }
    review = {
        "status": "APPROVED_PREFIX_TRANSFER_PREFIT",
        "scope": "native_prefix_transfer_fit",
        "role": "prefix_transfer",
        "evidence_kind": "REVIEWED_PREFIX_TRANSFER",
        "reviewer_session_id": "reviewer",
        "implementer_session_id": "builder",
        "coordinator_session_id": "root",
        "config": manifest["config"].copy(),
        "output_path": str(tmp_path / "out"),
        "receipt_path": str(tmp_path / "receipt"),
    }
    return manifest, review, tmp_path / "out", tmp_path / "receipt"


def test_source_only_guard_admits_exact_separate_prefix_cell(tmp_path):
    args = case(tmp_path)
    assert wrapper.validate_scope(*args) == args[0]["config"]


@pytest.mark.parametrize(
    "changed", ["device", "suffix", "reviewer", "output", "role", "evidence", "config", "updates"]
)
def test_reject_changed_scope_before_subprocess_or_numeric_decode(tmp_path, changed):
    manifest, review, output, receipt = case(tmp_path)
    if changed == "device":
        manifest["config"]["device"] = "cuda:1"
        review["config"] = manifest["config"].copy()
    elif changed == "suffix":
        manifest["suffix_numeric_path"] = "UNOPENED_SUFFIX.npz"
    elif changed == "reviewer":
        review["reviewer_session_id"] = "builder"
    elif changed == "output":
        review["output_path"] = str(tmp_path / "other")
    elif changed == "role":
        manifest["role"] = "final_test"
    elif changed == "evidence":
        manifest["evidence_kind"] = "SYNTHETIC_CORRECTNESS_ONLY"
    elif changed == "config":
        review["config"]["prefix_days"] = 30
    else:
        manifest["config"]["updates"] = 5001
        review["config"] = manifest["config"].copy()
    with pytest.raises(ValueError):
        wrapper.validate_scope(manifest, review, output, receipt)


def test_scope_rejects_boolean_seed(tmp_path):
    manifest, review, output, receipt = case(tmp_path)
    manifest["config"]["seed"] = True
    review["config"] = manifest["config"].copy()
    with pytest.raises(ValueError):
        wrapper.validate_scope(manifest, review, output, receipt)


def ledger(aggregate=95.9, band=11.99):
    return {
        "gpu_limit_hours": 96,
        "gpu_hours_spent_owned_scientific_jobs": aggregate,
        "runs": [
            {
                "id": "retained_failed_band_attempt",
                "status": "FAILED_REAL_BAND_CUDA_ATTEMPT",
                "budget_family": "native_band_v1",
                "resources_full_attempt": {"elapsed_full_attempt_seconds": band * 3600},
            }
        ],
    }


def test_band_failed_time_limits_prefix_to_lesser_remaining_allowance():
    deadline, aggregate, band = wrapper.resource_budget(ledger(), "band")
    assert deadline == pytest.approx(36)
    assert aggregate == 95.9 and band == pytest.approx(11.99)


def test_cf_prefix_uses_aggregate_instead_of_band_subbudget():
    deadline, aggregate, band = wrapper.resource_budget(ledger(), "cf")
    assert deadline == pytest.approx(360)
    assert aggregate == 95.9 and band is None


@pytest.mark.parametrize("bad", [96, float("nan"), -1, True])
def test_invalid_or_exhausted_aggregate_blocks_launch(bad):
    with pytest.raises(ValueError):
        wrapper.resource_budget(ledger(aggregate=bad), "core")


def test_band_budget_exhaustion_blocks_even_with_remaining_aggregate():
    with pytest.raises(ValueError):
        wrapper.resource_budget(ledger(aggregate=30, band=12), "band")


def test_active_owned_job_blocks_prefix_accounting_admission():
    data = ledger()
    data["runs"].append({"status": "RUNNING_CUDA"})
    with pytest.raises(ValueError):
        wrapper.resource_budget(data, "cf")
