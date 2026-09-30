"""Closed prefix histories must remain charged before subsequent Band transfers."""

import copy
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
name = os.environ.get("NATIVE_PREFIX_BUDGET_WRAPPER", "execute_native_prefix_job.py")
assert name in ("execute_native_prefix_job.py", "execute_native_prefix_job_v2.py")
SPEC = importlib.util.spec_from_file_location("_prefix_budget_history", ROOT / "tools" / name)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def case(status="COMPLETED_REAL_PREFIX_TRANSFER"):
    return {"gpu_limit_hours": 96, "gpu_hours_spent_owned_scientific_jobs": 9.0,
            "runs": [{"status": status, "budget_family": "native_band_v1",
                      "method": "band_SYNTHETIC_METADATA_ONLY", "elapsed_owned_seconds": 3600.0,
                      "resources_full_attempt": {"elapsed_full_attempt_seconds": 3600.0}}]}


@pytest.mark.parametrize("status", ["COMPLETED_REAL_PREFIX_TRANSFER", "PREFIX_TRANSFER_NOT_ASSESSABLE",
                                    "PREFIX_TRANSFER_FAILED_OR_BLOCKED"])
def test_every_closed_prefix_status_remains_fully_charged(status):
    ledger = case(status)
    before = copy.deepcopy(ledger)
    deadline, aggregate, band = wrapper.resource_budget(ledger, "band")
    assert (deadline, aggregate, band) == (7200, 9.0, 1.0)
    assert ledger == before


@pytest.mark.parametrize("damage", ["status", "active", "reconciliation", "negative", "bool",
                                    "conflict", "unaccounted", "aggregate"])
def test_bad_or_unreconciled_prefix_accounting_is_rejected(damage):
    ledger = case()
    record = ledger["runs"][0]
    if damage == "status":
        record["status"] = "UNKNOWN_TERMINAL"
    elif damage == "active":
        record["status"] = "RUNNING_CUDA"
    elif damage == "reconciliation":
        record["requires_reconciliation"] = True
    elif damage == "negative":
        record["elapsed_owned_seconds"] = -1
    elif damage == "bool":
        record["elapsed_owned_seconds"] = True
    elif damage == "conflict":
        record["elapsed_owned_seconds"] = 1
    elif damage == "unaccounted":
        record.pop("budget_family")
    else:
        ledger["gpu_hours_spent_owned_scientific_jobs"] = .5
    with pytest.raises((ValueError, TypeError)):
        wrapper.resource_budget(ledger, "band")
