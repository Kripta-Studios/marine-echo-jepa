"""Synthetic accounting contracts; these checks authorize no scientific fitting."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from native_cf_control_budget_v1 import limits


def ledger():
    return {
        "gpu_limit_hours": 96,
        "gpu_hours_spent_owned_scientific_jobs": 17.802355808369175,
        "runs": [],
    }


def attempt(seconds, *, status="FAILED_REAL_CF_CONTROL_ATTEMPT"):
    return {
        "id": "cf_random_frozen_seed7",
        "study": "CF_MATCHED_CONTROLS",
        "budget_family": "native_cf_matched_controls_v1",
        "status": status,
        "elapsed_owned_seconds": seconds,
        "resources_full_attempt": {
            "elapsed_full_attempt_seconds": seconds,
            "owned_tree_cleanup_verified": True,
        },
    }


def test_default_cap_preserves_existing_aggregate_and_evaluation_reserve():
    result = limits(ledger())
    assert result.aggregate_hours == 17.802355808369175
    assert result.extension_hours == 0
    assert result.deadline_seconds == 12 * 3600
    assert result.evaluation_reserve_hours == 12


def test_failed_and_resumed_attempts_share_the_same_extension_cap():
    value = ledger()
    value["runs"] = [attempt(3600), attempt(7200, status="COMPLETED_REAL_CF_CONTROL")]
    value["gpu_hours_spent_owned_scientific_jobs"] += 3
    result = limits(value)
    assert result.extension_hours == 3
    assert result.deadline_seconds == 9 * 3600


def test_aggregate_headroom_cannot_consume_evaluation_reserve():
    value = ledger()
    value["gpu_hours_spent_owned_scientific_jobs"] = 83
    assert limits(value).deadline_seconds == 3600
    value["gpu_hours_spent_owned_scientific_jobs"] = 84
    with pytest.raises(ValueError, match="budget"):
        limits(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_limit_hours", 100),
        ("gpu_limit_hours", True),
        ("gpu_hours_spent_owned_scientific_jobs", float("nan")),
        ("gpu_hours_spent_owned_scientific_jobs", -1),
        ("gpu_hours_spent_owned_scientific_jobs", True),
    ],
)
def test_no_budget_reset_or_nonfinite_accounting(field, value):
    record = ledger()
    record[field] = value
    with pytest.raises((ValueError, TypeError)):
        limits(record)


@pytest.mark.parametrize(
    "change",
    [
        "active",
        "unreconciled",
        "wrong_family",
        "missing_elapsed",
        "disagreeing_elapsed",
        "unverified_cleanup",
        "unaccounted_total",
        "exhausted",
    ],
)
def test_incomplete_or_ambiguous_owned_accounting_fails_closed(change):
    value = ledger()
    value["runs"] = [attempt(3600)]
    record = value["runs"][0]
    if change == "active":
        record["status"] = "RUNNING_CUDA"
    elif change == "unreconciled":
        record["requires_reconciliation"] = True
    elif change == "wrong_family":
        record["budget_family"] = "other"
    elif change == "missing_elapsed":
        del record["elapsed_owned_seconds"]
    elif change == "disagreeing_elapsed":
        record["elapsed_owned_seconds"] = 4000
    elif change == "unverified_cleanup":
        record["resources_full_attempt"]["owned_tree_cleanup_verified"] = False
    elif change == "unaccounted_total":
        value["gpu_hours_spent_owned_scientific_jobs"] = 0
    elif change == "exhausted":
        value["runs"] = [attempt(12 * 3600)]
    with pytest.raises((ValueError, TypeError)):
        limits(value)


def test_other_studies_are_not_recharged_or_modified():
    value = ledger()
    value["runs"] = [
        {
            "id": "band_shared_ssl",
            "budget_family": "native_band_v1",
            "status": "COMPLETED_REAL_BAND_CUDA",
            "elapsed_owned_seconds": 123,
        }
    ]
    before = deepcopy(value)
    assert limits(value).extension_hours == 0
    assert value == before
