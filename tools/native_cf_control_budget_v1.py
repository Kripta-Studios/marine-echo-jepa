"""Separate CF controls accounting inside the original aggregate budget."""

import math
from dataclasses import dataclass

FAMILY = "native_cf_matched_controls_v1"
STUDY = "CF_MATCHED_CONTROLS"
CONTROLS = ("cf_random_frozen", "cf_direct_supervised")
ACTIVE = {"RUNNING_CUDA", "RUNNING_CPU_FIT"}
CLOSED = {"COMPLETED_REAL_CF_CONTROL", "FAILED_REAL_CF_CONTROL_ATTEMPT"}


@dataclass(frozen=True)
class Allowance:
    aggregate_hours: float
    extension_hours: float
    deadline_seconds: float
    evaluation_reserve_hours: int = 12


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Explicit numeric {name} required")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Finite nonnegative {name} required")
    return float(value)


def limits(ledger):
    """No caller-selectable caps, reset, mutation or exclusion of failed attempts."""
    if type(ledger.get("gpu_limit_hours")) is not int or ledger["gpu_limit_hours"] != 96:
        raise ValueError("Original aggregate budget must remain exactly 96 hours")
    aggregate = number(ledger.get("gpu_hours_spent_owned_scientific_jobs"), "aggregate hours")
    runs = ledger.get("runs")
    if not isinstance(runs, list) or any(not isinstance(r, dict) for r in runs):
        raise TypeError("Complete run records required")
    if ledger.get("requires_reconciliation") or any(
        r.get("status") in ACTIVE or r.get("requires_reconciliation") for r in runs
    ):
        raise ValueError("Active or unreconciled scientific owner; no control launch")
    seconds = 0.0
    for record in runs:
        signature = (
            record.get("study") == STUDY
            or record.get("budget_family") == FAMILY
            or record.get("method") in CONTROLS
            or any(str(record.get("id", "")).startswith(c) for c in CONTROLS)
        )
        if not signature:
            continue
        if record.get("budget_family") != FAMILY or record.get("study") != STUDY:
            raise ValueError("CF control attempt missing explicit accounting family/study")
        if record.get("status") not in CLOSED:
            raise ValueError("Unclosed CF control attempt requires reconciliation")
        owned = number(record.get("elapsed_owned_seconds"), "owned attempt seconds")
        resources = record.get("resources_full_attempt", {})
        if resources.get("owned_tree_cleanup_verified") is not True:
            raise ValueError("Owned cleanup witness required for every closed attempt")
        observed = number(resources.get("elapsed_full_attempt_seconds"), "full attempt seconds")
        if owned != observed:
            raise ValueError("Full-owned accounting fields disagree")
        seconds += owned
    extension = seconds / 3600
    if aggregate < extension:
        raise ValueError("Extension time must already be charged to aggregate budget")
    remaining = min(12 - extension, 96 - aggregate - 12) * 3600
    if remaining <= 0:
        raise ValueError("No control budget remains after the evaluation reserve")
    return Allowance(aggregate, extension, remaining)
