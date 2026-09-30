"""Account for closed training, assessment and prefix operations in one Band allowance."""

import math

FAMILY = "native_band_v1"
ARCHITECTURE = "nonlinear_frequency_conditioned_v1"
CLOSED = {"COMPLETED_REAL_BAND_CUDA", "FAILED_REAL_BAND_CUDA_ATTEMPT", "COMPLETED_REAL_CUDA",
          "FAILED_REAL_CUDA_ATTEMPT", "COMPLETED_REAL_PREFIX_TRANSFER", "PREFIX_TRANSFER_NOT_ASSESSABLE",
          "PREFIX_TRANSFER_FAILED_OR_BLOCKED"}


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Finite nonnegative {name} required")
    return float(value)


def budget_totals(ledger):
    aggregate = number(ledger.get("gpu_hours_spent_owned_scientific_jobs"), "aggregate owned GPU hours")
    if ledger.get("gpu_limit_hours") != 96 or not isinstance(ledger.get("runs"), list):
        raise ValueError("Complete aggregate96-hour ledger required")
    band_seconds = 0.0
    for record in ledger["runs"]:
        if not isinstance(record, dict):
            raise TypeError("Exact operation record required")
        if str(record.get("status", "")).startswith("RUNNING_") or record.get("requires_reconciliation"):
            raise ValueError("Active or unreconciled operation blocks the next process")
        signature = (record.get("architecture") == ARCHITECTURE
                     or str(record.get("method", "")).startswith("band_")
                     or "native_band" in str(record.get("config", "")).lower()
                     or any("native_band" in str(value) for value in record.get("command", [])))
        if signature and record.get("budget_family") != FAMILY:
            raise ValueError("Unaccounted Band operation")
        if record.get("budget_family") != FAMILY:
            continue
        if record.get("status") not in CLOSED:
            raise ValueError("Unknown Band operation status")
        full = record.get("resources_full_attempt", {}).get("elapsed_full_attempt_seconds")
        seconds = number(record.get("elapsed_owned_seconds", full), "full owned Band seconds")
        if full is not None and not math.isclose(seconds, number(full, "supervised full time"), rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Conflicting full-owned elapsed counters")
        band_seconds += seconds
    if aggregate * 3600 + 1e-6 < band_seconds:
        raise ValueError("Aggregate undercounts full-owned Band operations")
    return aggregate, band_seconds / 3600, (12 - band_seconds / 3600) * 3600
