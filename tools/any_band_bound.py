"""TRAIN-only necessary support bound for any fixed 38 kHz range band."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_DAYS = np.arange("2020-02-17", "2020-05-27", dtype="datetime64[D]").astype(str).tolist()
_FREQUENCIES = (38000, 125000, 200000, 455000)
_EDGES = np.arange(0, 130, 2)
_OFFSETS = (0, 8, 20)
_CONTRACT = "evidence/continuation/any_band_bound_contract.json"
_REVIEW = "orchestration/reviews/ANY_BAND_BOUND_METHOD_20260927.json"
_MAP_REPORT = "evidence/continuation/train_support_map.json"
_MAP_ACCEPTANCE = "orchestration/reviews/TRAIN_SUPPORT_MAP_RESULT_20260927.json"
_EXECUTION = "evidence/continuation/train_census_v2_execution.json"
_OUTPUT = "evidence/continuation/any_band_bound.json"


def _local_file(root: Path, relative: str) -> Path:
    path = root / relative
    current = path
    while current != root:
        if current.is_symlink() or current.is_junction():
            raise ValueError(f"Linked evidence path: {relative}")
        current = current.parent
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(root: Path, relative: str) -> dict[str, Any]:
    document = json.loads(_local_file(root, relative).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"Expected JSON object: {relative}")
    return document


def any_band_cutoffs(valid: np.ndarray, effective: np.ndarray) -> list[int]:
    """Return hourly cutoffs passing an intentionally generous per-horizon cell bound."""
    if (
        valid.shape != (100 * 96, 64)
        or effective.shape != (100 * 96,)
        or valid.dtype.kind not in "iu"
        or effective.dtype.kind not in "iu"
    ):
        raise ValueError("Exact integer 100-day 38 kHz count grid required")
    counts = valid.astype(np.int64)
    denominator = effective.astype(np.int64)
    if np.any(denominator <= 0) or np.any(counts < 0) or np.any(counts > denominator[:, None]):
        raise ValueError("Corrupt valid/effective ping counts")
    cutoffs: list[int] = []
    for cutoff in range(0, 100 * 96 - 24 + 1, 4):
        for offset in _OFFSETS:
            cell_counts = counts[cutoff + offset : cutoff + offset + 4].sum(axis=0)
            available = int(denominator[cutoff + offset : cutoff + offset + 4].sum())
            if not np.any(cell_counts >= available - available // 5):
                break
        else:
            cutoffs.append(cutoff)
    return cutoffs


def cutoff_dates(cutoffs: list[int]) -> list[str]:
    if any(
        type(value) is not int or value < 0 or value > 100 * 96 - 24 or value % 4
        for value in cutoffs
    ):
        raise ValueError("Invalid hourly TRAIN cutoff")
    return sorted({_DAYS[value // 96] for value in cutoffs})


def bound_disposition(train_days: int) -> tuple[str, int]:
    if type(train_days) is not int or not 0 <= train_days <= 100:
        raise ValueError("Invalid TRAIN cutoff-date count")
    overall_upper = train_days + 67
    return (
        "D1_INELIGIBLE_ANY_38KHZ_BAND_SUPPORT_UPPER_BOUND"
        if overall_upper < 90
        else "NO_GLOBAL_INELIGIBILITY_CONCLUSION",
        overall_upper,
    )


def validate_day_counts(
    day: str,
    valid: np.ndarray,
    expected: np.ndarray,
    observed: np.ndarray,
    effective: np.ndarray,
    starts: np.ndarray,
    frequency: np.ndarray,
    edges: np.ndarray,
) -> None:
    """Check all native-grid count provenance before taking the 38 kHz slice."""
    if (
        day not in _DAYS
        or valid.shape != (96, 4, 64)
        or expected.shape != (96,)
        or observed.shape != (96,)
        or effective.shape != (96,)
        or any(array.dtype.kind not in "iu" for array in (valid, expected, observed, effective))
        or np.any(expected < 0)
        or np.any(observed < 0)
        or np.any(effective <= 0)
        or not np.array_equal(effective, np.maximum(expected, observed))
        or np.any(valid < 0)
        or np.any(valid > observed[:, None, None])
        or np.any(observed > effective)
        or not np.array_equal(frequency, _FREQUENCIES)
        or not np.array_equal(edges, _EDGES)
        or not np.array_equal(
            starts, np.datetime64(day, "ns") + np.arange(96) * np.timedelta64(15, "m")
        )
    ):
        raise ValueError("Invalid TRAIN observed/effective count, time, or fixed grid")


def verify_accepted_map(root: Path, recomputed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind the published complete map to a distinct result acceptance record."""
    published = _json(root, _MAP_REPORT)
    acceptance = _json(root, _MAP_ACCEPTANCE)
    if (
        acceptance.get("reviewer_session") != "/root/continuation_review"
        or acceptance.get("disposition") != "ACCEPT_TRAIN_SUPPORT_MAP_RESULT"
        or acceptance.get("support_map_sha256") != _sha(_local_file(root, _MAP_REPORT))
        or published.get("status") != "TRAIN_QC_ONLY_NOT_BENCHMARK"
        or published.get("census_generation") != "census-v2"
        or published.get("held_out_acoustic_payloads_processed") is not False
        or type(published.get("completed_benchmark_runs")) is not int
        or published["completed_benchmark_runs"] != 0
        or published.get("benchmark_eligible") is not False
        or type(published.get("processed_train_calendar_days")) is not int
        or published["processed_train_calendar_days"] != 100
    ):
        raise ValueError("Exact independently accepted complete TRAIN support map required")
    if recomputed is not None and published != recomputed:
        raise ValueError("Published support map differs from fully recomputed source lineage")
    return published


def audit_any_band_bound(root: Path) -> dict[str, Any]:
    """Reverify the accepted TRAIN map and read counts only for the fixed bound."""
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise ValueError("Unlinked local repository root required")
    root = root.resolve(strict=True)
    contract = _json(root, _CONTRACT)
    review = _json(root, _REVIEW)
    if (
        contract.get("id") != "train-any-fixed-38khz-band-necessary-bound-v1"
        or review.get("reviewer_session") != "/root/continuation_review"
        or review.get("disposition") != "APPROVE_ANY_BAND_BOUND_METHOD"
        or review.get("reviewed_contract_sha256") != _sha(_local_file(root, _CONTRACT))
        or review.get("reviewed_code_sha256") != _sha(Path(__file__))
    ):
        raise ValueError("Independent exact any-band method review required")
    verify_accepted_map(root)
    from train_support_map import audit_support_map

    recomputed = audit_support_map(root)
    verify_accepted_map(root, recomputed)
    from train_census_v2 import paths, verify_completed

    execution = _json(root, _EXECUTION)
    if (
        _sha(_local_file(root, _EXECUTION)) != recomputed.get("execution_report_sha256")
        or execution.get("status") != "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK"
        or execution.get("identity", {}).get("calendar") != _DAYS
        or list(execution.get("days", {})) != _DAYS
    ):
        raise ValueError("Completed v2 TRAIN execution differs from accepted support map")
    bindings = execution["identity"]["bindings"]
    valid_days: list[np.ndarray] = []
    effective_days: list[np.ndarray] = []
    for day in _DAYS:
        verify_completed(root, day, execution["days"][day], bindings)
        _, _, shard = paths(root, day)
        relative = shard.relative_to(root).as_posix()
        if _sha(_local_file(root, relative)) != recomputed["artifact_sha256"][relative]:
            raise ValueError("TRAIN shard changed after accepted support-map verification")
        with np.load(shard, allow_pickle=False) as data:
            valid = data["valid_ping_count"]
            expected = data["expected_ping_count"]
            observed = data["observed_ping_count"]
            effective = data["support_denominator_ping_count"]
            starts = data["bin_start"]
            frequency = data["frequency_hz"]
            edges = data["range_edges_m"]
        validate_day_counts(day, valid, expected, observed, effective, starts, frequency, edges)
        valid_days.append(valid[:, 0, :])
        effective_days.append(effective)
    if _sha(_local_file(root, _EXECUTION)) != recomputed["execution_report_sha256"]:
        raise ValueError("V2 TRAIN execution changed during audit")
    cutoffs = any_band_cutoffs(np.concatenate(valid_days), np.concatenate(effective_days))
    dates = cutoff_dates(cutoffs)
    status, upper = bound_disposition(len(dates))
    verify_accepted_map(root, recomputed)
    return {
        "status": status,
        "scope": "TRAIN_ONLY_ANY_FIXED_38KHZ_BAND_NECESSARY_SUPPORT_BOUND_NOT_BENCHMARK",
        "contract_sha256": _sha(_local_file(root, _CONTRACT)),
        "review_sha256": _sha(_local_file(root, _REVIEW)),
        "code_sha256": _sha(Path(__file__)),
        "support_map_sha256": _sha(_local_file(root, _MAP_REPORT)),
        "support_map_result_acceptance_sha256": _sha(_local_file(root, _MAP_ACCEPTANCE)),
        "execution_report_sha256": recomputed["execution_report_sha256"],
        "census_generation": "census-v2",
        "processed_train_calendar_days": 100,
        "frequency_hz": 38000,
        "range_cells": 64,
        "support_threshold": 0.8,
        "target_offsets_quarter_hours": list(_OFFSETS),
        "target_width_quarter_hours": 4,
        "anchor_stride_quarter_hours": 4,
        "any_band_supported_train_hourly_anchors_upper_bound": len(cutoffs),
        "any_band_supported_train_anchor_days_upper_bound": len(dates),
        "qualifying_cutoff_dates": dates,
        "unmeasured_nontrain_day_upper_bound": 67,
        "overall_eligible_day_upper_bound": upper,
        "minimum_overall_days": 90,
        "held_out_acoustic_payloads_processed": False,
        "completed_benchmark_runs": 0,
        "benchmark_eligible": False,
        "interpretation": "Necessary support bound only; different cells may satisfy different horizons and no band was selected",
    }


def main() -> int:
    result = audit_any_band_bound(ROOT)
    path = ROOT / _OUTPUT
    if path.parent.is_symlink() or path.parent.is_junction() or not path.parent.is_dir():
        raise ValueError("Unlinked existing evidence directory required")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".any_band_bound-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": result["status"], "path": _OUTPUT}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
