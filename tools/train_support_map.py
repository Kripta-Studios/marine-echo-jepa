"""Read-only complete TRAIN count-support map; never reads acoustic Sv arrays."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_DAYS = np.arange("2020-02-17", "2020-05-27", dtype="datetime64[D]").astype(str).tolist()
_FREQUENCIES = (38000, 125000, 200000, 455000)
_EDGES = np.arange(0, 130, 2)
_CONTRACT = "evidence/continuation/train_support_map_contract.json"
_REVIEW = "orchestration/reviews/TRAIN_SUPPORT_MAP_20260927.json"
_EXECUTION = "evidence/continuation/train_census_v2_execution.json"
_STRICT = "evidence/continuation/train_census_v2_eligibility.json"
_TARGET = "evidence/continuation/target_only_bound.json"
_TARGET_METHOD = "evidence/continuation/target_only_bound_contract.json"
_TARGET_INPUT = "evidence/continuation/target_only_bound_v2_input_contract.json"
_OUTPUT = "evidence/continuation/train_support_map.json"


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
    data = json.loads(_local_file(root, relative).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object: {relative}")
    return data


def summarize_support(valid: np.ndarray, effective: np.ndarray, days: list[str]) -> dict[str, Any]:
    """Summarize every fixed cell without selecting a frequency or range band."""
    if (
        days != _DAYS
        or valid.shape != (100, 96, 4, 64)
        or effective.shape != (100, 96)
        or valid.dtype.kind not in "iu"
        or effective.dtype.kind not in "iu"
    ):
        raise ValueError("Exact TRAIN calendar and integer native-grid counts required")
    numerator = valid.astype(np.int64)
    denominator = effective.astype(np.int64)
    if (
        np.any(denominator <= 0)
        or np.any(numerator < 0)
        or np.any(numerator > denominator[:, :, None, None])
    ):
        raise ValueError("Corrupt valid/effective ping counts")
    daily_valid = numerator.sum(axis=1)
    daily_effective = denominator.sum(axis=1)
    daily_fraction = daily_valid / daily_effective[:, None, None]
    sufficient = numerator >= (denominator - denominator // 5)[:, :, None, None]
    total_valid = daily_valid.sum(axis=0)
    total_effective = int(daily_effective.sum())
    bins_at_80 = sufficient.sum(axis=(0, 1))
    days_with_any = sufficient.any(axis=1).sum(axis=0)
    daily = [
        {
            "date": day,
            "valid_ping_count": daily_valid[index].tolist(),
            "effective_ping_count": np.full((4, 64), daily_effective[index]).tolist(),
        }
        for index, day in enumerate(days)
    ]
    cells = [
        {
            "frequency_hz": frequency,
            "range_start_m": int(_EDGES[range_index]),
            "range_end_m": int(_EDGES[range_index + 1]),
            "valid_ping_count": int(total_valid[channel, range_index]),
            "effective_ping_count": total_effective,
            "support_fraction": float(total_valid[channel, range_index] / total_effective),
            "quarter_hours_at_least_80_pct": int(bins_at_80[channel, range_index]),
            "utc_days_with_any_quarter_hour_at_least_80_pct": int(
                days_with_any[channel, range_index]
            ),
            "maximum_daily_support_fraction": float(daily_fraction[:, channel, range_index].max()),
            "median_daily_support_fraction": float(
                np.median(daily_fraction[:, channel, range_index])
            ),
        }
        for channel, frequency in enumerate(_FREQUENCIES)
        for range_index in range(64)
    ]
    return {
        "processed_train_calendar_days": 100,
        "quarter_hours_per_day": 96,
        "frequency_hz": list(_FREQUENCIES),
        "range_edges_m": _EDGES.tolist(),
        "cells": cells,
        "daily": daily,
    }


def _validated_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Require the separately reviewed, complete v2 census and target-only lineage."""
    contract = _json(root, _CONTRACT)
    review = _json(root, _REVIEW)
    if (
        contract.get("id") != "complete-train-native-grid-support-map-v1"
        or review.get("reviewer_session") != "/root/continuation_review"
        or review.get("disposition") != "APPROVE_TRAIN_SUPPORT_MAP_METHOD"
        or review.get("reviewed_contract_sha256") != _sha(_local_file(root, _CONTRACT))
        or review.get("reviewed_code_sha256") != _sha(Path(__file__))
    ):
        raise ValueError("Independent exact support-map method review required")
    from target_only_bound import validate_lineage
    from train_census_v2 import previous_execution, validate_ledger

    execution = _json(root, _EXECUTION)
    strict = _json(root, _STRICT)
    target = _json(root, _TARGET)
    input_contract = _json(root, _TARGET_INPUT)
    identity = execution.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("bindings"), dict):
        raise TypeError("Invalid v2 census identity")
    bindings = identity["bindings"]
    validate_ledger(execution, identity, _DAYS)
    if execution.get("status") != "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK":
        raise ValueError("Complete v2 TRAIN census required")
    if identity.get("calendar") != _DAYS or len(execution["days"]) != 100:
        raise ValueError("Exact original 100-day TRAIN calendar required")
    source_bindings = (
        ("driver_sha256", "tools/train_census_v2.py", identity),
        ("code_sha256", "tools/preprocess_train_candidate.py", bindings),
        ("contract_sha256", "evidence/continuation/train_census_v2_contract.json", bindings),
        ("compat_code_sha256", "src/marine_echo/data/azfp_compat.py", bindings),
        ("qc_code_sha256", "src/marine_echo/data/candidate_qc.py", bindings),
        ("regrid_code_sha256", "src/marine_echo/data/range_grid.py", bindings),
        ("protocol_sha256", "reports/active/protocol.json", bindings),
        (
            "method_review_sha256",
            "orchestration/reviews/TRAIN_CENSUS_METHOD_20260927.json",
            identity,
        ),
        (
            "singleton_review_sha256",
            "orchestration/reviews/AZFP_SINGLETON_COMPAT_20260927.json",
            identity,
        ),
        (
            "prior_failed_execution_sha256",
            "evidence/continuation/train_census_execution.json",
            identity,
        ),
    )
    for field, relative, document in source_bindings:
        if document.get(field) != _sha(_local_file(root, relative)):
            raise ValueError(f"V2 census source binding differs: {field}")
    if (
        input_contract.get("execution_path") != _EXECUTION
        or input_contract.get("strict_report_path") != _STRICT
        or input_contract.get("driver_sha256") != identity["driver_sha256"]
        or input_contract.get("preprocessor_sha256") != bindings["code_sha256"]
        or input_contract.get("census_contract_sha256") != bindings["contract_sha256"]
        or input_contract.get("compat_code_sha256") != bindings["compat_code_sha256"]
        or input_contract.get("method_contract_sha256") != _sha(_local_file(root, _TARGET_METHOD))
    ):
        raise ValueError("Original target-only v2 input contract differs")
    previous = previous_execution()
    validate_lineage(
        identity,
        strict,
        _sha(_local_file(root, "evidence/continuation/train_census_execution.json")),
        _sha(_local_file(root, "orchestration/reviews/TRAIN_CENSUS_METHOD_20260927.json")),
    )
    if (
        strict.get("execution_report_sha256") != _sha(_local_file(root, _EXECUTION))
        or strict.get("method_identity") != identity
        or type(strict.get("processed_train_calendar_days")) is not int
        or strict["processed_train_calendar_days"] != 100
        or type(strict.get("unresolved_train_days")) is not int
        or strict["unresolved_train_days"] != 0
        or target.get("census_generation") != "census-v2"
        or target.get("status")
        not in ("D1_INELIGIBLE_TARGET_SUPPORT_UPPER_BOUND", "NO_GLOBAL_INELIGIBILITY_CONCLUSION")
        or target.get("contract_sha256") != _sha(_local_file(root, _TARGET_METHOD))
        or target.get("input_contract_sha256") != _sha(_local_file(root, _TARGET_INPUT))
        or target.get("code_sha256") != _sha(_local_file(root, "tools/target_only_bound.py"))
        or target.get("execution_report_sha256") != _sha(_local_file(root, _EXECUTION))
        or target.get("strict_context_report_sha256") != _sha(_local_file(root, _STRICT))
        or target.get("held_out_acoustic_payloads_processed") is not False
        or type(target.get("completed_benchmark_runs")) is not int
        or target["completed_benchmark_runs"] != 0
        or target.get("benchmark_eligible") is not False
        or type(target.get("processed_train_calendar_days")) is not int
        or target["processed_train_calendar_days"] != 100
        or type(target.get("unmeasured_nontrain_day_upper_bound")) is not int
        or target["unmeasured_nontrain_day_upper_bound"] != 67
        or type(target.get("minimum_overall_days")) is not int
        or target["minimum_overall_days"] != 90
    ):
        raise ValueError("Completed original target-only v2 result required")
    return execution, target, previous


def audit_support_map(root: Path) -> dict[str, Any]:
    """Read verified TRAIN count fields only and return a fixed-order QC report."""
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise ValueError("Unlinked local repository root required")
    root = root.resolve(strict=True)
    execution, target, previous = _validated_evidence(root)
    from target_only_bound import target_only_cutoffs
    from train_census import paths as old_paths
    from train_census import verify_completed as verify_old_completed
    from train_census_v2 import paths, verify_completed

    bindings = execution["identity"]["bindings"]
    valid_days: list[np.ndarray] = []
    effective_days: list[np.ndarray] = []
    artifact_hashes: dict[str, str] = {}
    for day in _DAYS:
        entry = execution["days"][day]
        verify_completed(root, day, entry, bindings)
        manifest, configuration, shard = paths(root, day)
        for path in (manifest, configuration, shard):
            relative = path.relative_to(root).as_posix()
            _local_file(root, relative)
            artifact_hashes[relative] = _sha(path)
        old_entry = previous["days"].get(day)
        if old_entry is not None and old_entry["exit_code"] == 0:
            verify_old_completed(root, day, old_entry, previous["identity"]["bindings"])
            old_shard = old_paths(root, day)[2]
            equivalence_path = f"evidence/continuation/census-v2-equivalence/{day}.json"
            equivalence = _json(root, equivalence_path)
            if (
                equivalence.get("status") != "EXACT_ARRAY_EQUIVALENCE"
                or equivalence.get("date") != day
                or equivalence.get("previous_shard_sha256") != _sha(old_shard)
                or equivalence.get("new_shard_sha256")
                != artifact_hashes[shard.relative_to(root).as_posix()]
                or not isinstance(equivalence.get("fields"), list)
                or "linear_sv" not in equivalence["fields"]
                or "valid_ping_count" not in equivalence["fields"]
            ):
                raise ValueError("Preserved v1/v2 equivalence evidence differs")
            artifact_hashes[equivalence_path] = _sha(_local_file(root, equivalence_path))
        with np.load(shard, allow_pickle=False) as data:
            valid = data["valid_ping_count"]
            expected = data["expected_ping_count"]
            observed = data["observed_ping_count"]
            effective = data["support_denominator_ping_count"]
            starts = data["bin_start"]
            frequency = data["frequency_hz"]
            edges = data["range_edges_m"]
        if (
            valid.shape != (96, 4, 64)
            or expected.shape != (96,)
            or observed.shape != (96,)
            or effective.shape != (96,)
            or any(array.dtype.kind not in "iu" for array in (valid, expected, observed, effective))
            or not np.array_equal(effective, np.maximum(expected, observed))
            or np.any(expected < 0)
            or np.any(observed < 0)
            or not np.array_equal(frequency, _FREQUENCIES)
            or not np.array_equal(edges, _EDGES)
            or not np.array_equal(
                starts, np.datetime64(day, "ns") + np.arange(96) * np.timedelta64(15, "m")
            )
        ):
            raise ValueError("Unexpected TRAIN count, denominator, time, or grid schema")
        valid_days.append(valid)
        effective_days.append(effective)
    valid_all = np.stack(valid_days)
    effective_all = np.stack(effective_days)
    summary = summarize_support(valid_all, effective_all, _DAYS)
    primary = valid_all[:, :, 0, 5:50].reshape(-1, 45)
    cutoffs = target_only_cutoffs(primary, effective_all.reshape(-1))
    anchor_days = sorted({_DAYS[index // 96] for index in cutoffs})
    maximum = len(anchor_days) + 67
    if (
        type(target.get("target_supported_train_anchor_days_upper_bound")) is not int
        or target["target_supported_train_anchor_days_upper_bound"] != len(anchor_days)
        or type(target.get("target_supported_train_hourly_anchors_upper_bound")) is not int
        or target["target_supported_train_hourly_anchors_upper_bound"] != len(cutoffs)
        or target.get("target_supported_dates") != anchor_days
        or type(target.get("overall_eligible_day_upper_bound")) is not int
        or target["overall_eligible_day_upper_bound"] != maximum
        or target.get("status")
        != (
            "D1_INELIGIBLE_TARGET_SUPPORT_UPPER_BOUND"
            if maximum < 90
            else "NO_GLOBAL_INELIGIBILITY_CONCLUSION"
        )
    ):
        raise ValueError("Original target-only result differs from verified TRAIN counts")
    return {
        "status": "TRAIN_QC_ONLY_NOT_BENCHMARK",
        "scope": "Complete original TRAIN count-support grid; no selected band or target amendment",
        "census_generation": "census-v2",
        "contract_sha256": _sha(_local_file(root, _CONTRACT)),
        "review_sha256": _sha(_local_file(root, _REVIEW)),
        "code_sha256": _sha(Path(__file__)),
        "execution_report_sha256": _sha(_local_file(root, _EXECUTION)),
        "strict_context_report_sha256": _sha(_local_file(root, _STRICT)),
        "original_target_only_report_sha256": _sha(_local_file(root, _TARGET)),
        "artifact_sha256": artifact_hashes,
        "held_out_acoustic_payloads_processed": False,
        "completed_benchmark_runs": 0,
        "benchmark_eligible": False,
        **summary,
    }


def main() -> int:
    result = audit_support_map(ROOT)
    path = ROOT / _OUTPUT
    if path.parent.is_symlink() or path.parent.is_junction() or not path.parent.is_dir():
        raise ValueError("Unlinked existing evidence directory required")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "path": _OUTPUT}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
