"""Audit only counts/configuration for the prospectively fixed raw-code route."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from marine_echo.data.raw_response_support import audit_raw_partition

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "evidence/v2/raw-response-development-reviewed/index.json"
OUTPUT = ROOT / "evidence/v2/raw-response-support"
STUDY = "raw_response_development_v1"
RUN_ID = "raw_response_full_v2_20260927"
REFERENCE_CONFIG = "f90a1a2681898ad2c6dabb5eded4c7fa4c5f5f154aefd10b79dbc82629c74860"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_review(review: dict, index: dict) -> None:
    if (
        review.get("disposition") != "APPROVE_RAW_RESPONSE_SUPPORT_METHOD"
        or review.get("reviewer_session") != "/root/v2_reviewer"
        or review.get("driver_sha256") != sha(Path(__file__))
        or review.get("audit_sha256") != sha(ROOT / "src/marine_echo/data/raw_response_support.py")
        or review.get("contract_sha256") != sha(ROOT / "docs/adr/0007-raw-instrument-response-development.md")
        or review.get("processing_run_id") != RUN_ID
        or review.get("reference_configuration") != REFERENCE_CONFIG
        or review.get("index_sha256") != sha(INDEX)
        or review.get("bindings") != index.get("bindings")
        or index.get("data_kind") != "REAL"
        or index.get("study_id") != STUDY
        or index.get("processing_run_id") != RUN_ID
        or index.get("bindings", {}).get("contract") != review["contract_sha256"]
    ):
        raise ValueError("Exact independent raw-response support-method review required")


def validate_day_counts(
    day: str,
    document: dict,
    observed: np.ndarray,
    valid: np.ndarray,
    zero: np.ndarray,
    zero_affected: np.ndarray,
    nonfinite: np.ndarray,
    zero_samples: np.ndarray,
    nonfinite_samples: np.ndarray,
    expected: np.ndarray,
    starts: np.ndarray,
    times: np.ndarray,
    target_code_count: np.ndarray,
) -> None:
    first = np.datetime64(day, "ns")
    end = first + np.timedelta64(1, "D")
    if (
        any(x.shape != (96,) for x in (
            observed, valid, zero, zero_affected, nonfinite,
            zero_samples, nonfinite_samples, expected, starts, target_code_count,
        ))
        or (observed < 0).any()
        or (valid + zero + nonfinite != observed).any()
        or (zero_affected < zero).any()
        or (zero_affected > observed - valid).any()
        or (zero_samples < 0).any()
        or (nonfinite_samples < 0).any()
        or not np.array_equal(target_code_count, valid * 180)
        or not np.array_equal(expected, np.full(96, 60))
        or not np.array_equal(starts, first + np.arange(96) * np.timedelta64(15, "m"))
        or len(times) != int(observed.sum())
        or np.isnat(times).any()
        or (np.diff(times) <= np.timedelta64(0, "ns")).any()
        or (times < first).any()
        or (times >= end).any()
        or document.get("observed_pings") != int(observed.sum())
        or document.get("valid_target_pings") != int(valid.sum())
        or document.get("zero_or_undefined_pings") != int(zero.sum())
        or document.get("zero_affected_pings") != int(zero_affected.sum())
        or document.get("zero_affected_quarter_hours") != int((zero_affected > 0).sum())
        or document.get("nonfinite_pings") != int(nonfinite.sum())
        or document.get("zero_or_undefined_samples") != int(zero_samples.sum())
        or document.get("nonfinite_samples") != int(nonfinite_samples.sum())
    ):
        raise ValueError("Raw-response count/time/manifest identity differs")
    quarter = ((times - first) / np.timedelta64(15, "m")).astype(int)
    if not np.array_equal(np.bincount(quarter, minlength=96), observed):
        raise ValueError("Raw-response timestamp histogram differs from observed pings")


def load_count_hours(index: dict) -> tuple[list[dict[str, Any]], dict[str, str]]:
    expected_days = [(date(2020, 2, 17) + timedelta(days=i)).isoformat() for i in range(58)]
    if list(index.get("days", {})) != expected_days:
        raise ValueError("Exact fixed 58-day raw-response calendar required")
    hours: list[dict[str, Any]] = []
    hashes = {str(INDEX.relative_to(ROOT)): sha(INDEX)}
    for day, entry in index["days"].items():
        manifest_relative = f"evidence/v2/raw-response-development-reviewed/{day}.json"
        if Path(entry.get("manifest_path", "")).as_posix() != manifest_relative:
            raise ValueError("Raw-response manifest path differs")
        manifest = ROOT / manifest_relative
        if sha(manifest) != entry.get("manifest_sha256"):
            raise ValueError("Raw-response manifest hash differs")
        document = json.loads(manifest.read_text())
        shard_relative = f"data/processed/raw-response-development-reviewed/{day}.npz"
        if (
            document.get("data_kind") != "REAL"
            or document.get("study_id") != STUDY
            or document.get("processing_run_id") != RUN_ID
            or document.get("day") != day
            or document.get("bindings") != index["bindings"]
            or Path(document.get("shard_path", "")).as_posix() != shard_relative
            or document.get("shard_sha256") != entry.get("shard_sha256")
        ):
            raise ValueError("Raw-response day identity differs")
        old = ROOT / (
            f"evidence/continuation/train-candidate-{day.replace('-', '')}-census-v2/processing_manifest.json"
        )
        if sha(old) != document.get("historical_manifest_sha256"):
            raise ValueError("Historical source manifest digest differs")
        prior = json.loads(old.read_text())
        keys = ("source", "source_sha256", "configuration_sha256", "pings_in_day", "first_utc", "last_utc")
        if [tuple(x[k] for k in keys) for x in document["source_hours"]] != [
            tuple(x[k] for k in keys) for x in prior["source_hours"]
        ]:
            raise ValueError("Raw-response source list differs from historical record")
        shard = ROOT / shard_relative
        if sha(shard) != document["shard_sha256"]:
            raise ValueError("Raw-response shard hash differs")
        hashes[manifest_relative] = sha(manifest)
        hashes[shard_relative] = sha(shard)
        with np.load(shard, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["frequency_hz"], [38000, 125000, 200000, 455000])
            np.testing.assert_array_equal(data["averaged_bin_index"], np.arange(20, 200))
            observed = data["observed_ping_count"]
            valid = data["valid_target_ping_count"]
            zero = data["zero_or_undefined_ping_count"]
            zero_affected = data["zero_affected_ping_count"]
            nonfinite = data["nonfinite_ping_count"]
            zero_samples = data["zero_or_undefined_sample_count"]
            nonfinite_samples = data["nonfinite_sample_count"]
            config = data["configuration_id"]
            starts = data["bin_start"]
            times = data["raw_ping_time"]
            expected = data["expected_ping_count"]
            target_code_count = data["target_code_count"]
            if (
                config.shape != (96,)
            ):
                raise ValueError("Raw-response configuration shape differs")
            validate_day_counts(
                day, document, observed, valid, zero, zero_affected, nonfinite,
                zero_samples, nonfinite_samples, expected, starts, times, target_code_count,
            )
            for j in range(0, 96, 4):
                ids = {str(config[k]) for k in range(j, j + 4) if observed[k] > 0}
                identity = next(iter(ids)) if len(ids) == 1 else ("MIXED" if ids else None)
                hours.append({
                    "time": str(starts[j].astype("datetime64[h]")) + ":00:00Z",
                    "observed": int(observed[j : j + 4].sum()),
                    "valid": int(valid[j : j + 4].sum()),
                    "zero_affected_pings": int(zero_affected[j : j + 4].sum()),
                    "zero_samples": int(zero_samples[j : j + 4].sum()),
                    "nonfinite_pings": int(nonfinite[j : j + 4].sum()),
                    "nonfinite_samples": int(nonfinite_samples[j : j + 4].sum()),
                    "configuration": identity,
                })
    if len(hours) != 58 * 24:
        raise ValueError("Raw-response full calendar did not yield 1392 hours")
    return hours, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(INDEX.read_text())
    review = json.loads(args.review.read_text())
    validate_review(review, index)
    hours, hashes = load_count_hours(index)
    OUTPUT.mkdir(exist_ok=False)
    partitions = {}
    for name, start, end, minimum_days in (
        ("development_fit", "2020-02-17", "2020-04-01", 20),
        ("development_assessment", "2020-04-01", "2020-04-15", 5),
    ):
        result = audit_raw_partition(
            [hour for hour in hours if start <= hour["time"] < end],
            reference_configuration=REFERENCE_CONFIG,
            block_origin=start + "T00:00:00Z",
        )
        rows = result.pop("rows")
        row_path = OUTPUT / f"{name}_rows.json"
        with row_path.open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2)
            stream.write("\n")
        adequate = all(
            item["target_days"] >= minimum_days
            and (name != "development_assessment" or item["populated_48h_blocks"] >= 5)
            for item in result["horizons"].values()
        )
        partitions[name] = {
            "start": start,
            "end": end,
            "row_path": str(row_path.relative_to(ROOT)),
            "rows_sha256": sha(row_path),
            "minimum_target_days": minimum_days,
            "minimum_assessment_blocks_per_horizon": 5 if name == "development_assessment" else None,
            "adequate_all_horizons": adequate,
            **result,
        }
    report = {
        "status": "RAW_RESPONSE_DEVELOPMENT_ADEQUATE_PENDING_REVIEW"
        if all(item["adequate_all_horizons"] for item in partitions.values())
        else "RAW_RESPONSE_DEVELOPMENT_INELIGIBLE_STOP",
        "study_id": STUDY,
        "processing_run_id": RUN_ID,
        "quantity": "complete_positive_azfp_backscatter_r_code_mean; not calibrated Sv",
        "partitions": partitions,
        "artifact_sha256": hashes,
        "contract_sha256": sha(ROOT / "docs/adr/0007-raw-instrument-response-development.md"),
        "driver_sha256": sha(Path(__file__)),
        "audit_sha256": sha(ROOT / "src/marine_echo/data/raw_response_support.py"),
        "model_scores_computed": False,
        "non_train_acoustic_payloads_read": False,
    }
    report_path = OUTPUT / "eligibility_raw_response.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "partitions": partitions}, indent=2))


if __name__ == "__main__":
    main()
