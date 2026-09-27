"""Stream reviewed TRAIN AZFP raw-response codes into a separate study namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import echopype as ep  # type: ignore[import-untyped]
import numpy as np
import psutil

from marine_echo.data.azfp_compat import open_raw_azfp_compat
from marine_echo.data.raw_response import aggregate_code_response

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"
STUDY = "raw_response_development_v1"
PROCESSING_RUN_ID = "raw_response_full_v2_20260927"
EVIDENCE = ROOT / "evidence/v2/raw-response-development-reviewed"
OUTPUT = ROOT / "data/processed/raw-response-development-reviewed"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def registered_source_path(name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Raw source name must be a single registered basename")
    path = (RAW / name).resolve(strict=True)
    if not path.is_relative_to(RAW.resolve(strict=True)):
        raise ValueError("Raw source escapes registered archive")
    return path


def bindings() -> dict[str, Any]:
    paths = {
        "contract": ROOT / "docs/adr/0007-raw-instrument-response-development.md",
        "processor": Path(__file__),
        "aggregation": ROOT / "src/marine_echo/data/raw_response.py",
        "compat": ROOT / "src/marine_echo/data/azfp_compat.py",
        "parser": Path(ep.__file__).parent / "convert/parse_azfp.py",
        "group_builder": Path(ep.__file__).parent / "convert/set_groups_azfp.py",
        "xml": RAW / "20021600.XML",
        "historical_ledger": ROOT / "evidence/continuation/train_census_v2_execution.json",
    }
    result: dict[str, Any] = {key: sha(path) for key, path in paths.items()}
    ds = [
        float(element.text or "")
        for element in ElementTree.parse(paths["xml"]).findall(
            ".//LogAcousticCoefficients/Frequencies/Frequency/DS"
        )
    ]
    if ds != [
        0.02300000004470,
        0.02290000021458,
        0.02319999970496,
        0.02309999987483,
    ]:
        raise ValueError("XML detector-slope coefficients differ from raw contract")
    result["detector_slope_ds"] = ds
    prior = json.loads(paths["historical_ledger"].read_text())["identity"]["bindings"]
    if result["xml"] != prior["xml_sha256"] or result["compat"] != prior["compat_code_sha256"]:
        raise ValueError("Preserved XML or compatibility parser changed")
    return result


def validate_review(review: dict, bound: dict, start: str, end_exclusive: str) -> None:
    if (
        review.get("disposition") != "APPROVE_RAW_RESPONSE_PROCESSING_IMPLEMENTATION"
        or review.get("reviewer_session") != "/root/v2_reviewer"
        or review.get("bindings") != bound
        or not review.get("approved_start", "9999") <= start
        or not end_exclusive <= review.get("approved_end_exclusive", "0000")
    ):
        raise ValueError("Exact independent raw-response processing review is required")


def validate_index_identity(journal: dict, bound: dict) -> None:
    if (
        journal.get("data_kind") != "REAL"
        or journal.get("study_id") != STUDY
        or journal.get("processing_run_id") != PROCESSING_RUN_ID
        or journal.get("bindings") != bound
        or not isinstance(journal.get("days"), dict)
    ):
        raise ValueError("Raw-response index identity or processing bindings differ")


def validate_existing_entry(day: str, entry: dict, bound: dict) -> None:
    expected_manifest = (EVIDENCE / f"{day}.json").relative_to(ROOT).as_posix()
    expected_shard = (OUTPUT / f"{day}.npz").relative_to(ROOT).as_posix()
    if Path(entry.get("manifest_path", "")).as_posix() != expected_manifest:
        raise ValueError("Existing raw-response manifest path differs")
    manifest_path = ROOT / expected_manifest
    if sha(manifest_path) != entry.get("manifest_sha256"):
        raise ValueError("Completed raw-response manifest changed")
    document = json.loads(manifest_path.read_text())
    if (
        document.get("data_kind") != "REAL"
        or document.get("study_id") != STUDY
        or document.get("processing_run_id") != PROCESSING_RUN_ID
        or document.get("day") != day
        or document.get("bindings") != bound
        or Path(document.get("shard_path", "")).as_posix() != expected_shard
        or entry.get("shard_sha256") != document.get("shard_sha256")
        or sha(ROOT / expected_shard) != entry.get("shard_sha256")
    ):
        raise ValueError("Completed raw-response shard or identity changed")


def ensure_no_orphans(day: str) -> None:
    if (EVIDENCE / f"{day}.json").exists() or (OUTPUT / f"{day}.npz").exists():
        raise RuntimeError("Unindexed raw-response output must be recovered, not overwritten")


def process_day(day: str, bound: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    old_path = ROOT / (
        f"evidence/continuation/train-candidate-{day.replace('-', '')}-census-v2/processing_manifest.json"
    )
    old = json.loads(old_path.read_text())
    ledger = json.loads((ROOT / "evidence/continuation/train_census_v2_execution.json").read_text())
    if sha(old_path) != ledger["days"][day]["manifest_sha256"]:
        raise ValueError("Preserved source manifest digest differs")
    start = np.datetime64(day, "ns")
    end = start + np.timedelta64(1, "D")
    profile_sum = np.zeros((96, 4, 64), dtype=np.float64)
    profile_count = np.zeros((96, 4, 64), dtype=np.int64)
    target_sum = np.zeros(96, dtype=np.float64)
    target_count = np.zeros(96, dtype=np.int64)
    observed = np.zeros(96, dtype=np.int64)
    valid = np.zeros(96, dtype=np.int64)
    zero_ping = np.zeros(96, dtype=np.int64)
    zero_affected_ping = np.zeros(96, dtype=np.int64)
    nonfinite_ping = np.zeros(96, dtype=np.int64)
    zero_samples = np.zeros(96, dtype=np.int64)
    nonfinite_samples = np.zeros(96, dtype=np.int64)
    config_ids: list[set[str]] = [set() for _ in range(96)]
    times_all: list[int] = []
    seen: set[int] = set()
    sources = []
    peak_rss = 0
    for source in old["source_hours"]:
        path = registered_source_path(source["source"])
        if sha(path) != source["source_sha256"]:
            raise ValueError("Raw source digest differs")
        ed = open_raw_azfp_compat(path, xml_path=RAW / "20021600.XML")
        beam, vendor = ed["Sonar/Beam_group1"], ed["Vendor_specific"]
        np.testing.assert_array_equal(
            beam.frequency_nominal.values, [38000, 125000, 200000, 455000]
        )
        times = beam.ping_time.values.astype("datetime64[ns]")
        if (
            np.isnat(times).any()
            or (np.diff(times) <= np.timedelta64(0, "ns")).any()
            or (times < start).any()
            or (times >= end).any()
        ):
            raise ValueError("Invalid source acquisition times")
        integers = set(times.astype("int64").tolist())
        if seen & integers:
            raise ValueError("Duplicate source ping timestamp")
        seen.update(integers)
        actual_config = {
            name: vendor[name].values.tolist()
            for name in (
                "digitization_rate",
                "lock_out_index",
                "number_of_bins_per_channel",
                "number_of_samples_per_average_bin",
                "data_type",
                "average_pings_flag",
            )
        }
        actual_config.update(
            transmit_duration_nominal=np.unique(beam.transmit_duration_nominal.values).tolist(),
            xml_sha256=bound["xml"],
            ping_period_s=15.0,
        )
        config_hash = hashlib.sha256(json.dumps(actual_config, sort_keys=True).encode()).hexdigest()
        if config_hash != source["configuration_sha256"]:
            raise ValueError("Raw source configuration differs from historical manifest")
        if (
            actual_config["digitization_rate"] != [20000] * 4
            or actual_config["lock_out_index"] != [0] * 4
            or actual_config["number_of_bins_per_channel"] != [999, 999, 391, 235]
            or actual_config["number_of_samples_per_average_bin"] != [14] * 4
            or actual_config["data_type"] != [1] * 4
            or actual_config["average_pings_flag"] != 0
            or actual_config["transmit_duration_nominal"] != [0.001]
        ):
            raise ValueError("Raw response contract configuration differs")
        codes = beam.backscatter_r.values
        if codes.shape[0] != 4 or codes.shape[1] != len(times):
            raise ValueError("Raw codes do not align with four-channel ping times")
        quarters = ((times - start) / np.timedelta64(15, "m")).astype(int)
        for quarter in np.unique(quarters):
            q = int(quarter)
            result = aggregate_code_response(codes[:, quarters == q, :])
            profile_sum[q] += result["profile_code_sum"]
            profile_count[q] += result["profile_code_count"]
            target_sum[q] += result["target_code_sum"]
            target_count[q] += result["target_code_count"]
            observed[q] += result["observed_pings"]
            valid[q] += result["valid_target_pings"]
            zero_ping[q] += result["zero_target_pings"]
            zero_affected_ping[q] += result["zero_affected_target_pings"]
            nonfinite_ping[q] += result["nonfinite_target_pings"]
            zero_samples[q] += result["zero_target_samples"]
            nonfinite_samples[q] += result["nonfinite_target_samples"]
            config_ids[q].add(config_hash)
        times_all.extend(times.astype("int64").tolist())
        sources.append(
            {
                key: source[key]
                for key in (
                    "source",
                    "source_sha256",
                    "configuration_sha256",
                    "pings_in_day",
                    "first_utc",
                    "last_utc",
                )
            }
        )
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        if peak_rss >= 22 * 1024**3:
            raise MemoryError("Process RAM limit reached")
        del ed, beam, vendor, codes
    times_all.sort()
    if len(times_all) != old["observed_pings"] or int(observed.sum()) != len(times_all):
        raise ValueError("Observed ping count differs from historical evidence")
    if (valid + zero_ping + nonfinite_ping != observed).any() or any(
        len(value) > 1 for value in config_ids
    ):
        raise ValueError("Raw QC categories or slot configuration are not disjoint")
    ids = np.array([next(iter(value)) if value else "" for value in config_ids], dtype="U64")
    path = OUTPUT / f"{day}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            bin_start=start + np.arange(96) * np.timedelta64(15, "m"),
            bin_end=start + np.arange(1, 97) * np.timedelta64(15, "m"),
            profile_code_sum=profile_sum,
            profile_code_count=profile_count,
            target_code_sum=target_sum,
            target_code_count=target_count,
            observed_ping_count=observed,
            valid_target_ping_count=valid,
            zero_or_undefined_ping_count=zero_ping,
            zero_affected_ping_count=zero_affected_ping,
            nonfinite_ping_count=nonfinite_ping,
            zero_or_undefined_sample_count=zero_samples,
            nonfinite_sample_count=nonfinite_samples,
            expected_ping_count=np.full(96, 60, dtype=np.int64),
            raw_ping_time=np.array(times_all, dtype="int64").astype("datetime64[ns]"),
            frequency_hz=np.array([38000, 125000, 200000, 455000]),
            averaged_bin_index=np.arange(20, 200),
            configuration_id=ids,
        )
        stream.flush()
        os.fsync(stream.fileno())
    document = {
        "data_kind": "REAL",
        "study_id": STUDY,
        "processing_run_id": PROCESSING_RUN_ID,
        "status": "RAW_TRAIN_DEVELOPMENT_PROCESSED_NOT_APPROVED_FOR_FITTING",
        "day": day,
        "shard_path": str(path.relative_to(ROOT)),
        "shard_sha256": sha(path),
        "bindings": bound,
        "source_hours": sources,
        "historical_manifest_sha256": sha(old_path),
        "observed_pings": len(times_all),
        "valid_target_pings": int(valid.sum()),
        "zero_or_undefined_pings": int(zero_ping.sum()),
        "zero_affected_pings": int(zero_affected_ping.sum()),
        "zero_affected_quarter_hours": int((zero_affected_ping > 0).sum()),
        "zero_or_undefined_samples": int(zero_samples.sum()),
        "nonfinite_pings": int(nonfinite_ping.sum()),
        "nonfinite_samples": int(nonfinite_samples.sum()),
        "elapsed_seconds": time.monotonic() - started,
        "peak_sampled_process_rss_gib": peak_rss / 1024**3,
        "quantity": "complete_positive_azfp_backscatter_r_code_mean; not calibrated Sv",
    }
    manifest = EVIDENCE / f"{day}.json"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    return {"manifest_path": str(manifest.relative_to(ROOT)), "manifest_sha256": sha(manifest), "shard_sha256": document["shard_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-02-17")
    parser.add_argument("--end-exclusive", default="2020-04-15")
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    if not "2020-02-17" <= args.start < args.end_exclusive <= "2020-04-15":
        raise ValueError("Only fixed TRAIN development dates are permitted")
    bound = bindings()
    review = json.loads(args.review.read_text())
    validate_review(review, bound, args.start, args.end_exclusive)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index = EVIDENCE / "index.json"
    journal: dict[str, Any] = json.loads(index.read_text()) if index.exists() else {
        "data_kind": "REAL", "study_id": STUDY,
        "processing_run_id": PROCESSING_RUN_ID, "bindings": bound, "days": {}
    }
    validate_index_identity(journal, bound)
    current = date.fromisoformat(args.start)
    while current < date.fromisoformat(args.end_exclusive):
        day = current.isoformat()
        if day in journal["days"]:
            validate_existing_entry(day, journal["days"][day], bound)
        else:
            ensure_no_orphans(day)
            journal["days"][day] = process_day(day, bound)
            temporary = EVIDENCE / "index.tmp"
            temporary.write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
            temporary.replace(index)
            print(json.dumps({"day": day, **journal["days"][day]}), flush=True)
        current += timedelta(days=1)


if __name__ == "__main__":
    main()
