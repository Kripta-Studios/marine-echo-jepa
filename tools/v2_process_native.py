"""Stream the reviewed Candidate 2 TRAIN development interval into native summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import echopype as ep
import numpy as np
import psutil
from numpy.typing import NDArray

from marine_echo.data.azfp_compat import open_raw_azfp_compat
from marine_echo.data.candidate_qc import clean_per_ping, sample_edges
from marine_echo.data.native_observations import aggregate_detected_intervals

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def bindings() -> dict:
    freeze = ROOT / "evidence/v2/native_candidate_freeze_v3.json"
    document = json.loads(freeze.read_text())
    for entry in (document, document["inherited_contract"]):
        path = ROOT / entry.get("proposal_path", entry.get("path"))
        expected = entry.get("proposal_sha256", entry.get("sha256"))
        if sha(path) != expected:
            raise ValueError("Frozen native measurement contract changed")
    paths = {
        "native_freeze": freeze,
        "processor": Path(__file__),
        "aggregation": ROOT / "src/marine_echo/data/native_observations.py",
        "qc": ROOT / "src/marine_echo/data/candidate_qc.py",
        "compat": ROOT / "src/marine_echo/data/azfp_compat.py",
        "environment": ROOT / "evidence/continuation/depth_environment_sensitivity.json",
        "slant": ROOT / "evidence/continuation/slant_environment_sensitivity.json",
        "certificate": ROOT
        / "data/raw/calibration/azfp55170/ASL-AZFP55170-calibration-20190326.pdf",
        "xml": RAW / "20021600.XML",
        "upstream_noise": Path(ep.__file__).parent / "clean/api.py",
        "upstream_range": Path(ep.__file__).parent / "calibrate/range.py",
    }
    result = {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()}
    expected = json.loads(
        (ROOT / "evidence/continuation/train_census_v2_execution.json").read_text()
    )["identity"]["bindings"]
    for key, old in (
        ("qc", "qc_code_sha256"),
        ("compat", "compat_code_sha256"),
        ("environment", "environment_report_sha256"),
        ("slant", "slant_report_sha256"),
        ("certificate", "certificate_sha256"),
        ("xml", "xml_sha256"),
        ("upstream_noise", "upstream_noise_code_sha256"),
        ("upstream_range", "upstream_range_code_sha256"),
    ):
        if result[key]["sha256"] != expected[old]:
            raise ValueError(f"Reviewed calibration/QC dependency changed: {key}")
    return result


def process_day(day: str, evidence: Path, output: Path, bound: dict) -> dict:
    started = time.monotonic()
    compact = day.replace("-", "")
    old_path = (
        ROOT / f"evidence/continuation/train-candidate-{compact}-census-v2/processing_manifest.json"
    )
    old = json.loads(old_path.read_text())
    ledger = json.loads((ROOT / "evidence/continuation/train_census_v2_execution.json").read_text())
    if sha(old_path) != ledger["days"][day]["manifest_sha256"]:
        raise ValueError("Source manifest differs from historical digest")
    env = json.loads(Path(bound["environment"]["path"]).read_text())["nominal"]
    xml = Path(bound["xml"]["path"])
    start = np.datetime64(day, "ns")
    end = start + np.timedelta64(1, "D")
    sums = np.zeros((96, 4, 64))
    lengths = np.zeros_like(sums)
    observed: NDArray[np.int64] = np.zeros(96, dtype=np.int64)
    sample_count: NDArray[np.int64] = np.zeros(96, dtype=np.int64)
    configs: list[set[str]] = [set() for _ in range(96)]
    times_all = []
    seen: set[int] = set()
    sources = []
    peak_rss = 0
    reasons_total = {
        "native_band_sample_pairs": 0,
        "invalid": 0,
        "saturated": 0,
        "geometry": 0,
        "noise_unavailable": 0,
        "below_detection": 0,
        "detected": 0,
    }
    for source in old["source_hours"]:
        path = RAW / source["source"]
        if sha(path) != source["source_sha256"]:
            raise ValueError("Raw source differs from preserved acquisition")
        ed = open_raw_azfp_compat(path, xml_path=xml)
        beam, vendor, platform = ed["Sonar/Beam_group1"], ed["Vendor_specific"], ed["Platform"]
        np.testing.assert_array_equal(
            beam.frequency_nominal.values, [38000, 125000, 200000, 455000]
        )
        times = beam.ping_time.values.astype("datetime64[ns]")
        if (
            np.isnat(times).any()
            or (np.diff(times).astype("int64") <= 0).any()
            or (times < start).any()
            or (times >= end).any()
        ):
            raise ValueError("Invalid, unordered, or out-of-day acquisition timestamp")
        integer_times = set(times.astype("int64").tolist())
        if seen & integer_times:
            raise ValueError("Duplicate native ping across source files")
        seen.update(integer_times)
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
            xml_sha256=sha(xml),
            ping_period_s=15.0,
        )
        config_hash = hashlib.sha256(json.dumps(actual_config, sort_keys=True).encode()).hexdigest()
        if (
            config_hash != source["configuration_sha256"]
            or actual_config["average_pings_flag"] != 0
        ):
            raise ValueError("Acquisition configuration changed")
        calibrated = ep.calibrate.compute_Sv(ed, env_params=env)
        raw = calibrated.Sv.values.copy()
        counts = beam.backscatter_r.values
        ranges = calibrated.echo_range.values
        theta = np.hypot(platform.tilt_x.values, platform.tilt_y.values)
        if len(theta) != len(times):
            raise ValueError("Tilt acquisition does not align")
        reasons = np.zeros(raw.shape, dtype=np.uint16)
        reasons[~np.isfinite(raw) | ~np.isfinite(counts)] |= 1
        reasons[ranges < 10.0] |= 2
        reasons[counts >= 65535] |= 4
        reasons[:, ~np.isfinite(theta) | (theta >= 30), :] |= 8
        calibrated["Sv"] = calibrated.Sv.where(reasons == 0)
        cleaned = clean_per_ping(calibrated)
        corrected = cleaned.Sv_corrected.values
        reasons[~np.isfinite(corrected)] |= 16
        n = int(vendor.number_of_bins_per_channel.values[0])
        spacing = (
            env["sound_speed"]
            * float(vendor.number_of_samples_per_average_bin.values[0])
            / (2 * float(vendor.digitization_rate.values[0]))
        )
        if not np.allclose(ranges[0, :, :n], ranges[0, 0, :n], rtol=1e-12):
            raise ValueError("Range geometry varies inside file")
        edges = sample_edges(ranges[0, 0, :n], spacing)
        valid = reasons[0, :, :n] == 0
        agg = aggregate_detected_intervals(
            10 ** (corrected[0, :, :n] / 10), valid, edges, np.arange(0.0, 130.0, 2.0)
        )
        np.testing.assert_allclose(agg["available_range_m"][5:50], 2.0, rtol=0, atol=1e-12)
        quarters = (
            (times - start).astype("timedelta64[ns]").astype("int64") // 900_000_000_000
        ).astype(int)
        np.add.at(sums[:, 0], quarters, agg["linear_sum"])
        np.add.at(lengths[:, 0], quarters, agg["detected_range_m"])
        np.add.at(observed, quarters, 1)
        overlap_band = (edges[:-1] < 100) & (edges[1:] > 10)
        np.add.at(sample_count, quarters, valid[:, overlap_band].sum(axis=1))
        left = np.broadcast_to(overlap_band, valid.shape).copy()
        reasons_total["native_band_sample_pairs"] += int(left.sum())
        for key, mask in (
            ("invalid", (reasons[0, :, :n] & 1) != 0),
            ("saturated", (reasons[0, :, :n] & 4) != 0),
            ("geometry", (reasons[0, :, :n] & 10) != 0),
            ("noise_unavailable", ~np.isfinite(cleaned.Sv_noise.values[0, :, :n])),
            ("below_detection", ~np.isfinite(corrected[0, :, :n])),
            ("detected", valid),
        ):
            reasons_total[key] += int((left & mask).sum())
            left &= ~mask
        if left.any():
            raise ValueError("Native QC categories fail to partition samples")
        for q in np.unique(quarters):
            configs[int(q)].add(config_hash)
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
        rss = psutil.Process().memory_info().rss
        peak_rss = max(peak_rss, rss)
        if rss >= 22 * 1024**3:
            raise MemoryError("Process RAM limit reached")
        del ed, beam, vendor, platform, calibrated, cleaned
    times_all.sort()
    if len(times_all) != old["observed_pings"]:
        raise ValueError("Native ping count differs from preserved acquisition evidence")
    if any(len(x) > 1 for x in configs):
        raise ValueError("Mixed configuration quarter-hour requires explicit new processing")
    ids = np.array([next(iter(x)) if x else "" for x in configs], dtype="U64")
    means = np.full_like(sums, np.nan)
    np.divide(sums, lengths, out=means, where=lengths > 0)
    shard = output / f"{day}.npz"
    with shard.open("xb") as stream:
        np.savez_compressed(
            stream,
            bin_start=start + np.arange(96) * np.timedelta64(15, "m"),
            bin_end=start + np.arange(1, 97) * np.timedelta64(15, "m"),
            linear_sv=means,
            detected_range_ping_m=lengths,
            observed_ping_count=observed,
            expected_ping_count=np.full(96, 60, dtype=np.int64),
            raw_ping_time=np.array(times_all, dtype="int64").astype("datetime64[ns]"),
            frequency_hz=np.array([38000, 125000, 200000, 455000]),
            range_edges_m=np.arange(0.0, 130.0, 2.0),
            configuration_id=ids,
            native_detected_sample_count=sample_count,
        )
        stream.flush()
        os.fsync(stream.fileno())
    document = {
        "data_kind": "REAL",
        "study_id": "v2_candidate2",
        "status": "NATIVE_TRAIN_DEVELOPMENT_PROCESSED_NOT_APPROVED_FOR_FITTING",
        "day": day,
        "shard_path": str(shard.relative_to(ROOT)),
        "shard_sha256": sha(shard),
        "bindings": bound,
        "source_hours": sources,
        "historical_manifest_sha256": sha(old_path),
        "observed_pings": len(times_all),
        "native_qc_disjoint_counts": reasons_total,
        "elapsed_seconds": time.monotonic() - started,
        "peak_sampled_process_rss_gib": peak_rss / 1024**3,
        "available_time": "UNKNOWN",
        "replay_availability": "zero_latency_at_trailing_bin_end",
        "quantity": "detection-conditioned sampled Sv; range-length weighted; factory/regional assumption",
    }
    manifest = evidence / f"{day}.json"
    write_new(manifest, document)
    return {
        "manifest_path": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha(manifest),
        "shard_sha256": document["shard_sha256"],
        "elapsed_seconds": document["elapsed_seconds"],
    }


def validate_processing_review(review: dict, bound: dict, start: str, end: str) -> None:
    if (
        review.get("disposition") != "APPROVE_NATIVE_PROCESSING_IMPLEMENTATION"
        or review.get("processor_sha256") != bound["processor"]["sha256"]
        or review.get("aggregation_sha256") != bound["aggregation"]["sha256"]
        or review.get("reviewer_session") != "/root/v2_reviewer"
    ):
        raise ValueError("Exact distinct-session native implementation approval required")
    if not (
        review.get("approved_start", "9999") <= start < end <= review.get("approved_end", "0000")
    ):
        raise ValueError("Requested calendar exceeds approved processing scope")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-02-17")
    parser.add_argument("--end", default="2020-04-15")
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    if not "2020-02-17" <= args.start < args.end <= "2020-04-15":
        raise ValueError("Only the fixed TRAIN development interval is permitted")
    bound = bindings()
    review = json.loads(args.review.read_text())
    validate_processing_review(review, bound, args.start, args.end)
    evidence = ROOT / "evidence/v2/native-development"
    output = ROOT / "data/processed/v2-native-development"
    evidence.mkdir(exist_ok=True, parents=True)
    output.mkdir(exist_ok=True, parents=True)
    index = evidence / "index.json"
    journal: dict[str, Any] = (
        json.loads(index.read_text())
        if index.exists()
        else {"data_kind": "REAL", "study_id": "v2_candidate2", "bindings": bound, "days": {}}
    )
    if journal["bindings"] != bound:
        raise ValueError("Cannot resume native processing with different code/contract")
    current = date.fromisoformat(args.start)
    while current < date.fromisoformat(args.end):
        day = current.isoformat()
        if day in journal["days"]:
            entry = journal["days"][day]
            manifest = ROOT / entry["manifest_path"]
            if sha(manifest) != entry["manifest_sha256"]:
                raise ValueError("Completed native manifest changed")
            document = json.loads(manifest.read_text())
            if sha(ROOT / document["shard_path"]) != entry["shard_sha256"]:
                raise ValueError("Completed native shard changed")
        else:
            if (evidence / f"{day}.json").exists() or (output / f"{day}.npz").exists():
                raise RuntimeError(
                    f"Unindexed native output for {day}; preserved without overwrite. Verify and recover the interrupted day before resuming."
                )
            journal["days"][day] = process_day(day, evidence, output, bound)
            temporary = evidence / "index.tmp"
            temporary.write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
            temporary.replace(index)
            print(json.dumps({"day": day, **journal["days"][day]}), flush=True)
        current += timedelta(days=1)


if __name__ == "__main__":
    main()
