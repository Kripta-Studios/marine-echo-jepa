"""Execute the frozen one-day TRAIN candidate; no corpus promotion or test access."""

import argparse
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from datetime import date as calendar_date
from pathlib import Path

import echopype as ep
import numpy as np
import psutil
from continuation_records import write_report

from marine_echo.data.candidate_qc import clean_per_ping, sample_edges, support_denominator
from marine_echo.data.range_grid import regrid_linear_sv

ROOT = Path(__file__).resolve().parents[1]
DATE = "2020-03-03"
FREQUENCIES = np.array([38000, 125000, 200000, 455000])


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_npz(path, **arrays):
    # No implicit overwrite or promotion; rerun requires an explicitly new candidate version.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def main(date=DATE, variant="v2"):
    started = time.monotonic()
    if variant == "v2" and date != DATE:
        raise ValueError("Historical v2 is restricted to its predetermined day")
    if variant == "monthly-v1" and date not in (
        "2020-02-17",
        "2020-03-01",
        "2020-04-01",
        "2020-05-01",
    ):
        raise ValueError("Only the prospectively specified monthly TRAIN days are allowed")
    if variant in ("census-v1", "census-v2") and not (
        len(date) == 10
        and "2020-02-17" <= date < "2020-05-27"
        and calendar_date.fromisoformat(date).isoformat() == date
    ):
        raise ValueError("Census is restricted to the original TRAIN calendar")
    if variant not in ("v2", "monthly-v1", "census-v1", "census-v2"):
        raise ValueError("Unregistered candidate version")
    contract_name = {
        "v2": "train_candidate_contract_v2",
        "monthly-v1": "monthly_train_contract",
        "census-v1": "train_census_contract",
        "census-v2": "train_census_v2_contract",
    }[variant]
    contract_path = ROOT / f"evidence/continuation/{contract_name}.json"
    contract_hash = sha(contract_path)
    recorded = ROOT / f"evidence/continuation/{contract_name}.sha256"
    if contract_hash != recorded.read_text().strip():
        raise ValueError("Candidate contract differs from prospective digest")
    if variant in ("census-v1", "census-v2"):
        review = json.loads(
            (ROOT / "orchestration/reviews/TRAIN_CENSUS_METHOD_20260927.json").read_text()
        )
        if (
            review.get("disposition") != "APPROVE_TRAIN_ONLY_CENSUS_METHOD"
            or review.get("reviewer_session") != "/root/continuation_review"
        ):
            raise ValueError("Distinct-session TRAIN census method review required")
    if variant == "census-v2":
        compat_review = json.loads(
            (ROOT / "orchestration/reviews/AZFP_SINGLETON_COMPAT_20260927.json").read_text()
        )
        compat_hash = sha(ROOT / "src/marine_echo/data/azfp_compat.py")
        if (
            compat_review.get("disposition") != "APPROVE_PINNED_SINGLETON_COMPAT_TRAIN_ONLY"
            or compat_review.get("reviewer_session") != "/root/continuation_review"
            or compat_review.get("helper_sha256") != compat_hash
        ):
            raise ValueError("Exact independent singleton compatibility review required")
    protocol_path = ROOT / "reports/active/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    train = protocol["split"]["partitions"]["train"]
    start = np.datetime64(date, "ns")
    end = start + np.timedelta64(1, "D")
    if not (
        np.datetime64(train["start"].replace("Z", "")) <= start
        and end <= np.datetime64(train["end"].replace("Z", ""))
    ):
        raise ValueError("Hard-coded candidate day must be wholly inside TRAIN")
    depth_path = ROOT / "evidence/continuation/depth_environment_sensitivity.json"
    if sha(depth_path) != "bb3aa8c36b5a19b56bb30435743956ec842257e6e3ce6a921bfcbabd1a6a8018":
        raise ValueError("Reviewed environmental evidence changed")
    env = json.loads(depth_path.read_text())["nominal"]
    certificate = ROOT / "data/raw/calibration/azfp55170/ASL-AZFP55170-calibration-20190326.pdf"
    if sha(certificate) != "6541fbe21f6af64e0731f53784a2763b87d8527ff79ddcba178bd0dd5eb3f464":
        raise ValueError("Reviewed factory certificate changed")
    extracted = (
        ROOT.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"
    )
    inventory = json.loads((extracted / ".extraction-manifest.json").read_text())
    registered = {row["path"]: row["sha256"] for row in inventory["files"]}
    xml = extracted / "20021600.XML"
    if sha(xml) != registered[xml.name]:
        raise ValueError("Preserved configuration differs")
    period = float(ET.parse(xml).getroot().findtext(".//PingPeriod"))
    if period != 15.0:
        raise ValueError("This prospective candidate expects the registered15-second cadence")
    compact_date = date.replace("-", "")
    output = ROOT / f"data/processed/candidate-train-{compact_date}-{variant}"
    output.mkdir(parents=True, exist_ok=False)
    evidence = ROOT / (
        "evidence/continuation/train-candidate-v2"
        if variant == "v2"
        else f"evidence/continuation/train-candidate-{compact_date}-{variant}"
    )
    evidence.mkdir(parents=True, exist_ok=False)
    bindings = {
        "contract_sha256": contract_hash,
        "protocol_sha256": sha(protocol_path),
        "environment_report_sha256": sha(depth_path),
        "slant_report_sha256": sha(
            ROOT / "evidence/continuation/slant_environment_sensitivity.json"
        ),
        "certificate_sha256": sha(certificate),
        "xml_sha256": sha(xml),
        "code_sha256": sha(Path(__file__)),
        "qc_code_sha256": sha(ROOT / "src/marine_echo/data/candidate_qc.py"),
        "regrid_code_sha256": sha(ROOT / "src/marine_echo/data/range_grid.py"),
        "upstream_noise_code_sha256": sha(Path(ep.__file__).parent / "clean/api.py"),
        "upstream_range_code_sha256": sha(Path(ep.__file__).parent / "calibrate/range.py"),
    }
    if variant == "census-v2":
        bindings["compat_code_sha256"] = compat_hash
    write_report(
        evidence / "processing_manifest.json",
        {
            "status": "RUNNING_TRAIN_CANDIDATE",
            "bindings": bindings,
            "start": date,
            "benchmark_eligible": False,
        },
    )
    total = np.zeros((96, 4, 64))
    valid_counts = np.zeros((96, 4, 64), dtype=np.int64)
    observed_counts = np.zeros(96, dtype=np.int64)
    expected_counts = np.full(96, int(900 / period), dtype=np.int64)
    target_edges = np.arange(0.0, 130.0, 2.0)
    files, all_times, configs, previous_times = [], [], [], set()
    maximum_rss = 0
    for path in sorted(extracted.glob(compact_date[2:] + "*.01?")):
        digest = sha(path)
        if digest != registered.get(path.name):
            raise ValueError("TRAIN source differs from extraction inventory")
        if variant == "census-v2":
            from marine_echo.data.azfp_compat import open_raw_azfp_compat

            ed = open_raw_azfp_compat(path, xml_path=xml)
        else:
            ed = ep.open_raw(path, sonar_model="AZFP", xml_path=xml)
        beam, vendor, platform = ed["Sonar/Beam_group1"], ed["Vendor_specific"], ed["Platform"]
        np.testing.assert_array_equal(beam.frequency_nominal.values, FREQUENCIES)
        ping_times = beam.ping_time.values.astype("datetime64[ns]")
        if np.isnat(ping_times).any() or (np.diff(ping_times).astype("int64") <= 0).any():
            raise ValueError("Nonfinite, duplicate or unordered acquisition timestamps")
        keep = (ping_times >= start) & (ping_times < end)
        times = ping_times[keep]
        current = set(times.astype("int64").tolist())
        if previous_times & current:
            raise ValueError("Duplicate raw acquisition across TRAIN files")
        previous_times.update(current)
        config = {
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
        config["transmit_duration_nominal"] = np.unique(
            beam.transmit_duration_nominal.values
        ).tolist()
        config["xml_sha256"] = sha(xml)
        config["ping_period_s"] = period
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if int(vendor.average_pings_flag.values) != 0:
            raise ValueError("Time-averaged acquisition needs a separate availability policy")
        calibrated = ep.calibrate.compute_Sv(ed, env_params=env)
        counts = beam.backscatter_r.values
        raw_sv, ranges = calibrated.Sv.values.copy(), calibrated.echo_range.values
        tx, ty = platform.tilt_x.values, platform.tilt_y.values
        if len(tx) != len(ping_times) or len(ty) != len(ping_times):
            raise ValueError("Tilt telemetry timestamps do not align with pings")
        theta = np.hypot(tx, ty)
        geometry = ~np.isfinite(theta) | (theta >= 30)
        reasons = np.zeros(raw_sv.shape, dtype=np.uint16)
        reasons[~np.isfinite(raw_sv) | ~np.isfinite(counts)] |= 1
        reasons[ranges < 10.0] |= 2
        reasons[counts >= 65535] |= 4
        reasons[:, geometry, :] |= 8
        calibrated["Sv"] = calibrated.Sv.where(reasons == 0)
        cleaned = clean_per_ping(calibrated)
        corrected, noise = cleaned.Sv_corrected.values, cleaned.Sv_noise.values
        reasons[~np.isfinite(corrected)] |= 16
        grid_values = np.full((len(ping_times), 4, 64), np.nan)
        grid_mask = np.zeros(grid_values.shape, dtype=bool)
        range_support = []
        for channel in range(4):
            n = int(vendor.number_of_bins_per_channel.values[channel])
            c_ranges = ranges[channel, :, :n]
            if not np.allclose(c_ranges, c_ranges[0], rtol=1e-12):
                raise ValueError("Within-file range geometry changed")
            spacing = (
                env["sound_speed"]
                * float(vendor.number_of_samples_per_average_bin.values[channel])
                / (2 * float(vendor.digitization_rate.values[channel]))
            )
            source_edges = sample_edges(c_ranges[0], spacing)
            linear = 10 ** (corrected[channel, :, :n] / 10)
            valid = reasons[channel, :, :n] == 0
            grid = regrid_linear_sv(linear, valid, source_edges, target_edges)
            grid_values[:, channel], grid_mask[:, channel] = grid.sv_linear, grid.valid_mask
            range_support.append(
                {
                    "frequency_hz": int(FREQUENCIES[channel]),
                    "source_edges_min_max_m": [float(source_edges[0]), float(source_edges[-1])],
                    "geometric_supported_2m_cells": grid.supported_range.tolist(),
                }
            )
        quarters = (
            (times - start).astype("timedelta64[ns]").astype("int64") // 900_000_000_000
        ).astype(int)
        np.add.at(total, quarters, np.where(grid_mask[keep], grid_values[keep], 0.0))
        np.add.at(valid_counts, quarters, grid_mask[keep].astype(np.int64))
        np.add.at(observed_counts, quarters, 1)
        all_times.extend(times.astype("int64").tolist())
        configs.extend((int(q), config_hash) for q in np.unique(quarters))
        hour_path = output / (path.name + ".npz")
        if variant not in ("census-v1", "census-v2"):
            save_npz(
                hour_path,
                ping_time=times,
                pre_noise_Sv=raw_sv[:, keep],
                Sv_noise=noise[:, keep],
                corrected_Sv=corrected[:, keep],
                raw_qc_reason=reasons[:, keep],
                grid_linear_sv=grid_values[keep],
                grid_valid_mask=grid_mask[keep],
                tilt_x=tx[keep],
                tilt_y=ty[keep],
                tilt_x_count=vendor.tilt_x_count.values[keep],
                tilt_y_count=vendor.tilt_y_count.values[keep],
                range_edges_m=target_edges,
                echo_range=ranges[:, keep],
            )
        item = {
            "source": path.name,
            "source_sha256": digest,
            "candidate_file": str(hour_path.relative_to(ROOT))
            if not variant.startswith("census-")
            else None,
            "candidate_sha256": sha(hour_path) if not variant.startswith("census-") else None,
            "pings_in_day": len(times),
            "pings_outside_day_excluded": int((~keep).sum()),
            "configuration_sha256": config_hash,
            "configuration": config,
            "first_utc": str(times[0]) if len(times) else None,
            "last_utc": str(times[-1]) if len(times) else None,
            "tilt_magnitude_min_max_deg": [float(np.nanmin(theta)), float(np.nanmax(theta))],
            "range_support": range_support,
            "QC_reason_counts": {
                str(bit): int(((reasons[:, keep] & bit) > 0).sum()) for bit in (1, 2, 4, 8, 16)
            },
        }
        files.append(item)
        maximum_rss = max(maximum_rss, psutil.Process().memory_info().rss)
        print(
            json.dumps(
                {"file": path.name, "pings": len(times), "elapsed_s": time.monotonic() - started}
            ),
            flush=True,
        )
        del ed, beam, vendor, platform, calibrated, cleaned
    denominator = support_denominator(expected_counts, observed_counts)
    boundaries = np.zeros(96, dtype=bool)
    config_by_bin = []
    for i in range(96):
        values = {value for q, value in configs if q == i}
        config_by_bin.append(next(iter(values)) if len(values) == 1 else None)
        if len(values) > 1:
            total[i], valid_counts[i], boundaries[i] = 0.0, 0, True
        elif i and config_by_bin[i] != config_by_bin[i - 1]:
            boundaries[i] = True
    means = np.full(total.shape, np.nan)
    np.divide(total, valid_counts, out=means, where=valid_counts > 0)
    support = valid_counts / denominator[:, None, None]
    primary_support = support[:, 0, 5:50].mean(axis=1)
    daily_path = output / (date + ".npz")
    save_npz(
        daily_path,
        bin_start=start + np.arange(96) * np.timedelta64(15, "m"),
        bin_end=start + np.arange(1, 97) * np.timedelta64(15, "m"),
        linear_sv=means,
        valid_ping_count=valid_counts,
        expected_ping_count=expected_counts,
        observed_ping_count=observed_counts,
        ping_count=observed_counts,
        support_denominator_ping_count=denominator,
        excess_ping_count=np.maximum(observed_counts - expected_counts, 0),
        cell_support=support,
        raw_ping_time=np.array(all_times, dtype="int64").astype("datetime64[ns]"),
        primary_support=primary_support,
        range_edges_m=target_edges,
        frequency_hz=FREQUENCIES,
        configuration_boundary=boundaries,
    )
    (evidence / "coverage_by_day.csv").write_text(
        "date,expected_pings,observed_pings,primary_bins_at_least_80pct,approved_eligible_days\n"
        + f"{date},{int(expected_counts.sum())},{int(observed_counts.sum())},{int((primary_support >= 0.8).sum())},0\n"
    )
    write_report(
        evidence / "frequency_range_support.json",
        {
            "status": "TRAIN_CANDIDATE_NOT_APPROVED",
            "files": [{"source": r["source"], "range_support": r["range_support"]} for r in files],
        },
    )
    write_report(
        evidence / "configuration_boundaries.json",
        {
            "bin_configuration_ids": config_by_bin,
            "boundary_mask": boundaries.tolist(),
            "rule": "No window may cross a configuration boundary",
        },
    )
    report = {
        "status": "TRAIN_CANDIDATE_PROCESSED_REQUIRES_QC_REVIEW",
        "benchmark_eligible": False,
        "approved_eligible_days": 0,
        "dataset_id": "mosaic_azfp_down_2020",
        "dataset_version": "PANGAEA949811",
        "instrument_id": "AZFP55170",
        "orientation": "down",
        "latitude": None,
        "longitude": None,
        "position_observation_time": None,
        "available_time": "UNKNOWN",
        "replay_availability": "zero_latency_at_trailing_bin_end",
        "calibration_status": "FACTORY_FIXED_REGIONAL_ASSUMPTION_CANDIDATE",
        "bindings": bindings,
        "nominal_environment": env,
        "assumed_beam_face_depth_m": [0, 10],
        "source_hours": files,
        "daily_candidate_sha256": sha(daily_path),
        "daily_candidate_path": str(daily_path.relative_to(ROOT)),
        "expected_pings": int(expected_counts.sum()),
        "observed_pings": int(observed_counts.sum()),
        "primary_bins_at_least_80pct": int((primary_support >= 0.8).sum()),
        "total_quarter_hour_bins": 96,
        "limitations": [
            "Partial saturation unobservable after range averaging",
            "Bottom mask not applied; TRAIN review required",
            "Slant sensitivity accepted for TRAIN development; sampled bounds are not a continuous or field-calibration proof",
            "Not a promoted canonical corpus; no validation/calibration/test outcomes opened",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "peak_sampled_process_rss_gib_not_tree": maximum_rss / 1024**3,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    write_report(evidence / "processing_manifest.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "expected_pings",
                    "observed_pings",
                    "primary_bins_at_least_80pct",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DATE)
    parser.add_argument(
        "--variant", choices=["v2", "monthly-v1", "census-v1", "census-v2"], default="v2"
    )
    args = parser.parse_args()
    raise SystemExit(main(args.date, args.variant))
