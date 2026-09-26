"""Bounded, non-promotable AZFP numerical assay and environmental applicability audit.

Uses only one fixed training hour. No acoustic test outcomes are opened. The
homogeneous water-column scenarios below are sensitivity diagnostics, not an
approved physical calibration or eligible preprocessing product.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import echopype as ep
import netCDF4
import numpy as np
import psutil
from continuation_records import write_report
from echopype.utils import uwa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from marine_echo.data.environment import distance_km, parse_iridium_position, seawater_properties


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matlab_date(value: float) -> datetime:
    return (datetime.fromordinal(int(value)) + timedelta(days=value % 1 - 366)).replace(tzinfo=UTC)


def main() -> int:
    started = time.monotonic()
    extracted = (
        ROOT.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"
    )
    raw, xml = extracted / "20030300.01A", extracted / "20021600.XML"
    environment = ROOT / "data/raw/environment/mosaic_core/MOSAiC_daily_profiles.nc"
    iridium = ROOT / "data/raw/pangaea/949811/iridium-msgs.txt"
    inventory = json.loads((extracted / ".extraction-manifest.json").read_text())
    expected = {row["path"]: row["sha256"] for row in inventory["files"]}
    for path in (raw, xml):
        if sha(path) != expected[path.name]:
            raise ValueError("Selected input differs from the preserved extraction manifest.")
    environment_inventory = json.loads(
        (ROOT / "evidence/calibration/environment_inventory.json").read_text()
    )
    if sha(environment) != next(
        row["sha256"] for row in environment_inventory if row["name"] == environment.name
    ):
        raise ValueError("Environmental file differs from its registered SHA-256.")
    gps, rejected = [], 0
    for line in iridium.read_text().splitlines():
        if not line.strip():
            continue
        try:
            gps.append(parse_iridium_position(line))
        except (ValueError, IndexError):
            rejected += 1
    gps.sort()
    timestamp_counts = {}
    speeds, gaps = [], []
    for point in gps:
        timestamp_counts[point[0].isoformat()] = timestamp_counts.get(point[0].isoformat(), 0) + 1
    for left, right in pairwise(gps):
        hours = (right[0] - left[0]).total_seconds() / 3600
        gaps.append(hours)
        if hours > 0:
            speeds.append(distance_km(left[1], left[2], right[1], right[2]) / hours)
    xeos_comparisons = []
    xeos_path = (
        ROOT / "data/raw/pangaea/949811/xeos-rover-mosaicdown-transponder-16feb-02aug2020.csv"
    )
    with xeos_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Device"] != "MOSAICdown":
                continue
            match = re.search(
                r"Timestamp: ([^,]+).*Latitude: ([\d.-]+), Longitude: ([\d.-]+)", row["Message"]
            )
            if match:
                text, lat, lon = match.groups()
                stamp = datetime.fromisoformat(text)
                nearest = min(gps, key=lambda g: abs((g[0] - stamp).total_seconds()))
                minutes = abs((nearest[0] - stamp).total_seconds()) / 60
                if minutes <= 60:
                    xeos_comparisons.append(
                        distance_km(float(lat), float(lon), nearest[1], nearest[2])
                    )
    ds = netCDF4.Dataset("memory", memory=environment.read_bytes())
    dates = [matlab_date(float(t)) for t in ds["time"][:]]
    profile_rows = []
    for i, stamp in enumerate(dates):
        if not datetime(2020, 2, 17, tzinfo=UTC) <= stamp < datetime(2020, 8, 2, tzinfo=UTC):
            continue
        lat, lon = float(ds["latitude"][i]), float(ds["longitude"][i])
        nearest = min(gps, key=lambda g: abs((g[0] - stamp).total_seconds()))
        depth, sa, ct = [
            np.ma.filled(ds[n][:, i], np.nan) for n in ("depth", "salinity", "temperature")
        ]
        selected = (depth >= 10) & (depth <= 100)
        finite = selected & np.isfinite(sa) & np.isfinite(ct)
        distance = (
            distance_km(lat, lon, nearest[1], nearest[2]) if np.isfinite([lat, lon]).all() else None
        )
        profile_rows.append(
            {
                "date": stamp.isoformat(),
                "nearest_iridium_hours": abs((nearest[0] - stamp).total_seconds()) / 3600,
                "distance_km": distance,
                "finite_levels_10_100m": int(finite.sum()),
                "requested_levels_10_100m": int(selected.sum()),
                "available_at": None,
                "benchmark_eligible": False,
            }
        )
    # Predetermined earliest training date with a nearby XEOS record in the prior audit.
    # Selection is metadata-only, independent of acoustic outcomes/performance.
    index = dates.index(datetime(2020, 3, 3, tzinfo=UTC))
    depth, sa, ct = [
        np.ma.filled(ds[n][:, index], np.nan) for n in ("depth", "salinity", "temperature")
    ]
    selected = (depth >= 10) & (depth <= 100) & np.isfinite(sa) & np.isfinite(ct)
    depth, sa, ct = depth[selected], sa[selected], ct[selected]
    converted = seawater_properties(
        depth, sa, ct, float(ds["latitude"][index]), float(ds["longitude"][index])
    )
    metadata = {n: {k: str(ds[n].getncattr(k)) for k in ds[n].ncattrs()} for n in ds.variables}
    ds.close()
    if ep.__version__ != "0.11.1":
        raise ValueError("Pinned Echopype version required.")
    ed = ep.open_raw(raw, sonar_model="AZFP", xml_path=xml)
    beam, vendor = ed["Sonar/Beam_group1"], ed["Vendor_specific"]
    frequencies = beam.frequency_nominal.values
    np.testing.assert_array_equal(frequencies, [38000, 125000, 200000, 455000])
    xml_root = ET.parse(xml).getroot()
    coefficients = xml_root.findall(".//LogAcousticCoefficients/Frequencies/Frequency")
    mapping = []
    for channel, element in enumerate(coefficients):
        expected_frequency = float(element.findtext("kHz")) * 1000
        assert expected_frequency == frequencies[channel]
        row = {"frequency_hz": expected_frequency}
        for field in ("TVR", "VTX0", "EL", "DS"):
            expected = float(element.findtext(field))
            actual = float(vendor[field].isel(channel=channel))
            np.testing.assert_allclose(actual, expected, rtol=1e-10)
            row[field] = actual
        expected_bp = float(element.findtext("BP"))
        np.testing.assert_allclose(
            float(beam.equivalent_beam_angle.isel(channel=channel)), expected_bp
        )
        row["beam_solid_angle_sr"] = expected_bp
        mapping.append(row)
    scenarios = []
    values = []
    numerical_errors = []
    for reference_depth, override_sound_speed in (
        (10, None),
        (50, None),
        (100, None),
        (50, 1465.0),
    ):
        k = int(np.argmin(abs(depth - reference_depth)))
        env = {
            "temperature": float(converted["in_situ_temperature_c"][k]),
            "salinity": float(converted["practical_salinity"][k]),
            "pressure": float(converted["pressure_dbar"][k]),
            "sound_speed": float(converted["sound_speed_m_s"][k]),
        }
        if override_sound_speed is not None:
            env["sound_speed"] = override_sound_speed
        calibrated = ep.calibrate.compute_Sv(ed, env_params=env)
        alpha = calibrated.sound_absorption.values
        c = env["sound_speed"]
        for channel in range(4):
            sample, ping = 100, 0

            def scalar(array, channel=channel, ping=ping, sample=sample):
                return float(
                    array.isel(
                        **{
                            d: {"channel": channel, "ping_time": ping, "range_sample": sample}[d]
                            for d in array.dims
                        }
                    ).values
                )

            n, rate, lockout = [
                scalar(vendor[name])
                for name in (
                    "number_of_samples_per_average_bin",
                    "digitization_rate",
                    "lock_out_index",
                )
            ]
            duration = scalar(beam.transmit_duration_nominal)
            r = c * lockout / (2 * rate) + c / 4 * (
                ((2 * (sample + 1) - 1) * n - 1) / rate + duration
            )
            np.testing.assert_allclose(scalar(calibrated.echo_range), r, rtol=1e-12)
            coeff = mapping[channel]
            received = (
                coeff["EL"] - 2.5 / coeff["DS"] + scalar(beam.backscatter_r) / (26214 * coeff["DS"])
            )
            expected = (
                received
                - coeff["TVR"]
                - 20 * math.log10(coeff["VTX0"])
                + 20 * math.log10(r)
                + 2 * float(alpha[channel]) * r
                - 10 * math.log10(0.5 * c * duration * coeff["beam_solid_angle_sr"])
                + scalar(vendor.Sv_offset)
            )
            numerical_errors.append(abs(expected - scalar(calibrated.Sv)))
        ranges = calibrated.echo_range.values
        finite = np.isfinite(calibrated.Sv.values)
        values.append(
            np.where((ranges >= 10) & (ranges <= 100) & finite, calibrated.Sv.values, np.nan)
        )
        scenarios.append(
            {
                "homogeneous_reference_depth_m": float(depth[k]),
                "sound_speed_basis": "XML configured value sensitivity only"
                if override_sound_speed
                else "TEOS-10 measured profile scenario",
                "inputs": env,
                "absorption_db_per_m": alpha.tolist(),
                "sound_speed_azfp_formula_m_s": float(
                    uwa.calc_sound_speed(
                        **{k: v for k, v in env.items() if k != "sound_speed"},
                        formula_source="AZFP",
                    )
                ),
                "padded_formula_grid_range_m_by_channel_not_support": [
                    [
                        float(np.nanmin(calibrated.echo_range.values[ch])),
                        float(np.nanmax(calibrated.echo_range.values[ch])),
                    ]
                    for ch in range(4)
                ],
                "finite_sv_range_m_by_channel_not_qc_approval": [
                    [float(ranges[ch][finite[ch]].min()), float(ranges[ch][finite[ch]].max())]
                    for ch in range(4)
                ],
            }
        )
    stack = np.stack(values[:3])
    if max(numerical_errors) > 1e-8:
        raise ValueError("Independent scalar Sv reconstruction exceeds the numerical tolerance.")
    common = np.isfinite(stack).all(axis=0)
    spread = np.where(
        common,
        np.max(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
        - np.min(np.where(np.isfinite(stack), stack, np.inf), axis=0),
        np.nan,
    )
    partitions = json.loads((ROOT / "reports/active/protocol.json").read_text())["split"][
        "partitions"
    ]
    matching_summary = {}
    for name, bounds in partitions.items():
        start, end = [datetime.fromisoformat(bounds[key]) for key in ("start", "end")]
        rows = [r for r in profile_rows if start <= datetime.fromisoformat(r["date"]) < end]
        matching_summary[name] = {
            "calendar_days": bounds["calendar_days"],
            "profile_days": len(rows),
            "illustrative_within_5km_24h": sum(
                r["distance_km"] is not None
                and r["distance_km"] <= 5
                and r["nearest_iridium_hours"] <= 24
                for r in rows
            ),
            "illustrative_within_10km_24h": sum(
                r["distance_km"] is not None
                and r["distance_km"] <= 10
                and r["nearest_iridium_hours"] <= 24
                for r in rows
            ),
            "physically_approved_days": 0,
        }
    upstream = Path(inspect.getfile(ep)).parent
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "assay_code_sha256": sha(Path(__file__)),
        "environment_helper_sha256": sha(ROOT / "src/marine_echo/data/environment.py"),
        "status": "NUMERICAL_ASSAY_PASSED_PHYSICAL_APPLICABILITY_BLOCKED",
        "benchmark_eligible": False,
        "eligible_days": 0,
        "eligibility_meaning": "No physically approved corpus; not a measured claim of zero usable acoustic days.",
        "physical_calibration_distinct_from_interval_calibration": True,
        "source_hashes": {
            str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: sha(p)
            for p in (raw, xml, environment, iridium)
        },
        "upstream_version": ep.__version__,
        "gsw_version": "3.6.23",
        "upstream_sha256": {
            name: sha(upstream / name)
            for name in (
                "calibrate/calibrate_azfp.py",
                "calibrate/range.py",
                "calibrate/env_params.py",
                "utils/uwa.py",
                "convert/parse_azfp.py",
            )
        },
        "environment_metadata": metadata,
        "iridium_accepted": len(gps),
        "iridium_rejected": rejected,
        "iridium_quality_policy": "Only code 1 retained; receiver fix-code semantics are not independently certified. Audit positions are not quality-approved.",
        "trajectory_qc": {
            "duplicate_timestamps": {k: v for k, v in timestamp_counts.items() if v > 1},
            "successive_speed_km_h_p50_p95_max": np.percentile(speeds, [50, 95, 100]).tolist(),
            "gap_hours_p50_p95_max": np.percentile(gaps, [50, 95, 100]).tolist(),
            "xeos_pairs_within_60min": len(xeos_comparisons),
            "xeos_separation_km_p50_p95_max": np.percentile(
                xeos_comparisons, [50, 95, 100]
            ).tolist(),
            "acceptance": "DESCRIPTIVE_ONLY_NO_TRAJECTORY_APPROVAL",
        },
        "matching_by_partition": matching_summary,
        "profile_days": profile_rows,
        "coefficient_mapping": mapping,
        "assay_hour": {
            "file": raw.name,
            "first_ping": str(beam.ping_time.values[0]),
            "last_ping": str(beam.ping_time.values[-1]),
            "pings": int(beam.sizes["ping_time"]),
        },
        "scenarios": scenarios,
        "scalar_formula_max_absolute_error_db": max(numerical_errors),
        "scenario_spread_db_p50_p95_max_by_frequency": [
            np.nanpercentile(spread[ch], [50, 95, 100]).tolist() for ch in range(4)
        ],
        "xml_1465_vs_teos_50m_absolute_difference_db_p50_p95_max_by_frequency": [
            np.nanpercentile(np.abs(values[3][ch] - values[1][ch]), [50, 95, 100]).tolist()
            for ch in range(4)
        ],
        "scenario_support": "Common finite sample-index cells whose scenario-specific ranges all lie within 10–100 m; not range-regridded accuracy.",
        "assumptions": [
            "Homogeneous columns using measured 10/50/100 m profile values are illustrative sensitivity scenarios only.",
            "The daily profile is retrospective and unavailable at forecast issue time; never injected as predictor input.",
            "Range remains distance from transducer, never absolute depth.",
            "XML coefficient agreement and scalar recomputation establish implementation consistency, not field calibration accuracy.",
        ],
        "blockers": [
            "This numerical assay alone cannot approve physical calibration, QC or corpus eligibility.",
            "Its Iridium-only matching screen is historical; consult trajectory_matches.json for both trackers.",
            "Daily profiles are retrospective and cannot become issue-time environmental covariates.",
        ],
        "superseding_continuation_evidence": [
            "factory_certificate.json resolves factory serial55170 and coefficient provenance",
            "depth_environment_sensitivity.json and slant_environment_sensitivity.json assess the fixed regional method",
            "train-candidate-v2/processing_manifest.json reports actual TRAIN QC, not full corpus eligibility",
            "../../orchestration/reviews/R0-R1-continuation-addendum-20260926.md records the independent scope",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "final_process_rss_gib_not_peak": psutil.Process().memory_info().rss / 1024**3,
    }
    target = ROOT / "evidence/continuation/calibration_report.json"
    write_report(target, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "iridium_accepted",
                    "iridium_rejected",
                    "scalar_formula_max_absolute_error_db",
                    "scenario_spread_db_p50_p95_max_by_frequency",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
