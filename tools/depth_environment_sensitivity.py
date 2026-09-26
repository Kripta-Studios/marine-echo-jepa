"""Prospective depth-resolved regional assay; never promotes a physical corpus."""

import hashlib
import itertools
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import echopype as ep
import gsw
import netCDF4
import numpy as np
import pandas as pd
import psutil
from echopype.utils import uwa

from marine_echo.data.environment import seawater_properties

ROOT = Path(__file__).resolve().parents[1]
EDGES = np.arange(0.0, 132.0, 2.0)
FREQUENCIES = np.array([38000, 125000, 200000, 455000])


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vector_absorption(temperature, salinity, pressure):
    """Broadcast the pinned Echopype0.11.1 AZFP scalar branch without changing its equation."""
    temperature, salinity, pressure = [
        np.asarray(value)[:, None] for value in (temperature, salinity, pressure)
    ]
    frequency = FREQUENCIES[None, :]
    temp_k = temperature + 273.0
    f1 = 1320.0 * temp_k * np.exp(-1700 / temp_k)
    f2 = 1.55e7 * temp_k * np.exp(-3052 / temp_k)
    k = 1 + pressure / 10.0
    a = 8.95e-8 * (1 + temperature * (2.29e-2 - 5.08e-4 * temperature))
    b = (salinity / 35.0) * 4.88e-7 * (1 + 0.0134 * temperature) * (1 - 0.00103 * k + 3.7e-7 * k**2)
    c = (
        4.86e-13
        * (1 + temperature * (-0.042 + temperature * (8.53e-4 - temperature * 6.23e-6)))
        * (1 + k * (-3.84e-4 + k * 7.57e-8))
    )
    return np.where(
        salinity == 0,
        c * frequency**2,
        (a * f1 * frequency**2) / (f1**2 + frequency**2)
        + (b * f2 * frequency**2) / (f2**2 + frequency**2)
        + c * frequency**2,
    )


def segments(start, distance, edges):
    """Yield lengths and bin IDs; -1 explicitly denotes outside-source fallback."""
    stop = start + distance
    cursor = start
    while cursor < stop - 1e-10:
        index = int(np.searchsorted(edges, cursor, side="right") - 1)
        if index < 0:
            boundary, index = edges[0], -1
        elif index >= len(edges) - 1:
            boundary, index = stop, -1
        else:
            boundary = edges[index + 1]
        end = min(float(boundary), stop)
        yield end - cursor, index
        cursor = end


def distance_for_time(start, duration, edges, speed, fallback):
    """Invert a piecewise-constant one-way slowness integral exactly."""
    cursor, remaining = float(start), float(duration)
    while remaining > 1e-14:
        index = int(np.searchsorted(edges, cursor, side="right") - 1)
        if index < 0:
            boundary, local_speed = edges[0], fallback
        elif index >= len(speed):
            return cursor - start + remaining * fallback
        else:
            boundary, local_speed = edges[index + 1], speed[index]
        available = (boundary - cursor) / local_speed
        if remaining <= available:
            return cursor - start + remaining * local_speed
        cursor = float(boundary)
        remaining -= available
    return cursor - start


def absorption_integral(start, distance, edges, alpha, fallback):
    result = np.zeros(alpha.shape[1])
    for length, index in segments(start, distance, edges):
        result += length * (fallback if index < 0 else alpha[index])
    return result


def main():
    from continuation_records import write_report

    started = time.monotonic()
    contract = ROOT / "evidence/continuation/depth_environment_contract.json"
    # Hash recorded separately before executing any environmental calculations.
    expected = (
        (ROOT / "evidence/continuation/depth_environment_contract.sha256").read_text().strip()
    )
    if sha(contract) != expected:
        raise ValueError("Prospective contract differs from recorded hash")
    if ep.__version__ != "0.11.1" or gsw.__version__ != "3.6.23":
        raise ValueError("Pinned physical packages required")
    original = json.loads(
        (ROOT / "evidence/continuation/fixed_environment_sensitivity.json").read_text()
    )
    nominal = original["nominal"]
    c0 = nominal["sound_speed"]
    alpha0 = uwa.calc_absorption(frequency=FREQUENCIES, **nominal, formula_source="AZFP")
    lower, upper = np.full((65, 3), np.inf), np.full((65, 3), -np.inf)
    location_lo, location_hi = np.full((65, 2), np.inf), np.full((65, 2), -np.inf)
    cmin, cmax = np.full(65, np.inf), np.full(65, -np.inf)
    amin, amax = np.full((65, 4), np.inf), np.full((65, 4), -np.inf)
    counts = np.zeros(65, dtype=np.int64)
    source_days = [set() for _ in range(65)]
    day_sets = [set() for _ in range(65)]
    qc, sources = [], []
    sampled_rss = 0

    def update(depth, values, lat, lon, dates, source):
        nonlocal sampled_rss
        if not len(values):
            return
        lat, lon = np.broadcast_to(lat, (len(values),)), np.broadcast_to(lon, (len(values),))
        if not np.isfinite(np.column_stack([depth, values, lat, lon])).all():
            raise ValueError("Finite depth, physical triples and source coordinates required")
        sa = gsw.SA_from_SP(values[:, 1], values[:, 2], lon, lat)
        speed = gsw.sound_speed_t_exact(sa, values[:, 0], values[:, 2])
        alpha = vector_absorption(values[:, 0], values[:, 1], values[:, 2])
        ids = np.minimum((depth / 2).astype(int), 64)
        for index in np.unique(ids):
            mask = ids == index
            lower[index] = np.minimum(lower[index], values[mask].min(axis=0))
            upper[index] = np.maximum(upper[index], values[mask].max(axis=0))
            locations = np.column_stack([lat[mask], lon[mask]])
            location_lo[index] = np.minimum(location_lo[index], locations.min(axis=0))
            location_hi[index] = np.maximum(location_hi[index], locations.max(axis=0))
            cmin[index], cmax[index] = (
                min(cmin[index], speed[mask].min()),
                max(cmax[index], speed[mask].max()),
            )
            amin[index], amax[index] = (
                np.minimum(amin[index], alpha[mask].min(axis=0)),
                np.maximum(amax[index], alpha[mask].max(axis=0)),
            )
            counts[index] += int(mask.sum())
            days = set(np.asarray(dates)[mask].tolist())
            day_sets[index].update(days)
            source_days[index].update((source, day) for day in days)
        sampled_rss = max(sampled_rss, psutil.Process().memory_info().rss)

    inventory = json.loads((ROOT / "evidence/continuation/sit_inventory.json").read_text())
    for item in inventory["files"]:
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError("SIT source mismatch")
        sources.append({"path": item["path"], "sha256": item["sha256"]})
        with path.open(encoding="utf-8") as stream:
            for header, line in enumerate(stream, 1):
                if line.strip() == "*/":
                    break
                if header > 200:
                    raise ValueError("Publisher header not found")
        columns = [
            "Date/Time",
            "Latitude",
            "Longitude",
            "Temp [°C]",
            "Sal",
            "Press [dbar]",
            "Depth water [m]",
            "QF water temp",
            "QF sal",
            "QF water press",
            "QF water depth",
        ]
        accepted, rejected = 0, 0
        for chunk in pd.read_csv(
            path, sep="\t", skiprows=header, usecols=columns, chunksize=100_000
        ):
            relevant = (
                chunk["Date/Time"].ge("2020-02-17")
                & chunk["Date/Time"].lt("2020-08-02")
                & (-chunk["Depth water [m]"]).between(0, 130)
            )
            triples = chunk[["Temp [°C]", "Sal", "Press [dbar]"]].to_numpy(dtype=float)
            good = (
                relevant
                & chunk[columns[-4:]].isin([1, 2]).all(axis=1)
                & np.isfinite(triples).all(axis=1)
            )
            accepted += int(good.sum())
            rejected += int((relevant & ~good).sum())
            update(
                -chunk.loc[good, "Depth water [m]"].to_numpy(),
                triples[good],
                chunk.loc[good, "Latitude"].to_numpy(),
                chunk.loc[good, "Longitude"].to_numpy(),
                chunk.loc[good, "Date/Time"].str[:10].to_numpy(),
                item["buoy"],
            )
        qc.append(
            {
                "source": item["buoy"],
                "accepted": accepted,
                "rejected_relevant": rejected,
                "flags": "ALL_T_SP_P_DEPTH_1_OR_2",
            }
        )
        print(json.dumps(qc[-1]), flush=True)
    profile = ROOT / "data/raw/environment/mosaic_core/MOSAiC_daily_profiles.nc"
    registered = json.loads((ROOT / "evidence/calibration/environment_inventory.json").read_text())
    if sha(profile) != next(row["sha256"] for row in registered if row["name"] == profile.name):
        raise ValueError("Composite source mismatch")
    sources.append({"path": str(profile.relative_to(ROOT)), "sha256": sha(profile)})
    accepted, rejected = 0, 0
    with netCDF4.Dataset("memory", memory=profile.read_bytes()) as dataset:
        for i, value in enumerate(dataset["time"][:]):
            stamp = (
                (datetime.fromordinal(int(value)) + timedelta(days=float(value) % 1 - 366))
                .date()
                .isoformat()
            )
            if not "2020-02-17" <= stamp < "2020-08-02":
                continue
            lat, lon = float(dataset["latitude"][i]), float(dataset["longitude"][i])
            depth, sal, temp = [
                np.ma.filled(dataset[name][:, i], np.nan)
                for name in ("depth", "salinity", "temperature")
            ]
            relevant = (depth >= 0) & (depth <= 130)
            good = relevant & np.isfinite(sal) & np.isfinite(temp)
            accepted += int(good.sum())
            rejected += int((relevant & ~good).sum())
            converted = seawater_properties(depth[good], sal[good], temp[good], lat, lon)
            triples = np.column_stack(
                [
                    converted["in_situ_temperature_c"],
                    converted["practical_salinity"],
                    converted["pressure_dbar"],
                ]
            )
            update(
                depth[good], triples, lat, lon, np.full(int(good.sum()), stamp), "daily_composite"
            )
    qc.append(
        {
            "source": "daily_composite",
            "accepted": accepted,
            "rejected_relevant": rejected,
            "flags": "UNKNOWN_NOT_CERTIFIED",
        }
    )
    global_c = original["combined_sampled_speed_extrema_m_s"]
    global_a = [
        np.array(original[key]) for key in ("sampled_alpha_min_db_m", "sampled_alpha_max_db_m")
    ]
    coverage = []
    for index in range(65):
        missing = counts[index] == 0
        if missing:
            cmin[index], cmax[index] = global_c
            amin[index], amax[index] = global_a
        else:
            lo, hi = lower[index] - [0.05, 0.1, 1.0], upper[index] + [0.05, 0.1, 1.0]
            lo[2] = max(0.0, lo[2])
            grid = np.array(
                list(itertools.product(*(np.linspace(a, b, 17) for a, b in zip(lo, hi))))
            )
            alpha = vector_absorption(grid[:, 0], grid[:, 1], grid[:, 2])
            amin[index], amax[index] = (
                np.minimum(amin[index], alpha.min(axis=0)),
                np.maximum(amax[index], alpha.max(axis=0)),
            )
            for lat, lon in itertools.product(*zip(location_lo[index], location_hi[index])):
                sa = gsw.SA_from_SP(grid[:, 1], grid[:, 2], lon, lat)
                speed = gsw.sound_speed_t_exact(sa, grid[:, 0], grid[:, 2])
                cmin[index], cmax[index] = (
                    min(cmin[index], speed.min()),
                    max(cmax[index], speed.max()),
                )
        coverage.append(
            {
                "depth_edges_m": EDGES[index : index + 2].tolist(),
                "accepted_count": int(counts[index]),
                "distinct_days": len(day_sets[index]),
                "distinct_source_days": len(source_days[index]),
                "sources": sorted({source for source, _ in source_days[index]}),
                "global_fallback": bool(missing),
                "observed_T_SP_P_min": None if missing else lower[index].tolist(),
                "observed_T_SP_P_max": None if missing else upper[index].tolist(),
                "latitude_longitude_min": None if missing else location_lo[index].tolist(),
                "latitude_longitude_max": None if missing else location_hi[index].tolist(),
                "c_min_max_m_s": [float(cmin[index]), float(cmax[index])],
                "alpha_min_db_m": amin[index].tolist(),
                "alpha_max_db_m": amax[index].tolist(),
            }
        )
    results = []
    for start in np.arange(0.0, 30.01, 0.25):
        worst, displacement = np.zeros(4), 0.0
        for nominal_range in np.arange(10.0, 100.01, 0.5):
            low_range = distance_for_time(start, nominal_range / c0, EDGES, cmin, global_c[0])
            high_range = distance_for_time(start, nominal_range / c0, EDGES, cmax, global_c[1])
            displacement = max(
                displacement, abs(low_range - nominal_range), abs(high_range - nominal_range)
            )
            alpha_lo = absorption_integral(start, low_range, EDGES, amin, global_a[0])
            alpha_hi = absorption_integral(start, high_range, EDGES, amax, global_a[1])
            # Include both endpoints even where an endpoint equals a bin boundary.
            ids = set(
                index
                for _, index in segments(
                    start + low_range - 1e-9, high_range - low_range + 2e-9, EDGES
                )
            )
            local_min = min(global_c[0] if i < 0 else cmin[i] for i in ids)
            local_max = max(global_c[1] if i < 0 else cmax[i] for i in ids)
            delta_lo = (
                20 * math.log10(low_range / nominal_range)
                - 10 * math.log10(local_max / c0)
                + 2 * alpha_lo
                - 2 * alpha0 * nominal_range
            )
            delta_hi = (
                20 * math.log10(high_range / nominal_range)
                - 10 * math.log10(local_min / c0)
                + 2 * alpha_hi
                - 2 * alpha0 * nominal_range
            )
            worst = np.maximum(worst, np.maximum(np.abs(delta_lo), np.abs(delta_hi)))
        results.append(
            {
                "sampled_transducer_depth_m": float(start),
                "max_abs_Sv_delta_db": worst.tolist(),
                "max_abs_range_displacement_m": displacement,
                "primary_limits_passed": bool(worst[0] <= 1.0 and displacement <= 2.0),
            }
        )
    report = {
        "status": "DEPTH_RESOLVED_DIAGNOSTIC_REQUIRES_GEOMETRY_AND_R0",
        "benchmark_eligible": False,
        "physically_approved_days": 0,
        "contract_sha256": sha(contract),
        "code_sha256": sha(Path(__file__)),
        "parent_failed_report_sha256": sha(
            ROOT / "evidence/continuation/fixed_environment_sensitivity.json"
        ),
        "checked_at": datetime.now(UTC).isoformat(),
        "sources": sources,
        "QC": qc,
        "nominal": nominal,
        "depth_bin_coverage": coverage,
        "scenarios": results,
        "all_sampled_scenarios_passed": all(row["primary_limits_passed"] for row in results),
        "max_abs_Sv_delta_db_all_scenarios": np.max(
            [row["max_abs_Sv_delta_db"] for row in results], axis=0
        ).tolist(),
        "max_abs_range_displacement_m_all_scenarios": max(
            row["max_abs_range_displacement_m"] for row in results
        ),
        "limitations": [
            "No independently evidenced absolute transducer-depth interval yet",
            "Downward straight rays only; orientation and finite beam geometry not certified",
            "Sampled regional extrema and depth grid are not continuous physical bounds or local water observations",
            "No actual acoustic QC support invariance assessed",
            "No acoustic test values or forecast scores accessed",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "peak_sampled_process_rss_gib_not_tree": sampled_rss / 1024**3,
    }
    write_report(ROOT / "evidence/continuation/depth_environment_sensitivity.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "all_sampled_scenarios_passed",
                    "max_abs_Sv_delta_db_all_scenarios",
                    "max_abs_range_displacement_m_all_scenarios",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
