"""Execute the preregistered nonpromotable regional calibration sensitivity assay."""

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
from continuation_records import write_report
from echopype.utils import uwa

from marine_echo.data.environment import seawater_properties

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_HASH = "9808f38fc559af9b46b7c8c4dcf0fa5ee2f230ab313b4d6fade48f663ff76f86"
FREQUENCIES = np.array([38000, 125000, 200000, 455000])


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    started = time.monotonic()
    contract_path = ROOT / "evidence/continuation/fixed_environment_contract.json"
    if sha(contract_path) != CONTRACT_HASH:
        raise ValueError("Prospective contract changed; no assay execution.")
    addendum = ROOT / "evidence/continuation/fixed_environment_contract_addendum.json"
    if sha(addendum) != "452bd5c037d057fbbcc715ed8e5c7a5f17fd1d487ecf83f7e9077496c4a84110":
        raise ValueError("Prospective path/coordinate addendum changed; no execution.")
    contract = json.loads(contract_path.read_text())
    if ep.__version__ != "0.11.1" or gsw.__version__ != "3.6.23":
        raise ValueError("Pinned physical packages required.")
    nominal = contract["nominal"]
    env0 = {
        "temperature": nominal["temperature_in_situ_c"],
        "salinity": nominal["salinity_practical_pss78"],
        "pressure": nominal["pressure_dbar"],
        "sound_speed": nominal["sound_speed_m_s"],
    }
    minima, maxima = np.full(3, np.inf), np.full(3, -np.inf)
    observed_speed = [np.inf, -np.inf]
    sources, qc = [], []
    peak_sampled = 0

    def update(values: np.ndarray, latitude, longitude) -> None:
        nonlocal minima, maxima, peak_sampled
        if values.size:
            if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
                raise ValueError("Finite T/SP/P triples required for envelope.")
            minima, maxima = (
                np.minimum(minima, values.min(axis=0)),
                np.maximum(maxima, values.max(axis=0)),
            )
            absolute_salinity = gsw.SA_from_SP(values[:, 1], values[:, 2], longitude, latitude)
            speed = gsw.sound_speed_t_exact(absolute_salinity, values[:, 0], values[:, 2])
            if not np.isfinite(speed).all():
                raise ValueError("Source-coordinate sound speed conversion failed.")
            observed_speed[0] = min(observed_speed[0], float(speed.min()))
            observed_speed[1] = max(observed_speed[1], float(speed.max()))
        peak_sampled = max(peak_sampled, psutil.Process().memory_info().rss)

    inventory = json.loads((ROOT / "evidence/continuation/sit_inventory.json").read_text())
    for item in inventory["files"]:
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError("SIT source differs from inventory.")
        sources.append({"path": item["path"], "sha256": item["sha256"]})
        with path.open(encoding="utf-8") as stream:
            header = 0
            for line in stream:
                header += 1
                if line.strip() == "*/":
                    break
                if header > 200:
                    raise ValueError("Unexpected publisher header.")
        accepted, rejected = 0, 0
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
        for chunk in pd.read_csv(
            path, sep="\t", skiprows=header, usecols=columns, chunksize=100_000
        ):
            period = chunk["Date/Time"].ge("2020-02-17") & chunk["Date/Time"].lt("2020-08-02")
            depth = -chunk["Depth water [m]"]
            relevant = period & depth.between(0, 130)
            values = chunk[["Temp [°C]", "Sal", "Press [dbar]"]].to_numpy(dtype=np.float64)
            good = (
                relevant
                & chunk[["QF water temp", "QF sal", "QF water press", "QF water depth"]]
                .isin([1, 2])
                .all(axis=1)
                & np.isfinite(values).all(axis=1)
            )
            accepted += int(good.sum())
            rejected += int((relevant & ~good).sum())
            update(
                values[good],
                chunk.loc[good, "Latitude"].to_numpy(),
                chunk.loc[good, "Longitude"].to_numpy(),
            )
        qc.append(
            {
                "source": item["buoy"],
                "accepted_T_SP_P_triples": accepted,
                "rejected_relevant_rows": rejected,
            }
        )
        print(json.dumps(qc[-1]), flush=True)
    profile_path = ROOT / "data/raw/environment/mosaic_core/MOSAiC_daily_profiles.nc"
    registered = json.loads((ROOT / "evidence/calibration/environment_inventory.json").read_text())
    if sha(profile_path) != next(r["sha256"] for r in registered if r["name"] == profile_path.name):
        raise ValueError("Composite source differs from inventory.")
    sources.append({"path": str(profile_path.relative_to(ROOT)), "sha256": sha(profile_path)})
    composite_n, composite_rejected = 0, 0
    nominal_location = None
    with netCDF4.Dataset("memory", memory=profile_path.read_bytes()) as dataset:
        for i, value in enumerate(dataset["time"][:]):
            stamp = datetime.fromordinal(int(value)) + timedelta(days=float(value) % 1 - 366)
            if not "2020-02-17" <= stamp.date().isoformat() < "2020-08-02":
                continue
            lat, lon = float(dataset["latitude"][i]), float(dataset["longitude"][i])
            if stamp.date().isoformat() == "2020-03-03":
                nominal_location = (lat, lon)
            depth, salinity, temperature = [
                np.ma.filled(dataset[name][:, i], np.nan)
                for name in ("depth", "salinity", "temperature")
            ]
            relevant = (depth >= 0) & (depth <= 130)
            good = relevant & np.isfinite(salinity) & np.isfinite(temperature)
            composite_rejected += int((relevant & ~good).sum())
            if not good.any():
                continue
            converted = seawater_properties(
                depth[good], salinity[good], temperature[good], lat, lon
            )
            triples = np.column_stack(
                [
                    converted["in_situ_temperature_c"],
                    converted["practical_salinity"],
                    converted["pressure_dbar"],
                ]
            )
            composite_n += len(triples)
            update(triples, lat, lon)
    if nominal_location is None or not np.isfinite(minima).all():
        raise ValueError("No valid nominal location or regional envelope.")
    qc.append(
        {
            "source": "daily_composite",
            "accepted_T_SP_P_triples": composite_n,
            "rejected_relevant_rows": composite_rejected,
            "quality_flags": "UNKNOWN_NOT_CERTIFIED",
        }
    )
    allowances = np.array([0.05, 0.1, 1.0])
    lower, upper = minima - allowances, maxima + allowances
    lower[2] = max(0.0, lower[2])
    if lower[1] <= 0 or lower[0] < -3 or upper[0] > 40:
        raise ValueError("Envelope outside implemented seawater formula validity assumptions.")
    lat, lon = nominal_location
    nominal_absorption = np.asarray(
        uwa.calc_absorption(frequency=FREQUENCIES, **env0, formula_source="AZFP")
    )
    ranges = np.linspace(10, 100, 181)
    worst = np.zeros(4)
    worst_absorption = np.zeros(4)
    worst_range = 0.0
    max_case = [None] * 4
    corner_cases = []
    alpha_min, alpha_max = np.full(4, np.inf), np.full(4, -np.inf)
    speed_extrema = observed_speed.copy()
    for t, sal, pressure in itertools.product(
        *(np.linspace(lo, hi, 17) for lo, hi in zip(lower, upper))
    ):
        sa = gsw.SA_from_SP(sal, pressure, lon, lat)
        speed = float(gsw.sound_speed_t_exact(sa, t, pressure))
        absorption = np.asarray(
            uwa.calc_absorption(
                frequency=FREQUENCIES,
                temperature=t,
                salinity=sal,
                pressure=pressure,
                formula_source="AZFP",
            )
        )
        alpha_min, alpha_max = np.minimum(alpha_min, absorption), np.maximum(alpha_max, absorption)
        speed_extrema = [min(speed_extrema[0], speed), max(speed_extrema[1], speed)]
        ratio = speed / env0["sound_speed"]
        delta = 10 * np.log10(ratio) + 2 * ranges[:, None] * (
            absorption * ratio - nominal_absorption
        )
        abs_only = 2 * ranges[:, None] * (absorption - nominal_absorption)
        maxima_case = np.max(np.abs(delta), axis=0)
        for channel in range(4):
            if maxima_case[channel] > worst[channel]:
                max_case[channel] = {
                    "temperature": float(t),
                    "salinity": float(sal),
                    "pressure": float(pressure),
                    "sound_speed": speed,
                }
        worst = np.maximum(worst, maxima_case)
        worst_absorption = np.maximum(worst_absorption, np.max(np.abs(abs_only), axis=0))
        worst_range = max(worst_range, float(np.max(np.abs(ranges * (ratio - 1)))))
        if all(value in (lo, hi) for value, lo, hi in zip((t, sal, pressure), lower, upper)):
            corner_cases.append(
                {
                    "temperature": float(t),
                    "salinity": float(sal),
                    "pressure": float(pressure),
                    "sound_speed": speed,
                }
            )
    conditional_path_worst = np.zeros(4)
    for range_speed, local_speed, alpha in itertools.product(
        speed_extrema, speed_extrema, (alpha_min, alpha_max)
    ):
        ratio = range_speed / env0["sound_speed"]
        path_delta = (
            20 * math.log10(ratio)
            - 10 * math.log10(local_speed / env0["sound_speed"])
            + 2 * ranges[:, None] * (alpha * ratio - nominal_absorption)
        )
        conditional_path_worst = np.maximum(
            conditional_path_worst, np.max(np.abs(path_delta), axis=0)
        )
        worst_range = max(worst_range, float(np.max(np.abs(ranges * (ratio - 1)))))
    extracted = (
        ROOT.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"
    )
    raw, xml = extracted / "20030300.01A", extracted / "20021600.XML"
    manifest = json.loads((extracted / ".extraction-manifest.json").read_text())
    for path in (raw, xml):
        if sha(path) != next(
            row["sha256"] for row in manifest["files"] if row["path"] == path.name
        ):
            raise ValueError("Fixed TRAIN input differs from inventory.")
        sources.append({"path": path.name, "sha256": sha(path)})
    echodata = ep.open_raw(raw, sonar_model="AZFP", xml_path=xml)
    reference = ep.calibrate.compute_Sv(echodata, env_params=env0)
    baseline_sv = reference.Sv.values
    baseline_range = reference.echo_range.values
    numerical_errors, finite_edges = [], []
    for scenario in corner_cases:
        calibrated = ep.calibrate.compute_Sv(echodata, env_params=scenario)
        actual = calibrated.Sv.values - baseline_sv
        absorption = np.asarray(
            uwa.calc_absorption(frequency=FREQUENCIES, **scenario, formula_source="AZFP")
        )
        ratio = scenario["sound_speed"] / env0["sound_speed"]
        predicted = 10 * math.log10(ratio) + 2 * baseline_range * (
            absorption[:, None, None] * ratio - nominal_absorption[:, None, None]
        )
        finite = np.isfinite(actual) & (baseline_range >= 10) & (baseline_range <= 100)
        numerical_errors.append(float(np.max(np.abs(actual[finite] - predicted[finite]))))
        primary_valid = np.isfinite(calibrated.Sv.values[0])
        primary_ranges = calibrated.echo_range.values[0][primary_valid]
        finite_edges.append(
            {
                "scenario": scenario,
                "finite38_range_m": [float(primary_ranges.min()), float(primary_ranges.max())],
                "contains_10_100m": bool(
                    primary_ranges.min() <= 10 and primary_ranges.max() >= 100
                ),
            }
        )
    if max(numerical_errors) > 1e-8:
        raise ValueError("Independent difference formula disagrees with upstream calibration.")
    numeric_pass = bool(
        worst[0] <= 1.0
        and conditional_path_worst[0] <= 1.0
        and worst_range <= 2.0
        and all(row["contains_10_100m"] for row in finite_edges)
    )
    report = {
        "status": "SAMPLED_LIMITS_PASSED_REQUIRES_R0_REVIEW"
        if numeric_pass
        else "FIXED_ENVIRONMENT_SAMPLED_LIMITS_FAILED",
        "benchmark_eligible": False,
        "physically_approved_days": 0,
        "contract_sha256": CONTRACT_HASH,
        "code_sha256": sha(Path(__file__)),
        "sources": sources,
        "contract_addendum_sha256": sha(
            ROOT / "evidence/continuation/fixed_environment_contract_addendum.json"
        ),
        "checked_at": datetime.now(UTC).isoformat(),
        "nominal": env0,
        "nominal_lat_lon": nominal_location,
        "observed_min_T_SP_P": minima.tolist(),
        "observed_max_T_SP_P": maxima.tolist(),
        "expanded_lower_T_SP_P": lower.tolist(),
        "expanded_upper_T_SP_P": upper.tolist(),
        "QC": qc,
        "samples": 4913,
        "corners": len(corner_cases),
        "frequency_hz": FREQUENCIES.tolist(),
        "max_abs_Sv_delta_db": worst.tolist(),
        "max_abs_absorption_only_delta_db": worst_absorption.tolist(),
        "max_abs_range_displacement_m": worst_range,
        "worst_cases_per_frequency": max_case,
        "observed_own_coordinate_speed_extrema_m_s": observed_speed,
        "combined_sampled_speed_extrema_m_s": speed_extrema,
        "sampled_alpha_min_db_m": alpha_min.tolist(),
        "sampled_alpha_max_db_m": alpha_max.tolist(),
        "conditional_path_integral_at_100m_db_lower_upper_by_frequency": np.stack(
            [200 * alpha_min, 200 * alpha_max], axis=1
        ).tolist(),
        "conditional_path_max_abs_Sv_delta_db": conditional_path_worst.tolist(),
        "path_bound_premise": "Every local path condition lies within the regional envelope; alpha/speed extrema are sampled, not continuous-certified. Different extrema for path and local pulse volume intentionally overbound homogeneous scenarios.",
        "primary_numerical_limits_passed": numeric_pass,
        "upstream_scalar_delta_max_error_db": max(numerical_errors),
        "corner_finite_range_checks": finite_edges,
        "limitations": [
            "A sampled regional envelope does not prove a continuous bound or local water conditions",
            "Homogeneous rectangular extremes are conservative plausibility scenarios, not observed water columns",
            "Finite range support is not QC support and does not establish noise/saturation/edge eligibility invariance",
            "Factory calibration is not field calibration",
            "No acoustic test arrays or forecast scores were inspected",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "peak_sampled_process_rss_gib_not_tree": peak_sampled / 1024**3,
    }
    write_report(ROOT / "evidence/continuation/fixed_environment_sensitivity.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "observed_min_T_SP_P",
                    "observed_max_T_SP_P",
                    "max_abs_Sv_delta_db",
                    "max_abs_range_displacement_m",
                    "upstream_scalar_delta_max_error_db",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
