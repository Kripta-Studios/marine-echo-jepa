"""Bounded Echopype AZFP parse and calibration feasibility probe."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import tempfile
from pathlib import Path

import echopype as ep
import numpy as np


def _value_summary(variable: object) -> dict[str, object]:
    values = np.asarray(variable.values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "units": variable.attrs.get("units"),
        "size": int(values.size),
        "finite_count": int(finite.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe one real AZFP hour without assumed seawater parameters."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ed = ep.open_raw(args.raw, sonar_model="AZFP", xml_path=args.xml)
    beam = ed["Sonar/Beam_group1"]
    report: dict[str, object] = {
        "raw_file": args.raw.name,
        "xml_file": args.xml.name,
        "echopype_version": ep.__version__,
        "sonar_model": "AZFP",
        "frequency_hz": beam["frequency_nominal"].values.tolist(),
        "pings": int(beam.sizes["ping_time"]),
        "first_ping_utc": str(beam["ping_time"].values[0]) + "Z",
        "last_ping_utc": str(beam["ping_time"].values[-1]) + "Z",
        "range_samples": int(beam.sizes["range_sample"]),
        "environment_variables_present": sorted(ed["Environment"].data_vars),
        "environment_values": {
            name: _value_summary(ed["Environment"][name])
            for name in (
                "sound_speed_indicative",
                "absorption_indicative",
                "temperature",
            )
            if name in ed["Environment"]
        },
        "temperature_source_note": "Echopype derives Environment.temperature from AZFP ancillary thermistor counts and labels it sea_water_temperature. The source manual describes internal/sonar temperature; no evidence establishes it as seawater at the sampled range.",
        "vendor_variable_names": sorted(ed["Vendor_specific"].data_vars),
        "compute_sv_signature": str(inspect.signature(ep.calibrate.compute_Sv)),
        "azfp_calibration_requirement": "Echopype 0.11.1 get_env_params_AZFP requires user salinity and pressure; temperature is needed unless independently supplied sound speed and absorption are supplied. All require provenance before physical-unit use.",
        "physical_calibration_claim": "BLOCKED_UNVERIFIED_ENVIRONMENT",
        "reason": "Instrument XML supplies a configured sound speed but no verified seawater temperature, salinity, pressure, or absorption provenance.",
    }
    try:
        candidate = ep.calibrate.compute_Sv(ed)
    except (TypeError, ReferenceError, ValueError, KeyError) as error:
        report["no_environment_compute_sv"] = {
            "status": "ERROR",
            "type": type(error).__name__,
            "message": str(error)[:500],
        }
    else:
        report["no_environment_compute_sv"] = {
            "status": "RETURNED_UNVERIFIED",
            "note": "No physical-unit result is accepted without verified environmental provenance.",
            "sv_dims": dict(candidate["Sv"].sizes),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=args.output.name + ".stage.",
        delete=False,
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        staged = Path(stream.name)
    os.replace(staged, args.output)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
