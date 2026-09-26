"""Verify the real fallback parser/calibration path without benchmark promotion."""

import json
import time
from pathlib import Path

import echopype as ep
import numpy as np
import psutil


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "evidence/data/ooi_diagnostic_inventory.json").read_text())
    started = time.monotonic()
    ed = ep.open_raw(root / inventory["path"], sonar_model="EK60")
    sv = ep.calibrate.compute_Sv(ed)
    data = np.asarray(sv["Sv"].values)
    finite = np.isfinite(data)
    report = {
        "dataset_id": inventory["dataset_id"], "source_sha256": inventory["sha256"],
        "echopype_version": ep.__version__, "status": "CALIBRATION_CODE_PATH_EXECUTED",
        "calibration_basis": "EK60 instrument and indicative environment fields recorded in raw CON0/data, interpreted by pinned Echopype; no independent field calibration claim",
        "environment_variables": list(ed["Environment"].data_vars),
        "frequencies_hz": sv["frequency_nominal"].values.tolist(),
        "shape": list(data.shape), "finite_fraction": float(finite.mean()),
        "min_sv_db": float(data[finite].min()), "max_sv_db": float(data[finite].max()),
        "elapsed_seconds": time.monotonic() - started,
        "process_rss_gib": psutil.Process().memory_info().rss / 1024**3,
        "benchmark_eligible": False, "primary_dataset_unchanged": True,
        "physical_unit_benchmark_promotion": "BLOCKED_NO_ELIGIBLE_FALLBACK_CORPUS_OR_PROTOCOL",
    }
    output = root / "outputs/ooi_diagnostic_scipy.nc"
    output.parent.mkdir(parents=True, exist_ok=True)
    sv[["Sv", "frequency_nominal"]].isel(ping_time=slice(0, 8)).to_netcdf(output, engine="scipy")
    import xarray as xr
    with xr.open_dataset(output, engine="scipy") as restored:
        report["roundtrip_equal"] = bool(np.allclose(restored["Sv"], sv["Sv"].isel(ping_time=slice(0, 8)), equal_nan=True))
    (root / "evidence/data/ooi_calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if finite.any() and report["roundtrip_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
