"""Build a truthful, hash-addressed diagnostic release from executed evidence."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from marine_echo.evaluation.protocol import digest_file

REASON = "MOSAiC environmental profiles were found, but instrument geometry, environmental applicability and independent calibration review remain unresolved. No eligible physical-unit benchmark corpus exists."
FAMILIES = [
    "persistence",
    "seasonal",
    "ridge",
    "hist_gradient_boosting",
    "direct",
    "ema_jepa",
    "shared_sigreg",
    "ema_jepa_plus_raw",
    "shared_sigreg_plus_raw",
]


def run_registry(root: Path) -> dict[str, Any]:
    config = json.loads((root / "configs/experiments.json").read_text(encoding="utf-8"))
    rows = []
    for family in FAMILIES:
        seeds = config["seeds"] if family in ("direct", "ema_jepa", "shared_sigreg") else [7]
        for seed in seeds:
            rows.append(
                {
                    "run_id": f"{family}-seed{seed}",
                    "family": family,
                    "seed": seed,
                    "phase": "selected",
                    "status": "BLOCKED",
                    "metrics": None,
                    "reason": REASON,
                    "updates": 0,
                }
            )
    for family in ("direct", "ema_jepa", "shared_sigreg"):
        for configuration in range(2):
            rows.append(
                {
                    "run_id": f"{family}-development{configuration}-seed7",
                    "family": family,
                    "seed": 7,
                    "phase": "development",
                    "status": "BLOCKED",
                    "metrics": None,
                    "reason": REASON,
                    "updates": 0,
                }
            )
    for family in ("ema_jepa", "shared_sigreg"):
        for control in config["controls"]:
            rows.append(
                {
                    "run_id": f"{family}-{control}-seed7",
                    "family": family,
                    "seed": 7,
                    "phase": control,
                    "status": "BLOCKED",
                    "metrics": None,
                    "reason": REASON,
                    "updates": 0,
                }
            )
    return {
        "schema_version": "1.0",
        "protocol_id": config["protocol_id"],
        "status": "BLOCKED",
        "runs": rows,
        "completed_benchmark_runs": 0,
        "test_opened": False,
    }


def build_diagnostic(root: Path, output: Path) -> dict[str, Any]:
    """Publish once by same-filesystem rename; reject all existing output paths."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Release output exists. Choose a new directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".release-stage-", dir=output.parent))
    try:
        result = _build_diagnostic(root, staged)
        staged.rename(output)
        result["output"] = str(output)
        return result
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def _build_diagnostic(root: Path, output: Path) -> dict[str, Any]:
    raw = root / "evidence/data/diagnostic_raw_replay_2020-02-17.json"
    inventory = json.loads(
        (root / "data/manifests/mosaic_azfp_down_2020/source_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    observed = json.loads(raw.read_text(encoding="utf-8"))
    if (
        observed["calibrated"]
        or observed["units"] != "raw_counts"
        or observed["data_kind"] != "public_real"
    ):
        raise ValueError("Diagnostic release requires the verified real raw-count artifact.")
    artifacts = output / "artifacts"
    if (artifacts / "catalog.json").exists():
        raise ValueError("Release output already published. Use a new output directory.")
    artifacts.mkdir(parents=True, exist_ok=True)
    archive = next(f for f in inventory["files"] if f["name"].endswith(".zip"))
    protocol_path = root / "reports/active/protocol.json"
    evidence = {
        "title": "Marine Echo JEPA — executed evidence and open gates",
        "gates": {
            "G0_DATA": "BLOCKED_CALIBRATION",
            "G1_ENGINEERING": "DIAGNOSTIC_ONLY_PENDING_RELEASE_REVIEW",
            "G2_EXPERIMENT": "BLOCKED_NOT_RUN",
            "G3_INCREMENTAL_VALUE": "NOT_EVALUATED",
            "commercial_validation": "NOT_EVALUATED",
        },
        "source_files": inventory["files"],
        "protocol": json.loads(protocol_path.read_text()) if protocol_path.exists() else None,
        "attribution": "De La Torre, Pedro R; Berge, Jørgen; Granskog, Mats A; Katlein, Christian; Divine, Dmitry V; Raphael, Ian; Geoffroy, Maxime; Vogedes, Daniel; Itkin, Polona; Daase, Malin; Zolich, Artur; Cottier, Finlo (2022): Data from downward looking Acoustic zooplankton and fish profiler (AZFP) deployed on drifting sea ice in the Arctic during MOSAiC expedition. PANGAEA. https://doi.org/10.1594/PANGAEA.949811. CC BY 4.0.",
        "limitations": [
            "The full P0 forecasting MVP is incomplete. This package is an engineering diagnostic, not a trained forecasting release.",
            REASON,
            "The manual lists downward serial 55169; actual archive XML/DPL identify 55170. Manual is not an instrument calibration certificate.",
            "Raw counts are averaged over 64 sample-index groups and trailing 15-minute bins. Counts are not calibrated Sv, dB, range in metres, depth, fish or biomass.",
            "Replay is a predetermined first-full-day diagnostic (2020-02-17), selecting 24 stable .01B chunks. Other chunks remain preserved but are excluded here. This is not a complete scientific preprocessing pipeline.",
            "Replay assumes zero-latency availability; actual end-user delivery times are not known.",
            "Required families, seeds, controls, uncertainty, freshness-error study and sealed test are NOT_RUN. No JEPA win or negative hypothesis result exists.",
            "OOI access and one-file calibration code path work. Full fallback acquisition is RESOURCE_PLAN_BLOCKED under the current raw-data/disk budget and needs a separate prospective protocol.",
            "Arctic research deployment. Not tuna, catch or Marine Instruments validation.",
            "Local SHA-256 and CRC verify local integrity; no publisher checksum or signature was supplied.",
        ],
    }
    catalog: dict[str, Any] = {
        "schema_version": "1.0",
        "release_class": "ENGINEERING_DEMO_ONLY",
        "datasets": [
            {
                "id": observed["dataset_id"],
                "name": "MOSAiC downward AZFP 55170",
                "domain": "Arctic ice-tethered research deployment",
                "license": "CC BY 4.0",
                "calibration_status": "RAW_COUNTS_ONLY",
                "start": "2020-02-17T00:00:00Z",
                "end": "2020-02-18T00:15:00Z",
                "source_sha256": archive["sha256"],
                "observations_id": "observations",
            }
        ],
        "models": [
            {"id": family, "family": family, "status": "NOT_RUN", "reason": REASON}
            for family in FAMILIES
        ],
        "experiments": run_registry(root)["runs"],
        "artifacts": {},
        "exports": ["research-export"],
        "forecasts": {},
    }
    for key, body, kind in [
        ("observations", observed, "observations"),
        ("research", evidence, "evidence"),
        ("research-export", evidence, "export"),
    ]:
        path = artifacts / (key + ".json")
        path.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        catalog["artifacts"][key] = {"path": path.name, "sha256": digest_file(path), "kind": kind}
    if not (root / "web/dist/index.html").exists():
        raise ValueError("Build the frontend before publishing the release.")
    shutil.copytree(root / "web/dist", output / "web")
    (artifacts / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "ENGINEERING_DEMO_ONLY",
        "output": str(output),
        "raw_rows": len(observed["rows"]),
        "benchmark_runs": 0,
    }
