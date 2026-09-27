"""Build a truthful, hash-addressed diagnostic release from executed evidence."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from marine_echo.evaluation.protocol import digest_file
from marine_echo.serving.raw_development import build_raw_development_report

REASON = "No eligible independently reviewed physical-unit corpus or completed benchmark is available. Calibration assumptions and TRAIN-only QC evidence do not authorize forecasts."
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


def validate_recorded_registry(root: Path, registry: dict[str, Any]) -> None:
    """Reject ambiguous counters or an incomplete configured run inventory."""
    expected = run_registry(root)
    runs = registry.get("runs")
    count = registry.get("completed_benchmark_runs")
    if (
        registry.get("schema_version") != expected["schema_version"]
        or registry.get("protocol_id") != expected["protocol_id"]
        or not isinstance(registry.get("status"), str)
        or not registry["status"]
        or type(registry.get("test_opened")) is not bool
        or type(count) is not int
        or not 0 <= count <= len(expected["runs"])
        or not isinstance(runs, list)
        or len(runs) != len(expected["runs"])
    ):
        raise ValueError("Invalid recorded campaign registry")
    identities = {row["run_id"]: row for row in expected["runs"]}
    seen: set[str] = set()
    for row in runs:
        if not isinstance(row, dict) or not isinstance(row.get("run_id"), str):
            raise TypeError("Invalid recorded campaign row")
        run_id = row["run_id"]
        if run_id in seen or run_id not in identities:
            raise ValueError("Recorded campaign inventory differs")
        reference = identities[run_id]
        if (
            any(row.get(key) != reference[key] for key in ("family", "seed", "phase"))
            or type(row.get("seed")) is not int
            or type(row.get("updates")) is not int
            or row["updates"] < 0
            or not isinstance(row.get("status"), str)
            or not row["status"]
            or "metrics" not in row
        ):
            raise ValueError("Recorded campaign identity or counters differ")
        seen.add(run_id)


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


def build_v2_research(root: Path, output: Path) -> dict[str, Any]:
    """Publish reviewed raw-code development beside preserved blocked v1 evidence."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Research output exists. Choose a new directory.")
    report = build_raw_development_report(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".v2-research-stage-", dir=output.parent))
    try:
        _build_diagnostic(root, staged)
        artifacts = staged / "artifacts"
        catalog = json.loads((artifacts / "catalog.json").read_text(encoding="utf-8"))
        evidence = json.loads((artifacts / "research.json").read_text(encoding="utf-8"))
        evidence["title"] = "Marine Echo JEPA — v1 failure and v2 raw-code development"
        evidence["gates"].update(
            {
                "G0_DATA": "CALIBRATED_V2_INELIGIBLE_RAW_CODE_ENGINEERING_ONLY",
                "G1_ENGINEERING": "REAL_RAW_CODE_DEVELOPMENT_EXECUTED_OFFLINE_RELEASE_PENDING_REVIEW",
                "G2_EXPERIMENT": "BLOCKED_CALIBRATED_CORE_0_OF_25",
                "G3_INCREMENTAL_VALUE": "NOT_EVALUATED_NO_JEPA_RUN",
            }
        )
        evidence["raw_development"] = {
            "study_id": report["study_id"],
            "run_id": report["run_id"],
            "status": report["status"],
            "quantity": report["quantity"],
            "unit": report["unit"],
            "source_result_sha256": report["source_result_sha256"],
            "direct_checkpoint_128_sha256": report["direct_checkpoint_128_sha256"],
            "review_record_sha256": report["review_record_sha256"],
            "horizons": report["horizons"],
            "all_issued_rows_artifact": "raw-development",
        }
        evidence["limitations"].extend(report["limitations"])
        evidence["limitations"].append(
            "The preserved 25-slot v1 registry is a historical blocked registry; the separate raw-code ridge/direct development does not turn those runs into completed calibrated experiments."
        )
        catalog["release_class"] = "OFFLINE_RESEARCH_ENGINEERING_ONLY"
        evidence["release_class"] = catalog["release_class"]
        research_path = artifacts / "research.json"
        export_path = artifacts / "research-export.json"
        for path in (research_path, export_path):
            path.write_text(
                json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        catalog["artifacts"]["research"]["sha256"] = digest_file(research_path)
        catalog["artifacts"]["research-export"]["sha256"] = digest_file(export_path)
        report_path = artifacts / "raw-development.json"
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        catalog["artifacts"]["raw-development"] = {
            "path": report_path.name,
            "sha256": digest_file(report_path),
            "kind": "evidence",
        }
        catalog["artifacts"]["raw-development-export"] = {
            "path": report_path.name,
            "sha256": digest_file(report_path),
            "kind": "export",
        }
        catalog["exports"].append("raw-development-export")
        for path in (artifacts / "catalog.json", staged / "catalog.json"):
            path.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(output)
        return {
            "status": catalog["release_class"],
            "output": str(output),
            "raw_development_rows": len(report["rows"]),
            "calibrated_benchmark_runs": 0,
            "raw_development_sha256": digest_file(output / "artifacts/raw-development.json"),
        }
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
    registry_path = root / "reports/active/training_registry.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else run_registry(root)
    )
    validate_recorded_registry(root, registry)
    unattempted_fields = {
        "run_id",
        "family",
        "seed",
        "phase",
        "status",
        "reason",
        "updates",
        "metrics",
    }
    unattempted = (
        registry["status"] in {"BLOCKED", "NOT_RUN"}
        and registry.get("completed_benchmark_runs", 0) == 0
        and all(
            run.get("status") in {"BLOCKED", "NOT_RUN"}
            and run.get("updates", 0) == 0
            and run.get("metrics") is None
            and not any(value for key, value in run.items() if key not in unattempted_fields)
            for run in registry["runs"]
        )
    )
    evidence = {
        "title": "Marine Echo JEPA — executed evidence and open gates",
        "gates": {
            "G0_DATA": "BLOCKED_CORPUS_ELIGIBILITY_AND_REVIEW",
            "G1_ENGINEERING": "DIAGNOSTIC_ONLY_PENDING_RELEASE_REVIEW",
            "G2_EXPERIMENT": "BLOCKED_NOT_RUN" if unattempted else "INCOMPLETE_RECORDED_ATTEMPTS",
            "G3_INCREMENTAL_VALUE": "NOT_EVALUATED",
            "commercial_validation": "NOT_EVALUATED",
        },
        "source_files": inventory["files"],
        "run_registry": registry,
        "forecast_unavailability_reason": REASON,
        "protocol": json.loads(protocol_path.read_text()) if protocol_path.exists() else None,
        "attribution": "De La Torre, Pedro R; Berge, Jørgen; Granskog, Mats A; Katlein, Christian; Divine, Dmitry V; Raphael, Ian; Geoffroy, Maxime; Vogedes, Daniel; Itkin, Polona; Daase, Malin; Zolich, Artur; Cottier, Finlo (2022): Data from downward looking Acoustic zooplankton and fish profiler (AZFP) deployed on drifting sea ice in the Arctic during MOSAiC expedition. PANGAEA. https://doi.org/10.1594/PANGAEA.949811. CC BY 4.0.",
        "limitations": [
            "The full P0 forecasting MVP is incomplete. This package is an engineering diagnostic, not a trained forecasting release.",
            REASON,
            "Actual archive XML/DPL identify instrument 55170. Consult the current calibration evidence for factory certificate mapping and environmental assumptions; the historical manual alone does not establish calibration.",
            "Raw counts are averaged over 64 sample-index groups and trailing 15-minute bins. Counts are not calibrated Sv, dB, range in metres, depth, fish or biomass.",
            "Replay is a predetermined first-full-day diagnostic (2020-02-17), selecting 24 stable .01B chunks. Other chunks remain preserved but are excluded here. This is not a complete scientific preprocessing pipeline.",
            "Replay assumes zero-latency availability; actual end-user delivery times are not known.",
            "Required families, seeds, controls, uncertainty, freshness-error study and sealed test are NOT_RUN. No JEPA win or negative hypothesis result exists.",
            "OOI access and one-file calibration code path work. Full fallback acquisition is RESOURCE_PLAN_BLOCKED under the current raw-data/disk budget and needs a separate prospective protocol.",
            "Arctic research deployment. Not tuna, catch or Marine Instruments validation.",
            "Local SHA-256 and CRC verify local integrity; no publisher checksum or signature was supplied.",
        ],
    }
    summary_path = root / "evidence/continuation/public_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("schema_version") != "1.0"
            or summary.get("evidence_scope") != "TRAIN_QC_ONLY_NOT_BENCHMARK"
            or not isinstance(summary.get("limitations"), list)
            or not all(isinstance(item, str) for item in summary["limitations"])
            or not isinstance(summary.get("forecast_unavailability_reason"), str)
        ):
            raise ValueError("Unsupported continuation evidence summary")
        evidence["continuation"] = summary
        evidence["continuation_summary_sha256"] = digest_file(summary_path)
        evidence["limitations"].extend(summary["limitations"])
        evidence["forecast_unavailability_reason"] = summary["forecast_unavailability_reason"]
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
            {
                "id": family,
                "family": family,
                "status": "UNAVAILABLE",
                "reason": evidence["forecast_unavailability_reason"],
            }
            for family in FAMILIES
        ],
        "experiments": registry["runs"],
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
