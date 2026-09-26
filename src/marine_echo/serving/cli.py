"""Local command interface; blocked science exits nonzero and records the real reason."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

from marine_echo.evaluation.protocol import create_protocol, digest_file, verify_seal
from marine_echo.serving.release import REASON, build_diagnostic, run_registry

app = typer.Typer(no_args_is_help=True)
data = typer.Typer(no_args_is_help=True)
protocol_commands = typer.Typer(no_args_is_help=True)
benchmark = typer.Typer(no_args_is_help=True)
experiment = typer.Typer(no_args_is_help=True)
release = typer.Typer(no_args_is_help=True)
for name, command in [
    ("data", data),
    ("protocol", protocol_commands),
    ("benchmark", benchmark),
    ("experiment", experiment),
    ("release", release),
]:
    app.add_typer(command, name=name)


def emit(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, allow_nan=False))


def root_path() -> Path:
    return Path.cwd()


def blocked(operation: str) -> None:
    emit({"operation": operation, "status": "BLOCKED", "reason": REASON, "result": None})
    raise typer.Exit(2)


@app.command()
def doctor(config: Path = Path("configs/mvp.json")) -> None:
    """Inspect this runtime; CUDA smoke is a separate explicitly selected environment."""
    import platform
    import shutil

    import psutil

    emit(
        {
            "python": sys.version.split()[0],
            "platform": platform.system(),
            "config_sha256": digest_file(config),
            "free_disk_gib": shutil.disk_usage(root_path()).free / 1024**3,
            "available_ram_gib": psutil.virtual_memory().available / 1024**3,
            "process_rss_gib": psutil.Process().memory_info().rss / 1024**3,
            "cuda": "See evidence/runtime/existing_cuda.json; not imported by the app.",
        }
    )


@data.command()
def probe(config: Path = Path("configs/datasets.json")) -> None:
    """Check existing local source presence without redownloading owner files."""
    registry = json.loads(config.read_text(encoding="utf-8"))
    source = root_path() / "data/raw/pangaea/949811"
    expected = registry["datasets"][0]["download_items"]
    emit(
        {
            "dataset_id": registry["active_dataset"],
            "network_downloads": 0,
            "files": [
                {"name": item["name"], "present": (source / item["name"]).is_file()}
                for item in expected
            ],
        }
    )
    if not all((source / item["name"]).is_file() for item in expected):
        raise typer.Exit(2)


@data.command()
def inventory(dataset: str = "mosaic_azfp_down_2020") -> None:
    """Hash every immutable local source and compare the registered inventory."""
    from marine_echo.data.local_archive import inventory_local_source

    if dataset != "mosaic_azfp_down_2020":
        raise typer.BadParameter("Unknown or unregistered dataset.")
    result = inventory_local_source(root_path() / "data/raw/pangaea/949811")
    expected = json.loads(
        (root_path() / "data/manifests/mosaic_azfp_down_2020/source_inventory.json").read_text()
    )
    if result["files"] != expected["files"]:
        emit({"status": "INTEGRITY_MISMATCH"})
        raise typer.Exit(2)
    emit({"status": "LOCAL_INTEGRITY_VERIFIED", "files": result["files"], "redownloaded": 0})


@data.command()
def download(dataset: str = "mosaic_azfp_down_2020", resume: bool = False) -> None:
    """Reuse already supplied files after integrity checks; never redownload valid data."""
    inventory(dataset)


@data.command()
def preprocess(
    dataset: str = "mosaic_azfp_down_2020",
    resume: bool = False,
    diagnostic: bool = False,
    extracted: Path | None = None,
) -> None:
    """Generate a raw-count diagnostic; physical preprocessing requires calibration."""
    if not diagnostic:
        blocked("physical_preprocessing")
    if dataset != "mosaic_azfp_down_2020" or extracted is None:
        raise typer.BadParameter(
            "Diagnostic replay needs the primary dataset and --extracted directory."
        )
    command = [
        sys.executable,
        "tools/local_data_replay.py",
        "--extracted",
        str(extracted),
        "--day",
        "2020-02-17",
        "--output",
        "outputs/recomputed_diagnostic.json",
    ]
    result = subprocess.run(command, check=False)
    raise typer.Exit(result.returncode)


@data.command()
def validate(dataset: str = "mosaic_azfp_down_2020") -> None:
    """Report data readiness without treating raw counts as calibrated acoustic units."""
    blocked(f"validate:{dataset}")


@protocol_commands.command("create")
def protocol_create(config: Path = Path("configs/experiments.json")) -> None:
    emit(
        create_protocol(config, Path("configs/datasets.json"), Path("reports/active/protocol.json"))
    )


@benchmark.command("baselines")
def baselines(protocol: str = "active") -> None:
    blocked("baseline_benchmark")


@app.command()
def train(family: str, seed: int = 7, protocol: str = "active") -> None:
    """Training promotion is fail-closed while the physical-unit corpus is blocked."""
    if family not in ("direct", "ema_jepa", "shared_sigreg") or seed not in (7, 13, 23):
        raise typer.BadParameter("Family/seed is outside the declared P0 budget.")
    blocked(f"train:{family}:seed{seed}")


@experiment.command("complete-p0")
def complete_p0(protocol: str = "active", resume: bool = False) -> None:
    """Report the current campaign blocker without replacing attempt or exposure evidence.

    The scientific executor is not implemented while corpus/protocol approval is absent.
    This checkpoint operation must never be reported as a completed finite campaign.
    """
    if protocol != "active":
        raise typer.BadParameter("Only the explicit active protocol is registered.")
    path = Path("reports/active/training_registry.json")
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
        emit(
            {
                "operation": "complete-p0",
                "status": "BLOCKED",
                "reason": REASON,
                "executor_status": "NOT_IMPLEMENTED",
                "existing_registry_preserved": True,
                "registry": registry,
            }
        )
    else:
        registry = run_registry(root_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(registry, indent=2) + "\n")
        emit({"status": "BLOCKED", "executor_status": "NOT_IMPLEMENTED", "registry": registry})
    raise typer.Exit(2)


@protocol_commands.command("freeze")
def freeze(protocol: str = "active", review_id: str = "R2") -> None:
    """Reject promotion before independent R2 and calibrated-corpus acceptance."""
    blocked(f"freeze:{review_id}")


@app.command()
def evaluate(partition: str = "test", protocol: str = "active") -> None:
    """Sealed-test access requires a complete verified seal; never tune on its outcomes."""
    seal = Path("reports/active/seal.json")
    if seal.exists():
        verify_seal(Path("reports/active/protocol.json"), seal)
    blocked(f"evaluate:{partition}")


@app.command()
def report(protocol: str = "active") -> None:
    if protocol != "active":
        raise typer.BadParameter("Only the explicit active protocol is registered.")
    registry_path = Path("reports/active/training_registry.json")
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else run_registry(root_path())
    )
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
    unattempted = all(
        run.get("status") in {"BLOCKED", "NOT_RUN"}
        and run.get("updates", 0) == 0
        and run.get("metrics") is None
        and not any(value for key, value in run.items() if key not in unattempted_fields)
        for run in registry["runs"]
    )
    emit(
        {
            "release_class": "ENGINEERING_DEMO_ONLY",
            "G0_DATA": "BLOCKED",
            "G2_EXPERIMENT": "NOT_RUN" if unattempted else "INCOMPLETE",
            "G3_INCREMENTAL_VALUE": "NOT_EVALUATED",
            "commercial_validation": "NOT_EVALUATED",
            "reason": REASON,
            "registry": registry,
        }
    )


@release.command("build")
def release_build(protocol: str = "active", output: Path = Path("release/demo")) -> None:
    emit(build_diagnostic(root_path(), output))


@app.command()
def serve(
    host: str = "127.0.0.1", port: int = 8765, artifact_root: Path = Path("release/demo/artifacts")
) -> None:
    """Serve verified immutable artifacts and the production app on loopback."""
    import uvicorn

    from marine_echo.serving.api import create_app

    if host not in ("127.0.0.1", "localhost", "::1"):
        raise typer.BadParameter("Only loopback serving is authorized.")
    uvicorn.run(
        create_app(artifact_root, artifact_root.parent / "web"),
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    app()
