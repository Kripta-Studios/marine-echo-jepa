"""Run current scoped integration checks with fresh logs and explicit exit/resource records."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    output = ROOT / "evidence/continuation" / datetime.now(UTC).strftime("checks-%Y%m%dT%H%M%SZ")
    output.mkdir(exist_ok=False)
    python = ROOT / ".venv/Scripts/python.exe"
    gpu_python = ROOT.parent / "EVOCON_JEPA_Codex_Handoff/e-jepa-ttc/.venv/Scripts/python.exe"
    node = ROOT / ".runtime/node_modules/node/bin/node.exe"
    npm = Path("C:/Program Files/nodejs/node_modules/npm/bin/npm-cli.js")
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("UV_INTERNAL__PYTHONHOME", None)
    env.update(
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=str(ROOT / "src"),
        MARINE_ECHO_REAL_AZFP_HOUR=str(
            ROOT.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"
        ),
        MARINE_ECHO_REAL_AZFP_ZIP=str(ROOT / "data/raw/pangaea/949811/azfp55170-bioacoustics.zip"),
        PATH=str(node.parent) + os.pathsep + env["PATH"],
    )
    model_tests = [
        "tests/unit/test_models_forecast.py",
        "tests/unit/test_models_hybrid.py",
        "tests/scientific/test_model_gradients.py",
        "tests/scientific/test_model_controls.py",
        "tests/scientific/test_model_collapse.py",
        "tests/integration/test_training_resume.py",
        "tests/integration/test_training_gpu.py",
    ]
    commands = [
        ("ruff-format", [str(python), "-m", "ruff", "format", "--check", "src", "tests"], ROOT),
        ("ruff-check", [str(python), "-m", "ruff", "check", "src", "tests"], ROOT),
        ("mypy", [str(python), "tools/check_types.py"], ROOT),
        (
            "pytest-app-data",
            [
                str(python),
                "-m",
                "pytest",
                "tests/unit",
                "tests/integration",
                "tests/scientific",
                "tests/api",
                "tests/security",
                "-q",
                *["--ignore=" + p for p in model_tests],
            ],
            ROOT,
        ),
        (
            "pytest-model-software",
            [str(gpu_python), "-m", "pytest", *model_tests, "-q", "-rs"],
            ROOT,
        ),
        ("r2-integrity", [str(python), "release/meeting-20260926-r2/Verify-Release.py"], ROOT),
    ]
    for check in ("format:check", "lint", "typecheck", "test:run", "build"):
        commands.append(
            ("web-" + check.replace(":", "-"), [str(node), str(npm), "run", check], ROOT / "web")
        )
    if args.browser:
        commands.append(
            (
                "web-browser",
                [str(node), str(npm), "run", "test:e2e", "--", "--reporter=list"],
                ROOT / "web",
            )
        )
    results = []
    for name, command, cwd in commands:
        started = time.monotonic()
        peak = 0
        timed_out = False
        with (output / (name + ".log")).open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    peak = max(
                        peak,
                        sum(
                            p.memory_info().rss for p in [parent, *parent.children(recursive=True)]
                        ),
                    )
                    if time.monotonic() - started > 600 or peak > 22 * 1024**3:
                        timed_out = True
                        for child in reversed(parent.children(recursive=True)):
                            child.terminate()
                        parent.terminate()
                        process.wait(timeout=15)
                except psutil.Error:
                    pass
                time.sleep(0.2)
        result = {
            "name": name,
            "command": command,
            "exit_code": process.returncode,
            "resource_or_time_stop": timed_out,
            "seconds": time.monotonic() - started,
            "peak_sampled_process_tree_rss_gib": peak / 1024**3,
            "log": str((output / (name + ".log")).relative_to(ROOT)),
        }
        results.append(result)
        (output / "checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
    return (
        0
        if all(row["exit_code"] == 0 and not row["resource_or_time_stop"] for row in results)
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
