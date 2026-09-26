"""Execute repository checks with durable exit codes and bounded resource observations."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

# uv-managed interpreters inject a home path into children; do not pass a 3.12
# standard-library path to the explicitly selected existing 3.11 CUDA interpreter.
os.environ.pop("PYTHONHOME", None)
os.environ.pop("UV_INTERNAL__PYTHONHOME", None)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "evidence/tests/final"
    output.mkdir(parents=True, exist_ok=True)
    python = root / ".venv/Scripts/python.exe"
    node = root / ".runtime/node_modules/node/bin/node.exe"
    npm = Path("C:/Program Files/nodejs/node_modules/npm/bin/npm-cli.js")
    commands = [
        ("ruff-format", [str(python), "-m", "ruff", "format", "--check", "src", "tests"], root),
        ("ruff-check", [str(python), "-m", "ruff", "check", "src", "tests"], root),
        ("mypy", [str(python), "tools/check_types.py"], root),
        ("pytest", [str(python), "-m", "pytest", "tests/unit", "tests/integration", "tests/scientific", "tests/api", "tests/security", "--ignore=tests/unit/test_models_forecast.py", "--ignore=tests/unit/test_models_hybrid.py", "--ignore=tests/scientific/test_model_gradients.py", "--ignore=tests/scientific/test_model_controls.py", "--ignore=tests/scientific/test_model_collapse.py", "--ignore=tests/integration/test_training_resume.py", "--ignore=tests/integration/test_training_gpu.py", "--cov=marine_echo.contracts", "--cov=marine_echo.evaluation", "--cov=marine_echo.serving.api", "--cov-report=term-missing", "-q"], root),
    ]
    for check in ("format:check", "lint", "typecheck", "test:run", "build"):
        commands.append(("web-" + check.replace(":", "-"), [str(node), str(npm), "run", check], root / "web"))
    env = {**os.environ, "MARINE_ECHO_REAL_AZFP_HOUR": str(root.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"), "MARINE_ECHO_REAL_AZFP_ZIP": str(root / "data/raw/pangaea/949811/azfp55170-bioacoustics.zip"), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PATH": str(node.parent) + os.pathsep + os.environ["PATH"]}
    results = []
    for name, command, cwd in commands:
        started = time.monotonic()
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            peak = 0
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    peak = max(peak, sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)]))
                except psutil.Error:
                    pass
                if time.monotonic() - started > 600:
                    process.terminate()
                    process.wait(timeout=10)
                    break
                time.sleep(.5)
            result = {"name": name, "command": command, "exit_code": process.returncode, "seconds": time.monotonic() - started, "peak_process_tree_rss_gib": peak / 1024**3, "log": str((output / (name + ".log")).relative_to(root))}
            log.write("\n" + json.dumps(result) + "\n")
        results.append(result)
        print(json.dumps(result), flush=True)
        (output / "checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r["exit_code"] == 0 for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
