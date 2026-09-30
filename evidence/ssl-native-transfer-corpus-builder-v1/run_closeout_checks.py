"""Witness focused synthetic checks and lint; no scientific process or input."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
BUILDER = BASE.parents[1]
FILES = [
    "src/marine_echo/data/native_transfer_corpus.py",
    "tools/materialize_native_transfer_corpus.py",
    "tests/unit/test_native_transfer_corpus.py",
    "tests/integration/test_native_transfer_corpus.py",
    "evidence/ssl-native-transfer-corpus-builder-v1/root_supervised_fixture.py",
    "evidence/ssl-native-transfer-corpus-builder-v1/snapshot_sources.py",
    "evidence/ssl-native-transfer-corpus-builder-v1/inspect_sources.py",
    "evidence/ssl-native-transfer-corpus-builder-v1/run_closeout_checks.py",
]


def main():
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    commands = [
        (
            "format-closeout-v2",
            [sys.executable, "-B", "-m", "ruff", "format", "--no-cache", *FILES],
        ),
        (
            "green-actual-ownership-closeout-v2",
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "tests/unit/test_native_transfer_corpus.py",
                "-k",
                "root_only_fixture",
                "--import-mode=importlib",
                "-p",
                "no:cacheprovider",
                "-q",
            ],
        ),
        (
            "lint-closeout-v2",
            [sys.executable, "-B", "-m", "ruff", "check", "--no-cache", *FILES],
        ),
        (
            "format-check-closeout-v2",
            [sys.executable, "-B", "-m", "ruff", "format", "--check", "--no-cache", *FILES],
        ),
    ]
    records = []
    for label, command in commands:
        with (BASE / (label + ".log")).open("x", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=BUILDER,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        records.append({"label": label, "command": command, "exit_code": result.returncode})
        print(json.dumps(records[-1]), flush=True)
        if result.returncode:
            break
    with (BASE / "executed-closeout-checks-v2.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "checks": records}, stream, indent=2
        )
        stream.write("\n")
    return int(any(record["exit_code"] for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
