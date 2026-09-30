"""Stdlib-only source/metadata inspection; never numerical artifacts."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
PATHS = [
    "src/marine_echo/training/native_prefix_transfer.py",
    "tests/unit/test_native_prefix_transfer.py",
    "tests/integration/test_native_prefix_transfer.py",
    "src/marine_echo/evaluation/native_assessment.py",
    "tools/execute_bounded_native_assessment.py",
    "tools/execute_native_assessment_worker.py",
    "tools/execute_native_prefix_job.py",
    "tools/native_reference_supervisor.py",
    "src/marine_echo/training/native_band_replication_ssl.py",
    "src/marine_echo/training/native_band_replication_downstream.py",
    "src/marine_echo/inference/native_band_replication_acoustic.py",
    "src/marine_echo/inference/native_band_replication_encoder.py",
    "src/marine_echo/inference/native_band_replication_latent.py",
    "tools/execute_native_band_replication_job.py",
    "orchestration/native_band_budget_owner_resolution_v1.json",
    "docs/adr/0023-owner-resolved-band-budget.md",
]
mode = sys.argv[1]
if mode in ("export", "export_builder"):
    raw = ((MAIN if mode == "export" else BUILDER) / sys.argv[2]).read_bytes()
    print(json.dumps({"sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode("utf-8")}))
elif mode == "snapshot":
    group = sys.argv[2]
    paths = PATHS[:3] if group == "prefix" else PATHS[3:]
    records = {}
    for path in paths:
        raw = (MAIN / path).read_bytes()
        records[path] = {"sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode("utf-8")}
        if group == "prefix" and (BUILDER / path).read_bytes() != raw:
            raise ValueError("Builder prefix differs from current main: " + path)
    print(json.dumps(records))
else:
    path, start, end = sys.argv[1:]
    lines = (MAIN / path).read_text(encoding="utf-8").splitlines()
    print(
        "\n".join(f"{i + 1}: {lines[i]}" for i in range(int(start) - 1, min(int(end), len(lines))))
    )
