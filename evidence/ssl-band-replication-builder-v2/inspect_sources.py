"""Read-only source snapshot and bounded source inspection; no numerical files."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "marine-echo-jepa"
PATHS = [
    "src/marine_echo/training/native_band_ssl.py",
    "src/marine_echo/training/native_band_downstream.py",
    "src/marine_echo/inference/native_band_acoustic.py",
    "src/marine_echo/inference/native_latent.py",
    "tools/execute_native_band_job.py",
    "tools/prepare_native_band_configs.py",
    "tools/native_reference_supervisor.py",
    "src/marine_echo/models/native_band_temporal.py",
    "orchestration/native_band_budget_owner_resolution_v1.json",
    "docs/adr/0023-owner-resolved-band-budget.md",
]
if sys.argv[1] == "snapshot":
    print(
        json.dumps(
            {
                p: {
                    "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest(),
                    "text": (ROOT / p).read_text(encoding="utf-8"),
                }
                for p in PATHS
            }
        )
    )
else:
    p, start, end = sys.argv[1:]
    lines = (ROOT / p).read_text(encoding="utf-8").splitlines()
    print(
        "\n".join(f"{i + 1}: {lines[i]}" for i in range(int(start) - 1, min(int(end), len(lines))))
    )
