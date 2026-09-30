"""Freeze exact completed DEV prediction paths before distinct reconstruction review."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = {
        "shared_short_probe": "shared_ssl_seed7_h96_cuda0",
        "cf_short_probe": "cf_jepa_seed7_h96_deterministic",
        "shared_frozen_readout": "shared_ssl_frozen_readout_seed7_h96",
        "shared_full_finetune": "shared_ssl_full_finetune_seed7_h96",
        "direct_end_to_end": "direct_direct_end_to_end_seed7_h96",
        "lightgbm": "lightgbm_seed7_h96_text",
        "persistence": "persistence_h96",
        "seasonal24": "seasonal24_h96",
    }
    methods = {}
    for name, folder in names.items():
        path = ROOT / "outputs/native_acoustic_ssl_v1" / folder
        report = path / ("run.json" if (path / "run.json").exists() else "result.json")
        status = json.loads(report.read_text(encoding="utf-8"))["status"]
        if status not in ("COMPLETED", "COMPLETED_DEVELOPMENT_REFERENCE"):
            raise ValueError("Only closed complete actual artifacts may be compared.")
        prediction = path / "predictions.npz"
        if not prediction.is_file():
            raise ValueError("Completed prediction artifact absent.")
        methods[name] = str(prediction.resolve())
    manifest = {
        "role": "development",
        "evidence_kind": "REVIEWED_SAVED_PREDICTIONS",
        "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
        "root_coordinator_session_id": "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10",
        "scientific_protocol_path": str(
            (ROOT / "docs/adr/0015-native-acoustic-ssl-research.md").resolve()
        ),
        "bootstrap_seed": 20260929,
        "bootstrap_replicates": 2000,
        "block_hours": 48,
        "block_days": 2,
        "floor": 18,
        "methods": methods,
        "reference": "lightgbm",
        "interpretation": "DEV reconstruction; short selection probes are separately labeled and not strong transfer endpoints; one deployment, no final-test or SOTA claim",
    }
    path = ROOT / "orchestration/native_development_comparison_v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"status": "PREPARED_NOT_SCORED", "methods": len(methods), "manifest": str(path)}
        )
    )


if __name__ == "__main__":
    main()
