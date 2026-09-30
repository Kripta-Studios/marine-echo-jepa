"""Archive the closed unexecuted draft before protocol-conformance correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"


def main():
    folder = BUILDER / "evidence/ssl-comparison-builder-v1"
    handoff = folder / "handoff-verification.json"
    receipt = json.loads(handoff.read_text(encoding="utf-8"))
    captured = {}
    for raw, expected in receipt["files_sha256"].items():
        data = (BUILDER / raw).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Closed draft already changed; no snapshot may be invented.")
        captured[raw] = data
    captured["evidence/ssl-comparison-builder-v1/handoff-verification.json"] = handoff.read_bytes()
    destination = ROOT / "evidence/ssl-comparison-draft-v1"
    destination.mkdir()
    for raw, data in captured.items():
        path = destination / raw
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    note = {
        "status": "UNEXECUTED_DRAFT_PRESERVED_REQUIRES_PROTOCOL_CORRECTION",
        "scientific_comparison": "NOT_RUN",
        "reason": "Coordinator draft recipe contradicted ADR0015; fixed protocol takes precedence",
        "draft_block_days": 7,
        "draft_seed": 1729,
        "required_block_hours": 48,
        "required_seed": 20260929,
        "files_sha256": {raw: hashlib.sha256(data).hexdigest() for raw, data in captured.items()},
    }
    (destination / "preservation.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": note["status"], "files": len(captured)}))


if __name__ == "__main__":
    main()
