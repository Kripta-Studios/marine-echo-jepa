"""Copy the scoped completed builder deliverable into the coordinator checkout.

The builder's denied Git index and scratch locations are left untouched.
Coordinator integration writes only new, explicitly owned paths in its checkout.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT.parent / "marine-jepa-vnext-builder"
    files = [
        "src/marine_echo/models/native_temporal.py",
        "src/marine_echo/training/native_ssl.py",
        "tests/unit/test_native_temporal.py",
        "tests/integration/test_native_ssl.py",
        "orchestration/reports/NATIVE_SSL_IMPLEMENTATION_V1.md",
    ]
    evidence_source = source / "evidence/ssl-builder-v1"
    files += [
        str(p.relative_to(source)).replace("\\", "/")
        for p in evidence_source.rglob("*")
        if p.is_file()
    ]
    for relative in files:
        if (ROOT / relative).exists():
            raise ValueError(f"Preserve existing coordinator path: {relative}")
    copied = []
    for relative in files:
        original, target = source / relative, ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        expected = hashlib.sha256(original.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("Integrated builder bytes differ.")
        copied.append({"path": relative, "sha256": expected})
    report = {
        "status": "EXACT_SCOPED_BUILDER_BYTES_INTEGRATED_PARENT_VERIFICATION_PENDING",
        "builder_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
        "builder_git_index_denial": "PRESERVED_NOT_RETRIED",
        "builder_durable_scratch_denials": "PRESERVED_NOT_RETRIED",
        "files": copied,
    }
    (ROOT / "evidence/ssl-research-v1/builder-integration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "files": len(copied)}))


if __name__ == "__main__":
    main()
