"""Verify ORIGINAL handoff hashes, JSON/TOML and task dependencies (no network)."""
from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.reference_contracts import validate_forecast, validate_task_dag  # noqa: E402


def verify_manifest(root: Path, manifest: Path) -> int:
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = root / relative
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe manifest path: {relative}")
        if not target.is_file():
            raise ValueError(f"Missing handoff member: {relative}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Modified/corrupt original handoff member: {relative}")
        count += 1
    return count


def main() -> int:
    try:
        count = verify_manifest(ROOT, ROOT / "HANDOFF_SHA256SUMS")
        json_count = 0
        for folder in ("configs", "schemas", "orchestration", "tests_handoff/fixtures"):
            for file in (ROOT / folder).rglob("*.json"):
                json.loads(file.read_text(encoding="utf-8"))
                json_count += 1
        for file in (ROOT / ".codex").rglob("*.toml"):
            with file.open("rb") as stream:
                tomllib.load(stream)
        validate_task_dag(json.loads((ROOT / "orchestration/tasks.json").read_text())["tasks"])
        fixture = json.loads((ROOT / "tests_handoff/fixtures/forecast.synthetic.json").read_text())
        validate_forecast(fixture)
        print(f"PASS: {count} original handoff hashes, {json_count} JSON files, TOML, DAG and synthetic contract.")
        print("No model, application, account access or dataset download was validated by this command.")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
