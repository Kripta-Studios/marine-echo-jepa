"""Extract only routing fields from exact research sessions; never read credentials."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    evidence = ROOT / "evidence/ssl-research-v1"
    sessions = [("01a0ef1a-b166-7f83-91b7-2a2aff7c1b10", "primary_root")]
    for events in evidence.glob("*-events*.jsonl"):
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") == "thread.started":
                sessions.append((item["thread_id"], events.name))
                break
    routing = []
    for ident, events in sessions:
        contexts = []
        for path in (Path.home() / ".codex/sessions").rglob(f"*{ident}*"):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("type") == "turn_context":
                    contexts.append(
                        {
                            key: item["payload"].get(key)
                            for key in ("model", "effort", "sandbox_policy", "approval_policy")
                        }
                    )
        routing.append({"session_id": ident, "events": events, "observed_contexts": contexts})
    (evidence / "routing.json").write_text(json.dumps(routing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(routing, indent=2))
    pins = []
    for folder, url in (
        ("cf-jepa-vnext", "https://github.com/WDSLab/CF-JEPA"),
        ("lewm-vnext", "https://github.com/galilai-group/stable-worldmodel"),
    ):
        source = ROOT / "external" / folder
        if not (source / ".git").exists():
            continue
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=True
        )
        files = {}
        for path in source.rglob("*"):
            if path.is_file() and ".git" not in path.parts:
                files[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        pins.append(
            {
                "url": url,
                "commit": result.stdout.strip(),
                "files": files,
                "setup_scripts_executed": False,
                "dependencies_installed": False,
            }
        )
    (evidence / "source-pins.json").write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
