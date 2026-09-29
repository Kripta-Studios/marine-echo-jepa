"""Show bounded research-session progress without printing large tool payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", default="*-events*.jsonl")
    args = parser.parse_args()
    folder = ROOT / "evidence/ssl-research-v1"
    for path in sorted(folder.glob(args.match)):
        commands, messages = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item", {})
            if item.get("type") == "command_execution":
                commands.append(
                    {
                        "command": item.get("command", "")[:220],
                        "status": item.get("status"),
                        "exit_code": item.get("exit_code"),
                        "output_tail": item.get("aggregated_output", "")[-180:],
                    }
                )
            elif item.get("type") == "agent_message":
                messages.append(item.get("text", "")[-600:])
        print(
            json.dumps(
                {
                    "events": path.name,
                    "latest_commands": commands[-2:],
                    "latest_message": messages[-1:],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
