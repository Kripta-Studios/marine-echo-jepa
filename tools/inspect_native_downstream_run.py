"""Inspect a completed report or explicitly named immutable transfer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    report = args.output / "run.json"
    if report.exists():
        value = json.loads(report.read_text(encoding="utf-8"))
        fields = (
            "status",
            "mode",
            "supervised_updates",
            "selected_supervised_step",
            "selected_daily_dev_pinball",
            "parameters_optimized",
            "label_observations_processed",
            "sample_presentations",
            "selection_records",
            "resources_additional_downstream",
        )
        print(json.dumps({key: value[key] for key in fields}, indent=2))
        return
    if args.checkpoint is None:
        print(json.dumps({"status": "RUNNING_NO_IMMUTABLE_CHECKPOINT_SPECIFIED"}))
        return
    if (
        args.checkpoint.name == "latest.pt"
        or args.checkpoint.parent.resolve() != args.output.resolve()
    ):
        raise ValueError("Read only a named immutable checkpoint in this run.")
    value = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if value.get("kind") != "native_downstream_resume_v1":
        raise ValueError("Unsupported transfer checkpoint kind.")
    state = value["state"]
    print(
        json.dumps(
            {
                "step": state["step"],
                "best_step": state["best_step"],
                "best_score": state["best_score"],
                "selection_records": state["records"],
                "resources": value["resources"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
