"""Read owned, safe checkpoints to report actual completed training steps."""

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
        keys = (
            "status",
            "pretrain_steps",
            "readout_steps_all_probes",
            "optimizer_steps_total",
            "selected_pretrain_step",
            "selected_daily_dev_pinball",
            "resources",
            "inference_sha256",
        )
        print(json.dumps({key: value[key] for key in keys}, indent=2))
        return
    checkpoint = args.checkpoint or args.output / "latest.pt"
    if not checkpoint.exists():
        print(json.dumps({"status": "NO_DURABLE_CHECKPOINT_YET"}))
        return
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if value.get("kind") != "native_ssl_resume_v1":
        raise ValueError("Unsupported owned-run checkpoint kind.")
    state = value["state"]
    keys = (
        "phase",
        "pretrain_step",
        "readout_step",
        "optimizer_steps",
        "selected_pretrain_step",
        "best_score",
        "probe_best_score",
    )
    result = {key: state[key] for key in keys}
    result.update(
        resources=value["resources"],
        latest_training=state["sequence"][-1],
        development_selection=state["records"],
    )
    result["latest_training"].pop("context_indices", None)
    result["latest_training"].pop("target_indices", None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
