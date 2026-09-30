"""Freeze matched scratch-supervised seeds 13/23; no fit or data decoding."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    template = json.loads((ROOT / "configs/native_downstream_direct_direct_end_to_end_seed7_h96_v1.json").read_text(encoding="utf-8"))
    pending = []
    for seed in (13, 23):
        path = ROOT / f"configs/native_downstream_direct_direct_end_to_end_seed{seed}_h96_v1.json"
        if path.exists():
            raise ValueError("Preserve frozen replication configurations")
        pending.append((path, dict(template, seed=seed)))
    for path, config in pending:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps({"status": "DIRECT_REPLICATION_CONFIGS_FROZEN_NOT_FITTED",
                      "seeds": [13, 23], "same_recipe_except_seed": True,
                      "prefit": "NOT_RUN", "final_test_access": "NOT_RUN"}))


if __name__ == "__main__":
    main()
