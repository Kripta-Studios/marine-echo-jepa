"""Freeze strong CF seed13/23 endpoint configs; parents require actual reviews."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    pending = []
    for mode in ("frozen_readout", "full_finetune"):
        original = json.loads((ROOT / f"configs/native_downstream_cf_jepa_{mode}_seed7_h96_v1.json").read_text(encoding="utf-8"))
        for seed in (13, 23):
            path = ROOT / f"configs/native_downstream_cf_jepa_{mode}_seed{seed}_h96_v1.json"
            if path.exists():
                raise FileExistsError("Preserve frozen endpoint configs")
            pending.append((path, dict(original, seed=seed)))
    for path, config in pending:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps({"status": "STRONG_CF_CONFIGS_FROZEN_NOT_FITTED", "configs": len(pending),
                      "same_recipes_except_seed": True, "prefit": "NOT_RUN"}))


if __name__ == "__main__":
    main()
