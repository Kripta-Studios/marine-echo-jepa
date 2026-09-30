"""Freeze matched transfer configurations before their strong endpoint fits."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    pairs = [
        (method, mode)
        for method in ("cf_jepa", "masked_ssl")
        for mode in ("frozen_readout", "full_finetune")
    ]
    pairs += [(method, "frozen_readout") for method in ("permuted_ssl", "random_frozen", "direct")]
    pending = []
    for method, mode in pairs:
        template = ROOT / f"configs/native_downstream_shared_ssl_{mode}_seed7_h96_v1.json"
        config = json.loads(template.read_text(encoding="utf-8"))
        config["method"] = method
        destination = ROOT / f"configs/native_downstream_{method}_{mode}_seed7_h96_v1.json"
        if destination.exists():
            raise ValueError("Preserve all previously frozen transfer configurations.")
        pending.append((destination, config))
    for destination, config in pending:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(
        json.dumps(
            {
                "status": "CONFIGURATIONS_FROZEN_NOT_FITTED",
                "configs": [str(path) for path, _ in pending],
                "admission": "Separate exact completed-parent prefit still required",
                "final_test_access": "NOT_RUN",
            }
        )
    )


if __name__ == "__main__":
    main()
