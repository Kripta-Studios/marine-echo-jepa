"""Freeze exact finite first-screen configurations without numerical model fitting."""

from __future__ import annotations

import json
from pathlib import Path

from marine_echo.training.native_ssl import Config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    records = []
    for method in (
        "shared_ssl",
        "masked_ssl",
        "direct",
        "random_frozen",
        "permuted_ssl",
        "cf_jepa",
    ):
        options = {"method": method}
        if method == "direct":
            options.update(pretrain_updates=3000, pretrain_cadence=750)
        elif method == "random_frozen":
            options.update(pretrain_updates=0)
        config = Config(**options)
        config.validate()
        path = ROOT / f"configs/native_ssl_{method}_seed7_h96_v1.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
        records.append(
            {
                "method": method,
                "path": str(path.resolve()),
                "pretrain_updates": config.pretrain_updates,
                "readout_updates_per_probe": config.readout_updates,
                "checkpoint_cadence": config.pretrain_cadence,
            }
        )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
