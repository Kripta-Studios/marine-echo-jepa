"""Freeze separately reviewed strong transfer configurations without fitting."""

from __future__ import annotations

import json
from pathlib import Path

from marine_echo.training.native_downstream import DownstreamConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    for mode, method in (
        ("frozen_readout", "shared_ssl"),
        ("full_finetune", "shared_ssl"),
        ("direct_end_to_end", "direct"),
    ):
        config = DownstreamConfig(mode=mode, method=method)
        config.validate()
        path = ROOT / f"configs/native_downstream_{method}_{mode}_seed7_h96_v1.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(config.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(str(path))


if __name__ == "__main__":
    main()
