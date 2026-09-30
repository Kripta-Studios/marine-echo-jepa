"""Prepare only fixed shared/direct seed13/23 replications; no fit approval."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = "nonlinear_frequency_conditioned_v1"
SEEDS = (7, 13, 23)


def _definitions(root):
    sys.path.insert(0, str(root / "src"))
    ssl = importlib.import_module("marine_echo.training.native_band_replication_ssl")
    downstream = importlib.import_module("marine_echo.training.native_band_replication_downstream")
    for module in (ssl, downstream):
        if not Path(module.__file__).resolve().is_relative_to(root / "src"):
            raise ValueError("Replication definitions must resolve to the integrated checkout.")
    return ssl.Config, downstream.DownstreamConfig


def _catalog(ssl_config, downstream_config):
    screens, prospective = {}, {}
    for seed in SEEDS:
        c = ssl_config(seed=seed)
        c.validate()
        screens[f"native_band_shared_ssl_seed{seed}_v2.json"] = c.to_dict()
        c = downstream_config(method="direct", mode="direct_end_to_end", seed=seed)
        c.validate()
        screens[f"native_band_direct_end_to_end_seed{seed}_v2.json"] = c.to_dict()
        for mode in ("frozen_readout", "full_finetune"):
            c = downstream_config(method="shared_ssl", mode=mode, seed=seed)
            c.validate()
            prospective[f"native_band_shared_ssl_{mode}_seed{seed}_v2.json"] = c.to_dict()
    return screens, prospective, []


def _generate(output, root, definitions):
    output = Path(output).resolve()
    if not output.is_relative_to(root / "configs"):
        raise ValueError("Config output must remain inside integrated root configs.")
    screens, _, _ = _catalog(*definitions)
    files = {name: c for name, c in screens.items() if c["seed"] in (13, 23)}
    if any((output / name).exists() for name in [*files, "generation.json"]):
        raise FileExistsError("Exclusive generation preserves earlier configs.")
    output.mkdir(parents=True, exist_ok=True)
    for name, c in files.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(c, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    result = {
        "kind": "native_band_replication_config_generation_v2",
        "architecture": ARCHITECTURE,
        "status": "CONFIGS_READY_FOR_DISTINCT_REVIEW_NO_FITTING",
        "admitted_seeds": list(SEEDS),
        "generated_replication_seeds": [13, 23],
        "screens": [str(output / name) for name in files],
        "prospective_downstream": [],
        "later_endpoints": "Require completed version2 parents and separate exact review.",
        "budget_family": "native_band_v1",
        "budget": {
            "total_seed7_recipes": 11,
            "band_full_owned_gpu_hours": 12,
            "aggregate_full_owned_gpu_hours": 96,
        },
        "scientific_approval": False,
        "model_or_data_loaded": False,
    }
    with (output / "generation.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def generate(output=None):
    if not (ROOT / ".venv/Scripts/python.exe").is_file():
        raise ValueError("Generate only after integration into the actual root checkout.")
    return _generate(
        output or ROOT / "configs/native_band_replication_v2", ROOT, _definitions(ROOT)
    )


def _generate_for_test(output, root, definitions):
    caller = inspect.stack()[1]
    if (
        Path(caller.filename).name != "test_native_band_replication_execution.py"
        or not caller.function.startswith("test_")
        or "SYNTHETIC_CORRECTNESS_ONLY" not in str(root)
    ):
        raise ValueError("Private synthetic test root only; no production CLI override.")
    return _generate(output, root, definitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))


if __name__ == "__main__":
    main()
