"""Prepare owner-resolved band configs without data/model loading or fitting.

Five screens are recipe slots; prospective downstream endpoints remain separate
parent-specific review tasks and consume the same full-owned band allowance.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = "nonlinear_frequency_conditioned_v1"
SSL_METHODS = ("shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen")


def _definitions(root):
    sys.path.insert(0, str(root / "src"))
    ssl = importlib.import_module("marine_echo.training.native_band_ssl")
    downstream = importlib.import_module("marine_echo.training.native_band_downstream")
    for module in (ssl, downstream):
        if not Path(module.__file__).resolve().is_relative_to(root / "src"):
            raise ValueError("Config definitions must resolve to this integrated checkout.")
    return ssl.Config, downstream.DownstreamConfig


def _catalog(ssl_config, downstream_config):
    screens, prospective = {}, {}
    for method in SSL_METHODS:
        c = ssl_config(method=method, pretrain_updates=0 if method == "random_frozen" else 6000)
        c.validate()
        screens[f"native_band_{method}_screen_v1.json"] = c.to_dict()
    c = downstream_config(method="direct", mode="direct_end_to_end")
    c.validate()
    screens["native_band_direct_end_to_end_v1.json"] = c.to_dict()
    for method, modes in (
        ("shared_ssl", ("frozen_readout", "full_finetune")),
        ("masked_ssl", ("frozen_readout", "full_finetune")),
        ("permuted_ssl", ("frozen_readout",)),
        ("random_frozen", ("frozen_readout",)),
    ):
        for mode in modes:
            c = downstream_config(method=method, mode=mode)
            c.validate()
            prospective[f"native_band_{method}_{mode}_v1.json"] = c.to_dict()
    # Actual immutable loader accepts only native_band_ssl_selected_encoder_v1.
    # Scientific direct produces native_band_downstream_supervised_encoder_v1,
    # so Config accepting method=direct does not establish loadable ancestry.
    unsupported = [
        {
            "method": "direct",
            "mode": "frozen_readout",
            "status": "NOT_SUPPORTED",
            "reason": "Current downstream selected-ancestor loader requires native_band_ssl_selected_encoder_v1; direct_end_to_end emits native_band_downstream_supervised_encoder_v1. No direct SSL screen exists.",
        }
    ]
    return screens, prospective, unsupported


def _generate(output, root, definitions):
    output = Path(output).resolve()
    if not output.is_relative_to(root / "configs"):
        raise ValueError("Production config output must remain inside this checkout configs.")
    screens, prospective, unsupported = _catalog(*definitions)
    files = {**screens, **prospective}
    if any((output / name).exists() for name in [*files, "generation.json"]):
        raise FileExistsError("Exclusive config generation preserves earlier files.")
    output.mkdir(parents=True, exist_ok=True)
    for name, c in files.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(c, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    result = {
        "kind": "native_band_config_generation_v1",
        "architecture": ARCHITECTURE,
        "status": "CONFIGS_READY_FOR_DISTINCT_REVIEW_NO_FITTING",
        "seed": 7,
        "screens": [str(output / n) for n in screens],
        "prospective_downstream": [str(output / n) for n in prospective],
        "unsupported": unsupported,
        "screen_recipe_slots": 5,
        "prospective_endpoints_are_not_new_screen_slots": True,
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
        raise ValueError("Run this helper only after integration into the actual root checkout.")
    return _generate(output or ROOT / "configs/native_band_execution_v1", ROOT, _definitions(ROOT))


def _generate_for_test(output, root, definitions):
    caller = inspect.stack()[1]
    if (
        Path(caller.filename).name != "test_native_band_execution.py"
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
