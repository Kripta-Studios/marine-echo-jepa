"""Root-only new Unicode synthetic CPU fit/replay; grants no scientific approval."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    target = args.output.resolve()
    if (
        root.name != "marine-echo-jepa"
        or not target.is_relative_to(root / "evidence")
        or target.exists()
    ):
        raise ValueError("New root-owned evidence directory required; no retry/overwrite")
    spec = importlib.util.spec_from_file_location(
        "prefix_durable_support", root / "tests/unit/test_native_prefix_transfer.py"
    )
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    with pytest.MonkeyPatch.context() as patch:
        case = helper.support.case(helper.prefix, patch, target)
        files = dict(case.fs.files)
        paths = case.manifest_path, case.review_path, case.output
        arrays = case.prefix_arrays
    target.mkdir()
    for path, payload in files.items():
        if not path.is_relative_to(target):
            raise ValueError("Fixture escaped its new directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    result = helper.prefix.fit(*paths)
    assert result["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY"
    predictor = helper.prefix.PrefixPredictor(paths[2] / "inference.pt")
    predictions = predictor.forecast(
        arrays["x"], arrays["context_observed"], arrays["metadata"], arrays["query"]
    )
    assert predictions.shape == (len(arrays["x"]), 3, 5)
    assert helper.np.isfinite(predictions).all()
    try:
        helper.prefix.fit(*paths)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Protected output collision was not rejected")
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_kind": result["evidence_kind"],
                "synthetic_optimizer_updates": 4,
                "public_scientific_fit": False,
                "gpu_execution": False,
                "durable_replay": "PASSED",
                "output": str(paths[2]),
            }
        )
    )


if __name__ == "__main__":
    main()
