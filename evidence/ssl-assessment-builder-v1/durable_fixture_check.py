"""ROOT ONLY: new Unicode-safe synthetic disk fixture; no fit, weights or GPU."""

from __future__ import annotations

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
    repository = Path(__file__).resolve().parents[2]
    root = repository.parent / "marine-echo-jepa"
    if repository != root:
        raise RuntimeError(
            "Root must integrate this helper and run in its own authorized checkout."
        )
    target = args.output.resolve()
    if not target.is_relative_to(root / "evidence") or target.exists():
        raise ValueError("Root-owned NEW evidence fixture directory required; no retry/overwrite.")
    spec = importlib.util.spec_from_file_location(
        "assessment_durable_fixture_support", repository / "tests/unit/test_native_assessment.py"
    )
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    # Build reviewed fixture bytes through the explicit memory transport, then
    # restore real IO. This is NOT a fallback for a failed filesystem operation.
    with pytest.MonkeyPatch.context() as patch:
        fs = helper.MemoryFS(patch)
        fs.base, fs.dirs = target, {target}
        case = helper.Case(fs)
        case.spec["loader"] = "builtin"
        case.seal()
        files = dict(fs.files)
        paths = case.manifest_path, case.review_path, case.output
    target.mkdir(parents=False, exist_ok=False)
    for path, payload in files.items():
        if not path.is_relative_to(target) or path.parent != target:
            raise ValueError("Synthetic fixture escaped its authorized new directory.")
        with path.open("xb") as stream:
            stream.write(payload)
    result = helper.assessment.execute_assessment(*paths)
    assert result["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY"
    assert result["native_geometry"]["lower_m"] == 230
    assert result["status"] == "COMPLETED_FORECASTS"
    try:
        helper.assessment.execute_assessment(*paths)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Protected output collision was not rejected.")
    print(
        json.dumps(
            {
                "evidence_kind": result["evidence_kind"],
                "status": result["status"],
                "output": str(paths[2]),
                "scientific_fit": False,
            }
        )
    )


if __name__ == "__main__":
    main()
