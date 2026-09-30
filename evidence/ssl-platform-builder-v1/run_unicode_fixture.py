"""One NEW Unicode model path; SYNTHETIC_CORRECTNESS_ONLY, CPU LightGBM.

Any denied write propagates, with no fallback/relocation/retry. Main is never
written. A pre-existing fixture is rejected before fitting/writing.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

BUILDER = Path(__file__).resolve().parents[2]
TEST = BUILDER / "tests/unit/test_native_booster_serialization.py"


if __name__ == "__main__":
    output = Path(__file__).parent / "SYNTHETIC_CORRECTNESS_ONLY_Unicode_Á_é_booster_v1.txt"
    if output.exists():
        raise FileExistsError("This fixture is preserved; no overwrite/retry is permitted.")
    spec = importlib.util.spec_from_file_location("booster_unicode_correctness", TEST)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    booster, features = check.fit_tiny_booster()
    expected = booster.predict(features, num_threads=1)
    check.references.save_booster_text(booster, output)
    replay = check.references.load_booster_text(output)
    np.testing.assert_array_equal(replay.predict(features, num_threads=1), expected)
    text = output.read_text(encoding="utf-8")
    assert text == booster.model_to_string()
    print(
        json.dumps(
            {
                "status": "PASSED",
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "device": "cpu",
                "path": str(output.resolve()),
                "model_sha256": check.references.digest(output),
                "model_trees": booster.num_trees(),
                "prediction_rows": len(features),
                "prediction_roundtrip": "BIT_IDENTICAL",
                "native_filename_api_calls": 0,
                "original_failed_path_retried": False,
            },
            indent=2,
        )
    )
