"""Source/import provenance only; no prediction, corpus, weights or GPU access."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))

if __name__ == "__main__":
    candidate = BUILDER / "src/marine_echo/evaluation/native_comparison.py"
    spec = importlib.util.spec_from_file_location("native_comparison_candidate", candidate)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    imported = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if (name == "marine_echo" or name.startswith("marine_echo."))
        and getattr(module, "__file__", None)
    }
    authored = [candidate, BUILDER / "tests/unit/test_native_comparison.py", Path(__file__)]
    print(
        json.dumps(
            {
                "operation": "SOURCE_HASHES_ONLY_NO_SCIENTIFIC_ARTIFACT_ACCESS",
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "imports": imported,
                "required_review_source_bindings": api.IMPORTED_SOURCE_HASHES,
                "authored_sha256": {
                    str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in authored
                },
                "runtime": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "torch_imported": "torch" in sys.modules,
                    "executing_session_id": os.environ.get("CODEX_THREAD_ID"),
                    "authored_builder_session_id": api.IMPLEMENTER_SESSION_ID,
                },
            },
            indent=2,
        )
    )
