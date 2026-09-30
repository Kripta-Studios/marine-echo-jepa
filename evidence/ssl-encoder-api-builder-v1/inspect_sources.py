"""Read-only source hashes/import provenance; never inspect a real checkpoint."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))

if __name__ == "__main__":
    candidate = BUILDER / "src/marine_echo/inference/native_encoder.py"
    spec = importlib.util.spec_from_file_location("native_encoder_candidate", candidate)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    imported = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if (name == "marine_echo" or name.startswith("marine_echo."))
        and getattr(module, "__file__", None)
    }
    paths = [Path(path) for path in imported.values()]
    paths += [MAIN / "src/marine_echo/inference/native_acoustic.py"]
    authored = [candidate, BUILDER / "tests/unit/test_native_encoder.py", Path(__file__)]
    print(
        json.dumps(
            {
                "operation": "SOURCE_HASHES_ONLY_NO_CHECKPOINT_OR_CORPUS_ACCESS",
                "label": "SYNTHETIC_CORRECTNESS_ONLY",
                "imports": imported,
                "candidate_import": str(Path(api.__file__).resolve()),
                "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "authored_sha256": {
                    str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in authored
                },
                "runtime": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "cuda_initialized": torch.cuda.is_initialized(),
                },
            },
            indent=2,
        )
    )
