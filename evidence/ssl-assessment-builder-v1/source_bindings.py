"""Source-only local import provenance; no corpus, tensors or model construction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

repository = Path(__file__).resolve().parents[2]
main = repository.parent / "marine-echo-jepa"
sys.path.insert(0, str(main / "src"))
spec = importlib.util.spec_from_file_location(
    "marine_echo.evaluation.native_assessment",
    repository / "src/marine_echo/evaluation/native_assessment.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
bindings = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in module.required_sources()}
output = repository / "evidence/ssl-assessment-builder-v1/import-source-bindings-v1.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump(
        {
            "operation": "source_only_static_local_import_closure",
            "bindings": bindings,
            "dependency_lock": "Required per execution manifest; root-owned",
            "numerical_artifacts_read": False,
        },
        stream,
        indent=2,
        allow_nan=False,
    )
    stream.write("\n")
print(json.dumps({"source_files": len(bindings), "path": str(output)}))
