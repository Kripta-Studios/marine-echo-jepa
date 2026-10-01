"""Read-only source/protocol snapshot; no trained artifacts or acoustic data."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1].parent / "marine-echo-jepa"
pending = [ROOT / p for p in ("src/marine_echo/training/native_ssl.py", "src/marine_echo/training/native_downstream.py", "src/marine_echo/models/native_temporal.py", "src/marine_echo/inference/native_acoustic.py", "src/marine_echo/inference/native_encoder.py", "src/marine_echo/inference/native_band_acoustic.py", "src/marine_echo/data/native_ssl_corpus.py", "src/marine_echo/training/aeon_corpus.py", "src/marine_echo/evaluation/native_product.py", "tools/native_reference_supervisor.py")]
found = set()
while pending:
    path = pending.pop()
    if path in found:
        continue
    found.add(path)
    for node in ast.walk(ast.parse(path.read_bytes())):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] + [node.module + "." + a.name for a in node.names] if isinstance(node, ast.ImportFrom) and node.module else [])
        for name in names:
            if name.startswith("marine_echo"):
                parts = name.split(".")
                for count in range(1, len(parts) + 1):
                    base = ROOT / "src" / Path(*parts[:count])
                    pending += [p for p in (base.with_suffix(".py"), base / "__init__.py") if p.is_file()]
found.update(ROOT / p for p in ("orchestration/ssl_vnext_cf_matched_controls_builder_contract.txt", "docs/adr/0022-cf-backbone-matched-controls.md", "docs/adr/0025-cf-matched-controls-continuation.md"))
found.update(ROOT / "external/cf-jepa-vnext" / p for p in ("LICENSE", "baselines/cf_jepa/encoder.py", "baselines/cf_jepa/trainer.py", "baselines/cf_jepa/losses.py", "baselines/cf_jepa/cf_jepa.py"))
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(found)}
with (HERE / "protected-baseline-v1.json").open("x", encoding="utf-8") as stream:
    json.dump({"files": hashes, "numerical_access": False}, stream, indent=2, sort_keys=True)
print(json.dumps({"dependencies": len(hashes), "numerical_access": False}))
