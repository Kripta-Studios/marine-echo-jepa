"""Read-only source/metadata baseline, without tensor or corpus decoding."""

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
pending = [MAIN / "tools" / name for name in ("prepare_native_development_comparison_v4.py", "prepare_completed_native_inventory_v4.py", "execute_completed_native_inventory_owned_v4.py", "prepare_native_research_inventory.py", "prepare_native_seed7_assessment_metadata.py", "native_reference_supervisor.py")]
pending += [MAIN / path for path in ("src/marine_echo/evaluation/native_ancestry_inventory.py", "src/marine_echo/evaluation/native_assessment_replication.py", "src/marine_echo/training/native_prefix_transfer.py", "src/marine_echo/data/native_transfer_corpus.py", "src/marine_echo/training/native_references.py")]
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
                for length in range(1, len(parts) + 1):
                    base = MAIN / "src" / Path(*parts[:length])
                    pending += [p for p in (base.with_suffix(".py"), base / "__init__.py") if p.is_file()]
index = MAIN / "evidence/ssl-research-v1/original-source-archives-v3/index.json"
found.add(index)
for catalog in json.loads(index.read_bytes())["catalogs"]:
    found.add(Path(catalog))
    record = json.loads(Path(catalog).read_bytes())
    found.update(Path(record[k]) for k in ("path", "original_path"))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
proof = {"files": {str(p): sha(p) for p in sorted(found)}, "public_tensors_or_corpora_decoded": False}
for name in ("prepare_native_development_comparison_v4.py", "prepare_completed_native_inventory_v4.py", "execute_completed_native_inventory_owned_v4.py"):
    with (HERE / ("baseline-" + name)).open("xb") as stream:
        stream.write((MAIN / "tools" / name).read_bytes())
with (HERE / "protected-baseline-v1.json").open("x", encoding="utf-8") as stream:
    json.dump(proof, stream, indent=2, sort_keys=True)
print(json.dumps({"protected_files": len(found), "numeric_decoding": False}))
