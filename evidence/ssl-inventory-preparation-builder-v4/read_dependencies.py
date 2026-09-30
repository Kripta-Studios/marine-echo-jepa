"""Read only explicitly assigned contract sources; record immutable byte hashes."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent / 'marine-echo-jepa'
HERE = Path(__file__).resolve().parent
paths = [ROOT / name for name in (
    'tools/prepare_completed_native_inventory_v3.py',
    'tools/execute_completed_native_inventory_owned_v3.py',
    'tools/prepare_native_development_comparison_v4.py',
    'src/marine_echo/evaluation/native_ancestry_inventory.py',
    'tools/prepare_native_research_inventory.py',
    'tools/native_reference_supervisor.py',
    'evidence/ssl-research-v1/original-source-archives-v3/index.json',
)]
index = json.loads(paths[-1].read_bytes())
for name in index['catalogs']:
    catalog = Path(name)
    paths.extend((catalog, Path(json.loads(catalog.read_bytes())['path'])))
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with (HERE / 'dependency-baseline-v4.json').open('x', encoding='utf-8') as stream:
    json.dump(hashes, stream, indent=2)
source = (ROOT / 'src/marine_echo/evaluation/native_ancestry_inventory.py').read_text(encoding='utf-8')
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and node.name in {'_admit', 'resolve_original_source_binding', 'required_sources', 'validate_parent_graph'}:
        print(ast.get_source_segment(source, node))
