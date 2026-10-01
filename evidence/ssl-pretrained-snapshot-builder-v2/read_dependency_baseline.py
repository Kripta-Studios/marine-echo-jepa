"""Hash only assigned source/metadata/license dependencies; never read weights."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1].parent / 'marine-echo-jepa'
names = [
    'tools/package_native_pretrained_snapshot_v1.py',
    'tools/check_native_pretrained_snapshot_worker_v2.py',
    'tools/check_native_pretrained_snapshot_owned_v2.py',
    'tools/native_reference_supervisor.py',
    'outputs/native_pretrained_model_snapshot_v1/manifest.json',
    'outputs/native_pretrained_model_snapshot_v1/MODEL_CARD.md',
    'evidence/ssl-research-v1/original-source-archives-v3/index.json',
    'external/cf-jepa-vnext/LICENSE',
]
paths = {ROOT / name for name in names}
package = ROOT / 'src/marine_echo'
pending = [package / 'inference' / name for name in (
    'native_encoder.py', 'native_latent.py', 'native_acoustic.py',
    'native_band_acoustic.py', 'native_band_replication_acoustic.py',
    'native_band_replication_encoder.py', 'native_band_replication_latent.py',
)]
while pending:
    path = pending.pop()
    if path in paths:
        continue
    paths.add(path)
    for parent in (path.parent, *path.parent.parents):
        if parent.is_relative_to(package) and (parent / '__init__.py').is_file():
            pending.append(parent / '__init__.py')
    for node in ast.walk(ast.parse(path.read_bytes())):
        imports = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module, *[node.module+'.'+a.name for a in node.names]] if isinstance(node, ast.ImportFrom) and node.module else []
        for name in imports:
            if name.startswith('marine_echo.'):
                target = ROOT / 'src' / Path(*name.split('.'))
                for file in (target.with_suffix('.py'), target / '__init__.py'):
                    if file.is_file(): pending.append(file)
index = json.loads((ROOT/names[6]).read_bytes())
for name in index['catalogs']:
    path = Path(name)
    paths.update((path, Path(json.loads(path.read_bytes())['path'])))
baseline = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
with (HERE/'protected-baseline-v2.json').open('x', encoding='utf-8') as stream:
    json.dump({'files': baseline, 'v1_archive_owner_pin': 'ed1ba345a6fe3f04642d5bb2d8c32ef33f67132009295a6bca38cd73d8a3994b', 'v1_archive_byte_hash_validation': 'ROOT_PACKAGE_ONLY_NOT_RUN_BUILDER'}, stream, indent=2)
print(json.dumps({'protected_source_metadata_license_files': len(baseline), 'public_weights_read': False}))
