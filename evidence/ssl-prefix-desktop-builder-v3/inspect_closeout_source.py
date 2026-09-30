from pathlib import Path
import ast

p = Path('../marine-echo-jepa/tools/native_reference_supervisor.py')
s = p.read_text(encoding='utf-8')
for n in ast.parse(s).body:
    if isinstance(n, ast.FunctionDef) and n.name == 'supervise_owned':
        print(ast.get_source_segment(s, n))
print(Path('tests/integration/test_native_prefix_desktop_transfer_v3.py').read_text(encoding='utf-8'))
print(Path('evidence/ssl-prefix-desktop-builder-v3/protected-baseline-v3.json').read_text(encoding='utf-8')[:1600])
