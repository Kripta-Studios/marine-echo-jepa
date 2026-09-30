from pathlib import Path
import ast

p = Path('../marine-echo-jepa/src/marine_echo/training/native_ssl.py')
s = p.read_text(encoding='utf-8')
for n in ast.parse(s).body:
    if isinstance(n, ast.FunctionDef) and n.name in {'initialize_model', 'rng_state', 'restore_rng', 'seed_all'}:
        print(ast.get_source_segment(s, n))
