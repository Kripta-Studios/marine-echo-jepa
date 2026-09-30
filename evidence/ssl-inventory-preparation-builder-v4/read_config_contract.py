from pathlib import Path
import ast
path = Path('../marine-echo-jepa/src/marine_echo/evaluation/native_ancestry_inventory.py')
source = path.read_text(encoding='utf-8')
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and node.name in {'_config', '_decode_records'}:
        if node.name == '_config':
            print(ast.get_source_segment(source, node))
