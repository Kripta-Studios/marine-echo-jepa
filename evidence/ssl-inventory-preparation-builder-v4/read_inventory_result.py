from pathlib import Path
p=Path('../marine-echo-jepa/src/marine_echo/evaluation/native_ancestry_inventory.py')
print('\n'.join(p.read_text(encoding='utf-8').splitlines()[1239:1276]))
