from pathlib import Path
p = Path('../marine-echo-jepa/tests/unit/test_native_prefix_transfer.py')
print('\n'.join(p.read_text(encoding='utf-8').splitlines()[230:280]))
