from pathlib import Path
import ast

for relative in ['tests/integration/test_native_prefix_transfer.py', 'tools/execute_native_prefix_job_v2.py']:
    path = Path('../marine-echo-jepa') / relative
    text = path.read_text(encoding='utf-8')
    print(relative, '\n', '\n'.join(text.splitlines()[:85]))
for relative in ['tools/execute_native_prefix_desktop_job_v3.py', 'src/marine_echo/training/native_desktop_runtime_v3.py']:
    path = Path(relative)
    print(relative, '\n', path.read_text(encoding='utf-8'))
