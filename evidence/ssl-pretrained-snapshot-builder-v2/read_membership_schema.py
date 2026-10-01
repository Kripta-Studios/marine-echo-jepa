from pathlib import Path
p=Path('../marine-echo-jepa/src/marine_echo/training/native_ssl.py')
print('\n'.join(p.read_text(encoding='utf-8').splitlines()[1062:1080]))
