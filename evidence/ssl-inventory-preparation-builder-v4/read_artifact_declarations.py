from pathlib import Path
ROOT = Path(__file__).resolve().parents[2].parent / 'marine-echo-jepa'
for relative, start, end in (
    ('src/marine_echo/training/native_band_replication_ssl.py', 1255, 1291),
    ('src/marine_echo/training/native_band_replication_downstream.py', 936, 978),
    ('src/marine_echo/evaluation/native_ancestry_inventory.py', 0, 92),
):
    path = ROOT / relative
    print(relative, '\n', '\n'.join(path.read_text(encoding='utf-8').splitlines()[start:end]))
