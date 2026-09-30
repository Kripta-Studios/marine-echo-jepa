"""Read immutable schemas only, never fitted metadata/tensors or corpus arrays."""

from pathlib import Path

MAIN = Path(__file__).resolve().parents[3] / "marine-echo-jepa"
SLICES = {
    "src/marine_echo/training/native_ssl.py": [(44, 150), (300, 330)],
    "src/marine_echo/training/native_downstream.py": [(29, 86), (828, 842)],
    "src/marine_echo/training/native_band_replication_downstream.py": [(29, 145)],
    "src/marine_echo/data/native_ssl_corpus.py": [(500, 610)],
    "src/marine_echo/training/native_band_ssl.py": [(1120, 1170)],
}
for name, spans in SLICES.items():
    lines = (MAIN / name).read_text(encoding="utf-8").splitlines()
    for lo, hi in spans:
        print(name, lo, hi)
        print("\n".join(f"{i + 1}: {lines[i]}" for i in range(lo - 1, min(hi, len(lines)))))
