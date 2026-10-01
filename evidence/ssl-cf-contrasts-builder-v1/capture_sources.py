"""Source-only baseline capture. Never reads any scientific payload."""

import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
builder = here.parents[1]
root = builder.parent / "marine-echo-jepa"
names = ["src/marine_echo/evaluation/native_suffix_reconstruction_v1.py",
         "src/marine_echo/evaluation/native_comparison.py",
         "src/marine_echo/evaluation/native_product.py", "src/marine_echo/__init__.py",
         "src/marine_echo/evaluation/__init__.py"]
for name in ("pyproject.toml", "uv.lock"):
    if (root / name).is_file():
        names.append(name)
baseline = {}
for i, name in enumerate(names):
    raw = (root / name).read_bytes()
    baseline[str(root / name)] = hashlib.sha256(raw).hexdigest()
    if name.endswith(".py"):
        with (here / f"protected-source-{i:02d}.txt").open("xb") as stream:
            stream.write(raw)
with (here / "protected-baseline-v1.json").open("x", encoding="utf-8") as stream:
    json.dump(baseline, stream, indent=2, sort_keys=True)
with (here / "red-baseline-source.txt").open("xb") as stream:
    stream.write((builder / "src/marine_echo/evaluation/native_cf_matched_contrasts_v1.py").read_bytes())
print(json.dumps({"source_only": True, "baseline_files": len(baseline)}))
