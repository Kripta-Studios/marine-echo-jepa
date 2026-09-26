"""Type-check app and optional CUDA code against their actual installed environments."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.environ.pop("PYTHONHOME", None)
os.environ.pop("UV_INTERNAL__PYTHONHOME", None)
from mypy import api  # noqa: E402

common = root / "src/marine_echo"
output, errors, app_code = api.run([str(common / p) for p in ["contracts", "data", "evaluation", "features", "serving"]])
print(output, errors)
gpu = root.parent / "EVOCON_JEPA_Codex_Handoff/e-jepa-ttc/.venv/Scripts/python.exe"
output, errors, model_code = api.run(["--python-executable", str(gpu), str(common / "models"), str(common / "training")])
print(output, errors)
sys.exit(max(app_code, model_code))
