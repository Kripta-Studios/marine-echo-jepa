"""Repair only the isolated check's namespace-package transport; keep v1 failure."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
worker = ROOT / "tools/check_native_pretrained_snapshot_worker_v1.py"
old = '''        if name == "marine_echo" or name.startswith("marine_echo."):
            if not Path(module.__file__).resolve().is_relative_to(bundle / "src"):
                raise ValueError("Inference imported outside copied source closure")'''
new = '''        if name == "marine_echo" or name.startswith("marine_echo."):
            filename = getattr(module, "__file__", None)
            paths = [Path(filename)] if filename else [Path(p) for p in getattr(module, "__path__", [])]
            if not paths or any(not path.resolve().is_relative_to(bundle / "src") for path in paths):
                raise ValueError("Inference imported outside copied source closure")'''
raw = worker.read_text(encoding="utf-8")
if raw.count(old) != 1:
    raise ValueError("Exact failed namespace check source required")
with (ROOT / "tools/check_native_pretrained_snapshot_worker_v2.py").open("x", encoding="utf-8") as stream:
    stream.write(raw.replace(old, new))
parent = (ROOT / "tools/check_native_pretrained_snapshot_owned_v1.py").read_text(encoding="utf-8")
parent = parent.replace("native-pretrained-snapshot-isolated-cpu-v1", "native-pretrained-snapshot-isolated-cpu-v2")
parent = parent.replace("check_native_pretrained_snapshot_worker_v1.py", "check_native_pretrained_snapshot_worker_v2.py")
with (ROOT / "tools/check_native_pretrained_snapshot_owned_v2.py").open("x", encoding="utf-8") as stream:
    stream.write(parent)
print("Isolated namespace check v2 created; original failure preserved")
