"""Source-only desktop prefix baseline and exact Resources inspection."""

import ast
import base64
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
BUILDER = BASE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
PRIOR = BUILDER / "evidence/ssl-native-ancestry-inventory-builder-v1/protected-current-v1.json"
EXTRA = (
    "src/marine_echo/training/native_desktop_resources_v2.py",
    "src/marine_echo/training/native_resources.py",
    "src/marine_echo/training/native_prefix_transfer.py",
    "src/marine_echo/training/native_ssl.py",
    "tools/execute_native_prefix_job_v2.py",
    "tools/execute_native_prefix_worker.py",
    "tools/native_band_budget_history_v2.py",
    "tools/native_reference_supervisor.py",
    "tools/execute_native_band_job.py",
    "orchestration/native_vlc_desktop_owner_resolution_v1.json",
    "docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md",
)


def main():
    paths = {Path(name) for name in json.loads(PRIOR.read_bytes())}
    paths.update(MAIN / name for name in EXTRA)
    found = {}
    for path in paths:
        raw = path.read_bytes()
        found[str(path.resolve())] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                     "bytes_base64": base64.b64encode(raw).decode("ascii")}
    if found[str((MAIN / EXTRA[2]).resolve())]["sha256"] != "844159d11a1d9a410f7cb7f1a83c782360427e04aae965c19cb94a74d5a28d38":
        raise ValueError("Root current prefix source differs from explicit assignment")
    if found[str((MAIN / EXTRA[9]).resolve())]["sha256"] != "18e29745c7595a9ea50bccf23e3124b06c57106929ac20c40a8f04776ff598e4":
        raise ValueError("Owner resolution differs from explicit pin")
    with (BASE / "protected-baseline-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(found, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "SOURCE_ONLY", "protected_files": len(found)}))
    for filename, names in (
        ("src/marine_echo/training/native_ssl.py", {"Resources"}),
        ("src/marine_echo/training/native_prefix_transfer.py", {"required_sources"}),
    ):
        raw = (MAIN / filename).read_text(encoding="utf-8")
        for node in ast.parse(raw).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                print(filename)
                print(ast.get_source_segment(raw, node))


if __name__ == "__main__":
    main()
