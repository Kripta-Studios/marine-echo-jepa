"""Add operational executor copies while proving unchanged scientific functions."""

import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def definitions(raw):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def main():
    mapping = {
        "src/marine_echo/training/native_band_replication_ssl.py": ("src/marine_echo/training/native_band_operational_ssl.py", {"required_sources"}),
        "src/marine_echo/training/native_band_replication_downstream.py": ("src/marine_echo/training/native_band_operational_downstream.py", set()),
        "tools/execute_native_band_replication_job.py": ("tools/execute_native_band_operational_job.py", {"source_paths", "_admit", "_execute_for_test"}),
    }
    payloads, evidence = {}, {}
    for name, (destination, exceptions) in mapping.items():
        path = ROOT / name
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if name.endswith("ssl.py"):
            old = "from marine_echo.training import native_resources\n"
            if text.count(old) != 1:
                raise ValueError("Exact operational import required")
            text = text.replace(old, "from marine_echo.training import native_desktop_resources_v2 as native_resources\n")
            text = text.replace('Path(__file__).with_name("native_band_replication_downstream.py")', 'Path(__file__).with_name("native_band_operational_downstream.py")')
            text = text.replace("        Path(native_resources.__file__),", "        Path(native_resources.__file__),\n        Path(native_resources.original.__file__),\n        native_resources.OWNER_PATH,\n        MAIN / 'docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md',")
        elif name.endswith("downstream.py"):
            old = "from marine_echo.training import native_band_replication_ssl as core"
            if text.count(old) != 1:
                raise ValueError("Exact core import required")
            text = text.replace(old, "from marine_echo.training import native_band_operational_ssl as core")
        else:
            for before, after in (
                ("execute_native_band_replication_job.py", "execute_native_band_operational_job.py"),
                ("training/native_band_replication_ssl.py", "training/native_band_operational_ssl.py"),
                ("training/native_band_replication_downstream.py", "training/native_band_operational_downstream.py"),
                ('f"marine_echo.training.native_band_replication_{job.kind}"', 'f"marine_echo.training.native_band_operational_{job.kind}"'),
                ("test_native_band_replication_execution.py", "test_native_band_operational_execution.py"),
            ):
                if text.count(before) != 1:
                    raise ValueError(f"Exact operational routing site required: {before}")
                text = text.replace(before, after)
            text = text.replace("    return sorted(found)\n", "    for path in (root / 'orchestration/native_vlc_desktop_owner_resolution_v1.json',\n                 root / 'docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md'):\n        if not path.is_file():\n            raise ValueError('Owner desktop authority and ADR must be bound')\n        found.add(path.resolve())\n    return sorted(found)\n", 1)
        updated = text.encode("utf-8")
        before, after = definitions(raw), definitions(updated)
        changed = {key for key in before if before[key] != after.get(key)}
        if set(before) != set(after) or changed != exceptions:
            raise ValueError(f"Unexpected scientific AST changes: {name}: {changed}")
        if (ROOT / destination).exists():
            raise FileExistsError("Preserve earlier operational versions")
        payloads[destination] = updated
        evidence[name] = {"original_sha256": digest(raw), "original_bytes_base64": base64.b64encode(raw).decode("ascii"),
                          "new_path": destination, "new_sha256": digest(updated),
                          "changed_top_level_definitions": sorted(changed),
                          "unchanged_top_level_definition_count": len(before) - len(changed)}
    receipt_path = ROOT / "evidence/ssl-research-v1/band-desktop-operational-source-proof-v3.json"
    if receipt_path.exists():
        raise FileExistsError("Preserve earlier source proofs")
    for destination, raw in payloads.items():
        with (ROOT / destination).open("xb") as stream:
            stream.write(raw)
    for name, record in evidence.items():
        if digest((ROOT / name).read_bytes()) != record["original_sha256"]:
            raise ValueError("Historical fitted source was changed")
    receipt = {"status": "ADDITIVE_OPERATIONAL_COPIES_SCIENTIFIC_AST_UNCHANGED", "original_sources": evidence,
               "artifact_schema": "UNCHANGED_TYPED_V2", "scientific_recipe_changes": False,
               "new_prefit": "REQUIRED_NOT_GRANTED", "scientific_execution": "NOT_RUN"}
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "copies": len(payloads),
                      "unchanged_definition_counts": [v["unchanged_top_level_definition_count"] for v in evidence.values()]}))


if __name__ == "__main__":
    main()
