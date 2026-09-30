"""Add exactly the previously approved seed7 random frozen control to admission."""

import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "tools/execute_native_band_operational_job.py"
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    text = text.replace('root / "tools/execute_native_band_operational_job.py"',
                        'root / "tools/execute_native_band_operational_job_v4.py"')
    old = "    if config not in candidates:\n"
    if text.count(old) != 1:
        raise ValueError("Exact fixed-catalog admission site required")
    text = text.replace(old, "    if kind == 'downstream':\n        fixed_random = read_json(root / 'configs/native_band_execution_v1/native_band_random_frozen_frozen_readout_v1.json')\n        if (fixed_random.get('method'), fixed_random.get('mode'), fixed_random.get('seed')) != ('random_frozen', 'frozen_readout', 7):\n            raise ValueError('Only the original fixed seed7 random frozen control may be added')\n        candidates.append(fixed_random)\n" + old)
    text = text.replace("root / 'docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md'):",
                        "root / 'docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md',\n                 root / 'configs/native_band_execution_v1/native_band_random_frozen_frozen_readout_v1.json'):")
    updated = text.encode("utf-8")
    before = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(raw).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    after = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(updated).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    changed = {key for key in before if before[key] != after.get(key)}
    if set(before) != set(after) or changed != {"source_paths", "_check_config"}:
        raise ValueError("Only fixed control source/admission may change")
    destination = ROOT / "tools/execute_native_band_operational_job_v4.py"
    with destination.open("xb") as stream:
        stream.write(updated)
    proof = {"status": "EXACT_PREVIOUSLY_APPROVED_CONTROL_ADMISSION_ADDED_NO_NEW_RECIPE",
             "before_source_sha256": hashlib.sha256(raw).hexdigest(), "before_source_bytes_base64": base64.b64encode(raw).decode("ascii"),
             "after_source_sha256": hashlib.sha256(updated).hexdigest(), "changed_definitions": sorted(changed),
             "scientific_training_functions_changed": False, "prefit": "REQUIRED_NOT_GRANTED"}
    with (ROOT / "evidence/ssl-research-v1/band-operational-control-admission-proof-v4.json").open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": proof["status"], "changed_definitions": sorted(changed)}))


if __name__ == "__main__":
    main()
