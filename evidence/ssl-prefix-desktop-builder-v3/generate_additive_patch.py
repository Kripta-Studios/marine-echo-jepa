"""Generate a reviewable patch from immutable ROOT bytes; never runtime rewriting."""

import ast
import hashlib
from pathlib import Path

MAIN = Path(__file__).resolve().parents[3] / "marine-echo-jepa"


def resources_copy():
    original = (MAIN / "src/marine_echo/training/native_ssl.py").read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(original).body if isinstance(n, ast.ClassDef) and n.name == "Resources")
    header = '''"""Source-bound desktop operational runtime; original Resources science unchanged."""

import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import torch

from marine_echo.training import native_desktop_resources_v2 as native_resources
from marine_echo.training import native_resources as original_resources
from marine_echo.training import native_ssl as original_core

OWNER_SHA256 = "18e29745c7595a9ea50bccf23e3124b06c57106929ac20c40a8f04776ff598e4"


def operational_sources():
    root = Path(native_resources.__file__).resolve().parents[3]
    return [Path(__file__).resolve(), Path(native_resources.__file__).resolve(),
            Path(original_resources.__file__).resolve(), Path(original_core.__file__).resolve(),
            native_resources.OWNER_PATH, root / "docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md"]


def validate_operational_authority():
    if native_resources.OWNER_SHA256 != OWNER_SHA256:
        raise ValueError("Operational resource owner pin differs")
    raw = native_resources.OWNER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OWNER_SHA256:
        raise ValueError("Exact owner operational authority changed")
    native_resources.validate_owner_resolution(json.loads(raw))


'''
    return header + ast.get_source_segment(original, cls) + "\n"


def prefix_copy():
    path = MAIN / "src/marine_echo/training/native_prefix_transfer.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "844159d11a1d9a410f7cb7f1a83c782360427e04aae965c19cb94a74d5a28d38":
        raise ValueError("Assigned immutable prefix source changed")
    text = raw.decode("utf-8")
    # Keep the scientific core import intact; route only Resources to the new class.
    tree = ast.parse(text)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    anchor = max(n.end_lineno for n in imports)
    lines = text.splitlines(keepends=True)
    lines.insert(anchor, "from marine_echo.training import native_desktop_runtime_v3 as desktop_runtime\n")
    text = "".join(lines)
    old = '    pending = [Path(__file__).resolve(), root / "__init__.py"]'
    new = '''    desktop_runtime.validate_operational_authority()
    checkout = Path(__file__).resolve().parents[3]
    operational_root = Path(core.__file__).resolve().parents[3]
    pending = [Path(__file__).resolve(), root / "__init__.py", root / "training/native_prefix_transfer.py",
               *desktop_runtime.operational_sources(),
               checkout / "tools/execute_native_prefix_desktop_job_v3.py",
               checkout / "tools/execute_native_prefix_desktop_worker_v3.py",
               operational_root / "tools/execute_native_prefix_job_v2.py",
               operational_root / "tools/execute_native_prefix_worker.py",
               operational_root / "tools/native_reference_supervisor.py",
               operational_root / "tools/native_band_budget_history_v2.py",
               operational_root / "tools/execute_native_band_job.py",
               operational_root / "orchestration/native_band_budget_owner_resolution_v1.json"]'''
    assert text.count(old) == 1
    text = text.replace(old, new)
    # The closure now deliberately includes JSON and ADR resources, not just Python.
    token = '        for node in ast.walk(ast.parse(path.read_bytes())):'
    assert text.count(token) == 1
    text = text.replace(token, '        if path.suffix != ".py":\n            continue\n' + token)
    assert text.count("with core.Resources(config.device, output) as resources:") == 1
    return text.replace("with core.Resources(config.device, output) as resources:", "with desktop_runtime.Resources(config.device, output) as resources:")


def wrapper_copy():
    text = (MAIN / "tools/execute_native_prefix_job_v2.py").read_text(encoding="utf-8")
    text = text.replace("tools/execute_native_prefix_worker.py", "tools/execute_native_prefix_desktop_worker_v3.py")
    text = text.replace("src/marine_echo/training/native_prefix_transfer.py", "src/marine_echo/training/native_prefix_desktop_transfer_v3.py")
    text = text.replace("from marine_echo.training.native_prefix_transfer import admit", "from marine_echo.training.native_prefix_desktop_transfer_v3 import admit")
    marker = '        ROOT / "orchestration/native_prefix_execution_scope_v1.json",'
    assert text.count(marker) == 1
    text = text.replace(marker, marker + '''
        ROOT / "src/marine_echo/training/native_desktop_runtime_v3.py",
        ROOT / "src/marine_echo/training/native_desktop_resources_v2.py",
        ROOT / "src/marine_echo/training/native_prefix_transfer.py",
        ROOT / "src/marine_echo/training/native_ssl.py",
        ROOT / "tools/execute_native_prefix_job_v2.py",
        ROOT / "tools/execute_native_prefix_worker.py",
        ROOT / "tools/native_band_budget_history_v2.py",
        ROOT / "tools/execute_native_band_job.py",
        ROOT / "orchestration/native_band_budget_owner_resolution_v1.json",
        ROOT / "orchestration/native_vlc_desktop_owner_resolution_v1.json",
        ROOT / "docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md",''')
    marker = '    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))'
    assert text.count(marker) == 1
    text = text.replace(marker, '''    from marine_echo.training.native_desktop_runtime_v3 import validate_operational_authority

    validate_operational_authority()
''' + marker + '''
    validate_idle_desktop_ledger(ledger, LEDGER, ROOT / "evidence/ssl-builder-v1/gpu-owner.lock")''')
    helper = '''

def validate_idle_desktop_ledger(ledger, ledger_path, owner_lock):
    """Additional operational admission; never reconcile/remove unknown state."""
    from native_band_budget_history_v2 import budget_totals

    budget_totals(ledger)
    if ledger.get("requires_reconciliation"):
        raise ValueError("Ledger requires reconciliation before desktop prefix fit")
    if owner_lock.exists() or owner_lock.is_symlink():
        raise ValueError("Existing owner lock blocks desktop prefix fit")
    suffixes = (".pending", ".prefix-pending", ".band-pending", ".assessment-pending", ".reconciliation-pending")
    journals = {ledger_path.parent / "all.pending"}
    for suffix in suffixes:
        journals.add(ledger_path.with_suffix(suffix))
        journals.add(ledger_path.parent / suffix)
        journals.update(ledger_path.parent.glob("*" + suffix))
    if any(path.exists() or path.is_symlink() for path in journals):
        raise ValueError("Pending journal blocks desktop prefix fit")
'''
    text = text.replace("\ndef main():", helper + "\n\ndef main():")
    return text


def main():
    files = {
        "src/marine_echo/training/native_desktop_runtime_v3.py": resources_copy(),
        "src/marine_echo/training/native_prefix_desktop_transfer_v3.py": prefix_copy(),
        "tools/execute_native_prefix_desktop_job_v3.py": wrapper_copy(),
        "tools/execute_native_prefix_desktop_worker_v3.py": (MAIN / "tools/execute_native_prefix_worker.py").read_text(encoding="utf-8").replace("from marine_echo.training.native_prefix_transfer import fit", "from marine_echo.training.native_prefix_desktop_transfer_v3 import fit"),
    }
    print("*** Begin Patch")
    for name, text in files.items():
        print("*** Add File: " + name)
        for line in text.splitlines():
            print("+" + line)
    print("*** End Patch")


if __name__ == "__main__":
    main()
