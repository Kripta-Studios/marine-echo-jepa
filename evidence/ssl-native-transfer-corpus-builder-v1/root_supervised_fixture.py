"""ROOT-only actual owned CPU codec test; no public data, fit, or approval."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path


def check_idle(root):
    """Read actual ownership metadata; never reconcile or remove a lock/journal."""
    lock = root / "evidence/ssl-builder-v1/gpu-owner.lock"
    if lock.exists() or lock.is_symlink():
        raise RuntimeError("Active or unknown scientific owner lock; fixture is NOT_RUN.")
    orchestration = root / "orchestration"
    ledger = orchestration / "native_ssl_run_ledger_v1.json"
    suffixes = (
        ".pending",
        ".prefix-pending",
        ".band-pending",
        ".assessment-pending",
        ".reconciliation-pending",
    )
    journals = {orchestration / "all.pending"}
    for suffix in suffixes:
        journals.add(ledger.with_suffix(suffix))
        journals.add(orchestration / suffix)
        journals.update(orchestration.glob("*" + suffix))
    if any(path.exists() or path.is_symlink() for path in journals):
        raise RuntimeError("Pending scientific journal; reconciliation is required before fixture.")
    try:
        document = json.loads(ledger.read_bytes())
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Missing or unreadable actual scientific ledger; fixture is NOT_RUN."
        ) from error
    if not isinstance(document, dict) or not isinstance(document.get("runs"), list):
        raise RuntimeError("Malformed actual scientific ledger; fixture is NOT_RUN.")  # noqa: TRY004 -- uniform ownership-admission failure
    records = document["runs"]
    if any(not isinstance(record, dict) for record in records):
        raise RuntimeError("Malformed scientific ledger record; fixture is NOT_RUN.")
    if document.get("requires_reconciliation") or any(
        record.get("requires_reconciliation") for record in records
    ):
        raise RuntimeError("Scientific ledger requires reconciliation; fixture is NOT_RUN.")
    if any(record.get("status") in {"RUNNING_CUDA", "RUNNING_CPU_FIT"} for record in records):
        raise RuntimeError("Active scientific owner; do not run fixture concurrently.")


def main():
    root = Path(__file__).resolve().parents[2]
    if root.name != "marine-echo-jepa":
        raise RuntimeError("ROOT-only lifecycle check; NOT_RUN in builder.")
    check_idle(root)
    source = root / "tools/native_reference_supervisor.py"
    expected = "ec5456584ca6f6a805e13f4137365c2c0a785b2dbc52782b191c1528befb30dd"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Immutable supervisor binding changed.")
    spec = importlib.util.spec_from_file_location("root_native_transfer_supervisor", source)
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    directory = Path(__file__).parent / ("SYNTHETIC_CORRECTNESS_ONLY-owned-" + str(uuid.uuid4()))
    check_idle(root)
    directory.mkdir()
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "tests/integration/test_native_transfer_corpus.py",
        "-k",
        "actual_reader_source_metadata_and_two_source_equivalence",
        "--import-mode=importlib",
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    started = time.monotonic()
    with (directory / "stdout.log").open("x", encoding="utf-8") as output:
        check_idle(root)
        child = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
        resources = supervisor.supervise_owned(
            child, started=started, deadline_seconds=600, rss_limit_bytes=22 * 1024**3
        )
    with (directory / "resources.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "device": "cpu",
                "fitting": False,
                "command": command,
                "resources": resources,
            },
            stream,
            indent=2,
        )
    if (
        resources["exit_code"] != 0
        or resources["stopped_for"] is not None
        or not resources["owned_tree_cleanup_verified"]
    ):
        raise RuntimeError("Owned synthetic CPU fixture did not complete cleanly.")
    print(json.dumps({"status": "SYNTHETIC_CORRECTNESS_ONLY_COMPLETED", "output": str(directory)}))


if __name__ == "__main__":
    main()
