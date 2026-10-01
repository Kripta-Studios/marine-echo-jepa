"""Run exactly one bounded CPU artifact audit after immutable V6 reservation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from prepare_completed_native_inventory_v6 import (
    digest,
    document,
    fresh,
    idle_ledger,
    regular,
    reservation_path,
)

ROOT = Path(__file__).resolve().parents[1]


def execute(
    root, manifest_path, snapshot_path, output_path, receipt_path, *, launcher=None, supervisor=None
):
    """Private synthetic policy tests inject callables; production CLI never does."""
    started = time.monotonic()
    root = Path(root).absolute()
    output_path = fresh(output_path, root, "evidence")
    receipt_path = fresh(receipt_path, root, "evidence")
    if (
        output_path == receipt_path
        or output_path in receipt_path.parents
        or receipt_path in output_path.parents
    ):
        raise ValueError("Separate fresh output and receipt required")
    idle_ledger(root)
    manifest_path, snapshot_path = regular(manifest_path), regular(snapshot_path)
    doc, facts = document(manifest_path), document(reservation_path(snapshot_path))
    endpoints = doc.get("endpoints")
    if (
        doc.get("kind") != "native_research_inventory_manifest_v1"
        or doc.get("purpose") != "LOCAL_METADATA_DERIVATION_ONLY"
        or doc.get("execution") != "ROOT_LOCAL_COMPLETED_METADATA_AUDIT"
        or doc.get("device") != "cpu"
        or doc.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
        or doc.get("output_path") != str(output_path)
        or not isinstance(endpoints, dict)
        or len(endpoints) != 43
        or doc.get("references") != {}
        or facts.get("kind") != "native_completed_inventory_preparation_v6"
        or facts.get("manifest_path") != str(manifest_path)
        or facts.get("manifest_sha256") != digest(manifest_path)
        or facts.get("ledger_snapshot_path") != str(snapshot_path)
        or facts.get("ledger_snapshot_sha256") != digest(snapshot_path)
        or facts.get("neural_count") != 43
        or facts.get("method_count") != 47
        or facts.get("status") != "METADATA_RESERVED_NOT_APPROVAL"
        or any(
            facts.get(key) is not False
            for key in ("fit_authority", "final_selection", "numeric_decoding")
        )
        or facts.get("neural_methods") != sorted(endpoints)
        or set(facts.get("references", {}))
        != {"persistence", "seasonal24", "lightgbm", "chronos2_zero_shot"}
    ):
        raise ValueError("Exact immutable43/47 CPU reservation required")
    bindings = doc.get("bindings")
    required = {
        str(snapshot_path),
        str(Path(__file__).resolve()),
        str(Path(__file__).with_name("prepare_completed_native_inventory_v6.py").resolve()),
        str(Path(__file__).with_name("native_train_scaler_reservation_v1.py").resolve()),
        str(root / "tools/native_reference_supervisor.py"),
        str(root / "tools/prepare_native_research_inventory.py"),
        facts["matrix_path"],
    }
    if (
        not isinstance(bindings, dict)
        or not required <= bindings.keys()
        or str(root / "orchestration/native_ssl_run_ledger_v1.json") in bindings
    ):
        raise ValueError(
            "Immutable snapshot, executor, supervisor and audit-source bindings required"
        )
    for name, expected in bindings.items():
        if digest(name) != expected:
            raise ValueError(f"Changed immutable audit binding: {name}")
    if bindings[facts["matrix_path"]] != facts.get("matrix_sha256"):
        raise ValueError("Changed reserved comparison matrix")
    idle_ledger(root)  # Recheck immediately before any mutation/launch; snapshot stays immutable.
    receipt_path.mkdir()
    command = [
        str(root / ".venv/Scripts/python.exe"),
        "-B",
        str(root / "tools/prepare_native_research_inventory.py"),
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_path),
    ]
    record = {
        "kind": "native_completed_inventory_owned_attempt_v6",
        "status": "STARTING_CPU_AUDIT",
        "command": command,
        "device": "cpu",
        "fitting": False,
        "scientific_prefit": False,
        "manifest_sha256": digest(manifest_path),
        "ledger_snapshot_sha256": digest(snapshot_path),
        "reservation_sha256": digest(reservation_path(snapshot_path)),
        "deadline_seconds": 600,
        "rss_limit_bytes": 22 * 1024**3,
    }
    with (receipt_path / "attempt-start.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        if supervisor is None:
            from native_reference_supervisor import supervise_owned

            supervisor = supervise_owned
        if launcher is None:
            launcher = subprocess.Popen
        with (receipt_path / "stdout.log").open("x", encoding="utf-8") as log:
            idle_ledger(root)
            child = launcher(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            record["child_pid"] = child.pid
            resources = supervisor(
                child, started=started, deadline_seconds=600, rss_limit_bytes=22 * 1024**3
            )
        report_path = output_path / "inventory.json"
        result = document(report_path) if report_path.is_file() else {}
        peak, elapsed = (
            resources.get(key) for key in ("peak_process_rss_bytes", "elapsed_full_attempt_seconds")
        )
        bounded = (
            all(
                type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in (peak, elapsed)
            )
            and peak < 22 * 1024**3
            and elapsed < 600
        )
        completed = (
            resources.get("exit_code") == 0
            and resources.get("stopped_for") is None
            and resources.get("owned_tree_cleanup_verified") is True
            and bounded
            and result.get("kind") == "native_research_inventory_v1"
            and result.get("status") == "DERIVED_METADATA_NOT_FINAL_SELECTION"
            and result.get("manifest_sha256") == record["manifest_sha256"]
            and isinstance(result.get("models"), dict)
            and set(result["models"]) == set(endpoints)
            and result.get("numeric_corpus_decoded") is False
            and result.get("model_constructed") is False
        )
        if report_path.is_file():
            record["report_sha256"] = digest(report_path)
        record.update(
            {
                "status": "CPU_ARTIFACT_AUDIT_PASSED" if completed else "CPU_ARTIFACT_AUDIT_FAILED",
                "resources": resources,
                "elapsed_full_attempt_seconds": time.monotonic() - started,
                "source_bindings": bindings,
            }
        )
        with (receipt_path / "resources.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return (0 if completed else 1), record
    except BaseException as error:
        record.update(
            {
                "status": "CPU_AUDIT_RECONCILIATION_REQUIRED",
                "requires_reconciliation": True,
                "exception_type": type(error).__name__,
                "elapsed_full_attempt_seconds": time.monotonic() - started,
            }
        )
        with (receipt_path / "parent-exception.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        raise


def main():
    if ROOT.name != "marine-echo-jepa":
        raise RuntimeError("Real owned artifact audit is ROOT-only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "orchestration/native_completed_inventory_v6.json"
    )
    parser.add_argument(
        "--ledger-snapshot",
        type=Path,
        default=ROOT / "evidence/ssl-research-v1/native-completed-inventory-ledger-v6.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evidence/native-completed-inventory-v6"
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "evidence/ssl-research-v1/native-completed-inventory-owned-v4",
    )
    args = parser.parse_args()
    code, record = execute(ROOT, args.manifest, args.ledger_snapshot, args.output, args.receipt)
    print(
        json.dumps(
            {"status": record["status"], "actual_exit_code": record["resources"]["exit_code"]}
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
