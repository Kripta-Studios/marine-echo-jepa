"""Stage a finite explicit evidence list without the Windows argument-length limit."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_bytes())
    if any(str(run.get("status", "")).startswith("RUNNING_") or run.get("requires_reconciliation") for run in ledger["runs"]) or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists():
        raise ValueError("Do not stage active scientific accounting")
    paths = [
        ".gitattributes", "orchestration/STATUS.md", "orchestration/native_ssl_run_ledger_v1.json",
        "src/marine_echo/evaluation/native_ancestry_inventory.py", "tests/unit/test_native_inventory_direct_cadence.py",
        "docs/NATIVE_SSL_MODEL_RESEARCH_REPORT_ADDENDUM_V3.md",
        *["tools/" + name for name in (
            "prepare_closed_native_inventory_v3b.py", "version_closed_native_inventory_owned_v3b.py",
            "execute_closed_native_inventory_owned_v3b.py", "prepare_completed_native_inventory_v5.py",
            "version_completed_native_inventory_owned_v5.py", "execute_completed_native_inventory_owned_v5.py",
            "close_native_band_research_progress_v5.py", "close_native_seed23_strong_prefit_rejection_v3.py",
            "stage_closed_native_research_v5.py",
        )],
        *["orchestration/" + name for name in (
            "native_completed_inventory_v3a.json", "native_completed_inventory_v3b.json", "native_completed_inventory_v5.json",
            "native_band_replication_downstream_seed23_v3.json", "native_band_downstream_references_seed23_v3.json",
            "ssl_vnext_band_seed23_strong_prefit_v3.txt", "ssl_vnext_pretrained_snapshot_v2_builder.txt",
        )],
        *["configs/native_band_replication_downstream_v3/" + name for name in (
            "native_band_shared_ssl_frozen_readout_seed23_v3.json", "native_band_shared_ssl_full_finetune_seed23_v3.json",
        )],
        *["evidence/ssl-research-v1/" + name for name in (
            "band-seed23-strong-prefit-events-v3.jsonl", "band-seed23-strong-prefit-stderr-v3.log",
            "band-seed23-strong-prefit-rejection-closeout-v3.json", "native-research-real-jobs-closeout-v5.json",
            "band-fixed-replication-remaining-controller-console-v3.log", "band-remaining-original-controls-console-v1.log",
        )],
    ]
    folders = [
        "evidence/native-completed-inventory-v3b", "evidence/native-completed-inventory-v5",
        *["evidence/ssl-research-v1/" + name for name in (
            "native-fixed35-reservation-v3a", "native-completed40-reservation-v5",
            "native-completed-inventory-owned-v3a", "native-completed-inventory-owned-v3b", "native-completed-inventory-owned-v5",
            "band-fixed-replication-remaining-queue-attempt-01", "band-remaining-original-controls-queue-attempt-01",
            "band_shared_ssl_seed23_h96_replication_v2-attempt-01", "band_direct_end_to_end_seed23_h96_replication_v2-attempt-01",
            "band_shared_ssl_frozen_readout_seed13_h96_replication_v3-attempt-01",
            "band_shared_ssl_full_finetune_seed13_h96_replication_v3-attempt-01",
            "band_random_frozen_frozen_readout_seed7_h96_native_retry01-attempt-01",
        )],
    ]
    for folder in folders:
        paths.extend(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).iterdir() if p.is_file() and p.suffix in (".json", ".log"))
    paths.extend("evidence/ssl-native-ancestry-inventory-builder-v1/" + name for name in (
        "root-fixed35-owned-audit-v3a.log", "root-fixed35-owned-audit-v3b.log", "root-completed40-owned-audit-v5.log",
        "root-direct-cadence-red-v3b.log", "root-direct-cadence-green-v3b.log", "root-combined-checks-v3b.log",
        "root-combined-checks-importlib-v3b.log",
    ))
    if any(not (ROOT / path).is_file() for path in paths):
        raise FileNotFoundError("Every explicit evidence file must actually exist")
    pathlist = ROOT / "orchestration/native_closed_research_git_paths_v5.txt"
    with pathlist.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(sorted(set(paths))) + "\n")
    completed = subprocess.run(["git", "--literal-pathspecs", "add", "--pathspec-from-file=" + str(pathlist)], cwd=ROOT, check=False)
    print(json.dumps({"actual_git_exit_code": completed.returncode, "explicit_paths": len(set(paths))}))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
