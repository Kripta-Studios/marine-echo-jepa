"""Reconcile five actual completed jobs and the complete current neural ancestry audit."""

import hashlib
import json
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    queues = (
        (folder / "band-fixed-replication-remaining-queue-attempt-01/queue.json", "COMPLETED_TWO_ORIGINAL_SEED23_REPLICATIONS", 2),
        (folder / "band-remaining-original-controls-queue-attempt-01/queue.json", "COMPLETED_THREE_PREVIOUSLY_APPROVED_ORIGINAL_DOWNSTREAM_JOBS", 3),
    )
    bindings, jobs = {}, []
    for queue_path, status, count in queues:
        queue = json.loads(queue_path.read_bytes())
        if queue.get("status") != status or len(queue.get("jobs", [])) != count:
            raise ValueError("Actual successful closed serial queues required")
        bindings[str(queue_path)] = digest(queue_path)
        for job in queue["jobs"]:
            if job.get("status") != "VERIFIED_COMPLETED" or job.get("actual_exit_code") != 0:
                raise ValueError("Every actual job must have verified completion")
            run_path = ROOT / "outputs/native_acoustic_ssl_v1" / job["id"] / "run.json"
            attempt_path = folder / (job["id"] + "-attempt-01") / "attempt.json"
            run, attempt = (json.loads(path.read_bytes()) for path in (run_path, attempt_path))
            resources = attempt["resources_full_attempt"]
            device = run.get("resources", run.get("resources_additional_downstream"))
            if (digest(run_path) != job["run_sha256"] or digest(attempt_path) != job["attempt_sha256"]
                    or run.get("status") != "COMPLETED" or run.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
                    or resources.get("exit_code") != 0 or resources.get("stopped_for") is not None
                    or resources.get("owned_tree_cleanup_verified") is not True
                    or resources["peak_process_rss_bytes"] >= 22 * 1024**3
                    or any(device[key] >= 10 * 1024**3 for key in ("peak_allocated_bytes", "peak_reserved_bytes"))
                    or attempt.get("budget_family") != "native_band_v1"):
                raise ValueError("Actual bound completed scientific resource evidence required")
            review_path = Path(attempt["review"])
            if digest(review_path) != run["review_sha256"]:
                raise ValueError("Actual original independent review identity required")
            bindings.update({str(path): digest(path) for path in (run_path, attempt_path, review_path)})
            jobs.append({**job, "full_owned_seconds": resources["elapsed_full_attempt_seconds"],
                         "native_geometry_preserved": True, "metric_scope": "RUNNER_DEVELOPMENT_NOT_INDEPENDENT_RECONSTRUCTION"})
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(ledger_path.read_bytes())
    if (any(str(run.get("status", "")).startswith("RUNNING_") or run.get("requires_reconciliation") for run in ledger["runs"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
            or any(ledger_path.with_suffix(suffix).exists() for suffix in (".pending", ".prefix-pending", ".band-pending", ".assessment-pending", ".reconciliation-pending"))):
        raise ValueError("Every scientific ownership and journal must be closed")
    if not 0 <= ledger["native_band_gpu_hours_spent_full_owned"] < 12 or not 0 <= ledger["gpu_hours_spent_owned_scientific_jobs"] < 96:
        raise ValueError("Owner-resolved finite resource budgets required")
    inventory_path = ROOT / "evidence/native-completed-inventory-v5/inventory.json"
    resources_path = folder / "native-completed-inventory-owned-v5/resources.json"
    inventory, audit = (json.loads(path.read_bytes()) for path in (inventory_path, resources_path))
    owned = audit["resources"]
    if (len(inventory["models"]) != 40 or inventory.get("numeric_corpus_decoded") is not False
            or inventory.get("model_constructed") is not False or owned.get("exit_code") != 0
            or owned.get("owned_tree_cleanup_verified") is not True or owned.get("stopped_for") is not None
            or any(psutil.pid_exists(pid) for pid in owned["owned_process_pids"])):
        raise ValueError("Actual complete current40 metadata audit and verified closure required")
    manifest_path = ROOT / "orchestration/native_completed_inventory_v5.json"
    manifest = json.loads(manifest_path.read_bytes())
    paths = (inventory_path, resources_path, manifest_path, ledger_path, Path(__file__).resolve())
    bindings.update({str(path): digest(path) for path in paths})
    record = {
        "status": "FIVE_REAL_JOBS_AND_CURRENT40_NEURAL_ANCESTRY_VERIFIED_PROGRAMME_INCOMPLETE",
        "queue_actual_exit_witnesses": [{"session_id": 33144, "exit_code": 0, "chunk": "eb4538"}, {"session_id": 9450, "exit_code": 0, "chunk": "cbfe2a"}],
        "audit_actual_exit_witness": {"session_id": 81165, "exit_code": 0, "chunk": "320c8b"},
        "jobs": jobs, "neural_endpoints": 40, "fitted_parent_links": sum(e["parent"] is not None for e in manifest["endpoints"].values()),
        "planned_missing_neural_endpoints": ["band_direct_end_to_end_seed13_ownership_retry01", "band_shared_ssl_frozen_readout_seed23", "band_shared_ssl_full_finetune_seed23"],
        "band_full_owned_gpu_hours": ledger["native_band_gpu_hours_spent_full_owned"],
        "aggregate_full_owned_gpu_hours": ledger["gpu_hours_spent_owned_scientific_jobs"],
        "final_site_numeric_access": "NOT_RUN", "public_prefix_transfer": "NOT_RUN", "sota": "NOT_ESTABLISHED",
        "expanded_numeric_reconstruction": "NOT_RUN", "independent_ancestry_review": "NOT_RUN", "bindings": bindings,
    }
    with (folder / "native-research-real-jobs-closeout-v5.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "jobs": 5, "models": 40, "fitted_parent_links": record["fitted_parent_links"]}))


if __name__ == "__main__":
    main()
