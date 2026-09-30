"""Record both real CF replication endpoints from closed queue and ledger evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    queue_path = ROOT / "evidence/ssl-research-v1/cf-replication-queue-attempt-01/queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_text(encoding="utf-8"))
    if queue.get("status") != "COMPLETED_REVIEWED_SERIAL_QUEUE" or len(queue["jobs"]) != 2:
        raise ValueError("Closed completed two-job queue required")
    summaries = []
    for seed, job in zip((13, 23), queue["jobs"], strict=True):
        report_path = ROOT / job["report"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entry = next(r for r in ledger["runs"] if r["id"] == job["id"])
        if (job.get("exit_code") != 0 or report.get("status") != "COMPLETED"
                or report["config"]["seed"] != seed or digest(report_path) != job["report_sha256"]
                or entry.get("status") != "COMPLETED_REAL_CUDA"
                or not entry["resources_full_attempt"]["owned_tree_cleanup_verified"]):
            raise ValueError("Actual report/exit/cleanup provenance differs")
        summaries.append({"seed": seed, "id": job["id"], "run_sha256": digest(report_path),
                          "selected_pretrain_step": report["selected_pretrain_step"],
                          "short_probe_development_pinball_db": report["selected_daily_dev_pinball"],
                          "inference_sha256": report["inference_sha256"],
                          "resources_full_attempt": entry["resources_full_attempt"],
                          "strong_endpoints": "NOT_RUN", "test_values_opened": False})
    receipt = {"status": "VERIFIED_TWO_REAL_CF_REPLICATIONS", "queue_sha256": digest(queue_path),
               "actual_coordinator_exit_code": 0,
               "coordinator_exit_witness": "Root write_stdin session41887 completion chunk5c9c79; queue ledger independently confirms both child exits/cleanup",
               "runs": summaries, "all_seeds_preserved": True, "seed_selection": "NONE",
               "scientific_claim": "NOT_ESTABLISHED", "public_final_numeric_access": "NOT_RUN"}
    path = ROOT / "evidence/ssl-research-v1/cf-replication-completion-v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "runs": [{k: r[k] for k in ("seed", "selected_pretrain_step", "short_probe_development_pinball_db")} for r in summaries]}))


if __name__ == "__main__":
    main()
