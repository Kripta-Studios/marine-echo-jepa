"""Preserve every completed fresh supervised replication, without seed selection."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    queue_path = ROOT / "evidence/ssl-research-v1/direct-replication-queue-attempt-01/queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    ledger = json.loads(
        (ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_text(encoding="utf-8")
    )
    if queue.get("status") != "COMPLETED_REVIEWED_SERIAL_QUEUE" or len(queue["jobs"]) != 2:
        raise ValueError("Both actual supervised endpoints must be completed")
    records = []
    for seed, job in zip((13, 23), queue["jobs"], strict=True):
        report = json.loads((ROOT / job["report"]).read_text(encoding="utf-8"))
        run = next(r for r in ledger["runs"] if r["id"] == job["id"])
        if (
            job["exit_code"] != 0
            or digest(ROOT / job["report"]) != job["report_sha256"]
            or report.get("status") != "COMPLETED"
            or report["config"]["seed"] != seed
            or report.get("mode") != "direct_end_to_end"
            or run.get("status") != "COMPLETED_REAL_DOWNSTREAM"
            or not run["resources_full_attempt"]["owned_tree_cleanup_verified"]
        ):
            raise ValueError("Completed supervised identity/exit/cleanup evidence differs")
        records.append(
            {
                "seed": seed,
                "id": job["id"],
                "run_sha256": job["report_sha256"],
                "development_pinball_db": report["selected_daily_dev_pinball"],
                "selected_supervised_step": report["selected_supervised_step"],
                "resources_full_attempt": run["resources_full_attempt"],
                "initialization": "FRESH_NO_FITTED_PARENT",
                "test_access": "NOT_RUN",
            }
        )
    result = {
        "status": "VERIFIED_TWO_REAL_SUPERVISED_REPLICATIONS",
        "runs": records,
        "coordinator_exit_code": 0,
        "exit_witness": "Root write_stdin session19737 chunk5a9408 exit0",
        "queue_sha256": digest(queue_path),
        "seed_selection": "NONE",
        "scientific_claim": "NOT_ESTABLISHED",
        "cross_architecture_parameter_matching": False,
    }
    with (ROOT / "evidence/ssl-research-v1/direct-replication-completion-v1.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"status": result["status"], "scores": [r["development_pinball_db"] for r in records]}
        )
    )


if __name__ == "__main__":
    main()
