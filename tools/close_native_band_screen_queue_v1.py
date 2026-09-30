"""Record all five actual closed screens and immutable artifact/resource identities."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    queue_path = folder / "band-seed7-approved-screen-queue-attempt-01/queue.json"
    queue = json.loads(queue_path.read_bytes())
    manifest = json.loads((ROOT / "orchestration/native_band_seed7_approved_screen_queue_v1.json").read_bytes())
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_bytes())
    witness = json.loads((folder / "band-screen-queue-exit-witness-v1.json").read_bytes())
    if (witness.get("actual_cli_exit_code") != 0 or witness.get("session_id") != 82615
            or queue.get("status") != "COMPLETED_REVIEWED_BAND_SERIAL_QUEUE"
            or len(queue.get("jobs", [])) != 5 or len(manifest["jobs"]) != 5
            or queue["queue_sha256"] != digest(ROOT / "orchestration/native_band_seed7_approved_screen_queue_v1.json")
            or any(str(r.get("status", "")).startswith("RUNNING_") for r in ledger["runs"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()):
        raise ValueError("Actual five-job closure and released scientific owner required")
    rows = []
    for completed, job in zip(queue["jobs"], manifest["jobs"], strict=True):
        if completed["id"] != job["id"] or completed["exit_code"] != 0 or completed["status"] != "VERIFIED_COMPLETED":
            raise ValueError("Original exact job order or exit evidence differs")
        report_path = Path(job["report"])
        report = json.loads(report_path.read_bytes())
        review_path = Path(job["args"][job["args"].index("--review") + 1])
        review = json.loads(review_path.read_bytes())
        receipt_path = Path(review["runtime_arguments"]["receipt"]) / "attempt.json"
        attempt = json.loads(receipt_path.read_bytes())
        resources = attempt["resources_full_attempt"]
        if (completed["run_sha256"] != digest(report_path) or report["config"] != review["approved_config"]
                or report["status"] != "COMPLETED" or report["evidence_kind"] != "REAL_TRAIN_DEVELOPMENT_FIT"
                or report["architecture"] != "nonlinear_frequency_conditioned_v1"
                or report["inference_sha256"] != digest(report_path.parent / "inference.pt")
                or attempt["status"] != "COMPLETED_REAL_BAND_CUDA"
                or attempt["budget_family"] != "native_band_v1"
                or resources["exit_code"] != 0 or resources["stopped_for"] is not None
                or resources["owned_tree_cleanup_verified"] is not True):
            raise ValueError("Closed source, weights, approval or owned resource evidence differs")
        rows.append({"id": job["id"], "config": report["config"],
                     "development_pinball_db": report["selected_daily_dev_pinball"],
                     "artifacts_sha256": {name: digest(report_path.parent / name) for name in (
                         "run.json", "inference.pt", "selected_encoder.pt", "membership.json", "predictions.npz")},
                     "review_sha256": digest(review_path), "attempt_sha256": digest(receipt_path),
                     "resources_full_attempt": resources})
    result = {"status": "FIVE_REAL_BAND_SCREEN_ENDPOINTS_VERIFIED_CLOSED",
              "actual_exit_witness": witness, "jobs": rows,
              "band_screen_full_owned_gpu_hours": sum(r["resources_full_attempt"]["elapsed_full_attempt_seconds"] for r in rows) / 3600,
              "aggregate_charged_full_owned_gpu_hours": ledger["gpu_hours_spent_owned_scientific_jobs"],
              "metric_scope": "Runner development selection; independent reconstruction pending",
              "representation_transfer_value": "NOT_ESTABLISHED", "final_numeric_access": "NOT_RUN",
              "ledger_written": False}
    with (folder / "band-five-screen-completion-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "jobs": len(rows),
                      "band_screen_full_owned_gpu_hours": result["band_screen_full_owned_gpu_hours"],
                      "aggregate_charged_full_owned_gpu_hours": result["aggregate_charged_full_owned_gpu_hours"]}))


if __name__ == "__main__":
    main()
