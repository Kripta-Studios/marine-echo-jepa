"""Verify five completed fits and preserve the failed final control without claiming completion."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evidence/ssl-research-v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    witness = json.loads((FOLDER / "band-downstream-queue-exit-witness-v1.json").read_bytes())
    if witness.get("session_id") != 5351 or witness.get("actual_cli_exit_code") != 1:
        raise ValueError("Actual closed scientific controller required")
    queue_path = FOLDER / "band-downstream-serial-attempt-01/queue.json"
    queue = json.loads(queue_path.read_bytes())
    proposal = json.loads((ROOT / "orchestration/native_band_downstream_admission_v1.json").read_bytes())
    expected_pairs = [job["approval"]["approved_config"]["method"] + "/" + job["approval"]["approved_config"]["mode"]
                      for job in proposal["jobs"]]
    if (queue.get("status") != "STOPPED_AFTER_FAILURE"
            or [entry["id"] for entry in queue["jobs"]] != expected_pairs):
        raise ValueError("Exact original five completions followed by control failure required")
    if queue["jobs"][-1].get("status") != "FAILED_QUEUE_STOPPED" or queue["jobs"][-1].get("actual_exit_code") != 4294967295:
        raise ValueError("Preserved actual final wrapper failure required")
    rows = []
    for job, entry in zip(proposal["jobs"][:-1], queue["jobs"][:-1], strict=True):
        runtime = job["approval"]["runtime_arguments"]
        output = Path(runtime["output"])
        run_path, attempt_path = output / "run.json", Path(runtime["receipt"]) / "attempt.json"
        run, attempt = (json.loads(path.read_bytes()) for path in (run_path, attempt_path))
        config = job["approval"]["approved_config"]
        resources, peaks = attempt["resources_full_attempt"], run["resources_additional_downstream"]
        review_path = Path(runtime["review"])
        if (entry.get("status") != "VERIFIED_COMPLETED" or entry.get("actual_exit_code") != 0
                or entry["run_sha256"] != digest(run_path) or entry["attempt_sha256"] != digest(attempt_path)
                or run.get("status") != "COMPLETED" or run.get("config") != config
                or run.get("architecture") != "nonlinear_frequency_conditioned_v1"
                or run.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
                or run["supervised_updates"] != config["updates"]
                or run["optimizer_steps_total"] != config["updates"]
                or [record["step"] for record in run["selection_records"]] != list(range(config["cadence"], config["updates"] + 1, config["cadence"]))
                or run["selected_supervised_step"] not in [record["step"] for record in run["selection_records"]]
                or run["review_sha256"] != digest(review_path)
                or resources.get("exit_code") != 0 or resources.get("stopped_for") is not None
                or resources.get("owned_tree_cleanup_verified") is not True
                or resources["peak_process_rss_bytes"] >= 22 * 2**30
                or max(peaks["peak_allocated_bytes"], peaks["peak_reserved_bytes"]) >= 10 * 2**30
                or attempt.get("budget_family") != "native_band_v1"):
            raise ValueError("Actual fit/checkpoint-selection/resource completion differs")
        for name, expected in run["bindings"].items():
            if digest(name) != expected:
                raise ValueError(f"Actual scientific source binding changed: {name}")
        parent = Path(runtime["encoder"])
        if (run["supervised_ancestry"]["ancestor_encoder_sha256"] != digest(parent)
                or run["supervised_ancestry"]["ssl_only"] is not False
                or digest(output / "selected_encoder.pt") != run["selected_encoder_sha256"]
                or digest(output / "inference.pt") != run["inference_sha256"]
                or digest(output / "membership.json") != run["membership_sha256"]):
            raise ValueError("Actual parent/artifact ancestry differs")
        unchanged = digest(output / "selected_encoder.pt") == digest(parent)
        if config["mode"] == "frozen_readout" and not unchanged:
            raise ValueError("Frozen parent encoder byte preservation missing")
        rows.append({"id": job["id"], "run_sha256": digest(run_path), "attempt_sha256": digest(attempt_path),
                     "inference_sha256": run["inference_sha256"], "selected_encoder_sha256": run["selected_encoder_sha256"],
                     "selected_supervised_step": run["selected_supervised_step"], "runner_development_pinball_db": run["selected_daily_dev_pinball"],
                     "selected_encoder_bytes_equal_parent": unchanged,
                     "fine_tuning_tensor_update_check": "NOT_RUN_BY_THIS_METADATA_CLOSEOUT",
                     "full_owned_seconds": resources["elapsed_full_attempt_seconds"]})
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_bytes())
    failed = json.loads((FOLDER / "band_random_frozen_frozen_readout_seed7_h96-attempt-01/attempt.json").read_bytes())
    if failed.get("resources_full_attempt", {}).get("exit_code") != 3221227274:
        raise ValueError("Actual failed control resource receipt required")
    band_seconds = sum(record.get("elapsed_owned_seconds", record.get("resources_full_attempt", {}).get("elapsed_full_attempt_seconds", 0))
                       for record in ledger["runs"] if record.get("budget_family") == "native_band_v1")
    receipt = {"status": "FIVE_REAL_BAND_DOWNSTREAM_FITS_VERIFIED_FINAL_CONTROL_FAILED", "actual_exit_witness": witness,
               "queue_sha256": digest(queue_path), "jobs": rows, "band_full_owned_gpu_hours": band_seconds / 3600,
               "aggregate_full_owned_gpu_hours": ledger["gpu_hours_spent_owned_scientific_jobs"],
               "usage_snapshot_excludes_active_attempts": True,
               "failed_control_full_owned_seconds": failed["resources_full_attempt"]["elapsed_full_attempt_seconds"],
               "programme_or_six_fit_completion": False,
               "independent_numerical_reconstruction": "NOT_RUN", "held_out_transfer": "NOT_RUN", "sota": "NOT_ESTABLISHED"}
    with (FOLDER / "band-five-downstream-partial-closeout-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "jobs": len(rows), "band_hours": band_seconds / 3600}))


if __name__ == "__main__":
    main()
