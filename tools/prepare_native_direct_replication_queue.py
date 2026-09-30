"""Freeze the two approved matched scratch-supervised replication commands."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    review = "evidence/ssl-research-v1/direct-replication-prefit-review-final.json"
    approval = json.loads((ROOT / review).read_text(encoding="utf-8"))
    if (approval.get("status") != "APPROVED_DOWNSTREAM_PREFIT"
            or approval.get("allowed_methods") != ["direct"]
            or approval.get("allowed_modes") != ["direct_end_to_end"]
            or approval.get("allowed_seeds") != [13, 23]):
        raise ValueError("Exact fresh-direct seed13/23 approval required")
    jobs = []
    for seed in (13, 23):
        name = f"direct_direct_end_to_end_seed{seed}_h96_replication"
        jobs.append({"id": name, "entrypoint": "tools/execute_native_downstream_job.py",
                     "method": "direct", "mode": "direct_end_to_end",
                     "report": f"outputs/native_acoustic_ssl_v1/{name}/run.json",
                     "args": ["--config", f"configs/native_downstream_direct_direct_end_to_end_seed{seed}_h96_v1.json",
                              "--review", review, "--output", f"outputs/native_acoustic_ssl_v1/{name}",
                              "--receipt", f"evidence/ssl-research-v1/{name}-attempt-01"]})
    destination = ROOT / "orchestration/native_direct_replication_approved_queue_v1.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"kind": "reviewed_native_serial_queue_v1", "jobs": jobs,
                   "authorization": "Distinct76-binding fresh-direct seed13/23 prefit review",
                   "execution": "After CF replication queue exits and current allowance is reconciled; required wrappers only",
                   "final_test_access": "NOT_AUTHORIZED", "app_release_work": "FROZEN"}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "jobs": 2, "path": str(destination)}))


if __name__ == "__main__":
    main()
