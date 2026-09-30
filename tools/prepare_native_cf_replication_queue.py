"""Prepare two approved fresh CF replications; preserve seed7 and final tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    review = "evidence/ssl-research-v1/cf-replication-prefit-review-final.json"
    approval = json.loads((ROOT / review).read_text(encoding="utf-8"))
    if approval.get("status") != "APPROVED_PREFIT" or approval.get("allowed_seeds") != [13, 23]:
        raise ValueError("Exact separate replication approval required")
    jobs = []
    for seed in (13, 23):
        name = f"cf_jepa_seed{seed}_h96_replication"
        jobs.append({"id": name, "entrypoint": "tools/execute_native_ssl_job.py",
                     "method": "cf_jepa", "mode": None,
                     "report": f"outputs/native_acoustic_ssl_v1/{name}/run.json",
                     "args": ["--config", f"configs/native_ssl_cf_jepa_seed{seed}_h96_v1.json",
                              "--review", review,
                              "--output", f"outputs/native_acoustic_ssl_v1/{name}"]})
    destination = ROOT / "orchestration/native_cf_replication_approved_queue_v1.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"kind": "reviewed_native_serial_queue_v1", "jobs": jobs,
                   "authorization": "Distinct67-binding CF seed13/23 prefit",
                   "execution": "After approved control queue exits; exact required wrappers only",
                   "final_test_access": "NOT_AUTHORIZED", "app_release_work": "FROZEN"}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "jobs": 2, "path": str(destination)}))


if __name__ == "__main__":
    main()
