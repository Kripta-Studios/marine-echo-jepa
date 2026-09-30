"""Prepare exactly the two separately approved frozen control readouts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    review = "evidence/ssl-research-v1/random-direct-downstream-prefit-review-final.json"
    approval = json.loads((ROOT / review).read_text(encoding="utf-8"))
    if approval.get("status") != "APPROVED_DOWNSTREAM_PREFIT" or approval.get("allowed_modes") != ["frozen_readout"]:
        raise ValueError("Exact completed-parent frozen-only approval required")
    jobs = []
    for method in ("random_frozen", "direct"):
        name = f"{method}_frozen_readout_seed7_h96"
        jobs.append({"id": name, "entrypoint": "tools/execute_native_downstream_job.py",
                     "method": method, "mode": "frozen_readout",
                     "report": f"outputs/native_acoustic_ssl_v1/{name}/run.json",
                     "args": ["--config", f"configs/native_downstream_{name}_v1.json",
                              "--review", review, "--output", f"outputs/native_acoustic_ssl_v1/{name}",
                              "--receipt", f"evidence/ssl-research-v1/{name}-attempt-01",
                              "--encoder", f"outputs/native_acoustic_ssl_v1/{method}_seed7_h96_reviewed/selected_encoder.pt",
                              "--ancestor-review", "evidence/ssl-research-v1/prefit-review-final-03.json",
                              "--ancestor-config", f"configs/native_ssl_{method}_seed7_h96_v1.json"]})
    destination = ROOT / "orchestration/native_random_direct_approved_queue_v1.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"kind": "reviewed_native_serial_queue_v1", "jobs": jobs,
                   "authorization": "Distinct83-binding completed-parent frozen control review",
                   "execution": "After masked/permuted queue exits; approved wrappers only",
                   "final_test_access": "NOT_AUTHORIZED", "app_release_work": "FROZEN"}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "jobs": 2, "path": str(destination)}))


if __name__ == "__main__":
    main()
