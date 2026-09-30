"""Freeze the three separately approved original-control transfer commands."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    review = "evidence/ssl-research-v1/masked-permuted-downstream-prefit-review-final.json"
    approval = json.loads((ROOT / review).read_text(encoding="utf-8"))
    if approval.get("status") != "APPROVED_DOWNSTREAM_PREFIT":
        raise ValueError("Actual completed-parent approval required")
    jobs = []
    for method, mode in (("masked_ssl", "frozen_readout"),
                         ("masked_ssl", "full_finetune"),
                         ("permuted_ssl", "frozen_readout")):
        name = f"{method}_{mode}_seed7_h96"
        jobs.append({"id": name, "entrypoint": "tools/execute_native_downstream_job.py",
                     "method": method, "mode": mode,
                     "report": f"outputs/native_acoustic_ssl_v1/{name}/run.json",
                     "args": ["--config", f"configs/native_downstream_{name}_v1.json",
                              "--review", review, "--output", f"outputs/native_acoustic_ssl_v1/{name}",
                              "--receipt", f"evidence/ssl-research-v1/{name}-attempt-01",
                              "--encoder", f"outputs/native_acoustic_ssl_v1/{method}_seed7_h96_reviewed/selected_encoder.pt",
                              "--ancestor-review", "evidence/ssl-research-v1/prefit-review-final-03.json",
                              "--ancestor-config", f"configs/native_ssl_{method}_seed7_h96_v1.json"]})
    destination = ROOT / "orchestration/native_masked_permuted_approved_queue_v1.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"kind": "reviewed_native_serial_queue_v1", "jobs": jobs,
                   "authorization": "Distinct exact masked/permuted completed-parent approval",
                   "execution": "After CF endpoint queue exits; approved wrappers only",
                   "final_test_access": "NOT_AUTHORIZED", "app_release_work": "FROZEN"}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "jobs": 3, "path": str(destination)}))


if __name__ == "__main__":
    main()
