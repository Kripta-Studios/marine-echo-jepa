"""Prepare only the two separately approved CF transfer endpoints; never fit."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    review_path = "evidence/ssl-research-v1/cf-downstream-prefit-review-final.json"
    review = json.loads((ROOT / review_path).read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_DOWNSTREAM_PREFIT" or review.get("allowed_methods") != ["cf_jepa"]:
        raise ValueError("Actual separate CF downstream approval required")
    jobs = []
    for mode in ("frozen_readout", "full_finetune"):
        name = f"cf_jepa_{mode}_seed7_h96"
        jobs.append({
            "id": name, "entrypoint": "tools/execute_native_downstream_job.py",
            "args": ["--config", f"configs/native_downstream_{name}_v1.json",
                     "--review", review_path, "--output", f"outputs/native_acoustic_ssl_v1/{name}",
                     "--receipt", f"evidence/ssl-research-v1/{name}-attempt-01",
                     "--encoder", "outputs/native_acoustic_ssl_v1/cf_jepa_seed7_h96_deterministic/selected_encoder.pt",
                     "--ancestor-review", "evidence/ssl-research-v1/prefit-review-final-03.json",
                     "--ancestor-config", "configs/native_ssl_cf_jepa_seed7_h96_v1.json"],
            "method": "cf_jepa", "mode": mode,
            "report": f"outputs/native_acoustic_ssl_v1/{name}/run.json"})
    destination = ROOT / "orchestration/native_cf_downstream_approved_queue_v1.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"kind": "reviewed_native_serial_queue_v1", "jobs": jobs,
                   "authorization": "Existing separate89-binding CF downstream approval",
                   "execution": "After existing scientific queue exits; required wrappers only",
                   "final_test_access": "NOT_AUTHORIZED", "app_release_work": "FROZEN"}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "jobs": 2, "path": str(destination)}))


if __name__ == "__main__":
    main()
