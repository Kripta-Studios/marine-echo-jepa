"""Freeze two strong endpoints only after their actual parent-bound approval."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(13, 23), required=True)
    args = parser.parse_args()
    seed = args.seed
    review_path = f"evidence/ssl-research-v1/cf-seed{seed}-downstream-prefit-review-final.json"
    review = json.loads((ROOT / review_path).read_text(encoding="utf-8"))
    if (
        review.get("status") != "APPROVED_DOWNSTREAM_PREFIT"
        or review.get("allowed_methods") != ["cf_jepa"]
        or review.get("allowed_seeds") != [seed]
        or set(review.get("allowed_modes", [])) != {"frozen_readout", "full_finetune"}
        or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
    ):
        raise ValueError("Require exact actual-parent distinct strong-endpoint approval")
    for path, expected in review["bindings"].items():
        with Path(path).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Approved binding changed: {path}")
    parent = f"outputs/native_acoustic_ssl_v1/cf_jepa_seed{seed}_h96_replication"
    actual = json.loads((ROOT / parent / "run.json").read_text(encoding="utf-8"))
    if actual.get("status") != "COMPLETED" or actual.get("config", {}).get("seed") != seed:
        raise ValueError("Actual completed matching parent required")
    jobs = []
    for mode in ("frozen_readout", "full_finetune"):
        name = f"cf_jepa_{mode}_seed{seed}_h96"
        jobs.append(
            {
                "id": name,
                "entrypoint": "tools/execute_native_downstream_job.py",
                "args": [
                    "--config",
                    f"configs/native_downstream_{name}_v1.json",
                    "--review",
                    review_path,
                    "--output",
                    f"outputs/native_acoustic_ssl_v1/{name}",
                    "--receipt",
                    f"evidence/ssl-research-v1/{name}-attempt-01",
                    "--encoder",
                    f"{parent}/selected_encoder.pt",
                    "--ancestor-review",
                    "evidence/ssl-research-v1/cf-replication-prefit-review-final.json",
                    "--ancestor-config",
                    f"configs/native_ssl_cf_jepa_seed{seed}_h96_v1.json",
                ],
                "method": "cf_jepa",
                "mode": mode,
                "report": f"outputs/native_acoustic_ssl_v1/{name}/run.json",
            }
        )
    destination = ROOT / f"orchestration/native_cf_seed{seed}_strong_approved_queue_v1.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "kind": "reviewed_native_serial_queue_v1",
                "jobs": jobs,
                "authorization": f"Exact distinct completed-parent CF seed{seed} approval",
                "execution": "Only after active scientific queue exits; mandatory owned wrappers",
                "final_test_access": "NOT_AUTHORIZED",
                "app_release_work": "FROZEN",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "seed": seed, "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
