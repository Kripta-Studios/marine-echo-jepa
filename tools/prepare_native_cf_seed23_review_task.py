"""Write an actual-parent seed23 review request without approving or fitting it."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parent = ROOT / "outputs/native_acoustic_ssl_v1/cf_jepa_seed23_h96_replication/run.json"
    record = json.loads(parent.read_text(encoding="utf-8"))
    if record.get("status") != "COMPLETED" or record.get("config", {}).get("seed") != 23:
        raise ValueError("Actual completed seed23 parent required")
    task = (ROOT / "orchestration/ssl_vnext_cf_seed13_downstream_prefit_review.txt").read_text(
        encoding="utf-8"
    )
    first = task.index("Task: approve")
    body = task[first:]
    body = body.replace("seed13", "seed23").replace("seed=13", "seed=23").replace("[13]", "[23]")
    body = body.replace(
        "Parent runSHA6f1d512ad39bfa33abb24eaabeb10044b8ff17318b219bef495e0ba5ad75b171, inferenceSHAfa7250e1c7f988431b9032ed65eceea8021b8de96d037f5d935203f1c4bb8df3, selected1500, short-probe DEV0.9251928324563993. This is worse than seed7 short0.7457266;",
        f"Parent runSHA{hashlib.sha256(parent.read_bytes()).hexdigest()}, inferenceSHA{record['inference_sha256']}, selected{record['selected_pretrain_step']}, short-probe DEV{record['selected_daily_dev_pinball']}. Preserve the completed seed7/13/23 comparison;",
    )
    body = body.replace("seed23/23configs", "seed13/23configs").replace(
        "must support13", "must support23"
    )
    body = body.replace(
        "No prospective parent23 approval or parent7 reuse.",
        "No prospective parent approval or parent7 reuse.",
    )
    body = body.replace("NoCF23, band,", "No other CF seeds, band,")
    introduction = (
        "Continue SAME verified distinct reviewer GPT6.1Sol/high read-only/never only after its current task exits and root resumes. "
        "CF seeds13/23 serial queue closed exit0; both actual parents exist. Inspect metadata/hashes and safe tensor metadata only; "
        "no public input/prediction numerical decoding, fitting/GPU, source edits, installation, permission changes, credentials, "
        "denied-operation retries, process-tree operations or live ledger/lock writes. "
        "Approve only actual seed23 strong endpoints. Preserve all three seeds; no best-seed selection.\n\n"
    )
    path = ROOT / "orchestration/ssl_vnext_cf_seed23_downstream_prefit_review_v2.txt"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            introduction
            + body
            + "\n\nActual parent completion metadata:\n"
            + json.dumps(
                {
                    k: record[k]
                    for k in (
                        "status",
                        "config",
                        "selected_pretrain_step",
                        "selected_daily_dev_pinball",
                        "inference_sha256",
                    )
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps({"status": "ACTUAL_PARENT_REVIEW_TASK_PREPARED_NOT_APPROVED", "seed": 23}))


if __name__ == "__main__":
    main()
