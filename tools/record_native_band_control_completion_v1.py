"""Close an actual finished Band SSL control without writing the live ledger."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("masked_ssl", "permuted_ssl", "random_frozen"))
    args = parser.parse_args()
    folder = ROOT / "outputs/native_acoustic_ssl_v1" / f"band_{args.method}_seed7_h96_reviewed"
    receipt = ROOT / "evidence/ssl-research-v1" / f"band-{args.method.replace('_', '-')}-attempt-01" / "attempt.json"
    attempt = json.loads(receipt.read_text(encoding="utf-8"))
    report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    resources = attempt["resources_full_attempt"]
    expected_encoder_updates = 0 if args.method == "random_frozen" else 6000
    expected_readout_updates = 500 if args.method == "random_frozen" else 2000
    if (attempt.get("status") != "COMPLETED_REAL_BAND_CUDA"
            or attempt.get("budget_family") != "native_band_v1"
            or resources.get("exit_code") != 0 or resources.get("stopped_for") is not None
            or resources.get("owned_tree_cleanup_verified") is not True
            or report.get("status") != "COMPLETED"
            or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
            or report["config"]["method"] != args.method or report["config"]["seed"] != 7
            or report.get("pretrain_steps") != expected_encoder_updates
            or report.get("readout_steps_all_probes") != expected_readout_updates
            or report.get("inference_sha256") != digest(folder / "inference.pt")):
        raise ValueError("Actual completed control update/identity/cleanup evidence differs")
    result = {
        "status": "VERIFIED_REAL_BAND_CONTROL_COMPLETION", "method": args.method,
        "directory": str(folder), "evidence_kind": report["evidence_kind"],
        "architecture": report["architecture"], "config": report["config"],
        "selected_pretrain_step": report["selected_pretrain_step"],
        "pretrain_steps": expected_encoder_updates, "readout_steps_all_probes": expected_readout_updates,
        "development_pinball_db": report["selected_daily_dev_pinball"],
        "artifact_sha256": {name: digest(folder / name) for name in (
            "run.json", "inference.pt", "selected_encoder.pt", "membership.json", "predictions.npz", "scalers.json")},
        "attempt_sha256": digest(receipt), "resources_full_attempt": resources,
        "metric_scope": "Runner DEV selection; independent numerical reconstruction pending",
        "representation_transfer_value": "NOT_ESTABLISHED", "test_access": "NOT_RUN",
        "ledger_written": False,
    }
    target = ROOT / "evidence/ssl-research-v1" / f"band-{args.method.replace('_', '-')}-completion-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "method": args.method,
                      "development_pinball_db": result["development_pinball_db"],
                      "full_owned_gpu_hours": resources["elapsed_full_attempt_seconds"] / 3600}))


if __name__ == "__main__":
    main()
