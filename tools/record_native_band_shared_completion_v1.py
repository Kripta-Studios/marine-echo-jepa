"""Record a closed real Band parent while the serial queue owns live accounting."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "outputs/native_acoustic_ssl_v1/band_shared_ssl_seed7_h96_reviewed"
    attempt_path = ROOT / "evidence/ssl-research-v1/band-shared-ssl-attempt-01/attempt.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    resources = attempt["resources_full_attempt"]
    if (attempt.get("status") != "COMPLETED_REAL_BAND_CUDA"
            or attempt.get("budget_family") != "native_band_v1"
            or resources.get("exit_code") != 0
            or resources.get("stopped_for") is not None
            or resources.get("owned_tree_cleanup_verified") is not True
            or report.get("status") != "COMPLETED"
            or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
            or report.get("pretrain_steps") != 6000
            or report.get("readout_steps_all_probes") != 2000
            or report["config"]["method"] != "shared_ssl"
            or report["config"]["seed"] != 7
            or report.get("inference_sha256") != digest(folder / "inference.pt")):
        raise ValueError("Actual completed reviewed parent/optimizer/cleanup evidence differs")
    result = {
        "status": "VERIFIED_REAL_BAND_SHARED_SSL_COMPLETION",
        "directory": str(folder), "evidence_kind": report["evidence_kind"],
        "architecture": report["architecture"], "config": report["config"],
        "selected_pretrain_step": report["selected_pretrain_step"],
        "pretrain_steps": 6000, "readout_steps_all_probes": 2000,
        "development_pinball_db": report["selected_daily_dev_pinball"],
        "artifact_sha256": {name: digest(folder / name) for name in (
            "run.json", "inference.pt", "selected_encoder.pt", "membership.json", "predictions.npz", "scalers.json")},
        "attempt_sha256": digest(attempt_path), "resources_full_attempt": resources,
        "core_resources": report["resources"],
        "actual_wrapper_exit_witness": "Root queue session82615 chunk0d92bf child exit0",
        "queue_overall_status": "ACTIVE_OTHER_APPROVED_JOBS",
        "ledger_written_by_this_recorder": False,
        "metric_scope": "Runner DEV selection report; independent expanded numerical comparison pending",
        "representation_transfer_value": "NOT_ESTABLISHED", "test_access": "NOT_RUN",
    }
    target = ROOT / "evidence/ssl-research-v1/band-shared-ssl-completion-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "development_pinball_db": result["development_pinball_db"],
                      "full_owned_gpu_hours": resources["elapsed_full_attempt_seconds"] / 3600}))


if __name__ == "__main__":
    main()
