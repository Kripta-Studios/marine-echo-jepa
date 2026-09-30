"""Preserve a native failed attempt and admit one fresh exact-recipe retry."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    recovery = ROOT / "evidence/ssl-research-v1/cf-native-runtime-recovery-probe-01/completion.json"
    if json.loads(recovery.read_text(encoding="utf-8"))["status"] != "COMPLETED_SYNTHETIC_GPU_PREFLIGHT":
        raise ValueError("Actual current CUDA smoke required")
    if (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists():
        raise ValueError("Unknown owner remains")
    original = json.loads((ROOT / "orchestration/native_cf_seed13_strong_approved_queue_v1.json").read_text(encoding="utf-8"))
    failed = json.loads((ROOT / "evidence/ssl-research-v1/cf-seed13-strong-queue-attempt-01/queue.json").read_text(encoding="utf-8"))
    if failed.get("status") != "STOPPED_AFTER_FAILURE" or len(failed["jobs"]) != 1:
        raise ValueError("Exact native failed attempt must be preserved")
    frozen = original["jobs"][0]
    old_id = frozen["id"]
    new_id = old_id + "_runtime_retry01"
    frozen["id"] = new_id
    for flag, value in (("--output", f"outputs/native_acoustic_ssl_v1/{new_id}"),
                        ("--receipt", f"evidence/ssl-research-v1/{new_id}-attempt-01")):
        frozen["args"][frozen["args"].index(flag) + 1] = value
    frozen["report"] = f"outputs/native_acoustic_ssl_v1/{new_id}/run.json"
    original.update(authorization="Existing exact93-binding CFseed13 approval; unchanged config/data/source/parent/schedule. One fresh-output retry after native fatal exit and current bounded CUDA smoke.",
                    native_failure_cause="UNKNOWN; no scientific score/checkpoint was produced",
                    failed_attempt_preserved=old_id, retry_limit=1, final_test_access="NOT_AUTHORIZED")
    destination = ROOT / "orchestration/native_cf_seed13_strong_runtime_retry_queue_v1.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(original, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PREPARED_EXACT_RECIPE_RUNTIME_RETRY", "jobs": len(original["jobs"]), "failed_output_preserved": True}))


if __name__ == "__main__":
    main()
