"""Record actual root checks and physical saved-artifact replay without scientific approval."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder = ROOT / "evidence/ssl-replication-assessment-prefix-builder-v4"
    handoff = json.loads((folder / "handoff-v4.json").read_bytes())
    log = folder / "root-all-checks-v4.log"
    text = log.read_text(encoding="utf-8")
    if "391 passed in 674.98s" not in text or " skipped" in text or " failed" in text:
        raise ValueError("Actual full root checks without skips required")
    durable_log = folder / "root-durable-prefix-v4.log"
    durable = json.loads(durable_log.read_text(encoding="utf-8").strip())
    if (durable.get("evidence_kind") != "SYNTHETIC_CORRECTNESS_ONLY"
            or durable.get("durable_replay") != "PASSED" or durable.get("gpu_execution") is not False
            or durable.get("synthetic_optimizer_updates") != 4):
        raise ValueError("Actual physical synthetic CPU fit/replay required")
    physical = Path(durable["output"])
    if not physical.is_relative_to(folder) or not (physical / "inference.pt").is_file():
        raise ValueError("Durable protected output required")
    for name, sha in handoff["authored"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Authored root source changed after actual checks")
    for name, sha in handoff["source_closure"].items():
        if digest(name) != sha:
            raise ValueError("Immutable dependency source changed")
    receipt = {"status": "ROOT_391_CHECKS_AND_PHYSICAL_SYNTHETIC_PREFIX_REPLAY_PASSED",
               "checks": {"session_id": 3982, "actual_exit_code": 0, "actual_exit_chunk_id": "7969c2",
                          "passed": 391, "skipped": 0, "log_sha256": digest(log)},
               "durable": {"session_id": 52146, "actual_exit_code": 0, "actual_exit_chunk_id": "893d80",
                           "log_sha256": digest(durable_log), "completion": durable,
                           "inference_sha256": digest(physical / "inference.pt")},
               "current_authored_sha256": handoff["authored"], "source_closure": handoff["source_closure"],
               "owned_process_lifecycle": "NOT_RUN_PENDING_GPU_IDLE",
               "distinct_software_review": "NOT_RUN", "scientific_prefit": "NOT_RUN",
               "scientific_fitting_or_final_numeric_access": "NOT_RUN"}
    with (folder / "root-checks-closeout-v4.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "passed": 391}))


if __name__ == "__main__":
    main()
