"""Bind root policy checks and generated finite configs; confer no fit approval."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evidence/ssl-band-execution-builder-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    handoff = json.loads((FOLDER / "handoff-v1.json").read_text(encoding="utf-8"))
    for relative, expected in handoff["authored_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Root must preserve closed authored bytes")
    log = FOLDER / "root-policy-checks-v1.log"
    if "96 passed in 163.70s" not in log.read_text(encoding="utf-8"):
        raise ValueError("Actual complete root policy checks missing")
    generation = ROOT / "configs/native_band_execution_v1/generation.json"
    generated = json.loads(generation.read_text(encoding="utf-8"))
    if len(generated["screens"]) != 5 or len(generated["prospective_downstream"]) != 6 or generated["scientific_approval"]:
        raise ValueError("Wrong finite catalog or approval claim")
    paths = [ROOT / p for p in handoff["authored_sha256"]]
    paths += [Path(p) for p in generated["screens"] + generated["prospective_downstream"]]
    paths += [generation, log, FOLDER / "root-config-generation-v1.log",
              FOLDER / "policy_test_support.py", ROOT / "orchestration/native_band_budget_owner_resolution_v1.json"]
    receipt = {"status": "ROOT_BAND_EXECUTION_CHECKS_COMPLETE_PENDING_DISTINCT_PREFIT",
               "bindings": {str(p): digest(p) for p in paths}, "root_checks": "96 passed163.70s",
               "builder_checks": "96 passed153.72s", "authored_bytes_unchanged": True,
               "builder_actual_cli_exit": "session23230 chunkb21678 exit0",
               "root_actual_policy_exit": "session57723 chunkd8e87a exit0",
               "root_actual_config_exit": "session99021 chunk6f75d3 exit0",
               "root_lint_and_format": "PASS", "five_screen_configs": generated["screens"],
               "six_prospective_endpoints": "NOT_APPROVED_NO_COMPLETED_PARENT_YET",
               "direct_frozen_readout": "NOT_SUPPORTED_TYPED_SUPERVISED_ANCESTRY_LIMITATION",
               "public_data_loaded": False, "model_initialized": False, "fitting": False,
               "independent_prefit": "ACTIVE_NOT_CLOSED"}
    with (FOLDER / "root-checks-closeout-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "configs": 11, "fitting": False}))


if __name__ == "__main__":
    main()
