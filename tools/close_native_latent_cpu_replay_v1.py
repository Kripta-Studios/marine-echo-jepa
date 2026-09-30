"""Preserve observed process completion separately from a result file."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    output = ROOT / "evidence/ssl-research-v1/Unicode-actual-latent-Álvaro-cpu-replay-v1"
    result = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    if (result.get("status") != "PASSED_ACTUAL_IMMUTABLE_SSL_CPU_REPLAY"
            or result.get("optimizer_updates") != 0 or result.get("role") != "development"
            or len(result.get("records", [])) != 4 or result.get("test_access") != "NOT_RUN"):
        raise ValueError("Actual four-parent context-only result required")
    for record in result["records"]:
        if (record["selected_encoder_cpu_replay"] != "BITIDENTICAL"
                or record["saved_forward_head_cpu_replay"] != "BITIDENTICAL"
                or digest(output / (record["run"] + ".pt")) != record["artifact_sha256"]):
            raise ValueError("Actual saved-artifact replay identity differs")
    receipt = {
        "status": "ACTUAL_LATENT_CPU_REPLAY_CLOSED", "actual_cli_exit_code": 0,
        "process_exit_witness": "Root session67812 chunkd9f598 exit0",
        "completion_sha256": digest(output / "completion.json"),
        "console_sha256": digest(ROOT / "evidence/ssl-research-v1/native-latent-actual-cpu-replay-v1.log"),
        "completion": str(output / "completion.json"), "device": "cpu",
        "fitting": False, "final_numeric_access": False,
        "scope": "64 original DEV225 contexts; actual Shared7 and CF7/13/23 selected encoders and saved latent heads",
        "independent_result_review": "NOT_RUN", "representation_value": "NOT_ESTABLISHED",
    }
    with (ROOT / "evidence/ssl-research-v1/native-latent-actual-cpu-replay-closeout-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
