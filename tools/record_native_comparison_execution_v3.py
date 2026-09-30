"""Record observed comparator exit and immutable result identity."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    output = ROOT / "evidence/ssl-research-v1/development-comparison-v3.json"
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("role") != "development" or len(result.get("methods", {})) != 28:
        raise ValueError("Actual completed28-method DEV result required")
    receipt = {
        "status": "ACTUAL_EXPANDED_DEVELOPMENT_RECONSTRUCTION_CLOSED",
        "actual_cli_exit_code": 0, "exit_witness": "Root session8029 chunkb8e349 exit0",
        "result": str(output), "result_sha256": digest(output),
        "manifest_sha256": digest(ROOT / "orchestration/native_development_comparison_v3.json"),
        "review_sha256": digest(ROOT / "evidence/ssl-research-v1/development-comparison-v3-review-final.json"),
        "methods": 28, "role": "development", "fitting": False,
        "final_numeric_access": False, "independent_numerical_review": "NOT_RUN",
        "console_note": "Empty log is not independent proof of process exit; witness is coordinator tool observation",
    }
    with (ROOT / "evidence/ssl-research-v1/development-comparison-execution-receipt-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
