"""Read-only verification of the bounded builder handoff, not scientific review."""

import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
BUILDER = EVIDENCE.parents[1]

if __name__ == "__main__":
    provenance = json.loads((EVIDENCE / "final-source-provenance.json").read_text(encoding="utf-8"))
    checks = json.loads((EVIDENCE / "executed-checks.json").read_text(encoding="utf-8"))
    for field in ("sha256", "authored_sha256"):
        for filename, expected in provenance[field].items():
            actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            assert actual == expected, f"Source changed after verification: {filename}"
    final = next(
        check for check in checks["checks"] if check["stage"] == "final_green_default_dimensions"
    )
    log = (EVIDENCE / final["log"]).read_text(encoding="utf-8")
    assert final["exit_code"] == 0 and final["passed"] == 81
    assert "81 passed in 22.90s" in log
    report = (EVIDENCE / "NATIVE_ENCODER_IMPLEMENTATION_V1.md").read_text(encoding="utf-8")
    for filename in (
        "src/marine_echo/inference/native_encoder.py",
        "tests/unit/test_native_encoder.py",
    ):
        assert hashlib.sha256((BUILDER / filename).read_bytes()).hexdigest() in report
    deliverables = [
        BUILDER / "src/marine_echo/inference/native_encoder.py",
        BUILDER / "tests/unit/test_native_encoder.py",
    ]
    deliverables += sorted(
        p for p in EVIDENCE.iterdir() if p.is_file() and p.name != "handoff-verification.json"
    )
    print(
        json.dumps(
            {
                "operation": "HANDOFF_BYTES_VERIFICATION_NOT_INDEPENDENT_REVIEW",
                "status": "VERIFIED",
                "files_sha256": {
                    str(p.relative_to(BUILDER)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in deliverables
                },
            },
            indent=2,
        )
    )
