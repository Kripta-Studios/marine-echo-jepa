"""Verify corrected bytes and preservation; not a scientific/self-review."""

import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
BUILDER = EVIDENCE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    before = json.loads((EVIDENCE / "before-protocol-v2.json").read_text(encoding="utf-8"))
    # Redirection precreated the new snapshot output. Neither entry is closed V1.
    nonhistorical = {"before-protocol-v2.json", "snapshot_before_protocol_v2.py"}
    historical = {
        name: value
        for name, value in before["historical_evidence_sha256"].items()
        if name not in nonhistorical
    }
    for name, expected in historical.items():
        assert sha(EVIDENCE / name) == expected, f"Historical evidence changed: {name}"
    for filename, expected in before["original_source_sha256"].items():
        assert sha(EVIDENCE / (Path(filename).name + ".original-v1.txt")) == expected
    assert sha(Path(before["protocol"]["path"])) == before["protocol"]["sha256"]
    provenance = json.loads(
        (EVIDENCE / "final-source-provenance-protocol-v2.json").read_text(encoding="utf-8")
    )
    for field in ("required_review_source_bindings", "authored_sha256"):
        for filename, expected in provenance[field].items():
            assert sha(Path(filename)) == expected, f"Current bytes changed: {filename}"
    checks = json.loads((EVIDENCE / "executed-checks-protocol-v2.json").read_text(encoding="utf-8"))
    final = next(check for check in checks["checks"] if check["stage"] == "final_green")
    assert final["exit_code"] == 0 and final["passed"] == 68
    assert "68 passed in 27.90s" in (EVIDENCE / final["log"]).read_text(encoding="utf-8")
    report = (EVIDENCE / "NATIVE_COMPARISON_PROTOCOL_CORRECTION_V2.md").read_text(encoding="utf-8")
    source = BUILDER / "src/marine_echo/evaluation/native_comparison.py"
    tests = BUILDER / "tests/unit/test_native_comparison.py"
    assert sha(source) in report and sha(tests) in report
    assert "1729" not in source.read_text(encoding="utf-8")
    files = [
        source,
        tests,
        *sorted(
            p
            for p in EVIDENCE.iterdir()
            if p.is_file() and p.name != "handoff-verification-protocol-v2.json"
        ),
    ]
    print(
        json.dumps(
            {
                "operation": "CORRECTED_HANDOFF_BYTES_AND_HISTORY_PRESERVATION_NOT_INDEPENDENT_REVIEW",
                "status": "VERIFIED",
                "preserved_closed_v1_evidence_files": len(historical),
                "source_clock_basis": "two_source_calendar_days_48_nominal_hourly_intervals_not_verified_UTC",
                "files_sha256": {str(p.relative_to(BUILDER)): sha(p) for p in files},
            },
            indent=2,
        )
    )
