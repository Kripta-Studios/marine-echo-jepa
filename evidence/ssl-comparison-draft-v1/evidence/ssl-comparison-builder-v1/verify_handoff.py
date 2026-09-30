"""Verify handoff bytes/JSON/logs only; this is not an independent review."""

import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
BUILDER = EVIDENCE.parents[1]

if __name__ == "__main__":
    provenance = json.loads((EVIDENCE / "final-source-provenance.json").read_text(encoding="utf-8"))
    checks = json.loads((EVIDENCE / "executed-checks.json").read_text(encoding="utf-8"))
    for field in ("required_review_source_bindings", "authored_sha256"):
        for filename, expected in provenance[field].items():
            actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            assert actual == expected, f"Changed source bytes: {filename}"
    final = next(check for check in checks["checks"] if check["stage"] == "final_delivery_green")
    assert final["exit_code"] == 0 and final["passed"] == 60
    assert "60 passed in 25.25s" in (EVIDENCE / final["log"]).read_text(encoding="utf-8")
    report = (EVIDENCE / "NATIVE_COMPARISON_IMPLEMENTATION_V1.md").read_text(encoding="utf-8")
    source = BUILDER / "src/marine_echo/evaluation/native_comparison.py"
    tests = BUILDER / "tests/unit/test_native_comparison.py"
    for path in (source, tests):
        assert hashlib.sha256(path.read_bytes()).hexdigest() in report
    files = [
        source,
        tests,
        *sorted(
            p for p in EVIDENCE.iterdir() if p.is_file() and p.name != "handoff-verification.json"
        ),
    ]
    print(
        json.dumps(
            {
                "operation": "HANDOFF_BYTES_VERIFICATION_NOT_INDEPENDENT_REVIEW",
                "status": "VERIFIED",
                "files_sha256": {
                    str(p.relative_to(BUILDER)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
            },
            indent=2,
        )
    )
