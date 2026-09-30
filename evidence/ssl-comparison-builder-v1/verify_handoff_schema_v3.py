"""Verify delivered bytes and closed history; never scientific/self-review."""

import ast
import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
BUILDER = EVIDENCE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    before = json.loads((EVIDENCE / "before-schema-v3.json").read_text(encoding="utf-8"))
    for filename, expected in before["historical_evidence_sha256"].items():
        assert sha(EVIDENCE / filename) == expected, f"Closed evidence changed: {filename}"
    for identity in before["closed_source_snapshots"].values():
        assert sha(EVIDENCE / identity["snapshot"]) == identity["sha256"]
    provenance = json.loads(
        (EVIDENCE / "final-source-provenance-schema-v3.json").read_text(encoding="utf-8")
    )
    for field in ("required_review_source_bindings", "authored_sha256"):
        for filename, expected in provenance[field].items():
            assert sha(Path(filename)) == expected, f"Delivered source changed: {filename}"
    assert provenance["runtime"]["torch_imported"] is False
    assert (
        provenance["runtime"]["executing_session_id"]
        == provenance["runtime"]["authored_builder_session_id"]
    )
    source = BUILDER / "src/marine_echo/evaluation/native_comparison.py"
    tests = BUILDER / "tests/unit/test_native_comparison.py"
    report = (EVIDENCE / "NATIVE_COMPARISON_SCHEMA_CORRECTION_V3.md").read_text(encoding="utf-8")
    assert sha(source) in report and sha(tests) in report
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "RECIPE"
    }
    assert assignments["RECIPE"] == {
        "bootstrap_seed": 20260929,
        "bootstrap_replicates": 2000,
        "block_hours": 48,
        "block_days": 2,
        "floor": 18,
    }
    # Match ASCII summary bytes without decoding Windows-encoded traceback paths.
    assert (
        b"11 failed, 3 passed, 68 deselected in 10.64s"
        in (EVIDENCE / "red-schema-v3.txt").read_bytes()
    )
    assert "82 passed in 34.17s" in (EVIDENCE / "green-final-schema-v3.txt").read_text(
        encoding="utf-8"
    )
    assert "All checks passed" in (EVIDENCE / "lint-schema-v3.txt").read_text(encoding="utf-8")
    assert "3 files already formatted" in (EVIDENCE / "format-check-schema-v3.txt").read_text(
        encoding="utf-8"
    )
    protocol = MAIN / "docs/adr/0015-native-acoustic-ssl-research.md"
    assert sha(protocol) == "82deb36e88df2a6660a2389a653d077a186e5072fbb4ede11d9fbc06c9816148"
    generators = [
        MAIN / "src/marine_echo/training" / name
        for name in ("native_ssl.py", "native_downstream.py")
    ]
    files = [
        source,
        tests,
        *sorted(
            p
            for p in EVIDENCE.iterdir()
            if p.is_file() and p.name != "handoff-verification-delivery-schema-v3.json"
        ),
    ]
    print(
        json.dumps(
            {
                "operation": "SCHEMA_CORRECTED_BYTES_AND_CLOSED_HISTORY_NOT_INDEPENDENT_REVIEW",
                "status": "VERIFIED",
                "scientific_assessment_occurred": False,
                "preserved_preceding_evidence_files": len(before["historical_evidence_sha256"]),
                "protocol": {"path": str(protocol), "sha256": sha(protocol)},
                "read_only_generator_source_sha256": {str(p): sha(p) for p in generators},
                "files_sha256": {str(p.relative_to(BUILDER)): sha(p) for p in files},
            },
            indent=2,
        )
    )
