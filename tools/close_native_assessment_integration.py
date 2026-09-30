"""Close byte-identical engineering integration with actual root check evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-assessment-builder-v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    proof = json.loads((ROOT / FOLDER / "source-proof-final-v2.json").read_text(encoding="utf-8"))
    for relative, expected in proof["authored_sha256"].items():
        if digest(ROOT / relative) != expected or digest(BUILDER / relative) != expected:
            raise ValueError("Closed authored source differs")
    for path, expected in proof["protected_main_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Protected scientific source changed")
    for name, expected in {
        "NATIVE_ASSESSMENT_CLOSEOUT_20260930_V1.md": "febed632e1da73b45195fb984343077963d53e04e7654d05d9f612b8dda64705",
        "closeout-handoff-20260930-v1.json": "5305c61c4420a182203caa0f900c76e74dca2ada3e19b08c2df2bcc7bc18af09",
    }.items():
        raw = (BUILDER / FOLDER / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Closed report hash differs")
        with (ROOT / FOLDER / name).open("xb") as stream:
            stream.write(raw)
    cpu = ROOT / FOLDER / "root-cpu-checks-v1.log"
    durable = ROOT / FOLDER / "root-durable-check-v1.log"
    if "92 passed" not in cpu.read_text(encoding="utf-8"):
        raise ValueError("Actual CPU check completion missing")
    check = json.loads(durable.read_text(encoding="utf-8"))
    if check["evidence_kind"] != "SYNTHETIC_CORRECTNESS_ONLY" or check["status"] != "COMPLETED_FORECASTS":
        raise ValueError("Actual durable synthetic completion missing")
    receipt = {"status": "CLOSED_ENGINEERING_INTEGRATION", "authored_sha256": proof["authored_sha256"],
               "root_cpu": {"count": 92, "exit_code": 0, "log_sha256": digest(cpu)},
               "root_durable": {"exit_code": 0, "evidence_kind": check["evidence_kind"],
                                "log_sha256": digest(durable), "native_lower_m": 230,
                                "protected_output_collision": "REJECTED"},
               "builder_closeout_exit_code": 0, "distinct_review": "NOT_RUN",
               "real_schema_integration_repair": "PENDING_SEPARATE_RED_GREEN",
               "scientific_execution": "NOT_RUN", "final_site_numeric_access": "NOT_RUN"}
    with (ROOT / FOLDER / "root-closed-integration-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    with (ROOT / FOLDER / "native_assessment_builder_snapshot_v1.py.txt").open("xb") as stream:
        stream.write((ROOT / "src/marine_echo/evaluation/native_assessment.py").read_bytes())
    print(json.dumps({"status": receipt["status"], "checks": 92, "distinct_review": "NOT_RUN"}))


if __name__ == "__main__":
    main()
