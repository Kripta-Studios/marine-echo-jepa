"""Adapt immutable virtual policy checks to the additive operational executor."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    original_path = ROOT / "tests/unit/test_native_band_replication_execution.py"
    raw = original_path.read_bytes()
    text = raw.decode("utf-8")
    old_fixture = """@pytest.fixture
def case(monkeypatch):
    return support.policy_case(job, generator, monkeypatch, BUILDER, MAIN)
"""
    replacement = """LEGACY_SPEC = importlib.util.spec_from_file_location(
    "legacy_policy_fixture_job", BUILDER / "tools/execute_native_band_replication_job.py"
)
legacy_job = importlib.util.module_from_spec(LEGACY_SPEC)
sys.modules[LEGACY_SPEC.name] = legacy_job
LEGACY_SPEC.loader.exec_module(legacy_job)


@pytest.fixture
def case(monkeypatch):
    c = support.policy_case(legacy_job, generator, monkeypatch, BUILDER, MAIN)
    required = job.source_paths(MAIN)
    for path in required:
        c.fs.files[c.root / path.relative_to(MAIN)] = path.read_bytes()
    original_seal = c.seal

    def seal():
        original_seal()
        c.review['bindings'].update({str(c.root / path.relative_to(MAIN)):
                                    job.digest(c.root / path.relative_to(MAIN))
                                    for path in required})
        c.fs.files[c.request.review] = support.encode(c.review)

    c.seal = seal
    c.seal()
    return c
"""
    if text.count(old_fixture) != 1:
        raise ValueError("Exact original private fixture required")
    text = text.replace(old_fixture, replacement)
    # Preserve scientific artifact names and config factory; change only routing expectations.
    text = text.replace('"band_job_under_test", BUILDER / "tools/execute_native_band_replication_job.py"',
                        '"band_operational_job_under_test", BUILDER / "tools/execute_native_band_operational_job.py"')
    text = text.replace('"marine_echo.training.native_band_replication_ssl"', '"marine_echo.training.native_band_operational_ssl"')
    text = text.replace('"marine_echo.training.native_band_replication_downstream"', '"marine_echo.training.native_band_operational_downstream"')
    # Tamper/missing-binding cases now target the actual operational wrapper.
    text = text.replace('        "tools/execute_native_band_replication_job.py",', '        "tools/execute_native_band_operational_job.py",')
    destination = ROOT / "tests/unit/test_native_band_operational_execution.py"
    with destination.open("xb") as stream:
        stream.write(text.encode("utf-8"))
    receipt = {"status": "SYNTHETIC_POLICY_CHECK_ADAPTATION_NOT_SCIENTIFIC_APPROVAL",
               "original_test_sha256": hashlib.sha256(raw).hexdigest(),
               "new_test_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
               "adaptation": "Same virtual ledger/command/resource guards; explicit current operational source closure and route expectations",
               "original_test_modified": False}
    with (ROOT / "evidence/ssl-research-v1/band-operational-policy-test-transport-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
