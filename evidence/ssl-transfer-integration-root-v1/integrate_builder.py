"""Copy only the six authorized new files after exact source/dependency checks."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
SOURCE = BUILDER / "evidence/ssl-transfer-integration-builder-v1"
DEST = ROOT / "evidence/ssl-transfer-integration-builder-v1"
EXPECTED = {
    "src/marine_echo/training/native_prefix_matched_transfer_v4.py",
    "src/marine_echo/evaluation/native_prefix_suffix_assessment_v1.py",
    "tests/unit/test_native_prefix_matched_transfer_v4.py",
    "tests/integration/test_native_prefix_matched_transfer_v4.py",
    "tests/unit/test_native_prefix_suffix_assessment_v1.py",
    "tests/integration/test_native_prefix_suffix_assessment_v1.py",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    proof = json.loads((SOURCE / "closed-source-proof-v2.json").read_bytes())
    authored = proof["authored"]
    protected = proof["dependencies"]
    if set(authored) != EXPECTED:
        raise ValueError("Exact six-file builder contract required")
    for name, digest in protected.items():
        path = Path(name).resolve()
        if not path.is_relative_to(ROOT) or sha(path.read_bytes()) != digest:
            raise ValueError("Protected dependency changed: " + name)
    contents = {name: (BUILDER / name).read_bytes() for name in sorted(authored)}
    if any(sha(raw) != authored[name] or (ROOT / name).exists() for name, raw in contents.items()):
        raise ValueError("Changed source or occupied destination; preserve all existing work")
    evidence = {p.name: p.read_bytes() for p in SOURCE.iterdir()
                if p.is_file() and p.suffix in (".py", ".json", ".log", ".txt", ".patch")}
    if DEST.exists():
        raise FileExistsError("Fresh builder evidence destination required")
    for name, raw in contents.items():
        with (ROOT / name).open("xb") as stream:
            stream.write(raw)
    DEST.mkdir()
    for name, raw in evidence.items():
        with (DEST / name).open("xb") as stream:
            stream.write(raw)
    for name, digest in protected.items():
        if sha(Path(name).read_bytes()) != digest:
            raise ValueError("Protected dependency changed during copy")
    receipt = {"status": "EXACT_NEW_SOURCE_SNAPSHOT_INTEGRATED", "sources": authored,
               "protected_dependencies_unchanged": len(protected),
               "evidence": {name: sha(raw) for name, raw in evidence.items()},
               "independent_review": False, "real_fit_or_reserved_access": "NOT_RUN"}
    with (Path(__file__).parent / "integration.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "new_sources": len(contents), "protected": len(protected)}))


if __name__ == "__main__":
    main()
