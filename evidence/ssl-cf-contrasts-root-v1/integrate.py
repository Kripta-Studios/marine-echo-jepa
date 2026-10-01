"""Integrate only hash-closed new builder files; preserve originals and receipts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
DELIVERY = BUILDER / "evidence/ssl-cf-contrasts-builder-v1"
EXPECTED = {
    "src/marine_echo/evaluation/native_cf_matched_contrasts_v1.py",
    "tests/unit/test_native_cf_matched_contrasts_v1.py",
}


def main():
    handoff = json.loads((DELIVERY / "handoff-v1.json").read_text(encoding="utf-8"))
    if handoff["status"] != "CLOSED_ENGINEERING_DELIVERY" or set(handoff["authored"]) != EXPECTED:
        raise ValueError("Exact closed two-file delivery required")
    payloads = {}
    for name, digest in handoff["dependency_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("Protected ROOT dependency changed")
    for name, digest in handoff["authored"].items():
        path = ROOT / name
        if path.exists():
            raise FileExistsError("Do not overwrite ROOT authored files")
        raw = (BUILDER / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Builder authored source differs from closed receipt")
        payloads[path] = raw
    for name, digest in handoff["evidence"].items():
        raw = (DELIVERY / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Builder evidence changed")
        payloads[ROOT / "evidence/ssl-cf-contrasts-builder-v1" / name] = raw
    payloads[ROOT / "evidence/ssl-cf-contrasts-builder-v1/handoff-v1.json"] = (
        DELIVERY / "handoff-v1.json"
    ).read_bytes()
    for path, raw in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    snapshot = Path(__file__).parent / "builder-module-original-v1.txt"
    snapshot.write_bytes(
        (ROOT / "src/marine_echo/evaluation/native_cf_matched_contrasts_v1.py").read_bytes()
    )
    receipt = {
        "status": "CLOSED_BUILDER_SOURCE_INTEGRATED",
        "source_only": True,
        "authored": handoff["authored"],
        "copied_files": len(payloads),
        "protected_dependencies": len(handoff["dependency_hashes"]),
        "independent_approval": False,
        "real_numeric_access": False,
    }
    with (Path(__file__).parent / "integration.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print("Integrated two exact hash-closed sources and preserved builder evidence.")


if __name__ == "__main__":
    main()
