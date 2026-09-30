"""Recover exact recorded local source bytes from Git; never execute recovered code."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "tools/execute_native_ssl_job.py"
EXPECTED = "dd688444fa48b95e40ccb8b29b260550cc998b805273aa196f35909ff0175f9d"


def main():
    folder = ROOT / "evidence/ssl-research-v1/ancestor-source"
    source = folder / "execute_native_ssl_job_shared_screen_v1.py"
    receipt = folder / "execute_native_ssl_job_shared_screen_v1.manifest.json"
    witness = folder / "execute_native_ssl_job_shared_screen_v1.git-recovery.json"
    if any(path.exists() for path in (source, receipt, witness)):
        raise FileExistsError("Preserve every recovered source archive")
    revisions = subprocess.check_output(["git", "log", "--format=%H", "--", RELATIVE], cwd=ROOT, text=True).splitlines()
    recovered = None
    for revision in revisions:
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("Exact local Git revision required")
        response = subprocess.run(["git", "show", revision + ":" + RELATIVE], cwd=ROOT, capture_output=True, check=False)
        if response.returncode != 0:
            continue
        for label, candidate in (("EXACT_GIT_BLOB", response.stdout),
                                 ("RESTORED_CRLF_ONLY", response.stdout.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))):
            if hashlib.sha256(candidate).hexdigest() == EXPECTED:
                recovered = (candidate, revision, label)
                break
        if recovered is not None:
            break
    if recovered is None:
        raise ValueError("No exact recorded source bytes recovered; do not invent provenance")
    raw, revision, transformation = recovered
    with source.open("xb") as stream:
        stream.write(raw)
    record = {"original_path": str(ROOT / RELATIVE), "path": str(source), "sha256": EXPECTED,
              "methods": ["shared_ssl"], "status": "ORIGINAL_SOURCE_PRESERVED_NOT_COMPATIBILITY_APPROVAL"}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    with witness.open("x", encoding="utf-8") as stream:
        json.dump({"status": "EXACT_RECORDED_SOURCE_SHA256_RECOVERED_FROM_LOCAL_GIT", "revision": revision,
                   "git_path": RELATIVE, "transformation": transformation, "sha256": EXPECTED,
                   "source_executed": False, "historical_current_file_modified": False,
                   "compatibility_or_scientific_approval": False}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "EXACT_RECORDED_SOURCE_SHA256_RECOVERED_FROM_LOCAL_GIT", "revision": revision,
                      "transformation": transformation, "sha256": EXPECTED}))


if __name__ == "__main__":
    main()
