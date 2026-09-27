"""Package the reviewed v2 engineering research release without changing v1/r2."""

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = Path(sys.argv[1]).resolve()
research = Path(sys.argv[2]).resolve()
if output.exists() or output.with_suffix(".zip").exists():
    raise FileExistsError("Choose a new package name; releases are immutable.")
catalog = json.loads((research / "artifacts/catalog.json").read_text(encoding="utf-8"))
if catalog["release_class"] != "OFFLINE_RESEARCH_ENGINEERING_ONLY":
    raise ValueError("Expected the reviewed v2 research release class.")
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
dirty = bool(
    subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    ).strip()
)
if dirty:
    raise ValueError("Commit validated implementation before packaging.")
output.mkdir(parents=True)
for name in (
    "Run-V2-Research.ps1",
    "Verify-Release.py",
    "requirements-app.txt",
    "README_V2_RESEARCH.md",
    "THIRD_PARTY_NOTICES.md",
):
    shutil.copy2(root / "release" / name, output / name)
for source, target in (
    (research, "research"),
    (root / "release/wheelhouse", "wheelhouse"),
    (root / "src", "src"),
    (root / "outputs/raw-response-development-v1-deterministic-20260927", "provenance/run"),
    (root / "orchestration/reviews", "provenance/reviews"),
    (root / "docs/adr", "provenance/adr"),
    (root / "evidence/v2", "provenance/data-evidence"),
    (root / "evidence/v2-release/browser", "provenance/browser"),
    (root / "data/manifests", "provenance/source-manifests"),
):
    shutil.copytree(source, output / target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for name in (
    "docs/06_EXPERIMENT_PROTOCOL.md",
    "docs/20_DEFINITION_OF_DONE.md",
    "orchestration/reports/V2_RESEARCH_OUTCOME_20260927.md",
):
    destination = output / "provenance" / Path(name).name
    shutil.copy2(root / name, destination)
(output / "SOURCE_REVISION.json").write_text(
    json.dumps(
        {
            "commit": revision,
            "tracked_dirty": False,
            "release_class": catalog["release_class"],
            "independent_raw_result_review": "provenance/reviews/V2_RAW_RESPONSE_DEVELOPMENT_RESULT_20260927.json",
            "independent_final_release_review": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
entries = []
for path in sorted(output.rglob("*")):
    if path.is_file():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append(f"{digest}  {path.relative_to(output).as_posix()}")
(output / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
archive = output.with_suffix(".zip")
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as packed:
    for path in sorted(output.rglob("*")):
        if path.is_file():
            packed.write(path, path.relative_to(output).as_posix())
with archive.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
manifest = {
    "archive": str(archive),
    "bytes": archive.stat().st_size,
    "sha256": digest,
    "payload_files": len(entries),
    "source_commit": revision,
    "release_class": catalog["release_class"],
}
archive.with_suffix(".zip.sha256.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
