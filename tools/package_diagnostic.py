"""Create a new portable diagnostic payload and ZIP; never replace a published output."""
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = Path(sys.argv[1]).resolve()
demo_source = sys.argv[2] if len(sys.argv) > 2 else "release/demo-20260926"
if output.exists():
    raise FileExistsError("Choose a new package directory; published files are immutable.")
output.mkdir(parents=True)
for name in ["README_RUN.md", "REPORT.md", "DATA_CARD.md", "MODEL_CARD.md", "LIMITATIONS.md", "MARINE_DATA_REQUEST.md", "THIRD_PARTY_NOTICES.md", "requirements-app.txt", "Run-Demo.ps1", "Verify-Release.py"]:
    shutil.copy2(root / "release" / name, output / name)
for source, target in [(demo_source, "demo"), ("release/wheelhouse", "wheelhouse"), ("src", "src"), ("references/licenses", "licenses"), ("orchestration", "orchestration"), ("evidence/tests", "evidence/test_logs"), ("evidence/browser", "evidence/browser"), ("evidence/models", "evidence/models"), ("evidence/runtime", "evidence/runtime"), ("evidence/calibration", "evidence/calibration"), ("reports/active", "evidence/active"), ("data/manifests/mosaic_azfp_down_2020", "evidence/source_manifest")]:
    shutil.copytree(root / source, output / target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
locks = output / "requirements-locks"
locks.mkdir()
for name in ["pyproject.toml", "uv.lock", "web/package.json", "web/package-lock.json"]:
    shutil.copy2(root / name, locks / Path(name).name)
for name in ["react", "react-dom", "scheduler"]:
    source = root / "web/node_modules" / name / "LICENSE"
    if source.exists():
        target = output / "licenses/frontend" / (name + "-LICENSE.txt")
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
for name in ["docs/06_EXPERIMENT_PROTOCOL.md", "docs/20_DEFINITION_OF_DONE.md", "docs/adr/0003-environmental-research.md"]:
    target = output / "evidence" / Path(name).name
    shutil.copy2(root / name, target)
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip())
(output / "SOURCE_REVISION.json").write_text(json.dumps({"commit": revision, "tracked_dirty": dirty, "release_class": "ENGINEERING_DEMO_ONLY", "independent_final_approval": False}, indent=2)+"\n")
entries = []
for path in sorted(output.rglob("*")):
    if path.is_file():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append(f"{digest}  {path.relative_to(output).as_posix()}")
(output / "SHA256SUMS").write_text("\n".join(entries)+"\n", encoding="utf-8")
archive = output.with_suffix(".zip")
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(output.rglob("*")):
        if path.is_file():
            z.write(path, path.relative_to(output).as_posix())
with archive.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
manifest = {"archive":str(archive), "bytes":archive.stat().st_size, "sha256":digest, "payload_files":len(entries), "source_commit":revision, "tracked_dirty":dirty}
archive.with_suffix(".zip.sha256.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))
