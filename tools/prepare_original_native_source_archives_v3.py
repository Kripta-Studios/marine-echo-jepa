"""Preserve changed original Python source bindings from exact local Git bytes."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    matrix = json.loads((ROOT / "orchestration/native_development_comparison_v3.json").read_bytes())
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_bytes())
    folders = [Path(path).parent for name, path in matrix["methods"].items()
               if name not in {"persistence", "seasonal24", "lightgbm", "chronos2_zero_shot"}]
    base = ROOT / "outputs/native_acoustic_ssl_v1"
    folders.extend(base / ("band_" + method + "_seed7_h96_reviewed")
                   for method in ("shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen", "direct_end_to_end"))
    folders.extend(base / ("band_" + method + "_" + mode + "_seed7_h96") for method, mode in
                   (("shared_ssl", "frozen_readout"), ("shared_ssl", "full_finetune"),
                    ("masked_ssl", "frozen_readout"), ("masked_ssl", "full_finetune"),
                    ("permuted_ssl", "frozen_readout")))
    folders.append(base / "band_shared_ssl_seed13_h96_replication_v2")
    if len(folders) != 35:
        raise ValueError("Explicit35 currently completed neural runs required")
    changed, cache = {}, {}
    for folder in folders:
        run = json.loads((folder / "run.json").read_bytes())
        method = run.get("core_config", run["config"])["method"]
        reviews = {(ROOT / record["review"]).resolve() for record in ledger["runs"]
                   if record.get("output") and record.get("review") and (ROOT / record["output"]).resolve() == folder}
        reviews = [path for path in reviews if digest(path) == run["review_sha256"]]
        if len(reviews) != 1:
            raise ValueError("Exact original review must resolve")
        review = json.loads(reviews[0].read_bytes())
        for group in (run["bindings"], review["bindings"]):
            for name, recorded in group.items():
                path = Path(name)
                if name not in cache:
                    cache[name] = digest(path)
                if cache[name] == recorded:
                    continue
                if (path.suffix != ".py" or path.is_symlink()
                        or not (path.is_relative_to(ROOT / "src/marine_echo") or path.is_relative_to(ROOT / "tools"))):
                    raise ValueError("Changed non-source scientific input must not be archived as compatible")
                changed.setdefault((name, recorded), set()).add(method)
    folder = ROOT / "evidence/ssl-research-v1/original-source-archives-v3"
    if folder.exists():
        raise FileExistsError("Preserve every previous source archive")
    payloads, catalogs, witnesses = {}, [], []
    for (name, recorded), methods in sorted(changed.items()):
        path = Path(name)
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        revisions = subprocess.check_output(["git", "log", "--format=%H", "--", relative], cwd=ROOT, text=True).splitlines()
        recovered = None
        for revision in revisions:
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("Exact local Git revision required")
            response = subprocess.run(["git", "show", revision + ":" + relative], cwd=ROOT, capture_output=True, check=False)
            if response.returncode != 0:
                continue
            candidates = [("EXACT_GIT_BLOB", response.stdout),
                          ("RESTORED_CRLF_ONLY", response.stdout.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))]
            for transformation, raw in candidates:
                if hashlib.sha256(raw).hexdigest() == recorded:
                    recovered = raw, revision, transformation
                    break
            if recovered:
                break
        if recovered is None:
            raise ValueError(f"Exact original source cannot be recovered: {relative}, SHA256{recorded}")
        raw, revision, transformation = recovered
        source_name = path.stem + "_" + recorded[:16] + ".py"
        catalog_name = source_name + ".manifest.json"
        record = {"original_path": name, "path": str(folder / source_name), "sha256": recorded,
                  "methods": sorted(methods), "status": "ORIGINAL_SOURCE_PRESERVED_NOT_COMPATIBILITY_APPROVAL"}
        payloads[source_name] = raw
        payloads[catalog_name] = (json.dumps(record, indent=2) + "\n").encode()
        catalogs.append(str(folder / catalog_name))
        witnesses.append({"original_path": name, "sha256": recorded, "revision": revision,
                          "transformation": transformation, "methods": sorted(methods), "source_executed": False})
    index = {"status": "ALL_CHANGED_ORIGINAL_PYTHON_SOURCE_BYTES_RECOVERED_NOT_COMPATIBILITY_APPROVAL",
             "runs_inspected": 35, "catalogs": catalogs, "recovery": witnesses, "source_executed": False,
             "historical_current_files_modified": False, "scientific_approval": False}
    payloads["index.json"] = (json.dumps(index, indent=2) + "\n").encode()
    folder.mkdir()
    for name, raw in payloads.items():
        with (folder / name).open("xb") as stream:
            stream.write(raw)
    print(json.dumps({"status": index["status"], "original_source_versions": len(catalogs), "runs_inspected": 35}))


if __name__ == "__main__":
    main()
