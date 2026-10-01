"""Record actual copied-package checks without conferring scientific approval."""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/ssl-research-v1"
OUTPUT = EVIDENCE / "native-pretrained-snapshot-closeout-v2.json"
ARCHIVE_SHA256 = "22f116a426dbba384610b0bd60e9d450660dcf50d4111a56040989dd4e2d2d48"
V1_SHA256 = "ed1ba345a6fe3f04642d5bb2d8c32ef33f67132009295a6bca38cd73d8a3994b"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve the recorded closeout")
    receipt_path = EVIDENCE / "native-pretrained-model-snapshot-v2.json"
    receipt = json.loads(receipt_path.read_bytes())
    archive = Path(receipt["archive"])
    directory = Path(receipt["directory"])
    if archive != ROOT / "outputs/native_pretrained_model_snapshot_v2.zip":
        raise ValueError("Exact local snapshot archive required")
    if directory != ROOT / "outputs/native_pretrained_model_snapshot_v2":
        raise ValueError("Exact local snapshot directory required")
    if digest(archive) != ARCHIVE_SHA256 or receipt["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Executed archive bytes changed")
    expected_files = receipt["files"]
    if len(expected_files) != 97 or len(receipt["models"]) != 7:
        raise ValueError("Fixed seven-model, 97-file package required")
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected_files):
            raise ValueError("Archive entries differ or include duplicate paths")
        for name, expected in expected_files.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Unsafe archive path")
            actual = hashlib.sha256(zipped.read(name)).hexdigest()
            if actual != expected or digest(directory / name) != expected:
                raise ValueError(f"Copied directory/archive bytes differ: {name}")
    if digest(directory / "manifest.json") != receipt["manifest_sha256"]:
        raise ValueError("Copied manifest changed")
    for source, expected in receipt["source_bindings"].items():
        if digest(source) != expected:
            raise ValueError(f"Protected packaging dependency changed: {source}")
    old_archive = ROOT / "outputs/native_pretrained_model_snapshot_v1.zip"
    if digest(old_archive) != V1_SHA256:
        raise ValueError("Preserved original snapshot changed")

    qa_dir = EVIDENCE / "native-pretrained-snapshot-isolated-cpu-v3"
    resource_path = qa_dir / "resources.json"
    completion_path = qa_dir / "completion.json"
    resources = json.loads(resource_path.read_bytes())
    completion = json.loads(completion_path.read_bytes())
    owned = resources["resources"]
    if (
        resources["status"] != "COPIED_PACKAGE_CPU_QA_PASSED"
        or owned["exit_code"] != 0
        or owned["stopped_for"] is not None
        or owned["owned_tree_cleanup_verified"] is not True
        or owned["peak_process_rss_bytes"] >= 22 * 1024**3
        or owned["elapsed_full_attempt_seconds"] >= 600
        or resources["archive_sha256"] != ARCHIVE_SHA256
        or resources["completion_sha256"] != digest(completion_path)
    ):
        raise ValueError("Successful bounded owned CPU receipt required")
    if (
        completion["status"] != "COPIED_SEVEN_MODEL_PACKAGE_ISOLATED_CPU_CORRECTNESS_PASSED"
        or completion["evidence_kind"] != "SYNTHETIC_INPUT_CORRECTNESS_ONLY_REAL_PRETRAINED_WEIGHTS"
        or completion["all_imports_from_copied_source"] is not True
        or completion["rng_unchanged"] is not True
        or completion["cuda_initialized"] is not False
        or completion["scientific_performance_assessment"] is not False
        or completion["public_numerical_corpus_decoded"] is not False
        or completion["independent_package_review"] != "NOT_RUN"
        or {row["id"] for row in completion["records"]} != {row["id"] for row in receipt["models"]}
    ):
        raise ValueError("Exact scoped seven-model correctness evidence required")
    for row in completion["records"]:
        if row["selected_and_latent_encoder_cpu_replay"] != "BITIDENTICAL":
            raise ValueError("Copied selected and latent encoders differ")
        required_guard = (
            "NOT_APPLICABLE_TO_CF_ORDINAL_ZONES" if row["id"].startswith("cf_jepa_") else "PASSED"
        )
        if row["native230_query_guard"] != required_guard:
            raise ValueError("Native query guard semantics differ")
    checks_path = ROOT / "evidence/ssl-pretrained-snapshot-builder-v2/root-contract-closed-v2.log"
    if "54 passed in 15.22s" not in checks_path.read_text(encoding="utf-8"):
        raise ValueError("Actual root contract test log required")
    bindings = {
        str(path.relative_to(ROOT)): digest(path)
        for path in (receipt_path, resource_path, completion_path, checks_path, Path(__file__))
    }
    record = {
        "status": "SEVEN_REAL_PRETRAINED_MODELS_PACKAGED_LOCAL_CORRECTNESS_VERIFIED",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "models": [row["id"] for row in receipt["models"]],
        "archive": str(archive.relative_to(ROOT)),
        "archive_sha256": ARCHIVE_SHA256,
        "archive_size_bytes": archive.stat().st_size,
        "archive_entries_verified": len(expected_files),
        "directory_and_zip_entry_sha256_identical": True,
        "protected_packaging_inputs_unchanged": True,
        "preserved_v1_archive_sha256": V1_SHA256,
        "actual_cli_exit_observations": [
            {
                "operation": "root_54_contract_checks",
                "session_id": 87453,
                "exit_code": 0,
                "chunk": "b68693",
            },
            {"operation": "production_packaging", "exit_code": 0, "chunk": "2d71f1"},
            {"operation": "owned_isolated_cpu_qa", "exit_code": 0, "chunk": "8d6dc6"},
        ],
        "owned_cpu_resources": owned,
        "evidence_kind": completion["evidence_kind"],
        "fitting": False,
        "final_site_values_accessed": False,
        "public_prefix_transfer": "NOT_RUN",
        "independent_package_review": "NOT_RUN",
        "scientific_performance_assessment": False,
        "sota": "NOT_ESTABLISHED",
        "bindings": bindings,
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": record["status"],
                "models": len(record["models"]),
                "zip_entries": len(expected_files),
            }
        )
    )


if __name__ == "__main__":
    main()
