"""Review-gated, two-source, metadata-only AEON transfer inventory runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from marine_echo.training import aeon_external_metadata as scanner

ROOT = Path(__file__).resolve().parents[3]
STUDY = "aeon_external_transfer_20260928_v1"
ROLES = ("PRIMARY_CROSS_SITE_CONTEMPORANEOUS", "SECONDARY_PRIOR_YEAR_SAME_SITE")
STAGE0_PATH = ROOT / "orchestration/reviews/AEON_EXTERNAL_STAGE0_DESIGN_REVIEW_20260928.json"
AMENDMENT_PATH = ROOT / "orchestration/reviews/AEON_EXTERNAL_STAGE0_WORDING_AMENDMENT_REVIEW_20260928.json"
REVIEW_PATH = ROOT / "orchestration/reviews/AEON_EXTERNAL_STAGE1_RUNNER_REVIEW_20260928.json"
CONTRACT_PATH = ROOT / "configs/aeon_external_transfer.json"
ADR_PATH = ROOT / "docs/adr/0012-aeon-external-conditioned-product-transfer.md"
PRODUCTION_ARCHIVE_SHA = frozenset({
    "d0c57bcf73a09ad3fad1f6be3704d25263ce2a1842080dfe1ba680b57527364b",
    "b6d8380ed986c91d565ea7d669761ff8f66da8f2b5cd3e794fbf71c040bf1d37",
})
DEFAULT_OUTPUT_DIR = ROOT / "outputs/aeon_external_transfer_20260928_v1/metadata"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"AEON review or contract is not an object: {path.name}")
    return value


def _source_entries(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = contract.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("Stage 1 requires exactly two frozen source entries.")
    found: dict[str, dict[str, Any]] = {}
    for entry in sources:
        if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
            raise TypeError("Stage 1 source entry is malformed.")
        if entry["role"] in found:
            raise ValueError("Stage 1 repeats a source role.")
        found[entry["role"]] = entry
    if set(found) != set(ROLES):
        raise ValueError("Stage 1 source roles differ from freeze.")
    return found


def _preflight(
    *, contract_path: Path, adr_path: Path, stage0_path: Path,
    amendment_path: Path, review_path: Path, output_dir: Path,
    fixture_only: bool,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    """Validate all lineage, archives and outputs before any CSV data row opens."""
    if not fixture_only and tuple(path.resolve() for path in (
        contract_path, adr_path, stage0_path, amendment_path, review_path,
    )) != tuple(path.resolve() for path in (
        CONTRACT_PATH, ADR_PATH, STAGE0_PATH, AMENDMENT_PATH, REVIEW_PATH,
    )):
        raise ValueError("Real Stage 1 paths differ from fixed reviewed paths.")
    if not fixture_only and output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("Real Stage 1 output directory differs from fixed path.")
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise ValueError("Stage 1 output directory is a file or symlink.")
    paths = (contract_path, adr_path, stage0_path, amendment_path, review_path)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Stage 1 contract, ADR or review is absent.")
    digests = {
        "config_sha256": _sha(contract_path),
        "adr_sha256": _sha(adr_path),
        "stage0_review_sha256": _sha(stage0_path),
        "amendment_review_sha256": _sha(amendment_path),
        "stage1_review_sha256": _sha(review_path),
        "scanner_sha256": _sha(Path(scanner.__file__)),
        "runner_sha256": _sha(Path(__file__)),
    }
    contract = _json(contract_path)
    stage0 = _json(stage0_path)
    amendment = _json(amendment_path)
    review = _json(review_path)
    bindings = review.get("bindings")
    if not isinstance(bindings, dict):
        raise TypeError("Stage 1 review bindings are malformed.")
    expected_paths = {
        "config_path": "configs/aeon_external_transfer.json",
        "adr_path": "docs/adr/0012-aeon-external-conditioned-product-transfer.md",
        "stage0_review_path": "orchestration/reviews/AEON_EXTERNAL_STAGE0_DESIGN_REVIEW_20260928.json",
        "amendment_review_path": "orchestration/reviews/AEON_EXTERNAL_STAGE0_WORDING_AMENDMENT_REVIEW_20260928.json",
        "scanner_path": "src/marine_echo/training/aeon_external_metadata.py",
        "runner_path": "src/marine_echo/training/aeon_external_metadata_run.py",
    }
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("study_id") != ("synthetic_external_transfer" if fixture_only else STUDY)
        or contract.get("metadata_row_access") != "PENDING_INDEPENDENT_APPROVAL"
        or contract.get("numeric_external_sv_access") != "PROHIBITED"
        or stage0.get("verdict") != "APPROVE_STAGE0_DESIGN_ONLY"
        or stage0.get("reviewer_task") != "/root/external_reviewer"
        or stage0.get("reviewed_contracts", {}).get("config_sha256") != digests["config_sha256"]
        or amendment.get("verdict") != "APPROVE_STAGE0_WORDING_AMENDMENT_ONLY"
        or amendment.get("reviewer_task") != "/root/external_reviewer"
        or amendment.get("prior_stage0_review_sha256") != digests["stage0_review_sha256"]
        or amendment.get("amended_adr_sha256") != digests["adr_sha256"]
        or amendment.get("unchanged_config_sha256") != digests["config_sha256"]
        or amendment.get("scanner_source_sha256_after_wording_change") != digests["scanner_sha256"]
    ):
        raise ValueError("Stage 1 original and amended Stage 0 lineage differs.")
    expected_status = (
        "APPROVE_STAGE1_METADATA_RUNNER_FIXTURE" if fixture_only
        else "APPROVE_STAGE1_METADATA_RUNNER"
    )
    if (
        review.get("verdict") != expected_status
        or review.get("reviewer_task") != "/root/external_reviewer"
        or review.get("metadata_row_access") != "APPROVED_EXACT_TWO_SOURCE_SCAN"
        or review.get("numeric_external_sv_access") != "PROHIBITED"
        or review.get("study_id") != contract["study_id"]
        or (not fixture_only and any(bindings.get(key) != value for key, value in expected_paths.items()))
        or bindings.get("config_sha256") != digests["config_sha256"]
        or bindings.get("adr_sha256") != digests["adr_sha256"]
        or bindings.get("stage0_review_sha256") != digests["stage0_review_sha256"]
        or bindings.get("amendment_review_sha256") != digests["amendment_review_sha256"]
        or bindings.get("scanner_sha256") != digests["scanner_sha256"]
        or bindings.get("runner_sha256") != digests["runner_sha256"]
    ):
        raise ValueError("Stage 1 distinct runner approval or code binding differs.")
    sources = _source_entries(contract)
    reviewed_sources = bindings.get("sources")
    if not isinstance(reviewed_sources, list) or len(reviewed_sources) != 2:
        raise ValueError("Stage 1 review must bind both source identities.")
    reviewed = {item.get("role"): item for item in reviewed_sources if isinstance(item, dict)}
    if set(reviewed) != set(ROLES):
        raise ValueError("Stage 1 review source roles differ.")
    archives: dict[str, Path] = {}
    outputs: dict[str, Path] = {}
    for role in ROLES:
        source = sources[role]
        archive = (ROOT / source["archive_path"]).resolve() if not fixture_only else Path(source["archive_path"]).resolve()
        if not fixture_only and archive != (ROOT / source["archive_path"]).resolve():
            raise ValueError("Stage 1 source path differs.")
        if (
            reviewed[role].get("archive_path") != source["archive_path"]
            or reviewed[role].get("archive_sha256") != source["archive_sha256"]
            or reviewed[role].get("archive_bytes") != source["archive_bytes"]
            or reviewed[role].get("inventory_sha256")
            != source["hourly_full_depth_central_inventory_sha256"]
        ):
            raise ValueError("Stage 1 source identity differs from distinct review.")
        if not archive.is_file() or archive.stat().st_size != source["archive_bytes"]:
            raise ValueError("Stage 1 source archive missing or byte count differs.")
        if _sha(archive) != source["archive_sha256"]:
            raise ValueError("Stage 1 source archive SHA differs.")
        if fixture_only and source["archive_sha256"] in PRODUCTION_ARCHIVE_SHA:
            raise ValueError("Synthetic review cannot authorize a production archive.")
        archives[role] = archive
        output = output_dir / ("primary-metadata.json" if role == ROLES[0] else "secondary-metadata.json")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Stage 1 output already exists.")
        outputs[role] = output
    for role in ROLES:
        source = sources[role]
        with zipfile.ZipFile(archives[role]) as zf:
            months = scanner._month_range(source["first_month"], source["last_month"])
            prefix = source["hourly_member_prefix"]
            members = tuple(sorted(
                f"{prefix}{frequency}_{month.replace('-', '_')}_60minFullDepth.csv"
                for month in months for frequency in scanner.FREQUENCIES
            ))
            _, inventory = scanner._validate_zip(
                zf, members, max_member_bytes=512 * 1024 * 1024,
                max_total_bytes=8 * 1024 * 1024 * 1024,
                expected_inventory_sha256=source["hourly_full_depth_central_inventory_sha256"],
            )
            if inventory != reviewed[role]["inventory_sha256"]:
                raise ValueError("Stage 1 source inventory differs from review.")
    return contract, outputs, digests


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Stage 1 output already exists.")
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".stage1-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # Fails if the final path appeared; never replaces user data.
    finally:
        temporary.unlink(missing_ok=True)


def run_stage1(
    *, contract_path: Path = CONTRACT_PATH, adr_path: Path = ADR_PATH,
    stage0_path: Path = STAGE0_PATH, amendment_path: Path = AMENDMENT_PATH,
    review_path: Path = REVIEW_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fixture_only: bool = False,
) -> dict[str, Path]:
    contract, outputs, digests = _preflight(
        contract_path=contract_path, adr_path=adr_path,
        stage0_path=stage0_path, amendment_path=amendment_path,
        review_path=review_path, output_dir=output_dir, fixture_only=fixture_only,
    )
    reports: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        source = _source_entries(contract)[role]
        archive = (ROOT / source["archive_path"]) if not fixture_only else Path(source["archive_path"])
        report = scanner.scan_transfer_source_metadata(archive, contract, role)
        report["access_lineage"] = digests
        reports[role] = report
    for role in ROLES:
        _write_exclusive(outputs[role], reports[role])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_stage1(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
