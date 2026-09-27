"""Assemble a separate, hash-verified offline AEON research release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from marine_echo.serving.aeon_study import (
    _CAL_ARTIFACT_SHA,
    _CAL_REVIEW_NAME,
    _CAL_REVIEW_SHA,
    _REVIEW_FILES,
    build_aeon_research,
)

_SOURCE_FILES = (
    "marine_echo/__init__.py",
    "marine_echo/contracts/__init__.py",
    "marine_echo/contracts/forecast.py",
    "marine_echo/serving/__init__.py",
    "marine_echo/serving/api.py",
)
_TEMPLATE_FILES = (
    "Run-AEON-Research.ps1", "Serve-AEON-Research.py", "Verify-Release.py",
    "requirements-app.txt",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _upstream_wheels(source: Path, sums: Path) -> dict[str, str]:
    recorded: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator == "  " and name.startswith("wheelhouse/"):
            recorded[name.removeprefix("wheelhouse/")] = digest
    candidates = {p.name for p in source.iterdir() if p.is_file() and p.suffix == ".whl"}
    if not candidates or candidates != set(recorded):
        raise ValueError("wheel inventory differs from the historical verified release")
    for name, digest in recorded.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"wheel digest differs: {name}")
    return recorded


def verify_package(package: Path) -> int:
    """Check exact inventory and bytes, ignoring only local Python install byproducts."""
    expected: dict[str, str] = {}
    for line in (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or name in expected or not name:
            raise ValueError("Invalid package manifest")
        expected[name] = digest
    actual: dict[str, str] = {}
    for path in package.rglob("*"):
        if path.is_symlink():
            raise ValueError("Linked package path")
        if not path.is_file():
            continue
        name = path.relative_to(package).as_posix()
        if name == "SHA256SUMS" or name.startswith(".venv/") or "__pycache__" in path.parts:
            continue
        actual[name] = _sha256(path)
    if actual != expected:
        raise ValueError("Package inventory or digest mismatch")
    return len(actual)


def _archive(package: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as out:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                entry = zipfile.ZipInfo(f"{package.name}/{path.relative_to(package).as_posix()}")
                entry.date_time = (2026, 9, 27, 0, 0, 0)
                entry.compress_type = zipfile.ZIP_DEFLATED
                out.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_aeon_portable(
    root: Path, output: Path, wheelhouse: Path, upstream_sums: Path,
    payload: Path | None = None, calibration_artifact: Path | None = None,
    web_dist: Path | None = None,
) -> dict[str, Any]:
    """Build portable wrapper from reviewed study payload and verified local wheels."""
    archive = output.with_suffix(".zip")
    sidecar = output.with_suffix(".zip.sha256")
    if any(path.exists() or path.is_symlink() for path in (output, archive, sidecar)):
        raise FileExistsError("Portable release output already exists")
    if payload is not None and calibration_artifact is not None:
        raise ValueError("An external study payload cannot be combined with CAL input")
    wheels = _upstream_wheels(wheelhouse, upstream_sums)
    source_root = Path(__file__).resolve().parents[3]
    template_root = source_root / "release/aeon_portable"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aeon-portable-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        if payload is None:
            build_aeon_research(
                root, stage / "study", calibration_artifact=calibration_artifact,
                web_dist=web_dist,
            )
            payload_source = stage / "study"
        else:
            payload_source = payload
        for path in payload_source.iterdir():
            if path.name in {"SHA256SUMS", "study"}:
                continue
            destination = stage / path.name
            if path.is_dir():
                shutil.copytree(path, destination)
            else:
                shutil.copy2(path, destination)
        if payload is None:
            shutil.rmtree(payload_source)
        for name in _SOURCE_FILES:
            destination = stage / "src" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / "src" / name, destination)
        for name in _TEMPLATE_FILES:
            shutil.copy2(template_root / name, stage / name)
        (stage / "wheelhouse").mkdir()
        for name in sorted(wheels):
            shutil.copy2(wheelhouse / name, stage / "wheelhouse" / name)
        if payload is None:
            for review_name, expected, _ in _REVIEW_FILES.values():
                review = root / "orchestration/reviews" / review_name
                if _sha256(review) != expected:
                    raise ValueError(f"Reviewed AEON provenance digest differs: {review_name}")
                destination = stage / "provenance/reviews" / review_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(review, destination)
            if calibration_artifact is not None:
                review = root / "orchestration/reviews" / _CAL_REVIEW_NAME
                if _sha256(review) != _CAL_REVIEW_SHA or _sha256(calibration_artifact) != _CAL_ARTIFACT_SHA:
                    raise ValueError("CAL provenance differs from independently reviewed bytes")
                review_destination = stage / "provenance/reviews" / _CAL_REVIEW_NAME
                shutil.copy2(review, review_destination)
                artifact_destination = stage / "provenance/calibration/calibration.json"
                artifact_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(calibration_artifact, artifact_destination)
        frontend_licenses = {
            "react": "19.3.0", "react-dom": "19.3.0", "scheduler": "0.28.0",
        }
        for name, version in frontend_licenses.items():
            package = root / "web/node_modules" / name
            if package.exists():
                package_metadata = json.loads((package / "package.json").read_text())
                if package_metadata.get("version") != version:
                    raise ValueError(f"Frontend license version differs: {name}")
                destination = stage / "licenses/frontend" / f"{name}-{version}-MIT.txt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(package / "LICENSE", destination)
            elif payload is None:
                raise FileNotFoundError(f"Frontend license unavailable: {name}")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True,
        ).strip()
        catalog = json.loads((stage / "artifacts/catalog.json").read_text(encoding="utf-8"))
        if payload is None and (
            catalog.get("release_class") != "OFFLINE_RESEARCH_MIXED_STUDIES_DEVELOPMENT_ONLY"
            or catalog.get("forecasts") != {}
            or "aeon-study" not in catalog.get("artifacts", {})
        ):
            raise ValueError("AEON study payload contract differs")
        aeon_study_path = stage / "artifacts/aeon-study.json"
        study = json.loads(aeon_study_path.read_text()) if aeon_study_path.exists() else {}
        metadata = {
            "release_class": "OFFLINE_RESEARCH_MIXED_STUDIES_DEVELOPMENT_ONLY",
            "source_commit": commit,
            "source_archive_sha256": study.get("source", {}).get("archive_sha256"),
            "aeon_study_sha256": _sha256(aeon_study_path) if aeon_study_path.exists() else None,
            "calibration_artifact_sha256": _CAL_ARTIFACT_SHA if calibration_artifact else None,
            "calibration_outcome_review_sha256": _CAL_REVIEW_SHA if calibration_artifact else None,
            "historical_wheelhouse_sha256s_sha256": _sha256(upstream_sums),
            "wheel_count": len(wheels),
            "calibration_or_test_final_evaluation": False,
            "cached_aeon_forecasts": 0,
            "independent_final_release_review": False,
        }
        (stage / "SOURCE_REVISION.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
        calibration_notice = (
            "An independently reviewed 810-row CAL interval-widening result is included. "
            "Its raw scores and approximately 90% widened in-sample coverage are calibration "
            "diagnostics, not model selection or final TEST performance. The exact CAL artifact "
            "and independent review are bundled under `provenance/`. "
            if calibration_artifact is not None else "CAL outcomes are not included. "
        )
        (stage / "README_AEON_RESEARCH.md").write_text(
            "# AEON offline research release\n\n"
            "This is a development-only research artifact. The AEON hourly 38-kHz Sv study "
            "presents independently reviewed TRAIN/validation results. "
            + calibration_notice + "Retrospective TEST is not evaluated here. "
            "No AEON forecast is cached or served. "
            "Historical MOSAiC v1/v2 eligibility failures and blocked registry remain intact.\n\n"
            "On Windows, install Python 3.12 and uv locally, then run "
            "`Run-AEON-Research.ps1`. It verifies packaged bytes, installs from the bundled "
            "wheelhouse with `--offline --no-index`, and serves only 127.0.0.1. "
            "The host must already have an offline-available Python 3.12 interpreter.\n\n"
            "The source time zone and publication latency are unverified. Sv is an acoustics "
            "product, not species, biomass, catch or operational benefit. "
            "The original acoustic archive is not redistributed.\n\n"
            "`SHA256SUMS` covers every packaged file except itself; the adjacent `.zip.sha256` "
            "records the archive digest. `SOURCE_REVISION.json` records source and evidence. "
            "Exact development-outcome review records are in `provenance/reviews/`.\n",
            encoding="utf-8",
        )
        (stage / "THIRD_PARTY_NOTICES.md").write_text(
            "# Third-party and source provenance\n\n"
            "AEON AZFP hourly processed Sv: Hakai/AEON publisher source, "
            "`AEON3_GEB_Mar2024-Mar2025_AZFP_Sv.zip`; original data are not bundled. "
            "See the study artifact for exact archive hash, acquisition and QC limits.\n\n"
            "Historical MOSAiC raw-response replay derives from De La Torre et al. 2022, "
            "PANGAEA 949811, DOI 10.1594/PANGAEA.949811, CC BY 4.0. "
            "Environmental research references Schulz, Koenig and Muilwijk 2023, "
            "DOI 10.18739/A21J9790B. Existing derivative artifact provenance is bundled.\n\n"
            "Python wheels retain upstream distribution metadata, including licenses. "
            "React, ReactDOM and Scheduler MIT license texts are in `licenses/frontend/`.\n",
            encoding="utf-8",
        )
        hashes = {
            path.relative_to(stage).as_posix(): _sha256(path)
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        (stage / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
            encoding="utf-8",
        )
        count = verify_package(stage)
        stage.rename(output)
    _archive(output, archive)
    archive_sha = _sha256(archive)
    sidecar.write_text(f"{archive_sha}  {archive.name}\n", encoding="utf-8")
    return {
        "output": str(output), "archive": str(archive), "archive_sha256": archive_sha,
        "asset_count": count, "release_class": metadata["release_class"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--upstream-sums", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--web-dist", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build_aeon_portable(
        arguments.root, arguments.output, arguments.wheelhouse, arguments.upstream_sums,
        calibration_artifact=arguments.calibration_artifact, web_dist=arguments.web_dist,
    ), indent=2))
