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


def _write_final_documents(stage: Path, study: dict[str, Any]) -> None:
    """Write concise cards from the reviewed payload, preserving release-review status."""
    outcome = study["retrospective_test"]
    models = outcome["models"]
    direct = models["core_direct_equal_three_seed_ensemble"]["raw_primary_daily_mean_pinball_db"]
    ema = models["core_ema_equal_three_seed_ensemble"]["raw_primary_daily_mean_pinball_db"]
    documents = {
        "README_RUN.md": (
            "# Run this offline research package\n\n"
            "Final release review pending. On Windows with offline Python 3.12 and uv already "
            "available, run `Run-AEON-Research.ps1`; it verifies SHA256SUMS, installs only "
            "bundled wheels and serves 127.0.0.1. The adjacent ZIP checksum verifies the "
            "archive before extraction. Historical MOSAiC v1/v2 remains separate.\n"
        ),
        "REPORT.md": (
            "# Reviewed AEON retrospective study\n\n"
            f"Source: {study['source']['publisher']}. Archive SHA-256: "
            f"`{study['source']['archive_sha256']}`.\n\n"
            f"TEST issued {outcome['issued_rows']} rows; eligible source dates by +1/+3/+6 "
            f"interval: {outcome['eligible_days_per_horizon']}. Raw primary daily pinball: "
            f"direct {direct:.4f} dB, EMA-JEPA {ema:.4f} dB. The frozen EMA family passed "
            "its within-study retrospective comparison; selection was post hoc in development "
            "and frozen before TEST. Learned-representation attribution is unestablished. "
            "This is not sealed or external evaluation.\n\n"
            f"TEST score SHA-256: `{outcome['test_score_sha256']}`. Independent outcome "
            f"review SHA-256: `{outcome['test_outcome_review_sha256']}`. "
            "The full score, paired block draws and saved prediction arrays are packaged as "
            "reviewed evidence. Final package review remains pending.\n"
        ),
        "DATA_CARD.md": (
            "# AEON data card\n\n"
            "AEON3 Georges Basin fixed lander, published hourly processed AZFP Sv product, "
            "38 kHz 60minFullDepth, nominal 0–200 m. The target is source-reported conditioned "
            "Sv_mean in dB re 1 m^-1; source clock timezone and publication latency are unknown. "
            "Conditioning and manual exclusions were performed by the publisher. Absolute "
            "calibration is not independently field verified. The original source archive is "
            "not redistributed. Candidate metadata, row QC and forecast provenance are under "
            "`provenance/retrospective_test/`.\n"
        ),
        "MODEL_CARD.md": (
            "# Frozen forecast-family card\n\n"
            "The core comparison is an equal-three-seed direct neural ensemble versus an "
            "equal-three-seed EMA-JEPA ensemble, both using 24 past hourly four-frequency "
            "source products with masks to predict 38-kHz Sv quantiles at +1/+3/+6 source "
            "intervals. LightGBM is a post hoc exploratory challenger. Family selection was "
            "post hoc in development but frozen before TEST. The saved TEST arrays support "
            "truth-free historical replay only; there is no live forecasting service. "
            "A within-study EMA forecast gain does not isolate learned JEPA representation value.\n"
        ),
        "LIMITATIONS.md": (
            "# Research limitations\n\n"
            "This retrospective TEST is one deployment and not sealed or externally replicated. "
            "The source clock is not verified UTC and operational availability is unknown. "
            "Processed Sv is not species, biomass, catch, fuel savings or causal device impact. "
            "Publisher conditioning cannot be independently reconstructed from these products. "
            "No state-of-the-art, production or business-validation claim follows. "
            "The historical MOSAiC v1/v2 eligibility failures remain unchanged.\n"
        ),
        "MARINE_DATA_REQUEST.md": (
            "# Further marine data request\n\n"
            "No institution was contacted by this release process. Independent external "
            "replication would require a new source with documented clock timezone, "
            "acquisition latency, calibration coefficients, processing exclusions and sufficient "
            "hourly multi-frequency coverage. Any access request needs owner authorization.\n"
        ),
    }
    for name, content in documents.items():
        (stage / name).write_text(content, encoding="utf-8")


def build_aeon_portable(
    root: Path, output: Path, wheelhouse: Path, upstream_sums: Path,
    payload: Path | None = None, calibration_artifact: Path | None = None,
    web_dist: Path | None = None,
    test_score: Path | None = None, test_candidate: Path | None = None,
    test_review: Path | None = None, test_forecasts: Path | None = None,
    external_transfer: bool = False,
    scale_output: Path | None = None, expanded_output: Path | None = None,
) -> dict[str, Any]:
    """Build portable wrapper from reviewed study payload and verified local wheels."""
    archive = output.with_suffix(".zip")
    sidecar = output.with_suffix(".zip.sha256")
    if any(path.exists() or path.is_symlink() for path in (output, archive, sidecar)):
        raise FileExistsError("Portable release output already exists")
    test_inputs = (test_score, test_candidate, test_review, test_forecasts)
    if payload is not None and (
        calibration_artifact is not None or any(test_inputs) or external_transfer
        or scale_output is not None or expanded_output is not None
    ):
        raise ValueError("An external study payload cannot be combined with reviewed study inputs")
    if external_transfer and not all(value is not None for value in test_inputs):
        raise ValueError("External transfer packaging requires the reviewed retrospective study.")
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
                test_score=test_score, test_candidate=test_candidate,
                test_review=test_review, test_forecasts=test_forecasts,
                external_transfer=external_transfer,
                scale_output=scale_output, expanded_output=expanded_output,
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
        study = json.loads((stage / "artifacts/aeon-study.json").read_text(encoding="utf-8")) if payload is None else {}
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
            if all(value is not None for value in test_inputs):
                assert test_score is not None and test_candidate is not None
                assert test_review is not None and test_forecasts is not None
                for source, relative in (
                    (test_score, "provenance/retrospective_test/test-score.json"),
                    (test_candidate, "provenance/retrospective_test/metadata-candidates.json"),
                    (test_review, "provenance/reviews/AEON_RETROSPECTIVE_TEST_OUTCOME_REVIEW_20260927.json"),
                ):
                    destination = stage / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                for model_id in study["retrospective_test"]["models"]:
                    name = f"{model_id}-forecast.npz"
                    shutil.copy2(test_forecasts / name, stage / "provenance/retrospective_test" / name)
                outcome_report = root / "orchestration/reports/AEON_RETROSPECTIVE_OUTCOME_20260927.md"
                shutil.copy2(outcome_report, stage / "provenance/retrospective_test" / outcome_report.name)
                _write_final_documents(stage, study)
                evidence = stage / "evidence"
                (evidence / "protocol").mkdir(parents=True)
                (stage / "requirements-locks").mkdir()
                shutil.copy2(stage / "requirements-app.txt", stage / "requirements-locks/requirements-app.txt")
                for name in (
                    "0008-aeon-hourly-sv-study.md",
                    "0009-aeon-eligible-date-and-evaluation-freeze.md",
                    "0010-aeon-development-selection.md",
                    "0011-aeon-reviewed-development-selection.md",
                ):
                    shutil.copy2(root / "docs/adr" / name, evidence / "protocol" / name)
                for name in ("aeon_final_selection.json", "aeon_final_evaluation.json"):
                    shutil.copy2(root / "configs" / name, evidence / "protocol" / name)
                shutil.copy2(root / "orchestration/aeon_run_ledger.json", evidence / "aeon_run_ledger.json")
                shutil.copy2(test_score, evidence / "metrics.json")
                (evidence / "runtime.json").write_text(json.dumps({
                    "status": "NOT_MEASURED_DURING_PACKAGE_BUILD",
                    "offline_install": "REQUIRES_RELOCATED_SMOKE",
                    "api_and_browser": "REQUIRES_RELOCATED_SMOKE",
                    "live_cpu_inference": "NOT_PROVIDED_TRUTH_FREE_REPLAY_ONLY",
                }, indent=2) + "\n", encoding="utf-8")
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
        expected_class = (
            "OFFLINE_RESEARCH_MIXED_STUDIES_TEST_REVIEWED_RELEASE_REVIEW_PENDING"
            if all(value is not None for value in test_inputs)
            else "OFFLINE_RESEARCH_MIXED_STUDIES_DEVELOPMENT_ONLY"
        )
        if payload is None and (
            catalog.get("release_class") != expected_class
            or catalog.get("forecasts") != {}
            or "aeon-study" not in catalog.get("artifacts", {})
        ):
            raise ValueError("AEON study payload contract differs")
        aeon_study_path = stage / "artifacts/aeon-study.json"
        study = json.loads(aeon_study_path.read_text()) if aeon_study_path.exists() else {}
        metadata = {
            "release_class": expected_class,
            "source_commit": commit,
            "source_archive_sha256": study.get("source", {}).get("archive_sha256"),
            "aeon_study_sha256": _sha256(aeon_study_path) if aeon_study_path.exists() else None,
            "calibration_artifact_sha256": _CAL_ARTIFACT_SHA if calibration_artifact else None,
            "calibration_outcome_review_sha256": _CAL_REVIEW_SHA if calibration_artifact else None,
            "historical_wheelhouse_sha256s_sha256": _sha256(upstream_sums),
            "wheel_count": len(wheels),
            "retrospective_test_score_sha256": study.get("retrospective_test", {}).get("test_score_sha256"),
            "retrospective_test_outcome_review_sha256": study.get("retrospective_test", {}).get("test_outcome_review_sha256"),
            "retrospective_test_non_sealed": bool(study.get("retrospective_test")),
            "external_transfer_outcome_review_sha256": study.get("external_transfer", {}).get(
                "outcome_review_sha256"
            ),
            "external_transfer_manifest_sha256": study.get("external_transfer", {}).get(
                "manifest_sha256"
            ),
            "external_primary_jepa_gate": study.get("external_transfer", {}).get(
                "primary", {}
            ).get("jepa_value_gate"),
            "external_secondary_jepa_gate": study.get("external_transfer", {}).get(
                "secondary", {}
            ).get("jepa_value_gate"),
            "scaling_development_outcome_review_sha256": study.get("scaling_development", {}).get(
                "outcome_review_sha256"
            ),
            "scaling_development_manifest_sha256": study.get("scaling_development", {}).get(
                "manifest_sha256"
            ),
            "expanded_train_development_outcome_review_sha256": study.get(
                "expanded_train_development", {}
            ).get("outcome_review_sha256"),
            "expanded_train_development_manifest_sha256": study.get(
                "expanded_train_development", {}
            ).get("manifest_sha256"),
            "cached_aeon_forecasts": study.get("cached_forecasts", 0),
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
        test_notice = (
            "An independently reviewed retrospective TEST result and truth-free historical "
            "source-clock replay are included. TEST is not sealed or externally replicated. "
            "The EMA-JEPA family, selected post hoc in development and frozen before TEST, "
            "passed the within-study retrospective comparison; representation "
            "attribution, external generalization and business validation remain unproven. "
            "Exact TEST score, "
            "forecast arrays, metadata candidates and independent review are bundled under "
            "`provenance/`. Final package review remains pending. "
            if all(value is not None for value in test_inputs)
            else "Retrospective TEST is not evaluated here. No AEON forecast is cached or served. "
        )
        external_notice = (
            "The post-hoc external transfer includes a metadata-ineligible primary AEON2 "
            "Eastern Coastal Shelf target (no primary acoustic values opened or model scored) "
            "and a negative descriptive prior-year same-site AEON3 comparison: direct "
            "outperformed EMA-JEPA on daily mean pinball loss. This is not cross-site "
            "replication or a sealed holdout. Exact reviewed forecasts, issued-row support, "
            "scores, source contract, manifest and independent outcome review are bundled "
            "under `provenance/external_transfer/`; the original archives are excluded. "
            "Sources are Figshare AZFP article version 2, files 61937269 and 61937275, "
            "CC BY 4.0; the target is source-reported conditioned Sv_mean. "
            if external_transfer else ""
        )
        scaling_notice = (
            "The separate post-hoc same-cohort 30k seed-7 development endpoints worsened "
            "corrected validation loss versus original 3k endpoints in both direct and "
            "EMA-JEPA families. The prespecified gate did not authorize stage-2 seeds or "
            "a 50k extension. Final reviewed validation prediction arrays and endpoint "
            "checkpoints are bundled under `provenance/scaling_development/`; intermediate "
            "checkpoints and raw acoustic archives are excluded. "
            if scale_output is not None else ""
        )
        expanded_notice = (
            "The separate post-hoc expanded-TRAIN 3k study pooled prior-year and current "
            "same-site TRAIN deployments. Direct improved modestly on the unchanged original "
            "validation partition; EMA-JEPA worsened. This does not isolate data-volume "
            "effects or provide external or sealed evaluation for the derived models. "
            "Final reviewed validation prediction arrays and endpoint checkpoints are bundled "
            "under `provenance/expanded_train_development/`; intermediate checkpoints and "
            "raw acoustic archives are excluded. "
            if expanded_output is not None else ""
        )
        (stage / "README_AEON_RESEARCH.md").write_text(
            "# AEON offline research release\n\n"
            + ("This package awaits independent final release review. " if all(value is not None for value in test_inputs)
               else "This is a development-only research artifact. ")
            + "The AEON hourly 38-kHz Sv study "
            "presents independently reviewed TRAIN/validation results. "
            + calibration_notice + test_notice + external_notice + scaling_notice + expanded_notice
            +
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
            + (
                "External transfer uses Figshare AZFP article version 2, files 61937269 "
                "and 61937275, CC BY 4.0. Original external archives are excluded. "
                "The quantity is source-reported conditioned Sv_mean.\n\n"
                if external_transfer else ""
            )
            +
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
    parser.add_argument("--test-score", type=Path)
    parser.add_argument("--test-candidate", type=Path)
    parser.add_argument("--test-review", type=Path)
    parser.add_argument("--test-forecasts", type=Path)
    parser.add_argument("--external-transfer", action="store_true")
    parser.add_argument("--scale-output", type=Path)
    parser.add_argument("--expanded-output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build_aeon_portable(
        arguments.root, arguments.output, arguments.wheelhouse, arguments.upstream_sums,
        calibration_artifact=arguments.calibration_artifact, web_dist=arguments.web_dist,
        test_score=arguments.test_score, test_candidate=arguments.test_candidate,
        test_review=arguments.test_review, test_forecasts=arguments.test_forecasts,
        external_transfer=arguments.external_transfer,
        scale_output=arguments.scale_output,
        expanded_output=arguments.expanded_output,
    ), indent=2))
