"""Synthetic evidence bindings for the separate AEON external study payload."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from marine_echo.serving import aeon_external_study as external
from marine_echo.serving import aeon_study


def _write(path: Path, value: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    output = root / external._OUTPUT_RELATIVE
    output.mkdir(parents=True)
    primary: dict[str, object] = {
        "status": "METADATA_INELIGIBLE_NO_CANDIDATES",
        "comparison_role": "PRIMARY_FIXED_TARGET",
        "candidate_count": 0, "actual_issued_rows": 0,
        "numeric_sv_access": "NOT_RUN_METADATA_INELIGIBLE",
        "jepa_value_gate": "NOT_EVALUATED_METADATA_INELIGIBLE",
        "stage_1_38khz_geometry_histogram": {"0:230": 100},
        "fixed_target_layer_geometry_m": "0:200",
    }
    metrics = {
        "primary_daily_mean_pinball_db": 0.6,
        "scored_rows_per_horizon": [100, 100, 100],
        "eligible_scored_rows_per_horizon": [90, 90, 90],
    }
    secondary: dict[str, object] = {
        "status": "COMPLETED_ZERO_SHOT_EXTERNAL_TRANSFER",
        "comparison_role": "SECONDARY_DESCRIPTIVE_NO_FALLBACK",
        "candidate_count": 100, "actual_issued_rows": 100,
        "candidate_not_issued_reasons": {}, "site": "AEON3_GEB",
        "deployment": "Feb2023-Feb2024",
        "score": {
            "study_partition": "external_transfer",
            "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
            "issued_rows": 100, "eligible_dates_per_horizon": [90, 90, 90],
            "direct_raw_metrics": metrics,
            "ema_raw_metrics": {**metrics, "primary_daily_mean_pinball_db": 0.7},
            "relative_loss_reduction": -1 / 6,
            "paired_bootstrap_status": "COMPLETED_2000_DRAWS_SEED_20260928",
            "paired_95pct_difference_interval_db": [0.01, 0.2],
            "jepa_value_gate": "DESCRIPTIVE_ONLY_NOT_PRIMARY_GATE",
        },
    }
    _write(output / external._PRIMARY, primary)
    _write(output / external._SECONDARY, secondary)
    for name in (external._ISSUED, external._DIRECT, external._EMA):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic fixture {name}".encode("ascii"))
    fields = (
        "contract_sha256", "selection_sha256", "runner_code_sha256",
        "evaluator_code_sha256", "evaluation_code_sha256", "corpus_code_sha256",
        "adapter_composite_sha256",
    )
    lineage = dict.fromkeys(fields, "a" * 64)
    contract_sha = _write(root / "configs/aeon_external_transfer.json", {
        "study_id": external._STUDY_ID,
        "publisher_article": "https://figshare.com/articles/dataset/AZFP/29247113",
        "publisher_article_version": 2, "license": "CC BY 4.0",
        "sources": [
            {"role": "PRIMARY_CROSS_SITE_CONTEMPORANEOUS", "site": "AEON2_ECS",
             "site_name": "AEON2 Eastern Coastal Shelf", "file_id": 61937269},
            {"role": "SECONDARY_PRIOR_YEAR_SAME_SITE", "site": "AEON3_GEB",
             "file_id": 61937275},
        ],
    })
    lineage["contract_sha256"] = contract_sha
    manifest: dict[str, object] = {
        "status": "COMPLETED_EXTERNAL_TRANSFER_PENDING_INDEPENDENT_OUTCOME_REVIEW",
        "study_id": external._STUDY_ID,
        "classification": external._CLASSIFICATION,
        "output_directory": external._OUTPUT_RELATIVE,
        "stage_2_review_sha256": "b" * 64,
        "stage_1_outcome_review_sha256": "c" * 64,
        **lineage,
    }
    review: dict[str, object] = {
        "study_id": external._STUDY_ID,
        "status": "APPROVED_AEON_EXTERNAL_STAGE2_OUTCOME",
        "verdict": "APPROVE_DESCRIPTIVE_SECONDARY_NEGATIVE_COMPARISON_ONLY",
        "reviewer_session": "/root/external_reviewer",
        "classification": external._CLASSIFICATION,
        "promotion_authorization": (
            "The exact reviewed descriptive outcome may be integrated into the research app "
            "and offline release only with the primary NOT_EVALUATED state, secondary "
            "negative result and claim limits preserved."
        ),
        "retry_review_sha256": "b" * 64,
        "stage_1_outcome_review_sha256": "c" * 64,
        **lineage,
        "primary_outcome": {
            "candidate_count": 0, "actual_issued_rows": 0,
            "numeric_sv_access": "NOT_RUN_METADATA_INELIGIBLE",
            "jepa_value_gate": "NOT_EVALUATED_METADATA_INELIGIBLE",
            "observed_38khz_geometry_histogram": {"0:230": 100},
            "frozen_target_geometry": "0:200",
            "cross_site_replication_result": "ABSENT",
            "model_result_classification": "NOT_A_NEGATIVE_MODEL_RESULT",
        },
        "secondary_outcome": {
            "role": "SECONDARY_PRIOR_YEAR_SAME_SITE_DESCRIPTIVE",
            "metadata_candidates": 100, "actual_issued_rows": 100,
            "scored_rows_per_horizon": [100, 100, 100],
            "eligible_scored_rows_per_horizon": [90, 90, 90],
            "eligible_source_dates_per_horizon": [90, 90, 90],
            "direct_daily_mean_pinball_db": 0.6,
            "ema_jepa_daily_mean_pinball_db": 0.7,
            "relative_loss_reduction": -1 / 6,
            "paired_95pct_ema_minus_direct_interval_db": [0.01, 0.2],
            "paired_bootstrap_status": "COMPLETED_2000_DRAWS_SEED_20260928",
            "jepa_value_gate": "DESCRIPTIVE_ONLY_NOT_PRIMARY_GATE",
        },
    }
    _bind(root, manifest, review, monkeypatch)
    return primary, secondary, review


def _bind(root: Path, manifest: dict[str, object], review: dict[str, object],
          monkeypatch: pytest.MonkeyPatch) -> None:
    output = root / external._OUTPUT_RELATIVE
    hashes = {name: external._sha256(output / name) for name in external._OUTPUT_FILES}
    manifest["files"] = hashes
    manifest_sha = _write(output / "manifest.json", manifest)
    review["artifacts"] = {
        "output_directory": external._OUTPUT_RELATIVE,
        "manifest_path": external._OUTPUT_RELATIVE + "/manifest.json",
        "manifest_sha256": manifest_sha,
        **{field: hashes[name] for name, field in external._ARTIFACT_FIELDS.items()},
    }
    review_sha = _write(root / external._REVIEW_RELATIVE, review)
    monkeypatch.setattr(external, "_REVIEW_SHA256", review_sha)


def test_reviewed_negative_external_summary_and_file_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixture(tmp_path, monkeypatch)
    result = external.load_reviewed_external_study(tmp_path)
    assert result["primary"]["candidate_count"] == 0
    assert result["primary"]["jepa_value_gate"] == "NOT_EVALUATED_METADATA_INELIGIBLE"
    assert result["secondary"]["direct_daily_pinball_db"] == 0.6
    assert result["secondary"]["ema_daily_pinball_db"] == 0.7
    assert result["secondary"]["ema_relative_loss_reduction"] < 0
    assert result["source"]["file_ids"] == [61937269, 61937275]
    assert set(external.reviewed_external_files(tmp_path)) == {
        "outcome-review.json", "source-contract.json", "manifest.json", *external._OUTPUT_FILES,
    }


def test_missing_or_mutated_review_blocks_packaging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixture(tmp_path, monkeypatch)
    review_path = tmp_path / external._REVIEW_RELATIVE
    review_path.write_text(review_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="outcome review digest"):
        external.load_reviewed_external_study(tmp_path)


def test_forecast_bytes_must_match_review_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixture(tmp_path, monkeypatch)
    (tmp_path / external._OUTPUT_RELATIVE / external._EMA).write_bytes(b"mutated forecast")
    with pytest.raises(ValueError, match="binding differs"):
        external.load_reviewed_external_study(tmp_path)


def test_hardlinked_external_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixture(tmp_path, monkeypatch)
    os.link(tmp_path / external._OUTPUT_RELATIVE / external._EMA, tmp_path / "same-forecast.npz")
    with pytest.raises(ValueError, match="unsafe"):
        external.load_reviewed_external_study(tmp_path)


def test_primary_ineligible_state_cannot_be_promoted_even_if_resigned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, _, review = _fixture(tmp_path, monkeypatch)
    primary["jepa_value_gate"] = "PASSED"
    _write(tmp_path / external._OUTPUT_RELATIVE / external._PRIMARY, primary)
    manifest = external._json(tmp_path / external._OUTPUT_RELATIVE / "manifest.json")
    _bind(tmp_path, manifest, review, monkeypatch)
    with pytest.raises(ValueError, match="primary metadata-ineligible"):
        external.load_reviewed_external_study(tmp_path)


@pytest.mark.parametrize("mutate_at_copy", [False, True])
def test_research_package_copies_exact_reviewed_external_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate_at_copy: bool,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _fixture(root, monkeypatch)
    monkeypatch.setattr(aeon_study, "build_aeon_development_report", lambda _root: {
        "study_id": "aeon3_geb_2024_hourly_sv_v1", "limitations": [
            "No AEON forecasts are cached or served by this release."
        ], "cached_forecasts": 0,
    })
    monkeypatch.setattr(aeon_study, "build_aeon_calibration_report", lambda *_args: {})
    monkeypatch.setattr(aeon_study, "load_reviewed_test", lambda *_args: (
        {"status": "INDEPENDENTLY_REVIEWED_RETROSPECTIVE_TEST", "models": {}}, {"rows": []}
    ))

    def base_package(_root: Path, base: Path) -> None:
        artifacts = base / "artifacts"
        artifacts.mkdir(parents=True)
        (base / "web").mkdir()
        catalog = {
            "release_class": "OFFLINE_RESEARCH_ENGINEERING_ONLY", "forecasts": {},
            "artifacts": {}, "studies": [], "experiments": [],
        }
        _write(artifacts / "catalog.json", catalog)
        _write(base / "catalog.json", catalog)

    monkeypatch.setattr(aeon_study, "build_v2_research", base_package)
    if mutate_at_copy:
        def mutated_files(source_root: Path) -> dict[str, Path]:
            files = external.reviewed_external_files(source_root)
            files[external._EMA].write_bytes(b"changed after reviewed load")
            return files

        monkeypatch.setattr(aeon_study, "reviewed_external_files", mutated_files)
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("synthetic web", encoding="utf-8")
    def package() -> dict[str, object]:
        return aeon_study.build_aeon_research(
            root, tmp_path / "release", calibration_artifact=tmp_path / "calibration.json",
            web_dist=web, test_score=tmp_path / "score.json",
            test_candidate=tmp_path / "candidate.json", test_review=tmp_path / "review.json",
            test_forecasts=tmp_path / "forecasts", external_transfer=True,
        )

    if mutate_at_copy:
        with pytest.raises(ValueError, match="packaged provenance"):
            package()
        assert not (tmp_path / "release").exists()
        return
    result = package()
    assert result["status"] == "OFFLINE_RESEARCH_MIXED_STUDIES_TEST_REVIEWED_RELEASE_REVIEW_PENDING"
    package = tmp_path / "release"
    study = external._json(package / "artifacts/aeon-study.json")
    catalog = external._json(package / "artifacts/catalog.json")
    assert catalog["studies"][-1] == external._STUDY_ID
    assert study["external_transfer"]["primary"]["jepa_value_gate"] == (
        "NOT_EVALUATED_METADATA_INELIGIBLE"
    )
    for name, source in external.reviewed_external_files(root).items():
        assert (package / "provenance/external_transfer" / name).read_bytes() == source.read_bytes()
