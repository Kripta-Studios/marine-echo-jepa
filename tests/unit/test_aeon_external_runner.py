"""Two-source preaccess and zero-shot persistence tests with synthetic fixtures."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training import aeon_external_metadata, aeon_forecast_adapters
from marine_echo.training import aeon_external_runner as runner
from marine_echo.training.aeon_corpus import AeonHourlySlot


def _write(path: Path, value: dict[str, object]) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return runner._sha256(path)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(aeon_forecast_adapters, "adapter_composite_sha256", lambda: "c" * 64)
    contract_path = tmp_path / "contract.json"
    selection_path = tmp_path / "selection.json"
    review_path = tmp_path / "review.json"
    stage_1_outcome_review_path = tmp_path / "stage1-outcome-review.json"
    archives = {}
    reports = {}
    sources = []
    review_sources = {}
    stage_1_sources = {}
    for index, role in enumerate(runner.SOURCE_ROLES):
        archive = tmp_path / f"source{index}.zip"
        archive.write_bytes(f"synthetic archive {index}".encode("ascii"))
        archive_sha = runner._sha256(archive)
        archives[role] = archive
        candidates = []
        for cutoff in (24, 25):
            stamp = np.datetime64("2022-01-01", "us") + np.timedelta64(cutoff - 1, "h")
            candidates.append({
                "cutoff_interval_id": cutoff,
                "cutoff_source_timestamp": str(stamp),
                "cutoff_source_date": str(stamp.astype("datetime64[D]")),
                "row_id": hashlib.sha256(
                    f"aeon-external-full-depth:{archive_sha}:{cutoff}:24:1,3,6".encode("ascii")
                ).hexdigest(),
                "target_interval_ids": [cutoff + 1, cutoff + 3, cutoff + 6],
                "actual_issued_status": "UNKNOWN",
                "target_scoring_status": "UNKNOWN",
            })
        report_path = tmp_path / f"candidate{index}.json"
        report_value = {
            "classification": "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED",
            "source_role": role,
            "source": {"archive_sha256": archive_sha},
            "hourly_full_depth_central_inventory_sha256": "d" * 64,
            "scanner_code_sha256": runner._sha256(
                Path(runner.aeon_external_metadata.__file__)
            ),
            "candidate_inventory_sha256": "e" * 64,
            "candidate_sha256": "f" * 64,
            "candidate_count": 2,
            "source_interval_id_min": 1,
            "source_interval_id_max": 32,
            "source_interval_count": 32,
            "candidate_source_dates": ["2022-01-01", "2022-01-02"],
            "metadata_exclusion_reasons_by_cutoff": {"1": ["CONTEXT_CROSSES_ARCHIVE_START"]},
            "actual_issued_rows": "UNKNOWN_NUMERIC_QC_NOT_OPENED",
            "actual_scored_rows": "UNKNOWN_NUMERIC_QC_NOT_OPENED",
            "candidate_rows": candidates,
        }
        report_sha = _write(report_path, report_value)
        reports[role] = report_path
        review_sources[role] = {
            "archive_sha256": archive_sha, "candidate_report_sha256": report_sha,
        }
        stage_1_sources[role] = {
            "archive_sha256": archive_sha,
            "candidate_report_sha256": report_sha,
            "candidate_inventory_sha256": "e" * 64,
            "candidate_sha256": "f" * 64,
            "candidate_count": 2,
            "source_interval_id_min": 1,
            "source_interval_id_max": 32,
            "source_interval_count": 32,
            "candidate_source_dates": ["2022-01-01", "2022-01-02"],
            "metadata_exclusion_sha256": hashlib.sha256(json.dumps(
                report_value["metadata_exclusion_reasons_by_cutoff"], sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        }
        sources.append({
            "role": role, "site": f"Synthetic site {index}", "deployment": "Synthetic",
            "archive_sha256": archive_sha, "archive_bytes": archive.stat().st_size,
            "hourly_full_depth_central_inventory_sha256": "d" * 64,
        })
    artifacts = {}
    selected = []
    components = {}
    for model_id, family in runner.MODEL_FAMILIES.items():
        seeds = []
        for seed in (7, 13, 23):
            artifact_id = f"{family}_seed{seed}"
            path = tmp_path / f"{artifact_id}.bin"
            path.write_bytes(artifact_id.encode("ascii"))
            digest = runner._sha256(path)
            artifacts[artifact_id] = path
            components[artifact_id] = digest
            seeds.append({"artifact_id": artifact_id, "sha256": digest})
        selected.append({
            "model_id": model_id, "adapter": "core_neural_ensemble",
            "ensemble_weights": [1 / 3] * 3,
            "adapter_options": {"family": family, "device": "cpu"},
            "component_artifacts": seeds,
        })
    selection_sha = _write(selection_path, {
        "status": "FROZEN_AEON_MODEL_SELECTION",
        "adapter_composite_sha256": "c" * 64,
        "models": selected,
    })
    contract_sha = _write(contract_path, {
        "study_id": "aeon_external_transfer_20260928_v1",
        "numeric_external_sv_access": "PROHIBITED",
        "models": {
            "model_ids": list(runner.MODEL_FAMILIES),
            "selection_freeze_sha256": selection_sha,
            "external_fit_calibration_or_model_selection": "PROHIBITED",
        },
        "sources": sources,
    })
    stage_1_outcome_review_sha = _write(stage_1_outcome_review_path, {
        "status": "APPROVED_AEON_EXTERNAL_STAGE1_METADATA_OUTCOME",
        "study_id": "aeon_external_transfer_20260928_v1",
        "classification": "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED",
        "reviewer_session": "/root/external_reviewer",
        "config_sha256": contract_sha,
        "adr_sha256": runner._ADR_SHA256,
        "stage_0_design_review_sha256": runner._STAGE0_DESIGN_SHA256,
        "stage_0_wording_review_sha256": runner._STAGE0_WORDING_SHA256,
        "scanner_code_sha256": runner._sha256(
            Path(runner.aeon_external_metadata.__file__)
        ),
        "numeric_external_sv_access":
            "PROHIBITED_PENDING_SEPARATE_STAGE2_PRENUMERIC_REVIEW",
        "stage_1_runner_review_sha256": "1" * 64,
        "metadata_runner_code_sha256": "2" * 64,
        "sources": stage_1_sources,
    })
    review_sha = _write(review_path, {
        "status": "APPROVED_AEON_EXTERNAL_STAGE2_SYNTHETIC_FIXTURE",
        "reviewer_session": "/root/aeon_reviewer",
        "contract_sha256": contract_sha,
        "stage_1_outcome_review_sha256": stage_1_outcome_review_sha,
        "selection_sha256": selection_sha,
        "runner_code_sha256": runner._sha256(Path(runner.__file__)),
        "evaluator_code_sha256": runner._sha256(Path(runner.aeon_external_evaluator.__file__)),
        "metadata_scanner_code_sha256": runner._sha256(Path(runner.aeon_external_metadata.__file__)),
        "adapter_composite_sha256": "c" * 64,
        "checkpoint_sha256": components,
        "sources": review_sources,
    })
    return {
        "contract_path": contract_path, "selection_path": selection_path,
        "source_archives": archives, "candidate_reports": reports,
        "checkpoint_paths": artifacts, "review_path": review_path,
        "stage_1_outcome_review_path": stage_1_outcome_review_path,
        "review_sha256": review_sha, "output_directory": tmp_path / "output",
        "fixture_only": True,
    }


def test_both_sources_and_all_six_checkpoint_bytes_preflight_before_numeric_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    contract, selection, reports, artifacts = runner.preflight_external_access(**kwargs)
    assert len(reports) == 2
    assert len(artifacts) == 2
    assert sum(len(items) for items in artifacts.values()) == 6
    assert contract["study_id"] == "aeon_external_transfer_20260928_v1"
    assert selection["status"] == "FROZEN_AEON_MODEL_SELECTION"
    archives = kwargs["source_archives"]
    assert isinstance(archives, dict)
    archives[runner.SOURCE_ROLES[1]].write_bytes(b"changed secondary")
    monkeypatch.setattr(zipfile, "ZipFile", lambda *_args, **_kwargs: pytest.fail("ZIP opened"))
    with pytest.raises(ValueError, match="source or Stage-1"):
        runner.run_external_transfer(**kwargs)


def test_stage1_outcome_review_change_blocks_all_numeric_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    stage_1_review = kwargs["stage_1_outcome_review_path"]
    assert isinstance(stage_1_review, Path)
    stage_1_review.write_text(stage_1_review.read_text(encoding="utf-8") + " ",
                              encoding="utf-8")
    monkeypatch.setattr(runner, "_read_numeric_slots", lambda *_args:
                        pytest.fail("numeric row opened"))
    with pytest.raises(ValueError, match="code/contract review"):
        runner.run_external_transfer(**kwargs)


def test_fixture_run_persists_both_cohorts_and_external_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    sources = {item["role"]: item for item in json.loads(
        kwargs["contract_path"].read_text(encoding="utf-8")
    )["sources"]}

    def slots(_archive: Path, source: dict[str, object], _report: dict[str, object]) -> list[AeonHourlySlot]:
        return [AeonHourlySlot(
            interval_id=interval,
            source_timestamp=np.datetime64("2022-01-01", "us")
            + np.timedelta64(interval - 1, "h"),
            sv_db=np.full(4, -70.0), observed_mask=np.ones(4, dtype=bool),
            qc_status=("OBSERVED_SOURCE_PRODUCT",) * 4,
            member_names=("synthetic.csv",),
            archive_sha256=str(source["archive_sha256"]),
        ) for interval in range(1, 33)]

    monkeypatch.setattr(runner, "_read_numeric_slots", slots)
    kwargs["fixture_forecaster"] = lambda rows: {
        model: np.stack([np.full((3, 5), -70.0) for _ in rows])
        for model in runner.MODEL_FAMILIES
    }
    result = runner.run_external_transfer(**kwargs)
    assert result["manifest"]["status"].endswith("PENDING_INDEPENDENT_OUTCOME_REVIEW")
    for role in runner.SOURCE_ROLES:
        cohort = kwargs["output_directory"] / role.lower()
        assert (cohort / "score.json").exists()
        assert (cohort / "issued-rows.npz").exists()
        assert json.loads((cohort / "score.json").read_text())["score"]["study_partition"] == (
            "external_transfer"
        )
        assert result["cohorts"][role]["source_archive_sha256"] == sources[role][
            "archive_sha256"
        ]


def test_synthetic_numeric_reader_reconstructs_stage1_metadata_inventory(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    metadata = {
        "Date_M": "20220101", "Time_M": "00:00:00.000", "Interval": "1",
        "Layer": "1", "Layer_depth_min": "0", "Layer_depth_max": "200",
        "Ping_S": "1", "Ping_E": "150", "Sv_mean": "-70.0",
    }
    member_names = [
        f"AEON4_12345_{frequency}_2022_01_60minFullDepth.csv"
        for frequency in aeon_external_metadata.FREQUENCIES
    ]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for name in member_names:
            row = ",".join(
                metadata.get(field, "SECRET_UNUSED_FIELD")
                for field in aeon_external_metadata.EXACT_CSV_HEADER
            )
            stream.writestr(name, ",".join(aeon_external_metadata.EXACT_CSV_HEADER) + "\n"
                            + row + "\n")
    manifest = {
        "schema_version": "1.0", "publisher": "Synthetic publisher",
        "publisher_file_id": "1234", "publisher_url": "https://example.invalid/1234",
        "site": "AEON4_SYNTHETIC", "deployment": "Synthetic",
        "archive_sha256": runner._sha256(archive), "months": ["2022-01"],
        "members": sorted(member_names),
    }
    report = aeon_external_metadata.scan_external_metadata(archive, manifest)
    source = {
        "archive_sha256": runner._sha256(archive),
        "hourly_full_depth_central_inventory_sha256": report[
            "hourly_full_depth_central_inventory_sha256"
        ],
        "first_month": "2022_01", "last_month": "2022_01",
        "hourly_full_depth_member_count": 4,
        "hourly_member_prefix": "AEON4_12345_",
    }
    slots = runner._read_numeric_slots(archive, source, report)
    assert len(slots) == 1
    assert slots[0].observed_mask.tolist() == [True, True, True, True]
    assert slots[0].sv_db.tolist() == [-70.0] * 4
    assert "SECRET_UNUSED_FIELD" not in repr(slots)
