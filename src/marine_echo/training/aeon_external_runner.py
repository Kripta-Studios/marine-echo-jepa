"""Review-gated one-pass zero-shot evaluation of both external AEON archives.

No real archive rows may be opened before a distinct Stage-2 review binds every
source, Stage-1 inventory, frozen model, and executable code byte hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from marine_echo.training import (
    aeon_external_evaluator,
    aeon_external_metadata,
    aeon_forecast_adapters,
)
from marine_echo.training.aeon_corpus import SPECIAL_SV, AeonHourlySlot
from marine_echo.training.aeon_external_evaluator import (
    adapter_compatible_rows,
    materialize_candidates,
    score_external_predictions,
)
from marine_echo.training.aeon_forecast_adapters import FrozenArtifact, adapt_core_neural_ensemble

MODEL_FAMILIES = {
    "core_direct_equal_three_seed_ensemble": "direct",
    "core_ema_equal_three_seed_ensemble": "ema_jepa",
}
SOURCE_ROLES = ("PRIMARY_CROSS_SITE_CONTEMPORANEOUS", "SECONDARY_PRIOR_YEAR_SAME_SITE")
_REVIEW_STATUS = "APPROVED_AEON_EXTERNAL_STAGE2_NUMERIC_ACCESS"
_FIXTURE_STATUS = "APPROVED_AEON_EXTERNAL_STAGE2_SYNTHETIC_FIXTURE"
_ADR_SHA256 = "3abd740a0597c712fbf67f0265737afecd31acc645be834bb329cf8d271491c7"
_STAGE0_DESIGN_SHA256 = "b9c1ebf40c5a924e6fb2efcb869dca29b42e0c7baf7d2b7a788c70a10dad259d"
_STAGE0_WORDING_SHA256 = "001c961f49dc5b4cafb61b66ea0aaf79c54b0e549325e0aa25a7170c62b14d5a"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("AEON external JSON review/contract is not an object.")
    return value


def _expected_members(source: Mapping[str, Any]) -> tuple[str, ...]:
    months = aeon_external_metadata._month_range(source["first_month"], source["last_month"])
    if len(months) * 4 != source["hourly_full_depth_member_count"]:
        raise ValueError("AEON external source member count differs from contract.")
    return tuple(sorted(
        f"{source['hourly_member_prefix']}{frequency}_{month.replace('-', '_')}_60minFullDepth.csv"
        for month in months for frequency in aeon_external_metadata.FREQUENCIES
    ))


def preflight_external_access(
    *,
    contract_path: Path,
    selection_path: Path,
    source_archives: Mapping[str, Path],
    candidate_reports: Mapping[str, Path],
    checkpoint_paths: Mapping[str, Path],
    stage_1_outcome_review_path: Path,
    review_path: Path,
    review_sha256: str,
    output_directory: Path,
    fixture_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, list[FrozenArtifact]]]:
    """Validate both sources and all six checkpoint bytes before any CSV row read."""
    if output_directory.exists():
        raise FileExistsError("AEON external output already exists.")
    if set(source_archives) != set(SOURCE_ROLES) or set(candidate_reports) != set(SOURCE_ROLES):
        raise ValueError("AEON external transfer requires both independent archive roles.")
    if _sha256(review_path) != review_sha256:
        raise ValueError("AEON external Stage-2 review digest differs.")
    review = _json(review_path)
    stage_1_review_sha256 = _sha256(stage_1_outcome_review_path)
    stage_1_review = _json(stage_1_outcome_review_path)
    contract = _json(contract_path)
    selection = _json(selection_path)
    if (
        review.get("status") != (_FIXTURE_STATUS if fixture_only else _REVIEW_STATUS)
        or review.get("reviewer_session") != "/root/external_reviewer"
        or review.get("contract_sha256") != _sha256(contract_path)
        or review.get("stage_1_outcome_review_sha256") != stage_1_review_sha256
        or review.get("selection_sha256") != _sha256(selection_path)
        or review.get("runner_code_sha256") != _sha256(Path(__file__))
        or review.get("evaluator_code_sha256") != _sha256(
            Path(aeon_external_evaluator.__file__)
        )
        or review.get("metadata_scanner_code_sha256") != _sha256(
            Path(aeon_external_metadata.__file__)
        )
        or review.get("adapter_composite_sha256")
        != aeon_forecast_adapters.adapter_composite_sha256()
    ):
        raise ValueError("AEON external numeric access lacks exact distinct code/contract review.")
    if (
        contract.get("study_id") != "aeon_external_transfer_20260928_v1"
        or contract.get("numeric_external_sv_access") != "PROHIBITED"
        or contract.get("models", {}).get("model_ids") != list(MODEL_FAMILIES)
        or contract.get("models", {}).get("selection_freeze_sha256") != _sha256(
            selection_path
        )
        or contract.get("models", {}).get("external_fit_calibration_or_model_selection")
        != "PROHIBITED"
        or selection.get("status") != "FROZEN_AEON_MODEL_SELECTION"
        or selection.get("adapter_composite_sha256")
        != aeon_forecast_adapters.adapter_composite_sha256()
    ):
        raise ValueError("AEON external model/selection freeze differs.")
    if (
        stage_1_review.get("status") != "APPROVED_AEON_EXTERNAL_STAGE1_METADATA_OUTCOME"
        or stage_1_review.get("study_id") != contract["study_id"]
        or stage_1_review.get("classification")
        != "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED"
        or stage_1_review.get("reviewer_session") != "/root/external_reviewer"
        or stage_1_review.get("config_sha256") != _sha256(contract_path)
        or stage_1_review.get("adr_sha256") != _ADR_SHA256
        or stage_1_review.get("stage_0_design_review_sha256") != _STAGE0_DESIGN_SHA256
        or stage_1_review.get("stage_0_wording_review_sha256") != _STAGE0_WORDING_SHA256
        or stage_1_review.get("scanner_code_sha256") != _sha256(
            Path(aeon_external_metadata.__file__)
        )
        or stage_1_review.get("numeric_external_sv_access")
        != "PROHIBITED_PENDING_SEPARATE_STAGE2_PRENUMERIC_REVIEW"
        or not isinstance(stage_1_review.get("stage_1_runner_review_sha256"), str)
        or len(stage_1_review["stage_1_runner_review_sha256"]) != 64
        or not isinstance(stage_1_review.get("metadata_runner_code_sha256"), str)
        or len(stage_1_review["metadata_runner_code_sha256"]) != 64
    ):
        raise ValueError("AEON external Stage-1 outcome lacks distinct metadata review.")
    if not fixture_only:
        repository = Path(__file__).resolve().parents[3]
        lineage_files = {
            "adr_sha256": repository / "docs/adr/0012-aeon-external-conditioned-product-transfer.md",
            "stage_0_design_review_sha256": repository
            / "orchestration/reviews/AEON_EXTERNAL_STAGE0_DESIGN_REVIEW_20260928.json",
            "stage_0_wording_review_sha256": repository
            / "orchestration/reviews/AEON_EXTERNAL_STAGE0_WORDING_AMENDMENT_REVIEW_20260928.json",
            "metadata_runner_code_sha256": repository
            / "src/marine_echo/training/aeon_external_metadata_run.py",
        }
        if any(_sha256(path) != stage_1_review[key] for key, path in lineage_files.items()):
            raise ValueError("AEON external Stage-1 reviewed source lineage differs.")
        review_relative = stage_1_review.get("stage_1_runner_review_path")
        if not isinstance(review_relative, str):
            raise ValueError("AEON external Stage-1 runner review path is missing.")
        allowed_review_root = (repository / "orchestration/reviews").resolve(strict=True)
        runner_review = (repository / review_relative).resolve(strict=True)
        if not runner_review.is_relative_to(allowed_review_root) or _sha256(
            runner_review
        ) != stage_1_review["stage_1_runner_review_sha256"]:
            raise ValueError("AEON external Stage-1 runner review bytes differ.")
    source_specs = contract.get("sources")
    if not isinstance(source_specs, list) or {item.get("role") for item in source_specs} != set(
        SOURCE_ROLES
    ) or len(source_specs) != 2:
        raise ValueError("AEON external contract lacks exactly two source roles.")
    specs = {item["role"]: item for item in source_specs}
    reports: dict[str, dict[str, Any]] = {}
    review_sources = review.get("sources")
    if not isinstance(review_sources, dict) or set(review_sources) != set(SOURCE_ROLES):
        raise ValueError("AEON external Stage-2 review lacks both source bindings.")
    stage_1_sources = stage_1_review.get("sources")
    if not isinstance(stage_1_sources, dict) or set(stage_1_sources) != set(SOURCE_ROLES):
        raise ValueError("AEON external Stage-1 outcome lacks both source bindings.")
    for role in SOURCE_ROLES:
        source = specs[role]
        archive = source_archives[role]
        report_path = candidate_reports[role]
        reviewed = review_sources[role]
        if (
            not isinstance(reviewed, dict)
            or reviewed.get("archive_sha256") != source.get("archive_sha256")
            or reviewed.get("candidate_report_sha256") != _sha256(report_path)
            or archive.stat().st_size != source.get("archive_bytes")
            or _sha256(archive) != source.get("archive_sha256")
        ):
            raise ValueError("AEON external source or Stage-1 report digest differs before access.")
        report = _json(report_path)
        if (
            report.get("classification") != "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED"
            or report.get("source_role") != role
            or report.get("source", {}).get("archive_sha256") != source["archive_sha256"]
            or report.get("hourly_full_depth_central_inventory_sha256")
            != source["hourly_full_depth_central_inventory_sha256"]
            or report.get("scanner_code_sha256") != _sha256(
                Path(aeon_external_metadata.__file__)
            )
            or not isinstance(report.get("candidate_rows"), list)
            or not isinstance(report.get("candidate_inventory_sha256"), str)
        ):
            raise ValueError("AEON external Stage-1 candidate report differs.")
        prior = stage_1_sources[role]
        exclusion_sha256 = hashlib.sha256(json.dumps(
            report.get("metadata_exclusion_reasons_by_cutoff"), sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if (
            not isinstance(prior, dict)
            or prior.get("archive_sha256") != source["archive_sha256"]
            or prior.get("candidate_report_sha256") != _sha256(report_path)
            or prior.get("candidate_inventory_sha256") != report["candidate_inventory_sha256"]
            or prior.get("candidate_sha256") != report.get("candidate_sha256")
            or prior.get("candidate_count") != report.get("candidate_count")
            or prior.get("source_interval_id_min") != report.get("source_interval_id_min")
            or prior.get("source_interval_id_max") != report.get("source_interval_id_max")
            or prior.get("source_interval_count") != report.get("source_interval_count")
            or prior.get("candidate_source_dates") != report.get("candidate_source_dates")
            or prior.get("metadata_exclusion_sha256") != exclusion_sha256
            or report.get("actual_issued_rows") != "UNKNOWN_NUMERIC_QC_NOT_OPENED"
            or report.get("actual_scored_rows") != "UNKNOWN_NUMERIC_QC_NOT_OPENED"
        ):
            raise ValueError("AEON external Stage-1 outcome review/report lineage differs.")
        reports[role] = report
    selected = selection.get("models")
    if not isinstance(selected, list):
        raise TypeError("AEON external selected models are malformed.")
    selected_by_id = {item.get("model_id"): item for item in selected if isinstance(item, dict)}
    model_artifacts: dict[str, list[FrozenArtifact]] = {}
    reviewed_components = review.get("checkpoint_sha256")
    if not isinstance(reviewed_components, dict):
        raise TypeError("AEON external review lacks checkpoint bytes.")
    required_components: set[str] = set()
    for model_id, family in MODEL_FAMILIES.items():
        model = selected_by_id.get(model_id)
        if (
            not isinstance(model, dict)
            or model.get("adapter") != "core_neural_ensemble"
            or model.get("ensemble_weights") != [1 / 3] * 3
            or model.get("adapter_options") != {"family": family, "device": "cpu"}
            or not isinstance(model.get("component_artifacts"), list)
            or len(model["component_artifacts"]) != 3
        ):
            raise ValueError("AEON external frozen neural model specification differs.")
        components = []
        for component in model["component_artifacts"]:
            artifact_id = component["artifact_id"]
            expected_sha = component["sha256"]
            required_components.add(artifact_id)
            path = checkpoint_paths.get(artifact_id)
            if path is None or reviewed_components.get(artifact_id) != expected_sha or _sha256(
                path
            ) != expected_sha:
                raise ValueError("AEON external frozen checkpoint bytes differ.")
            components.append(FrozenArtifact(path=path, sha256=expected_sha))
        model_artifacts[model_id] = components
    if set(checkpoint_paths) != required_components or set(reviewed_components) != required_components:
        raise ValueError("AEON external checkpoint inventory differs from six selected seeds.")
    return contract, selection, reports, model_artifacts


def _source_timestamp(row: Mapping[str, str]) -> np.datetime64:
    value = row["Date_M"] + row["Time_M"].strip()
    for pattern in ("%Y%m%d%H:%M:%S.%f", "%Y%m%d%H:%M:%S"):
        try:
            return np.datetime64(datetime.strptime(value, pattern), "us")  # noqa: DTZ007
        except ValueError:
            continue
    raise ValueError("AEON external source timestamp is malformed.")


def _read_numeric_slots(
    archive: Path, source: Mapping[str, Any], expected_report: Mapping[str, Any],
) -> list[AeonHourlySlot]:
    """Open approved CSV rows once after the complete two-source preflight."""
    expected = _expected_members(source)
    archive_sha = source["archive_sha256"]
    if _sha256(archive) != archive_sha:
        raise ValueError("AEON external archive changed after numeric-access preflight.")
    records: dict[int, dict[str, list[tuple[dict[str, str], str]]]] = {}
    metadata_records: dict[int, dict[str, list[dict[str, str]]]] = {}
    with zipfile.ZipFile(archive) as zf:
        aeon_external_metadata._validate_zip(
            zf, expected, max_member_bytes=512 * 1024 * 1024,
            max_total_bytes=8 * 1024 * 1024 * 1024,
            expected_inventory_sha256=source["hourly_full_depth_central_inventory_sha256"],
        )
        for name in expected:
            match = aeon_external_metadata._MEMBER.fullmatch(name)
            assert match is not None
            frequency = match.group(2)
            with io.TextIOWrapper(zf.open(name), encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                if next(reader, None) != list(aeon_external_metadata.EXACT_CSV_HEADER):
                    raise ValueError("AEON external numeric reader header differs from freeze.")
                positions = {field: aeon_external_metadata.EXACT_CSV_HEADER.index(field)
                             for field in aeon_external_metadata.METADATA_FIELDS}
                sv_position = aeon_external_metadata.EXACT_CSV_HEADER.index("Sv_mean")
                for row_count, raw in enumerate(reader, start=1):
                    if len(raw) != len(aeon_external_metadata.EXACT_CSV_HEADER):
                        raise ValueError("AEON external numeric row width differs.")
                    metadata = {field: raw[index] for field, index in positions.items()}
                    sv = raw[sv_position]
                    del raw
                    interval_id = aeon_external_metadata._integer(metadata["Interval"])
                    metadata_records.setdefault(interval_id, {}).setdefault(frequency, []).append(
                        metadata
                    )
                    records.setdefault(interval_id, {}).setdefault(frequency, []).append(
                        (metadata | {"Sv_mean": sv}, name)
                    )
                    if row_count > 10_000:
                        raise ValueError("AEON external numeric monthly row limit exceeded.")
    reconstructed = aeon_external_metadata._candidate_universe(metadata_records, archive_sha)
    for field in (
        "candidate_sha256", "candidate_inventory_sha256", "candidate_rows",
        "source_date_interval_counts", "metadata_exclusion_reasons_by_cutoff",
    ):
        if reconstructed[field] != expected_report.get(field):
            raise ValueError("AEON external numeric rows differ from Stage-1 metadata inventory.")
    slots = []
    previous_time: np.datetime64 | None = None
    for interval_id, channels in sorted(records.items()):
        all_rows = [row for member_rows in channels.values() for row, _ in member_rows]
        representative = channels.get("038", [(all_rows[0], "")])[0][0]
        source_time = _source_timestamp(representative)
        if previous_time is not None and source_time <= previous_time:
            raise ValueError("AEON external source interval times are nonmonotone.")
        previous_time = source_time
        values = np.full(4, np.nan, dtype=np.float64)
        mask = np.zeros(4, dtype=bool)
        statuses = []
        member_names: set[str] = set()
        for index, frequency in enumerate(aeon_external_metadata.FREQUENCIES):
            channel = channels.get(frequency, [])
            member_names.update(name for _, name in channel)
            if not channel:
                statuses.append("MISSING_ROW")
                continue
            if len(channel) != 1:
                statuses.append("DUPLICATE_ROW")
                continue
            row = channel[0][0]
            delta = abs(_source_timestamp(row) - source_time)
            if delta > np.timedelta64(5, "m"):
                statuses.append("SOURCE_TIME_MISMATCH")
                continue
            if (
                aeon_external_metadata._integer(row["Layer"]) != 1
                or aeon_external_metadata._geometry(row["Layer_depth_min"]) != "0"
                or aeon_external_metadata._geometry(row["Layer_depth_max"]) != "200"
            ):
                statuses.append("INVALID_GEOMETRY")
                continue
            if (
                aeon_external_metadata._integer(row["Ping_E"])
                - aeon_external_metadata._integer(row["Ping_S"]) + 1 != 150
            ):
                statuses.append("PARTIAL_SOURCE_INTERVAL")
                continue
            raw_value = row["Sv_mean"].strip()
            try:
                value = float(raw_value) if raw_value else float("nan")
            except ValueError:
                value = float("nan")
            if not np.isfinite(value) or value in SPECIAL_SV:
                statuses.append("INVALID_OR_SPECIAL_SV")
                continue
            values[index] = value
            mask[index] = True
            statuses.append("OBSERVED_SOURCE_PRODUCT")
        slots.append(AeonHourlySlot(
            interval_id=interval_id, source_timestamp=source_time,
            sv_db=values, observed_mask=mask,
            qc_status=tuple(statuses),  # type: ignore[arg-type]
            member_names=tuple(sorted(member_names)), archive_sha256=archive_sha,
        ))
    return slots


def run_external_transfer(
    *,
    contract_path: Path,
    selection_path: Path,
    source_archives: Mapping[str, Path],
    candidate_reports: Mapping[str, Path],
    checkpoint_paths: Mapping[str, Path],
    stage_1_outcome_review_path: Path,
    review_path: Path,
    review_sha256: str,
    output_directory: Path,
    fixture_only: bool = False,
    fixture_forecaster: Callable[[list[Any]], dict[str, NDArray[np.float64]]] | None = None,
) -> dict[str, Any]:
    """Execute both frozen cohorts once and persist every outcome and forecast."""
    if fixture_forecaster is not None and not fixture_only:
        raise ValueError("Injected AEON forecasts are synthetic-fixture only.")
    contract, _, reports, artifacts = preflight_external_access(
        contract_path=contract_path, selection_path=selection_path,
        source_archives=source_archives, candidate_reports=candidate_reports,
        checkpoint_paths=checkpoint_paths, review_path=review_path,
        stage_1_outcome_review_path=stage_1_outcome_review_path,
        review_sha256=review_sha256, output_directory=output_directory,
        fixture_only=fixture_only,
    )
    sources = {item["role"]: item for item in contract["sources"]}
    results: dict[str, Any] = {}
    predictions: dict[str, dict[str, NDArray[np.float64]]] = {}
    issued_windows: dict[str, list[Any]] = {}
    for role in SOURCE_ROLES:
        source = sources[role]
        slots = _read_numeric_slots(source_archives[role], source, reports[role])
        windows, not_issued = materialize_candidates(
            slots, reports[role]["candidate_rows"], source["archive_sha256"]
        )
        issued_windows[role] = windows
        if not windows:
            results[role] = {
                "status": "NO_ISSUED_ROWS_NOT_A_NEGATIVE_TRANSFER_RESULT",
                "candidate_count": len(reports[role]["candidate_rows"]),
                "actual_issued_rows": 0,
                "candidate_not_issued_reasons": not_issued,
            }
            predictions[role] = {}
            continue
        if fixture_forecaster is not None:
            model_predictions = fixture_forecaster(windows)
        else:
            adapter_rows = adapter_compatible_rows(windows)
            model_predictions = {
                model_id: adapt_core_neural_ensemble(
                    adapter_rows, artifacts[model_id], family=family, device="cpu"
                )
                for model_id, family in MODEL_FAMILIES.items()
            }
        if set(model_predictions) != set(MODEL_FAMILIES):
            raise ValueError("AEON external forecast family inventory differs.")
        for forecast in model_predictions.values():
            if (
                forecast.shape != (len(windows), 3, 5)
                or not np.isfinite(forecast).all()
                or (np.diff(forecast, axis=-1) < 0).any()
            ):
                raise ValueError("AEON external model forecast shape or quantiles differ.")
        truth = np.stack([row.target_db for row in windows])
        observed = np.stack([row.target_mask for row in windows])
        times = np.stack([row.target_source_timestamps for row in windows])
        score = score_external_predictions(
            truth, observed, times,
            model_predictions["core_direct_equal_three_seed_ensemble"],
            model_predictions["core_ema_equal_three_seed_ensemble"],
        )
        results[role] = {
            "status": "COMPLETED_ZERO_SHOT_EXTERNAL_TRANSFER"
            if score["cohort_eligible"] else "EXECUTED_COHORT_INELIGIBLE",
            "source_archive_sha256": source["archive_sha256"],
            "candidate_report_sha256": _sha256(candidate_reports[role]),
            "stage_1_candidate_inventory_sha256": reports[role]["candidate_inventory_sha256"],
            "candidate_count": len(reports[role]["candidate_rows"]),
            "actual_issued_rows": len(windows),
            "site": source["site"],
            "deployment": source["deployment"],
            "candidate_not_issued_reasons": not_issued,
            "issued_row_ids": [row.row_id for row in windows],
            "issued_cutoff_interval_ids": [row.cutoff_interval_id for row in windows],
            "target_qc_status_by_row": {row.row_id: list(row.target_qc_status) for row in windows},
            "score": score,
        }
        predictions[role] = model_predictions
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError("AEON external output appeared during evaluation.")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output_directory.name + ".stage.",
                                  dir=output_directory.parent))
    try:
        for role in SOURCE_ROLES:
            cohort = stage / role.lower()
            cohort.mkdir()
            (cohort / "score.json").write_text(
                json.dumps(results[role], indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            if "issued_row_ids" in results[role]:
                with (cohort / "issued-rows.npz").open("xb") as stream:
                    np.savez_compressed(
                        stream,
                        row_ids=np.asarray(results[role]["issued_row_ids"]),
                        target_db=np.stack([row.target_db for row in issued_windows[role]]),
                        target_mask=np.stack([row.target_mask for row in issued_windows[role]]),
                        target_source_timestamps=np.stack([
                            row.target_source_timestamps for row in issued_windows[role]
                        ]),
                    )
            for model_id, forecast in predictions[role].items():
                with (cohort / f"{model_id}.npz").open("xb") as stream:
                    np.savez_compressed(
                        stream, row_ids=np.asarray(results[role]["issued_row_ids"]),
                        quantiles_db=forecast,
                    )
        manifest = {
            "status": "COMPLETED_EXTERNAL_TRANSFER_PENDING_INDEPENDENT_OUTCOME_REVIEW",
            "study_id": contract["study_id"],
            "classification": "POST_HOC_INITIATED_EXTERNAL_TRANSFER_NOT_SEALED",
            "source_roles": list(SOURCE_ROLES),
            "contract_sha256": _sha256(contract_path),
            "selection_sha256": _sha256(selection_path),
            "stage_2_review_sha256": review_sha256,
            "stage_1_outcome_review_sha256": _sha256(stage_1_outcome_review_path),
            "runner_code_sha256": _sha256(Path(__file__)),
            "evaluator_code_sha256": _sha256(Path(aeon_external_evaluator.__file__)),
            "metadata_scanner_code_sha256": _sha256(Path(aeon_external_metadata.__file__)),
            "adapter_composite_sha256": aeon_forecast_adapters.adapter_composite_sha256(),
            "checkpoint_sha256": {
                artifact_id: _sha256(path) for artifact_id, path in checkpoint_paths.items()
            },
            "files": {
                str(path.relative_to(stage)).replace("\\", "/"): _sha256(path)
                for path in stage.rglob("*") if path.is_file()
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(stage, output_directory)
    except BaseException:  # noqa: TRY203
        # Keep the stage for forensic inspection; never replace prior output.
        raise
    return {"manifest": manifest, "cohorts": results}


def main() -> None:
    """Run only with the separately approved real Stage-2 access review."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "contract", "selection", "primary_archive", "secondary_archive",
        "primary_candidates", "secondary_candidates", "stage_1_outcome_review",
        "review", "output", "direct_seed7", "direct_seed13", "direct_seed23",
        "ema_jepa_seed7", "ema_jepa_seed13", "ema_jepa_seed23",
    ):
        parser.add_argument("--" + option.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    arguments = parser.parse_args()
    result = run_external_transfer(
        contract_path=arguments.contract,
        selection_path=arguments.selection,
        source_archives={
            SOURCE_ROLES[0]: arguments.primary_archive,
            SOURCE_ROLES[1]: arguments.secondary_archive,
        },
        candidate_reports={
            SOURCE_ROLES[0]: arguments.primary_candidates,
            SOURCE_ROLES[1]: arguments.secondary_candidates,
        },
        checkpoint_paths={
            name: getattr(arguments, name)
            for name in (
                "direct_seed7", "direct_seed13", "direct_seed23",
                "ema_jepa_seed7", "ema_jepa_seed13", "ema_jepa_seed23",
            )
        },
        stage_1_outcome_review_path=arguments.stage_1_outcome_review,
        review_path=arguments.review,
        review_sha256=arguments.review_sha256,
        output_directory=arguments.output,
    )
    print(json.dumps({
        "status": result["manifest"]["status"],
        "source_roles": result["manifest"]["source_roles"],
        "output_directory": str(arguments.output),
    }, indent=2))


if __name__ == "__main__":
    main()
