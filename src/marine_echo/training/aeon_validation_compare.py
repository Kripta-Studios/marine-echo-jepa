"""Review-gated, neutral comparison of immutable AEON validation predictions.

Input JSON is a prospective artifact manifest, not a source of model scores. It
must list all five source groups, their exact report/review digests, and every
prediction path/digest. A separate reviewer approval binds this code and input
manifest before this module opens prediction arrays. No CAL or TEST reader is
imported here. The output never selects a model or claims inferential success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from marine_echo.evaluation.aeon import daily_pinball, paired_48h_bootstrap
from marine_echo.training.aeon_rescore import _load_prediction, _row_digest, _same_rows

CORE_SLOTS = (
    "persistence", "seasonal_24_source_intervals", "ridge", "hist_gradient_boosting",
    *(f"{family}_seed{seed}" for family in ("direct", "ema_jepa", "shared_sigreg") for seed in (7, 13, 23)),
    "random_encoder_ema_seed7", "random_encoder_shared_sigreg_seed7",
    "temporally_shuffled_pretrain_target_ema_seed7",
    "temporally_shuffled_pretrain_target_shared_sigreg_seed7",
)
HYBRID_SLOTS = (
    *(f"{mode}_pretrain_raw_latent_hgb_seed{seed}" for mode in ("ema", "shared_sigreg") for seed in (7, 13, 23)),
    "random_encoder_ema_raw_latent_hgb_seed7",
    "random_encoder_shared_sigreg_raw_latent_hgb_seed7",
    "shuffled_target_ema_raw_latent_hgb_seed7",
    "shuffled_target_shared_sigreg_raw_latent_hgb_seed7",
)
FORWARD_SLOTS = (
    *(f"forward_ema_seed{seed}" for seed in (7, 13, 23)),
    "random_encoder_seed7", "temporally_shuffled_future_target_seed7",
)
SOURCE_CONTRACTS = {
    "core": ("APPROVED_AEON_VALIDATION_RESCORE_OUTCOME_NO_SELECTION", "rescore_report", set(CORE_SLOTS)),
    "hybrid": ("APPROVED_AEON_HYBRID_OUTCOME_NO_SELECTION", "hybrid_report", set(HYBRID_SLOTS)),
    "lightgbm": ("APPROVED_AEON_POST_HOC_SUPERVISED_OUTCOME_NO_SELECTION", "report", {"post_hoc_lightgbm"}),
    "forward": ("APPROVED_AEON_FORWARD_OUTCOME_NO_SELECTION", "forward_manifest", set(FORWARD_SLOTS)),
    "chronos": ("APPROVED_AEON_CHRONOS2_OUTCOME_NO_SELECTION", "chronos_manifest", {"post_hoc_chronos2"}),
}
ENSEMBLES = {
    "core_direct_equal_three_seed_ensemble": tuple(f"direct_seed{seed}" for seed in (7, 13, 23)),
    "core_ema_equal_three_seed_ensemble": tuple(f"ema_jepa_seed{seed}" for seed in (7, 13, 23)),
    "core_shared_equal_three_seed_ensemble": tuple(f"shared_sigreg_seed{seed}" for seed in (7, 13, 23)),
    "hybrid_ema_equal_three_seed_ensemble": tuple(f"ema_pretrain_raw_latent_hgb_seed{seed}" for seed in (7, 13, 23)),
    "hybrid_shared_equal_three_seed_ensemble": tuple(f"shared_sigreg_pretrain_raw_latent_hgb_seed{seed}" for seed in (7, 13, 23)),
    "forward_ema_equal_three_seed_ensemble": tuple(f"forward_ema_seed{seed}" for seed in (7, 13, 23)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _comparison_code_sha256() -> str:
    """Bind the comparator and the imported scoring and row-integrity contracts."""
    root = Path(__file__).resolve().parents[1]
    dependencies = (
        Path(__file__),
        root / "evaluation/aeon.py",
        Path(__file__).with_name("aeon_rescore.py"),
    )
    digest = hashlib.sha256()
    for path in dependencies:
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _bound_path(spec: dict[str, Any]) -> Path:
    if set(spec) != {"path", "sha256"} or not isinstance(spec["path"], str):
        raise ValueError("Comparison artifact path and digest must be explicit.")
    expected = spec["sha256"]
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Comparison artifact digest is malformed.")
    path = Path(spec["path"]).resolve(strict=True)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError("Comparison artifact digest differs.")
    return path


def _verify_outer_approval(
    review: dict[str, Any], code_sha: str, input_sha: str, source_hashes: dict[str, str]
) -> None:
    if (
        review.get("status") != "APPROVED_AEON_VALIDATION_COMPARISON_EXECUTION"
        or review.get("reviewer_session") != "/root/aeon_reviewer"
        or review.get("comparison_code_sha256") != code_sha
        or review.get("comparison_input_sha256") != input_sha
        or review.get("source_report_sha256") != source_hashes
        or review.get("calibration_access") != "PROHIBITED"
        or review.get("test_access") != "PROHIBITED"
    ):
        raise ValueError("AEON validation comparison lacks exact independent approval.")


def _report_prediction_hash(kind: str, report: dict[str, Any], slot: str) -> str:
    if kind in ("core", "hybrid"):
        return str(report["slots"][slot]["prediction_sha256"])
    if kind == "lightgbm" or kind == "chronos":
        return str(report["prediction_sha256"])
    return str(report["slots"][slot]["prediction_sha256"])


def _report_metric(kind: str, report: dict[str, Any], slot: str) -> dict[str, Any]:
    if kind in ("core", "hybrid"):
        return report["slots"][slot]["protocol_validation_metrics"]
    if kind == "chronos":
        return report["metrics"]
    if kind == "lightgbm":
        return report["protocol_validation_metrics"]
    raise ValueError("Forward slot metric needs its immutable slot report.")


def _source_report_status(kind: str) -> str:
    return {
        "core": "PROTOCOL_VALIDATION_RESCORE_PENDING_INDEPENDENT_SELECTION_REVIEW",
        "hybrid": "COMPLETED_TRAIN_VALIDATION_HYBRID_DEVELOPMENT_PENDING_SELECTION_REVIEW",
        "lightgbm": "POST_HOC_SUPERVISED_VALIDATION_COMPLETED_PENDING_INDEPENDENT_REVIEW",
        "forward": "COMPLETED_POST_HOC_FORWARD_TRAIN_VALIDATION_DEVELOPMENT_NOT_FINAL_EVALUATION",
        "chronos": "COMPLETED_POST_HOC_TRAIN_VALIDATION_ZERO_SHOT_DEVELOPMENT",
    }[kind]


def _reviewed_report_digest(kind: str, review: dict[str, Any], review_key: str) -> Any:
    if kind in ("forward", "chronos"):
        return review.get("manifest_sha256")
    return review.get("artifact_sha256", {}).get(review_key)


def _report_lineage(kind: str, report: dict[str, Any]) -> dict[str, Any]:
    return report.get("binding", {}) if kind == "chronos" else report


def _validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema_version") != "1.0"
        or config.get("study_id") != "aeon3_geb_2024_hourly_sv_v1"
        or config.get("phase") != "validation_comparison_only"
        or config.get("status") != "PROPOSED_FOR_INDEPENDENT_COMPARISON_REVIEW"
        or config.get("calibration_access") != "PROHIBITED"
        or config.get("test_access") != "PROHIBITED"
        or config.get("assessment_partition") != "validation"
        or config.get("validation_source_date_bounds_inclusive") != ["2024-10-08", "2024-11-30"]
        or config.get("validation_issued_rows") != 1219
        or config.get("bootstrap") != {"block_hours": 48, "draws": 2000, "seed": 20260926}
        or not isinstance(config.get("cohort_sha256"), str)
        or len(config["cohort_sha256"]) != 64
        or not isinstance(config.get("validation_row_sha256"), str)
        or len(config["validation_row_sha256"]) != 64
        or not isinstance(config.get("source_archive_sha256"), str)
        or len(config["source_archive_sha256"]) != 64
        or set(config.get("sources", {})) != set(SOURCE_CONTRACTS)
    ):
        raise ValueError("AEON comparison manifest differs from finite validation contract.")
    return config["sources"]


def _load_sources(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    reports: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for kind, source in _validate_manifest(config).items():
        status, review_key, slots = SOURCE_CONTRACTS[kind]
        if set(source) != {"report", "review", "predictions"} or set(source["predictions"]) != slots:
            raise ValueError(f"AEON {kind} source lacks its complete finite slot set.")
        report_path = _bound_path(source["report"])
        review_path = _bound_path(source["review"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        review = json.loads(review_path.read_text(encoding="utf-8"))
        lineage = _report_lineage(kind, report)
        if (
            report.get("status") != _source_report_status(kind)
            or review.get("status") != status
            or review.get("reviewer_session") != "/root/aeon_reviewer"
            or _reviewed_report_digest(kind, review, review_key) != source["report"]["sha256"]
            or review.get("test_access") != "PROHIBITED"
            or report.get("test_access") != "PROHIBITED"
            or (kind != "forward" and lineage.get("validation_row_sha256") != config["validation_row_sha256"])
            or (kind != "core" and lineage.get("cohort_sha256") != config["cohort_sha256"])
            or (kind not in ("core", "forward") and lineage.get("source_archive_sha256") != config["source_archive_sha256"])
        ):
            raise ValueError(f"AEON {kind} outcome lacks exact independent review.")
        report_slots = report.get("slots", {})
        if kind in ("core", "hybrid", "forward") and set(report_slots) != slots and not (
            kind == "hybrid" and set(report_slots) == slots | {"raw_only_hgb"}
        ):
            raise ValueError(f"AEON {kind} report slot set differs.")
        for slot, prediction in source["predictions"].items():
            if prediction.get("sha256") != _report_prediction_hash(kind, report, slot):
                raise ValueError(f"AEON {kind}/{slot} prediction differs from reviewed report.")
            if kind == "forward":
                slot_report = prediction.get("slot_report")
                if (
                    not isinstance(slot_report, dict)
                    or slot_report.get("sha256") != report["slots"][slot]["slot_sha256"]
                ):
                    raise ValueError(f"AEON {kind}/{slot} slot report differs from manifest.")
            elif set(prediction) != {"path", "sha256"}:
                raise ValueError(f"AEON {kind}/{slot} prediction spec differs.")
        reports[kind] = report
        hashes[kind] = source["report"]["sha256"]
    return reports, hashes


def _daily_differences(base: dict[str, Any], candidate: dict[str, Any]) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for base_dates, candidate_dates in zip(
        base["daily_pinball_db_by_horizon_date_quantile"],
        candidate["daily_pinball_db_by_horizon_date_quantile"], strict=True,
    ):
        if [row["source_date"] for row in base_dates] != [row["source_date"] for row in candidate_dates]:
            raise ValueError("AEON compared models differ in eligible source-date support.")
        result.append([
            {
                "source_date": left["source_date"],
                "candidate_minus_reference_db": float(np.mean(right["quantile_losses_db"]) - np.mean(left["quantile_losses_db"])),
            }
            for left, right in zip(base_dates, candidate_dates, strict=True)
        ])
    return result


def _compare_arrays(
    truth: np.ndarray, mask: np.ndarray, source_times: np.ndarray,
    forecasts: dict[str, np.ndarray],
) -> dict[str, Any]:
    if not forecasts or any(array.shape != (len(truth), 3, 5) for array in forecasts.values()):
        raise ValueError("AEON forecasts need identical validation support.")
    scores = {name: daily_pinball(truth, prediction, mask, source_times) for name, prediction in forecasts.items()}
    for name, seeds in ENSEMBLES.items():
        if set(seeds).issubset(forecasts):
            ensemble = np.mean(np.stack([forecasts[seed] for seed in seeds]), axis=0)
            ensemble.sort(axis=-1)
            forecasts[name] = ensemble
            scores[name] = daily_pinball(truth, ensemble, mask, source_times)
    reference_name = "core_direct_equal_three_seed_ensemble"
    if reference_name not in scores:
        raise ValueError("AEON matched direct three-seed reference is incomplete.")
    reference = scores[reference_name]
    support = reference["eligible_source_dates_by_horizon"]
    date_hashes = [
        hashlib.sha256(("\n".join(dates) + "\n").encode("ascii")).hexdigest()
        for dates in support
    ]
    distinct_days = sorted(set().union(*(set(dates) for dates in support)))
    first_day = np.datetime64(min(distinct_days), "D")
    last_day = np.datetime64(max(distinct_days), "D")
    eligible_day_set = {np.datetime64(day, "D") for day in distinct_days}
    calendar_days = int((last_day - first_day) / np.timedelta64(1, "D")) + 1
    nonempty_blocks = sum(
        any(first_day + np.timedelta64(index + offset, "D") in eligible_day_set
            for offset in range(min(2, calendar_days - index)))
        for index in range(0, calendar_days, 2)
    )
    seed_variability: dict[str, Any] = {}
    for ensemble_name, members in ENSEMBLES.items():
        if ensemble_name not in scores:
            continue
        values = np.asarray([scores[member]["primary_daily_mean_pinball_db"] for member in members])
        seed_variability[ensemble_name] = {
            "individual_seed_primary_pinball_db": values.astype(float).tolist(),
            "mean_individual_seed_primary_pinball_db": float(values.mean()),
            "standard_deviation_population_db": float(values.std(ddof=0)),
            "ensemble_primary_pinball_db": scores[ensemble_name]["primary_daily_mean_pinball_db"],
        }
    comparisons: dict[str, Any] = {}
    for name, score in scores.items():
        if score["eligible_source_dates_by_horizon"] != support or score["eligible_scored_rows_per_horizon"] != reference["eligible_scored_rows_per_horizon"]:
            raise ValueError("AEON compared models differ in eligible source-date support.")
        samples = np.zeros(2000) if name == reference_name else paired_48h_bootstrap(
            truth, forecasts[reference_name], forecasts[name], mask, source_times,
        )
        point = float(score["primary_daily_mean_pinball_db"] - reference["primary_daily_mean_pinball_db"])
        comparisons[name] = {
            "protocol_validation_metrics": score,
            "reference": reference_name,
            "point_difference_db": point,
            "point_percent_change_vs_reference": 100 * point / reference["primary_daily_mean_pinball_db"],
            "paired_daily_differences_db_by_horizon": _daily_differences(reference, score),
            "bootstrap_48h": {
                "draws": 2000, "seed": 20260926,
                "candidate_minus_reference_95_percent_interval_db": np.quantile(samples, [0.025, 0.975]).astype(float).tolist(),
            },
        }
    return {
        "model_selection": "NOT_PERFORMED",
        "reference": reference_name,
        "eligible_source_dates_by_horizon": support,
        "eligible_source_date_sha256_by_horizon": date_hashes,
        "distinct_eligible_source_dates": len(distinct_days),
        "nonempty_48h_source_date_blocks": nonempty_blocks,
        "seed_variability": seed_variability,
        "source_clock_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
        "validation_bootstrap_scope": "DESCRIPTIVE_SELECTION_INDUCED_OPTIMISM_NOT_EXTERNAL_GENERALIZATION",
        "comparisons": comparisons,
    }


def compare_validation(input_path: Path, approval_path: Path, output_path: Path) -> dict[str, Any]:
    """Compare reviewed development artifacts only after outer approval; no winner."""
    input_path = input_path.resolve(strict=True)
    input_sha = _sha256(input_path)
    config = json.loads(input_path.read_text(encoding="utf-8"))
    reports, source_hashes = _load_sources(config)
    code_sha = _comparison_code_sha256()
    approval = json.loads(approval_path.resolve(strict=True).read_text(encoding="utf-8"))
    _verify_outer_approval(approval, code_sha, input_sha, source_hashes)
    rows_reference: dict[str, np.ndarray] | None = None
    forecasts: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for kind, source in config["sources"].items():
        for slot, spec in source["predictions"].items():
            path = _bound_path({"path": spec["path"], "sha256": spec["sha256"]})
            rows, forecast = _load_prediction(path)
            if rows_reference is None:
                rows_reference = rows
            elif not _same_rows(rows_reference, rows):
                raise ValueError("AEON compared artifacts differ in validation row identity, truth or mask.")
            if kind == "forward":
                slot_path = _bound_path(spec["slot_report"])
                slot_report = json.loads(slot_path.read_text(encoding="utf-8"))
                if (
                    slot_report.get("slot_id") != slot
                    or slot_report.get("prediction_sha256") != spec["sha256"]
                ):
                    raise ValueError(f"AEON forward/{slot} slot report differs from prediction.")
                metric = slot_report["protocol_validation_metrics"]
            else:
                metric = _report_metric(kind, reports[kind], slot)
            recomputed = daily_pinball(rows["truth_db"], forecast, rows["target_mask"], rows["target_source_timestamps"])
            if metric != recomputed:
                raise ValueError(f"AEON {kind}/{slot} reviewed score differs from saved prediction.")
            name = slot if kind == "core" else f"{kind}:{slot}"
            if name in forecasts:
                raise ValueError("Duplicate AEON comparison candidate.")
            forecasts[name] = forecast
            provenance[name] = {"kind": kind, "source_slot": slot, "prediction_sha256": spec["sha256"], "classification": "DIAGNOSTIC_CONTROL" if slot.startswith(("random_encoder", "temporally_shuffled", "shuffled_target")) else "CORE_DEVELOPMENT" if kind == "core" else "POST_HOC_DEVELOPMENT" if kind in ("lightgbm", "forward", "chronos") else "HYBRID_DEVELOPMENT"}
    assert rows_reference is not None
    if len(rows_reference["row_ids"]) != config["validation_issued_rows"]:
        raise ValueError("AEON issued validation row count differs from reviewed support.")
    lower = np.datetime64(config["validation_source_date_bounds_inclusive"][0], "D")
    upper = np.datetime64(config["validation_source_date_bounds_inclusive"][1], "D")
    cutoffs: np.ndarray = rows_reference["cutoff_source_timestamps"].astype("datetime64[D]")
    targets: np.ndarray = rows_reference["target_source_timestamps"].astype("datetime64[D]")
    observed = rows_reference["target_mask"]
    if (
        np.isnat(cutoffs).any()
        or (cutoffs < lower).any() or (cutoffs > upper).any()
        or (targets[observed] < lower).any() or (targets[observed] > upper).any()
    ):
        raise ValueError("AEON comparison encountered a non-validation source date.")
    reviewed_row_sha = config["validation_row_sha256"]
    if reviewed_row_sha != _row_digest(rows_reference):
        raise ValueError("AEON validation rows differ from reviewed core rescore support.")
    # Normalize names solely for fixed ensemble members; preserve source in provenance.
    for kind in ("hybrid", "forward"):
        for slot in config["sources"][kind]["predictions"]:
            forecasts[slot] = forecasts.pop(f"{kind}:{slot}")
            provenance[slot] = provenance.pop(f"{kind}:{slot}")
    comparison = _compare_arrays(
        rows_reference["truth_db"], rows_reference["target_mask"],
        rows_reference["target_source_timestamps"], forecasts,
    )
    if comparison["eligible_source_date_sha256_by_horizon"] != reports["core"].get(
        "eligible_source_date_sha256_by_horizon"
    ):
        raise ValueError("AEON eligible-date hashes differ from independently reviewed rescore.")
    for ensemble, members in ENSEMBLES.items():
        provenance[ensemble] = {
            "kind": "equal_weight_three_seed_ensemble",
            "members": list(members),
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "classification": "POST_HOC_DEVELOPMENT" if ensemble.startswith("forward_") else "CORE_DEVELOPMENT" if ensemble.startswith("core_") else "HYBRID_DEVELOPMENT",
        }
    report = {
        "status": "NEUTRAL_VALIDATION_COMPARISON_NO_SELECTION",
        "classification": "TRAIN_VALIDATION_DEVELOPMENT_NOT_FINAL_EVALUATION",
        "calibration_access": "PROHIBITED", "test_access": "PROHIBITED",
        "study_id": config["study_id"], "input_sha256": input_sha,
        "approval_sha256": _sha256(approval_path), "code_sha256": code_sha,
        "evaluation_code_sha256": _sha256(Path(daily_pinball.__code__.co_filename)),
        "validation_row_sha256": _row_digest(rows_reference),
        "source_report_sha256": source_hashes,
        "candidate_provenance": provenance,
        **comparison,
    }
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError("AEON neutral comparison output already exists.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent, prefix=output_path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_validation(args.input, args.approval, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
