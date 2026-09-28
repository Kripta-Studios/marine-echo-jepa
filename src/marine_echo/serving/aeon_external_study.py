"""Bound the reviewed external AEON outcome before offline packaging."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

_STUDY_ID = "aeon_external_transfer_20260928_v1"
_CLASSIFICATION = "POST_HOC_INITIATED_EXTERNAL_TRANSFER_NOT_SEALED"
_REVIEW_RELATIVE = "orchestration/reviews/AEON_EXTERNAL_STAGE2_OUTCOME_REVIEW_20260928.json"
_REVIEW_SHA256 = "ec1ad3ecc437d28f47c53063d6abc6c8f9efe08c36a0f39dec4bda8ef9ad255d"
_OUTPUT_RELATIVE = "outputs/aeon_external_transfer_20260928_v1/zero_shot_secondary"
_PRIMARY = "primary_cross_site_contemporaneous/score.json"
_SECONDARY = "secondary_prior_year_same_site/score.json"
_ISSUED = "secondary_prior_year_same_site/issued-rows.npz"
_DIRECT = "secondary_prior_year_same_site/core_direct_equal_three_seed_ensemble.npz"
_EMA = "secondary_prior_year_same_site/core_ema_equal_three_seed_ensemble.npz"
_OUTPUT_FILES = (_PRIMARY, _SECONDARY, _ISSUED, _DIRECT, _EMA)
_ARTIFACT_FIELDS = {
    _PRIMARY: "primary_score_sha256", _SECONDARY: "secondary_score_sha256",
    _ISSUED: "issued_rows_sha256", _DIRECT: "direct_forecast_sha256",
    _EMA: "ema_forecast_sha256",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bounded(root: Path, relative: str, max_bytes: int) -> Path:
    path = root / relative
    if (
        path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"AEON external evidence path is unsafe: {relative}")
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(
        ValueError("AEON external evidence contains a nonfinite JSON constant.")
    ))
    if not isinstance(value, dict):
        raise ValueError("AEON external evidence is not a JSON object.")
    return value


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("AEON external reviewed metric is not finite.")
    return float(value)


def reviewed_external_files(root: Path) -> dict[str, Path]:
    """Return only the reviewed evidence inventory for provenance copying."""
    output = _OUTPUT_RELATIVE + "/"
    return {
        "outcome-review.json": _bounded(root, _REVIEW_RELATIVE, 128_000),
        "source-contract.json": _bounded(root, "configs/aeon_external_transfer.json", 128_000),
        "manifest.json": _bounded(root, output + "manifest.json", 128_000),
        **{name: _bounded(root, output + name, 8_000_000) for name in _OUTPUT_FILES},
    }


def load_reviewed_external_study(root: Path) -> dict[str, Any]:
    """Read the independently reviewed compact outcome; never open forecast arrays."""
    files = reviewed_external_files(root)
    if _sha256(files["outcome-review.json"]) != _REVIEW_SHA256:
        raise ValueError("AEON external outcome review digest differs from pinned approval.")
    review = _json(files["outcome-review.json"])
    contract = _json(files["source-contract.json"])
    manifest = _json(files["manifest.json"])
    primary = _json(files[_PRIMARY])
    secondary = _json(files[_SECONDARY])
    artifacts = review.get("artifacts", {})
    primary_review = review.get("primary_outcome", {})
    secondary_review = review.get("secondary_outcome", {})
    score = secondary.get("score", {})
    if (
        review.get("study_id") != _STUDY_ID
        or review.get("status") != "APPROVED_AEON_EXTERNAL_STAGE2_OUTCOME"
        or review.get("verdict") != "APPROVE_DESCRIPTIVE_SECONDARY_NEGATIVE_COMPARISON_ONLY"
        or review.get("reviewer_session") != "/root/external_reviewer"
        or review.get("classification") != _CLASSIFICATION
        or review.get("contract_sha256") != _sha256(files["source-contract.json"])
        or review.get("promotion_authorization")
        != "The exact reviewed descriptive outcome may be integrated into the research app and offline release only with the primary NOT_EVALUATED state, secondary negative result and claim limits preserved."
        or artifacts.get("output_directory") != _OUTPUT_RELATIVE
        or artifacts.get("manifest_path") != _OUTPUT_RELATIVE + "/manifest.json"
        or artifacts.get("manifest_sha256") != _sha256(files["manifest.json"])
        or manifest.get("status") != "COMPLETED_EXTERNAL_TRANSFER_PENDING_INDEPENDENT_OUTCOME_REVIEW"
        or manifest.get("study_id") != _STUDY_ID
        or manifest.get("classification") != _CLASSIFICATION
        or manifest.get("output_directory") != _OUTPUT_RELATIVE
        or manifest.get("stage_2_review_sha256") != review.get("retry_review_sha256")
        or manifest.get("stage_1_outcome_review_sha256")
        != review.get("stage_1_outcome_review_sha256")
        or any(manifest.get(key) != review.get(key) for key in (
            "contract_sha256", "selection_sha256", "runner_code_sha256",
            "evaluator_code_sha256", "evaluation_code_sha256", "corpus_code_sha256",
            "adapter_composite_sha256",
        ))
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != set(_OUTPUT_FILES)
        or any(
            artifacts.get(_ARTIFACT_FIELDS[name]) != manifest["files"].get(name)
            or _sha256(files[name]) != manifest["files"][name]
            for name in _OUTPUT_FILES
        )
    ):
        raise ValueError("AEON external manifest or distinct outcome review binding differs.")
    sources = contract.get("sources")
    if (
        contract.get("study_id") != _STUDY_ID
        or contract.get("publisher_article")
        != "https://figshare.com/articles/dataset/AZFP/29247113"
        or contract.get("publisher_article_version") != 2
        or contract.get("license") != "CC BY 4.0"
        or not isinstance(sources, list) or len(sources) != 2
        or {(item.get("role"), item.get("site"), item.get("file_id")) for item in sources}
        != {
            ("PRIMARY_CROSS_SITE_CONTEMPORANEOUS", "AEON2_ECS", 61937269),
            ("SECONDARY_PRIOR_YEAR_SAME_SITE", "AEON3_GEB", 61937275),
        }
        or sources[0].get("site_name") != "AEON2 Eastern Coastal Shelf"
    ):
        raise ValueError("AEON external publisher source attribution differs from reviewed contract.")
    observed = primary_review.get("observed_38khz_geometry_histogram")
    if (
        primary.get("status") != "METADATA_INELIGIBLE_NO_CANDIDATES"
        or primary.get("comparison_role") != "PRIMARY_FIXED_TARGET"
        or not (primary.get("candidate_count") == primary_review.get("candidate_count") == 0)
        or not (primary.get("actual_issued_rows") == primary_review.get("actual_issued_rows") == 0)
        or not (
            primary.get("numeric_sv_access") == primary_review.get("numeric_sv_access")
            == "NOT_RUN_METADATA_INELIGIBLE"
        )
        or not (
            primary.get("jepa_value_gate") == primary_review.get("jepa_value_gate")
            == "NOT_EVALUATED_METADATA_INELIGIBLE"
        )
        or primary.get("stage_1_38khz_geometry_histogram") != observed
        or not (
            primary.get("fixed_target_layer_geometry_m")
            == primary_review.get("frozen_target_geometry") == "0:200"
        )
        or not isinstance(observed, dict) or set(observed) != {"0:230"}
        or not isinstance(observed["0:230"], int) or observed["0:230"] <= 0
        or primary_review.get("cross_site_replication_result") != "ABSENT"
        or primary_review.get("model_result_classification") != "NOT_A_NEGATIVE_MODEL_RESULT"
    ):
        raise ValueError("AEON external primary metadata-ineligible state differs from review.")
    direct = score.get("direct_raw_metrics", {})
    ema = score.get("ema_raw_metrics", {})
    metrics = (
        ("direct_daily_mean_pinball_db", direct.get("primary_daily_mean_pinball_db")),
        ("ema_jepa_daily_mean_pinball_db", ema.get("primary_daily_mean_pinball_db")),
        ("relative_loss_reduction", score.get("relative_loss_reduction")),
    )
    if (
        secondary.get("status") != "COMPLETED_ZERO_SHOT_EXTERNAL_TRANSFER"
        or secondary.get("comparison_role") != "SECONDARY_DESCRIPTIVE_NO_FALLBACK"
        or secondary_review.get("role") != "SECONDARY_PRIOR_YEAR_SAME_SITE_DESCRIPTIVE"
        or secondary.get("candidate_count") != secondary_review.get("metadata_candidates")
        or secondary.get("actual_issued_rows") != secondary_review.get("actual_issued_rows")
        or secondary.get("candidate_not_issued_reasons") != {}
        or score.get("study_partition") != "external_transfer"
        or score.get("source_time_basis") != "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC"
        or score.get("issued_rows") != secondary_review.get("actual_issued_rows")
        or score.get("eligible_dates_per_horizon")
        != secondary_review.get("eligible_source_dates_per_horizon")
        or not (
            direct.get("scored_rows_per_horizon") == ema.get("scored_rows_per_horizon")
            == secondary_review.get("scored_rows_per_horizon")
        )
        or not (
            direct.get("eligible_scored_rows_per_horizon")
            == ema.get("eligible_scored_rows_per_horizon")
            == secondary_review.get("eligible_scored_rows_per_horizon")
        )
        or score.get("paired_95pct_difference_interval_db")
        != secondary_review.get("paired_95pct_ema_minus_direct_interval_db")
        or not (
            score.get("paired_bootstrap_status")
            == secondary_review.get("paired_bootstrap_status")
            == "COMPLETED_2000_DRAWS_SEED_20260928"
        )
        or not (
            score.get("jepa_value_gate") == secondary_review.get("jepa_value_gate")
            == "DESCRIPTIVE_ONLY_NOT_PRIMARY_GATE"
        )
        or any(value != secondary_review.get(key) for key, value in metrics)
    ):
        raise ValueError("AEON external secondary descriptive score differs from review.")
    direct_loss = _finite(direct["primary_daily_mean_pinball_db"])
    ema_loss = _finite(ema["primary_daily_mean_pinball_db"])
    reduction = _finite(score["relative_loss_reduction"])
    interval = score["paired_95pct_difference_interval_db"]
    eligible_dates = score["eligible_dates_per_horizon"]
    if (
        not 0 <= direct_loss < ema_loss
        or reduction >= 0
        or not isinstance(interval, list) or len(interval) != 2
        or not 0 < _finite(interval[0]) <= _finite(interval[1])
        or not isinstance(eligible_dates, list) or len(eligible_dates) != 3
        or any(not isinstance(day, int) or day < 90 for day in eligible_dates)
    ):
        raise ValueError("AEON external secondary result is not the reviewed eligible negative comparison.")
    return {
        "study_id": _STUDY_ID,
        "status": "INDEPENDENTLY_REVIEWED_EXTERNAL_TRANSFER",
        "classification": _CLASSIFICATION,
        "source": {
            "publisher_article": contract["publisher_article"],
            "publisher_article_version": 2,
            "file_ids": [61937269, 61937275],
            "license": "CC BY 4.0",
            "original_archives_bundled": False,
            "acoustic_quantity": "source-reported conditioned Sv_mean",
        },
        "primary": {
            "status": primary["status"], "site": sources[0]["site_name"],
            "candidate_count": 0, "fixed_target_layer_geometry_m": "0:200",
            "observed_layer_geometry_m": "0:230",
            "observed_38khz_rows": observed["0:230"],
            "numeric_sv_access": primary["numeric_sv_access"],
            "jepa_value_gate": primary["jepa_value_gate"],
        },
        "secondary": {
            "status": secondary["status"], "site": secondary["site"],
            "deployment": secondary["deployment"],
            "candidate_count": secondary["candidate_count"],
            "issued_rows": secondary["actual_issued_rows"],
            "eligible_dates_per_horizon": eligible_dates,
            "direct_daily_pinball_db": direct_loss,
            "ema_daily_pinball_db": ema_loss,
            "ema_relative_loss_reduction": reduction,
            "paired_95pct_ema_minus_direct_db": interval,
            "jepa_value_gate": score["jepa_value_gate"],
        },
        "outcome_review_sha256": _REVIEW_SHA256,
        "manifest_sha256": artifacts["manifest_sha256"],
    }
