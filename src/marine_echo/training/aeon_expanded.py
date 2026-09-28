"""Review-gated, post-hoc AEON3 two-deployment TRAIN development study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from torch import nn

from marine_echo.evaluation.aeon import QUANTILES, daily_pinball
from marine_echo.models.aeon_ssl import AeonDirect, AeonTemporalSSL
from marine_echo.training import (
    aeon_campaign,
    aeon_corpus,
    aeon_development,
    aeon_external_evaluator,
    aeon_external_metadata,
    aeon_external_runner,
    aeon_scale,
)
from marine_echo.training.aeon_rescore import _date_hashes
from marine_echo.training.aeon_windows import AeonHourlyWindow

PRIOR_SHA256 = "b6d8380ed986c91d565ea7d669761ff8f66da8f2b5cd3e794fbf71c040bf1d37"
CURRENT_SHA256 = aeon_corpus.AEON_SOURCE_SHA256
_STUDY = "aeon3_geb_expanded_train_3k_development_v1"
_METADATA_SHA256 = "38dd82cdda273c451f894571ba18b75ff3d6ede9b9919f4ee32ce77d8bf46177"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    protocol = Path(__file__).resolve().parents[3] / "docs/adr/0014-aeon-expanded-data-development.md"
    prior = config.get("prior_source", {})
    current = config.get("current_source", {})
    neural = config.get("neural", {})
    if (
        config.get("study_id") != _STUDY
        or config.get("classification") != "POST_HOC_TRAIN_VALIDATION_DEVELOPMENT_NOT_EXTERNAL_EVALUATION"
        or config.get("protocol_path") != "docs/adr/0014-aeon-expanded-data-development.md"
        or config.get("protocol_sha256") != _sha256(protocol)
        or config.get("publisher_article") != "https://figshare.com/articles/dataset/AZFP/29247113"
        or config.get("publisher_article_version") != 2
        or config.get("license") != "CC BY 4.0"
        or prior != {
            "role": "PRIOR_AEON3_GEB_TRAIN_RECLASSIFIED",
            "publisher_file_id": 61937275,
            "filename_serial_identifier": "55146",
            "archive_sha256": PRIOR_SHA256,
            "hourly_member_prefix": "AEON3_55146_",
            "first_month": "2023_02", "last_month": "2024_02",
            "hourly_full_depth_member_count": 52,
            "hourly_full_depth_central_inventory_sha256": "5f775eec41dcf1bd704d101467f0c65469ac1fcfac23d2277fdd9a37436a4323",
            "metadata_report_sha256": _METADATA_SHA256,
        }
        or current != {
            "role": "CURRENT_AEON3_GEB_ORIGINAL_TRAIN_AND_VALIDATION",
            "publisher_file_id": 61937281,
            "filename_serial_identifier": "55144",
            "archive_sha256": CURRENT_SHA256,
        }
        or config.get("fit_partitions") != ["prior_reclassified_train", "original_train"]
        or config.get("assessment_partition") != "original_validation"
        or config.get("calibration_access") != "PROHIBITED"
        or config.get("test_access") != "PROHIBITED"
        or config.get("target") != "source_reported_conditioned_38khz_0_200m_full_depth_Sv_mean"
        or config.get("input") != "24x4_past_full_depth_Sv_mean_plus_observed_mask"
        or config.get("horizon_interval_steps") != [1, 3, 6]
        or config.get("quantiles") != QUANTILES.tolist()
        or config.get("minimum_scored_anchors_per_target_source_date") != 18
        or config.get("sampling") != {
            "source_policy": "natural_pooled_window_proportions",
            "ssl_batch_source_policy": "one_source_per_batch_proportional_to_train_window_count",
            "supervised_batch_source_policy": "pooled_labelled_train_rows",
        }
        or neural != {
            "seeds": [7], "encoder_width": 128, "encoder_layers": 3,
            "batch_size": 64, "direct_supervised_updates": 3000,
            "ssl_pretrain_updates": 1500, "ssl_supervised_updates": 1500,
            "checkpoint_every_updates": 500, "checkpoint_selection": "final_endpoint_only",
            "optimizer": "AdamW", "learning_rate": 0.0003,
            "weight_decay": 0.0001, "gradient_clip_norm": 1.0,
            "ema_teacher_momentum": 0.996, "ema_sigreg_weight": 0.03,
            "normalizer": "joint_train_only_observed_values_and_targets",
            "pretrain_view": "first_18_vs_last_6_of_24_past_source_products",
        }
        or config.get("device_policy") != {
            "preferred": "cuda", "minimum_free_gpu_bytes": 2 * 1024**3,
            "fallback": "cpu", "peak_process_rss_limit_bytes": 22 * 1024**3,
            "peak_gpu_reserved_limit_bytes": 10 * 1024**3, "one_training_process": True,
        }
    ):
        raise ValueError("AEON expanded contract differs from the reviewed 3k plan.")


def _code_sha256() -> str:
    digest = hashlib.sha256()
    files = (
        Path(__file__), Path(aeon_external_runner.__file__),
        Path(aeon_external_metadata.__file__), Path(aeon_external_evaluator.__file__),
        Path(aeon_corpus.__file__), Path(aeon_development.__file__),
        Path(daily_pinball.__code__.co_filename), Path(_date_hashes.__code__.co_filename),
        Path(aeon_scale.__file__),
        Path(__file__).resolve().parents[1] / "models/aeon_ssl.py",
    )
    for path in files:
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    digest.update(aeon_campaign._campaign_code_sha256().encode("ascii"))
    return digest.hexdigest()


def _new_train_id(row: AeonHourlyWindow, source: dict[str, Any], license_name: str) -> str:
    identity = (
        f"aeon-expanded-v1|{source['role']}|{row.source_archive_sha256}|"
        f"{source['publisher_file_id']}|{source['filename_serial_identifier']}|"
        f"{license_name}|{row.cutoff_interval_id}"
    )
    return hashlib.sha256(identity.encode("ascii")).hexdigest()


def _validate_source_rows(
    rows: list[AeonHourlyWindow], *, source_sha: str, serial: str,
    partition: str, start: str, end: str,
) -> None:
    if not rows:
        raise ValueError("AEON expanded source cohort is empty.")
    lower, upper = np.datetime64(start, "us"), np.datetime64(end, "us")
    member_prefix = f"AEON3_{serial}_"
    for row in rows:
        if row.partition != partition:
            raise ValueError("AEON expanded partition differs from source allocation.")
        members = row.past_members + row.target_members
        if not row.past_members or any(not Path(name).name.startswith(member_prefix) for name in members):
            raise ValueError("AEON expanded member serial or deployment differs.")
        if row.source_archive_sha256 != source_sha:
            raise ValueError("AEON expanded source archive differs.")
        if (
            row.context_db.shape != (24, 4)
            or row.context_mask.shape != (24, 4)
            or row.context_source_timestamps.shape != (24,)
            or row.target_db.shape != (3,)
            or row.target_mask.shape != (3,)
            or row.target_source_timestamps.shape != (3,)
            or row.context_mask.dtype.kind != "b"
            or row.target_mask.dtype.kind != "b"
            or not np.array_equal(row.context_interval_ids, np.arange(row.cutoff_interval_id - 23, row.cutoff_interval_id + 1))
            or not np.array_equal(row.target_interval_ids, row.cutoff_interval_id + np.array([1, 3, 6]))
            or not np.array_equal(row.context_mask, np.isfinite(row.context_db))
            or not np.array_equal(row.target_mask, np.isfinite(row.target_db))
            or not row.context_mask[:, 0].all()
        ):
            raise ValueError("AEON expanded past/target geometry or QC mask differs.")
        if (
            row.context_source_timestamps[0] < lower
            or row.context_source_timestamps[-1] != row.cutoff_source_timestamp
            or row.cutoff_source_timestamp >= upper
            or np.any(row.target_source_timestamps[~np.isnat(row.target_source_timestamps)] >= upper)
            or np.any(row.target_source_timestamps[~np.isnat(row.target_source_timestamps)] < lower)
        ):
            raise ValueError("AEON expanded row crosses a deployment or partition boundary.")
        deltas = np.diff(row.context_source_timestamps)
        if np.any(deltas < np.timedelta64(55, "m")) or np.any(deltas > np.timedelta64(65, "m")):
            raise ValueError("AEON expanded past source timestamps are discontinuous.")
        for index, step in enumerate((1, 3, 6)):
            if row.target_mask[index]:
                delta = row.target_source_timestamps[index] - row.cutoff_source_timestamp
                if not np.timedelta64(55 * step, "m") <= delta <= np.timedelta64(65 * step, "m"):
                    raise ValueError("AEON expanded scored target source time is discontinuous.")


def join_training_cohorts(
    prior: list[AeonHourlyWindow], current: list[AeonHourlyWindow],
    validation: list[AeonHourlyWindow], *, config: dict[str, Any] | None = None,
) -> tuple[list[AeonHourlyWindow], list[AeonHourlyWindow], str, dict[str, int]]:
    """Keep each window within its source and preserve original validation rows."""
    _validate_source_rows(prior, source_sha=PRIOR_SHA256, serial="55146", partition="train", start="2023-02-01", end="2024-03-01")
    _validate_source_rows(current, source_sha=CURRENT_SHA256, serial="55144", partition="train", start="2024-03-06", end="2024-10-08")
    _validate_source_rows(validation, source_sha=CURRENT_SHA256, serial="55144", partition="validation", start="2024-10-08", end="2024-12-01")
    prior_spec = {"role": "PRIOR_AEON3_GEB_TRAIN_RECLASSIFIED", "publisher_file_id": 61937275, "filename_serial_identifier": "55146"}
    current_spec = {"role": "CURRENT_AEON3_GEB_ORIGINAL_TRAIN_AND_VALIDATION", "publisher_file_id": 61937281, "filename_serial_identifier": "55144"}
    fit = [replace(row, row_id=_new_train_id(row, prior_spec, "CC BY 4.0")) for row in prior]
    fit.extend(replace(row, row_id=_new_train_id(row, current_spec, "CC BY 4.0")) for row in current)
    assess = list(validation)
    all_rows = fit + assess
    if len({row.row_id for row in all_rows}) != len(all_rows):
        raise ValueError("AEON expanded row identity overlaps fit or validation.")
    fit_keys = {
        (row.source_archive_sha256, int(interval))
        for row in fit for interval in np.r_[row.context_interval_ids, row.target_interval_ids]
    }
    val_keys = {
        (row.source_archive_sha256, int(interval))
        for row in assess for interval in np.r_[row.context_interval_ids, row.target_interval_ids]
    }
    if fit_keys & val_keys:
        raise ValueError("AEON expanded raw interval support overlaps validation.")
    digest = hashlib.sha256()
    if config is not None:
        digest.update(json.dumps({key: config[key] for key in ("prior_source", "current_source", "license")}, sort_keys=True).encode("utf-8"))
    for row in all_rows:
        digest.update(row.row_id.encode("ascii"))
        digest.update(row.partition.encode("ascii"))
        digest.update(row.source_archive_sha256.encode("ascii"))
        for array in (row.context_db, row.context_mask, row.context_interval_ids,
                      row.context_source_timestamps, row.target_interval_ids,
                      row.target_source_timestamps, row.target_db, row.target_mask):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update("\n".join(row.past_members + row.target_members).encode("utf-8"))
    counts = {
        "prior_train_windows": len(prior), "current_train_windows": len(current),
        "validation_windows": len(validation),
    }
    return fit, assess, digest.hexdigest(), counts


def sample_ssl_batch(rng: np.random.Generator, sources: np.ndarray, batch_size: int) -> np.ndarray:
    """Natural source proportions with a source-homogeneous SSL batch."""
    if sources.ndim != 1 or len(sources) < 2 or batch_size < 2:
        raise ValueError("AEON expanded SSL sampling cohort is invalid.")
    unique, counts = np.unique(sources, return_counts=True)
    if len(unique) != 2:
        raise ValueError("AEON expanded SSL needs both TRAIN deployments.")
    source = rng.choice(unique, p=counts / counts.sum())
    pool = np.flatnonzero(sources == source)
    return rng.choice(pool, size=batch_size, replace=len(pool) < batch_size)


def _preaccess_gate(
    config_path: Path, prior_archive: Path, current_archive: Path,
    metadata_report_path: Path, metadata_outcome_review_path: Path,
    split_review_path: Path, cohort_review_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    metadata_review = json.loads(metadata_outcome_review_path.read_text(encoding="utf-8"))
    metadata_report = json.loads(metadata_report_path.read_text(encoding="utf-8"))
    review = json.loads(cohort_review_path.read_text(encoding="utf-8"))
    if (
        _sha256(prior_archive) != PRIOR_SHA256
        or _sha256(current_archive) != CURRENT_SHA256
        or _sha256(metadata_report_path) != _METADATA_SHA256
        or metadata_review.get("status") != "APPROVED_AEON_EXTERNAL_STAGE1_METADATA_OUTCOME"
        or metadata_review.get("sources", {}).get("SECONDARY_PRIOR_YEAR_SAME_SITE", {}).get("candidate_report_sha256") != _METADATA_SHA256
        or metadata_report.get("source", {}).get("archive_sha256") != PRIOR_SHA256
        or metadata_report.get("source_role") != "SECONDARY_PRIOR_YEAR_SAME_SITE"
        or metadata_report.get("candidate_count") != len(metadata_report.get("candidate_rows", []))
        or not isinstance(review.get("reviewer_session"), str)
        or not review["reviewer_session"].startswith("/root/")
        or review["reviewer_session"] == "/root/scale_builder"
        or review.get("configured_model") != "gpt-5.6-sol"
        or review.get("configured_reasoning_effort") != "high"
        or any(review.get(key) != value for key, value in {
            "status": "APPROVED_AEON_EXPANDED_NUMERIC_COHORT_ONLY",
            "study_id": _STUDY,
            "config_sha256": _sha256(config_path),
            "protocol_sha256": config["protocol_sha256"],
            "code_sha256": _code_sha256(),
            "prior_archive_sha256": PRIOR_SHA256,
            "current_archive_sha256": CURRENT_SHA256,
            "metadata_report_sha256": _METADATA_SHA256,
            "metadata_outcome_review_sha256": _sha256(metadata_outcome_review_path),
            "split_review_sha256": _sha256(split_review_path),
            "calibration_access": "PROHIBITED",
            "test_access": "PROHIBITED",
        }.items())
    ):
        raise ValueError("AEON expanded numeric cohort lacks exact independent preaccess approval.")
    return config, metadata_report


def _load_joint_cohort(
    config: dict[str, Any], prior_archive: Path, current_archive: Path,
    metadata_report: dict[str, Any], split_review_path: Path,
) -> tuple[list[AeonHourlyWindow], list[AeonHourlyWindow], str, dict[str, Any]]:
    original_fit, validation, original_digest, _ = aeon_campaign.load_cohort(
        current_archive, split_review_path
    )
    prior_slots = aeon_external_runner._read_numeric_slots(
        prior_archive, config["prior_source"], metadata_report
    )
    prior_external, not_issued = aeon_external_evaluator.materialize_candidates(
        prior_slots, metadata_report["candidate_rows"], PRIOR_SHA256
    )
    prior_train = [replace(row, partition="train") for row in prior_external]
    fit, assess, digest, counts = join_training_cohorts(
        prior_train, original_fit, validation, config=config
    )
    counts.update({
        "prior_source_intervals": len(prior_slots),
        "prior_metadata_candidates": len(metadata_report["candidate_rows"]),
        "prior_numeric_qc_not_issued": len(not_issued),
        "current_train_rejected_candidate_count": "NOT_AVAILABLE_FROM_ORIGINAL_READER_NO_COMPARABLE_CANDIDATE_UNIVERSE",
        "validation_rejected_candidate_count": "NOT_AVAILABLE_FROM_ORIGINAL_READER_NO_COMPARABLE_CANDIDATE_UNIVERSE",
        "prior_issued_with_any_observed_target": int(sum(bool(row.target_mask.any()) for row in prior_train)),
        "current_train_with_any_observed_target": int(sum(bool(row.target_mask.any()) for row in original_fit)),
        "joint_train_windows": len(fit),
    })
    for name, rows in (
        ("prior_train", prior_train), ("current_train", original_fit),
        ("validation", validation),
    ):
        counts[f"{name}_observed_targets_by_horizon"] = np.stack(
            [row.target_mask for row in rows]
        ).sum(axis=0).astype(int).tolist()
        counts[f"{name}_source_dates_with_18_scored_anchors_by_horizon"] = list(
            aeon_external_evaluator.eligible_source_dates(
                np.stack([row.target_source_timestamps for row in rows]),
                np.stack([row.target_mask for row in rows]),
            ).values()
        )
    if len(prior_train) + len(not_issued) != counts["prior_metadata_candidates"]:
        raise ValueError("AEON expanded prior candidate accounting differs.")
    detail = {
        "counts": counts,
        "original_2024_train_validation_cohort_sha256": original_digest,
        "prior_not_issued_reasons": not_issued,
        "prior_train_row_ids_sha256": hashlib.sha256(("\n".join(row.row_id for row in fit[:len(prior_train)]) + "\n").encode("ascii")).hexdigest(),
        "current_train_row_ids_sha256": hashlib.sha256(("\n".join(row.row_id for row in fit[len(prior_train):]) + "\n").encode("ascii")).hexdigest(),
        "validation_original_row_ids_sha256": hashlib.sha256(("\n".join(row.row_id for row in assess) + "\n").encode("ascii")).hexdigest(),
        "validation_raw_interval_support_sha256": hashlib.sha256(
            ("\n".join(str(interval) for interval in sorted({
                int(value) for row in assess for value in np.r_[row.context_interval_ids, row.target_interval_ids]
            })) + "\n").encode("ascii")
        ).hexdigest(),
    }
    return fit, assess, digest, detail


def build_cohort(
    *, config_path: Path, prior_archive: Path, current_archive: Path,
    metadata_report_path: Path, metadata_outcome_review_path: Path,
    split_review_path: Path, cohort_review_path: Path, output: Path,
) -> dict[str, Any]:
    """Persist real QC and row lineage only after distinct numeric preaccess review."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("AEON expanded cohort output already exists.")
    config, metadata_report = _preaccess_gate(
        config_path, prior_archive, current_archive, metadata_report_path,
        metadata_outcome_review_path, split_review_path, cohort_review_path,
    )
    fit, assess, digest, detail = _load_joint_cohort(
        config, prior_archive, current_archive, metadata_report, split_review_path
    )
    report = {
        "status": "NUMERIC_COHORT_BUILT_NO_MODEL_FIT_PENDING_INDEPENDENT_PREFIT_REVIEW",
        "study_id": _STUDY,
        "classification": config["classification"],
        "config_sha256": _sha256(config_path),
        "code_sha256": _code_sha256(),
        "protocol_sha256": config["protocol_sha256"],
        "preaccess_review_sha256": _sha256(cohort_review_path),
        "metadata_report_sha256": _sha256(metadata_report_path),
        "prior_archive_sha256": PRIOR_SHA256,
        "current_archive_sha256": CURRENT_SHA256,
        "cohort_sha256": digest,
        "fit_normalizer_status": "NOT_FIT_COHORT_STAGE",
        "model_status": "NOT_RUN",
        "calibration_access": "PROHIBITED",
        "test_access": "PROHIBITED",
        **detail,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        (staged / "cohort.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(staged, output)
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    # Keep actual arrays transient; only reviewed digests and counts are persisted.
    assert len(fit) == report["counts"]["joint_train_windows"] and len(assess) == report["counts"]["validation_windows"]
    return report


def _forecast(
    model: nn.Module, assess: list[AeonHourlyWindow],
    x_assess: torch.Tensor, m_assess: torch.Tensor,
    scaler: tuple[float, float, float, float], device: str, batch_size: int,
) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for first in range(0, len(assess), batch_size):
            part = slice(first, first + batch_size)
            result.append(
                model(x_assess[part].to(device), m_assess[part].to(device)).cpu().numpy()
                * scaler[3] + scaler[2]
            )
    return np.concatenate(result)


def _score_validation(path: Path, assess: list[AeonHourlyWindow]) -> tuple[dict[str, Any], list[str]]:
    aeon_development._verify_and_score(path, assess)  # schema/row validation; raw score discarded
    with np.load(path, allow_pickle=False) as stored:
        score = daily_pinball(
            stored["truth_db"], stored["quantiles_db"], stored["target_mask"],
            stored["target_source_timestamps"],
        )
    return score, _date_hashes(score)


def _train_slot(
    family: str, fit: list[AeonHourlyWindow], assess: list[AeonHourlyWindow],
    config: dict[str, Any], config_sha: str, cohort_sha: str,
    stage: Path, device: str,
) -> dict[str, Any]:
    """One original-geometry seed-7 fit, with source-homogeneous SSL batches."""
    if family not in ("direct", "ema_jepa") or len(fit) < 2 or not assess:
        raise ValueError("AEON expanded slot or cohort is invalid.")
    neural = config["neural"]
    seed = neural["seeds"][0]
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    try:
        if family == "direct":
            model: nn.Module = AeonDirect(
                width=neural["encoder_width"], layers=neural["encoder_layers"]
            ).to(device)
        else:
            model = AeonTemporalSSL(
                mode="ema", width=neural["encoder_width"],
                layers=neural["encoder_layers"],
                regularizer_weight=neural["ema_sigreg_weight"],
            ).to(device)
        scaler = aeon_development._normalizer(fit)
        x_fit, m_fit, y_fit, y_mask = aeon_development._tensors(fit, scaler)
        x_assess, m_assess = aeon_development._context_tensors(assess, scaler)
        labels = np.asarray([0 if row.source_archive_sha256 == PRIOR_SHA256 else 1 for row in fit])
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("AEON expanded TRAIN lacks both source deployments.")
        candidates = np.flatnonzero(y_mask.any(dim=1).numpy())
        if not len(candidates):
            raise ValueError("AEON expanded TRAIN lacks observed supervised targets.")
        checkpoints: list[dict[str, Any]] = []
        checks: list[dict[str, Any]] = []
        sampled_pretrain = np.zeros(2, dtype=np.int64)
        sampled_supervised = np.zeros(2, dtype=np.int64)
        reference_date_hashes: list[str] | None = None
        diagnostic: dict[str, Any] | None = None

        def guard_resources() -> None:
            nonlocal peak_rss
            peak_rss = max(peak_rss, process.memory_info().rss)
            if peak_rss >= config["device_policy"]["peak_process_rss_limit_bytes"]:
                raise MemoryError("AEON expanded fit reached the 22 GiB process RAM limit.")
            if device == "cuda" and torch.cuda.max_memory_reserved() >= config["device_policy"]["peak_gpu_reserved_limit_bytes"]:
                raise MemoryError("AEON expanded fit reached the 10 GiB GPU reserve limit.")

        def checkpoint(phase: str, step: int, optimizer: torch.optim.Optimizer) -> Path:
            path = stage / f"checkpoint-{phase}-{step}.pt"
            torch.save({
                "study_id": _STUDY, "family": family, "seed": seed,
                "phase": phase, "step": step, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_joint_train_only": scaler,
                "source_archive_sha256": [PRIOR_SHA256, CURRENT_SHA256],
                "cohort_sha256": cohort_sha, "protocol_sha256": config["protocol_sha256"],
                "config_sha256": config_sha,
            }, path)
            checkpoints.append({"phase": phase, "step": step, "path": path.name, "sha256": _sha256(path)})
            return path

        guard_resources()
        pretrain_seconds = 0.0
        if family == "ema_jepa":
            assert isinstance(model, AeonTemporalSSL)
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=neural["learning_rate"], weight_decay=neural["weight_decay"],
            )
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            begin = time.perf_counter()
            for step in range(1, neural["ssl_pretrain_updates"] + 1):
                model.train()
                selected = sample_ssl_batch(rng, labels, neural["batch_size"])
                sampled_pretrain += np.bincount(labels[selected], minlength=2)
                loss = model.pretrain_loss(x_fit[selected].to(device), m_fit[selected].to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("AEON expanded SSL loss is non-finite.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    neural["gradient_clip_norm"], error_if_nonfinite=True,
                )
                optimizer.step()
                model.update_teacher(momentum=neural["ema_teacher_momentum"])
                guard_resources()
                if step % neural["checkpoint_every_updates"] == 0:
                    checkpoint("pretrain", step, optimizer)
            pretrain_seconds = time.perf_counter() - begin
            diagnostic = aeon_campaign._representation_diagnostics(
                model, fit, x_fit, m_fit, device=device
            )
            diagnostic.pop("source_archive_sha256")
            selected = np.linspace(0, len(fit) - 1, min(512, len(fit)), dtype=int)
            diagnostic["source_archive_sha256_by_row"] = [fit[index].source_archive_sha256 for index in selected]
            guard_resources()

        supervised_updates = neural["direct_supervised_updates" if family == "direct" else "ssl_supervised_updates"]
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=neural["learning_rate"], weight_decay=neural["weight_decay"],
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        quantiles = torch.as_tensor(QUANTILES, dtype=torch.float32, device=device)
        begin = time.perf_counter()
        final_prediction: np.ndarray | None = None
        final_checkpoint: Path | None = None
        final_metrics: dict[str, Any] | None = None
        for step in range(1, supervised_updates + 1):
            model.train()
            selected = rng.choice(
                candidates, size=neural["batch_size"],
                replace=len(candidates) < neural["batch_size"],
            )
            sampled_supervised += np.bincount(labels[selected], minlength=2)
            predicted = model(x_fit[selected].to(device), m_fit[selected].to(device))
            truth = y_fit[selected].to(device)
            valid = y_mask[selected].to(device)
            error = truth[:, :, None] - predicted
            loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)[valid].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("AEON expanded supervised loss is non-finite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                neural["gradient_clip_norm"], error_if_nonfinite=True,
            )
            optimizer.step()
            guard_resources()
            if step % neural["checkpoint_every_updates"] == 0:
                saved = checkpoint("supervised", step, optimizer)
                prediction = _forecast(
                    model, assess, x_assess, m_assess, scaler, device, neural["batch_size"]
                )
                path = stage / f"validation-at-supervised-{step}.npz"
                aeon_development._save_predictions(path, assess, prediction)
                score, date_hashes = _score_validation(path, assess)
                if reference_date_hashes is None:
                    reference_date_hashes = date_hashes
                elif date_hashes != reference_date_hashes:
                    raise ValueError("AEON expanded validation eligible-date support changed.")
                checks.append({
                    "step": step, "path": path.name, "sha256": _sha256(path),
                    "protocol_primary_daily_mean_pinball_db": score["primary_daily_mean_pinball_db"],
                    "eligible_source_date_sha256_by_horizon": date_hashes,
                })
                if step == supervised_updates:
                    final_prediction, final_checkpoint, final_metrics = prediction, saved, score
        supervised_seconds = time.perf_counter() - begin
        if final_prediction is None or final_checkpoint is None or final_metrics is None:
            raise ValueError("AEON expanded fit lacks a final supervised endpoint.")
        prediction_path = stage / "validation-predictions.npz"
        aeon_development._save_predictions(prediction_path, assess, final_prediction)
        score, hashes = _score_validation(prediction_path, assess)
        if score != final_metrics or hashes != reference_date_hashes:
            raise ValueError("AEON expanded final prediction differs from scheduled endpoint.")
        result = {
            "run_id": f"{family}_seed{seed}", "family": family, "seed": seed,
            "study_id": _STUDY, "selection": "FINAL_ENDPOINT_ONLY",
            "cohort_sha256": cohort_sha,
            "source_archive_sha256": [PRIOR_SHA256, CURRENT_SHA256],
            "pretrain_updates": neural["ssl_pretrain_updates"] if family == "ema_jepa" else 0,
            "supervised_updates": supervised_updates,
            "prediction_path": prediction_path.name, "prediction_sha256": _sha256(prediction_path),
            "model_path": final_checkpoint.name, "model_sha256": _sha256(final_checkpoint),
            "protocol_validation_metrics": score,
            "primary_daily_mean_pinball_db": score["primary_daily_mean_pinball_db"],
            "eligible_source_date_sha256_by_horizon": hashes,
            "validation_checks": checks, "checkpoints": checkpoints,
            "sampled_pretrain_rows_by_source": {"prior": int(sampled_pretrain[0]), "current": int(sampled_pretrain[1])},
            "sampled_supervised_rows_by_source": {"prior": int(sampled_supervised[0]), "current": int(sampled_supervised[1])},
            "scaler_joint_train_only": scaler,
            "final_pretrain_joint_train_representation_diagnostics": diagnostic,
            "pretrain_seconds": pretrain_seconds,
            "supervised_seconds": supervised_seconds,
            "fit_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "device": device,
        }
        (stage / "slot.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result
    finally:
        torch.set_num_threads(prior_threads)


def _verify_done_slot(output: Path, entry: dict[str, Any]) -> dict[str, Any]:
    aeon_campaign._verify_done_slot(output, entry)
    result = json.loads((output / entry["run_id"] / "slot.json").read_text(encoding="utf-8"))
    if result.get("study_id") != _STUDY or result.get("selection") != "FINAL_ENDPOINT_ONLY":
        raise ValueError("AEON expanded completed slot identity differs.")
    return result


def run_training(
    *, config_path: Path, prior_archive: Path, current_archive: Path,
    metadata_report_path: Path, metadata_outcome_review_path: Path,
    split_review_path: Path, cohort_review_path: Path, train_review_path: Path,
    baseline_output: Path, baseline_rescore_path: Path,
    baseline_rescore_review_path: Path, output: Path,
) -> dict[str, Any]:
    """Fit exactly two seed-7 models after a second cohort-bound review."""
    config, metadata_report = _preaccess_gate(
        config_path, prior_archive, current_archive, metadata_report_path,
        metadata_outcome_review_path, split_review_path, cohort_review_path,
    )
    cohort_path = output / "cohort.json"
    cohort_report = json.loads(cohort_path.read_text(encoding="utf-8"))
    train_review = json.loads(train_review_path.read_text(encoding="utf-8"))
    if (
        cohort_report.get("status") != "NUMERIC_COHORT_BUILT_NO_MODEL_FIT_PENDING_INDEPENDENT_PREFIT_REVIEW"
        or cohort_report.get("study_id") != _STUDY
        or cohort_report.get("classification") != config["classification"]
        or cohort_report.get("config_sha256") != _sha256(config_path)
        or cohort_report.get("protocol_sha256") != config["protocol_sha256"]
        or cohort_report.get("code_sha256") != _code_sha256()
        or cohort_report.get("prior_archive_sha256") != PRIOR_SHA256
        or cohort_report.get("current_archive_sha256") != CURRENT_SHA256
        or cohort_report.get("preaccess_review_sha256") != _sha256(cohort_review_path)
        or cohort_report.get("metadata_report_sha256") != _sha256(metadata_report_path)
        or cohort_report.get("calibration_access") != "PROHIBITED"
        or cohort_report.get("test_access") != "PROHIBITED"
        or not isinstance(train_review.get("reviewer_session"), str)
        or not train_review["reviewer_session"].startswith("/root/")
        or train_review["reviewer_session"] == "/root/scale_builder"
        or train_review.get("configured_model") != "gpt-5.6-sol"
        or train_review.get("configured_reasoning_effort") != "high"
        or any(train_review.get(key) != value for key, value in {
            "status": "APPROVED_AEON_EXPANDED_TRAIN_VALIDATION_ONLY",
            "study_id": _STUDY,
            "config_sha256": _sha256(config_path),
            "protocol_sha256": config["protocol_sha256"],
            "code_sha256": _code_sha256(),
            "cohort_report_sha256": _sha256(cohort_path),
            "cohort_sha256": cohort_report["cohort_sha256"],
            "cohort_preaccess_review_sha256": _sha256(cohort_review_path),
            "baseline_manifest_sha256": _sha256(baseline_output / "manifest.json"),
            "baseline_rescore_sha256": _sha256(baseline_rescore_path),
            "baseline_rescore_outcome_review_sha256": _sha256(baseline_rescore_review_path),
            "calibration_access": "PROHIBITED",
            "test_access": "PROHIBITED",
        }.items())
    ):
        raise ValueError("AEON expanded fit lacks exact independent cohort-bound prefit approval.")
    old_scores, old_dates = aeon_scale._baseline_scores(
        baseline_output, baseline_rescore_path, baseline_rescore_review_path,
        train_review,
    )
    fit, assess, digest, detail = _load_joint_cohort(
        config, prior_archive, current_archive, metadata_report, split_review_path
    )
    if (
        digest != cohort_report["cohort_sha256"]
        or detail != {key: cohort_report[key] for key in detail}
        or json.loads((baseline_output / "manifest.json").read_text(encoding="utf-8")).get("cohort_sha256")
        != detail["original_2024_train_validation_cohort_sha256"]
    ):
        raise ValueError("AEON expanded fit cohort differs from reviewed real QC.")
    device, _ = aeon_campaign._resolve_device(config)
    manifest_path = output / "manifest.json"
    identity = {
        "study_id": _STUDY, "config_sha256": _sha256(config_path),
        "code_sha256": _code_sha256(), "cohort_sha256": digest,
        "cohort_report_sha256": _sha256(cohort_path),
        "train_review_sha256": _sha256(train_review_path),
        "baseline_rescore_sha256": _sha256(baseline_rescore_path),
        "device": device,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("AEON expanded training resume identity differs.")
    else:
        manifest = {
            **identity,
            "status": "TRAIN_VALIDATION_FIT_IN_PROGRESS",
            "slots": {},
            "original_corrected_seed7_validation_primary": old_scores,
            "eligible_source_date_sha256_by_horizon": old_dates,
        }
        aeon_campaign._atomic_json(manifest_path, manifest)
    for family in ("direct", "ema_jepa"):
        run_id = f"{family}_seed7"
        prior = manifest["slots"].get(run_id)
        if prior is not None and prior.get("status") == "DONE":
            result = _verify_done_slot(output, prior)
            if result["eligible_source_date_sha256_by_horizon"] != old_dates:
                raise ValueError("AEON expanded resumed validation support differs.")
            continue
        if (output / run_id).exists():
            raise FileExistsError("AEON expanded slot exists without a completed ledger entry.")
        staged = Path(tempfile.mkdtemp(prefix=run_id + ".stage.", dir=output))
        try:
            result = _train_slot(
                family, fit, assess, config, _sha256(config_path), digest, staged, device
            )
            if result["eligible_source_date_sha256_by_horizon"] != old_dates:
                raise ValueError("AEON expanded validation eligible-date support differs from original 3k.")
            os.replace(staged, output / run_id)
            manifest["slots"][run_id] = {
                "status": "DONE", "run_id": run_id,
                "slot_sha256": _sha256(output / run_id / "slot.json"),
                "primary_daily_mean_pinball_db": result["primary_daily_mean_pinball_db"],
            }
            aeon_campaign._atomic_json(manifest_path, manifest)
        except BaseException:
            if staged.exists() and staged.parent == output:
                shutil.rmtree(staged)
            manifest["slots"][run_id] = {"status": "FAILED", "run_id": run_id}
            aeon_campaign._atomic_json(manifest_path, manifest)
            raise
    manifest["status"] = "COMPLETED_POST_HOC_TRAIN_VALIDATION_PENDING_INDEPENDENT_OUTCOME_REVIEW"
    manifest["relative_validation_loss_reduction_vs_original_3k_seed7"] = {
        family: (old_scores[family] - manifest["slots"][f"{family}_seed7"]["primary_daily_mean_pinball_db"]) / old_scores[family]
        for family in ("direct", "ema_jepa")
    }
    aeon_campaign._atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cohort", "train"), required=True)
    for name in (
        "config", "prior-archive", "current-archive", "metadata-report",
        "metadata-outcome-review", "split-review", "cohort-review", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "train-review", "baseline-output", "baseline-rescore",
        "baseline-rescore-review",
    ):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    common = {
        "config_path": args.config,
        "prior_archive": args.prior_archive,
        "current_archive": args.current_archive,
        "metadata_report_path": args.metadata_report,
        "metadata_outcome_review_path": args.metadata_outcome_review,
        "split_review_path": args.split_review,
        "cohort_review_path": args.cohort_review,
        "output": args.output,
    }
    if args.phase == "cohort":
        result = build_cohort(**common)
    else:
        if any(value is None for value in (
            args.train_review, args.baseline_output, args.baseline_rescore,
            args.baseline_rescore_review,
        )):
            parser.error("Training needs train review and all corrected 3k baseline paths.")
        result = run_training(
            **common,
            train_review_path=args.train_review,
            baseline_output=args.baseline_output,
            baseline_rescore_path=args.baseline_rescore,
            baseline_rescore_review_path=args.baseline_rescore_review,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
