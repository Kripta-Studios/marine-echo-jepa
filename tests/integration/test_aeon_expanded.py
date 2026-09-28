"""Expanded AEON TRAIN cohort and gate tests without acoustic access."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training import aeon_expanded
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _config() -> dict:
    return json.loads((Path(__file__).resolve().parents[2] / "configs/aeon_expanded_3k.json").read_text())


def _row(source: str, date: str, *, partition: str, serial: str, cutoff: int) -> AeonHourlyWindow:
    t = np.datetime64(date + "T12:00", "us")
    return AeonHourlyWindow(
        row_id=f"{source}:{cutoff}", partition=partition,
        cutoff_source_timestamp=t, cutoff_interval_id=cutoff,
        context_db=np.full((24, 4), -80.0), context_mask=np.ones((24, 4), dtype=bool),
        context_interval_ids=np.arange(cutoff - 23, cutoff + 1),
        context_source_timestamps=t - np.arange(23, -1, -1) * np.timedelta64(1, "h"),
        target_interval_ids=np.array([cutoff + 1, cutoff + 3, cutoff + 6]),
        target_source_timestamps=t + np.array([1, 3, 6]) * np.timedelta64(1, "h"),
        target_db=np.full(3, -79.0), target_mask=np.ones(3, dtype=bool),
        target_qc_status=("OBSERVED_SOURCE_PRODUCT",) * 3,
        source_archive_sha256=source,
        past_members=(f"AEON3_{serial}_2023_06_60minFullDepth.csv",),
        target_members=(f"AEON3_{serial}_2023_06_60minFullDepth.csv",),
    )


def test_expanded_config_is_posthoc_3k_train_validation_only() -> None:
    config = _config()
    aeon_expanded.validate_config(config)
    assert config["neural"]["checkpoint_selection"] == "final_endpoint_only"
    assert config["neural"]["direct_supervised_updates"] == 3000
    assert (config["neural"]["ssl_pretrain_updates"], config["neural"]["ssl_supervised_updates"]) == (1500, 1500)
    config["test_access"] = "ALLOWED"
    with pytest.raises(ValueError, match="expanded contract"):
        aeon_expanded.validate_config(config)
    config = _config()
    config["sampling"]["source_policy"] = "balanced"
    with pytest.raises(ValueError, match="expanded contract"):
        aeon_expanded.validate_config(config)


def test_join_rejects_cross_deployment_member_and_preserves_val() -> None:
    prior = _row(aeon_expanded.PRIOR_SHA256, "2023-06-15", partition="train", serial="55146", cutoff=100)
    current = _row(aeon_expanded.CURRENT_SHA256, "2024-06-15", partition="train", serial="55144", cutoff=200)
    validation = _row(aeon_expanded.CURRENT_SHA256, "2024-10-15", partition="validation", serial="55144", cutoff=300)
    fit, assess, digest, counts = aeon_expanded.join_training_cohorts(
        [prior], [current], [validation]
    )
    assert len(fit) == 2 and assess == [validation]
    assert counts == {"prior_train_windows": 1, "current_train_windows": 1, "validation_windows": 1}
    assert len(digest) == 64
    with pytest.raises(ValueError, match="member"):
        aeon_expanded.join_training_cohorts(
            [replace(prior, past_members=current.past_members)], [current], [validation]
        )
    with pytest.raises(ValueError, match="partition"):
        aeon_expanded.join_training_cohorts(
            [prior], [replace(current, partition="validation")], [validation]
        )
    with pytest.raises(ValueError, match="geometry"):
        aeon_expanded.join_training_cohorts(
            [replace(prior, target_source_timestamps=prior.target_source_timestamps[:2])],
            [current], [validation],
        )
    invalid_target_time = prior.target_source_timestamps.copy()
    invalid_target_time[0] = np.datetime64("2024-03-02T01:00", "us")
    invalid_mask = prior.target_mask.copy()
    invalid_mask[0] = False
    invalid_values = prior.target_db.copy()
    invalid_values[0] = np.nan
    with pytest.raises(ValueError, match="boundary"):
        aeon_expanded.join_training_cohorts(
            [replace(prior, target_source_timestamps=invalid_target_time,
                     target_mask=invalid_mask, target_db=invalid_values)],
            [current], [validation],
        )
    shifted = prior.context_source_timestamps.copy()
    shifted[-2] += np.timedelta64(5, "m") + np.timedelta64(1, "s")
    with pytest.raises(ValueError, match="discontinuous"):
        aeon_expanded.join_training_cohorts(
            [replace(prior, context_source_timestamps=shifted)], [current], [validation]
        )


def test_prior_ssl_batches_are_source_homogeneous() -> None:
    rng = np.random.default_rng(7)
    source = np.array([0] * 8 + [1] * 4)
    counts = np.zeros(2, dtype=int)
    for _ in range(100):
        batch = aeon_expanded.sample_ssl_batch(rng, source, 4)
        assert np.unique(source[batch]).size == 1
        counts[source[batch[0]]] += 1
    assert counts.min() > 0


def test_expanded_validation_score_uses_18_anchor_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        aeon_expanded.aeon_development, "_verify_and_score",
        lambda path, rows: {"non_protocol": True},
    )
    times = np.full((19, 3), np.datetime64("2024-10-15T12:00", "us"))
    times[-1] = np.datetime64("2024-10-16T12:00", "us")
    forecast = np.full((19, 3, 5), -80.0)
    forecast[-1] = -40.0
    path = tmp_path / "validation.npz"
    np.savez_compressed(
        path,
        truth_db=np.full((19, 3), -80.0), quantiles_db=forecast,
        target_mask=np.ones((19, 3), dtype=bool),
        target_source_timestamps=times,
    )
    score, date_hashes = aeon_expanded._score_validation(path, [])
    assert score["primary_daily_mean_pinball_db"] == 0.0
    assert score["eligible_days_per_horizon"] == [1, 1, 1]
    assert len(date_hashes) == 3


def test_fit_rejects_tampered_cohort_report_before_numeric_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs/aeon_expanded_3k.json"
    config = _config()
    monkeypatch.setattr(aeon_expanded, "_preaccess_gate", lambda *args: (config, {}))
    monkeypatch.setattr(
        aeon_expanded, "_load_joint_cohort",
        lambda *args: pytest.fail("numeric cohort was opened before report validation"),
    )
    output = tmp_path / "expanded"
    output.mkdir()
    paths = {name: tmp_path / f"{name}.json" for name in (
        "metadata_report", "metadata_review", "split_review", "cohort_review",
        "baseline_rescore", "baseline_review",
    )}
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    baseline_output = tmp_path / "baseline"
    baseline_output.mkdir()
    (baseline_output / "manifest.json").write_text("{}", encoding="utf-8")
    report = {
        "status": "NUMERIC_COHORT_BUILT_NO_MODEL_FIT_PENDING_INDEPENDENT_PREFIT_REVIEW",
        "study_id": config["study_id"], "classification": config["classification"],
        "config_sha256": aeon_expanded._sha256(config_path),
        "protocol_sha256": config["protocol_sha256"],
        "code_sha256": aeon_expanded._code_sha256(),
        "prior_archive_sha256": "tampered",
        "current_archive_sha256": aeon_expanded.CURRENT_SHA256,
        "preaccess_review_sha256": aeon_expanded._sha256(paths["cohort_review"]),
        "metadata_report_sha256": aeon_expanded._sha256(paths["metadata_report"]),
        "calibration_access": "PROHIBITED", "test_access": "PROHIBITED",
        "cohort_sha256": "c" * 64,
    }
    cohort_path = output / "cohort.json"
    cohort_path.write_text(json.dumps(report), encoding="utf-8")
    review = {
        "status": "APPROVED_AEON_EXPANDED_TRAIN_VALIDATION_ONLY",
        "reviewer_session": "/root/scale_reviewer",
        "configured_model": "gpt-5.6-sol",
        "configured_reasoning_effort": "high",
        "study_id": config["study_id"],
        "config_sha256": aeon_expanded._sha256(config_path),
        "protocol_sha256": config["protocol_sha256"],
        "code_sha256": aeon_expanded._code_sha256(),
        "cohort_report_sha256": aeon_expanded._sha256(cohort_path),
        "cohort_sha256": report["cohort_sha256"],
        "cohort_preaccess_review_sha256": aeon_expanded._sha256(paths["cohort_review"]),
        "baseline_manifest_sha256": aeon_expanded._sha256(baseline_output / "manifest.json"),
        "baseline_rescore_sha256": aeon_expanded._sha256(paths["baseline_rescore"]),
        "baseline_rescore_outcome_review_sha256": aeon_expanded._sha256(paths["baseline_review"]),
        "calibration_access": "PROHIBITED", "test_access": "PROHIBITED",
    }
    train_review = tmp_path / "train_review.json"
    train_review.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort-bound prefit approval"):
        aeon_expanded.run_training(
            config_path=config_path, prior_archive=tmp_path / "prior.zip",
            current_archive=tmp_path / "current.zip",
            metadata_report_path=paths["metadata_report"],
            metadata_outcome_review_path=paths["metadata_review"],
            split_review_path=paths["split_review"],
            cohort_review_path=paths["cohort_review"], train_review_path=train_review,
            baseline_output=baseline_output,
            baseline_rescore_path=paths["baseline_rescore"],
            baseline_rescore_review_path=paths["baseline_review"],
            output=output,
        )
