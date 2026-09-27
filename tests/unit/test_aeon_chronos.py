"""Contract tests for the frozen AEON Chronos-2 validation baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training.aeon_chronos import (
    CHRONOS_QUANTILES,
    _code_sha256,
    execute_zero_shot,
    prepare_multivariate_context,
)
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _row(index: int, *, missing_channel: bool = False) -> AeonHourlyWindow:
    cutoff = np.datetime64("2024-10-08T00:00:00", "us") + index * np.timedelta64(1, "h")
    context = np.arange(96, dtype=np.float64).reshape(24, 4) + index
    mask = np.ones((24, 4), dtype=bool)
    if missing_channel:
        mask[5, 2] = False
        context[5, 2] = 12345.0
    targets = cutoff + np.asarray([1, 3, 6]) * np.timedelta64(1, "h")
    return AeonHourlyWindow(
        row_id=f"{index:064x}",
        partition="validation",
        cutoff_source_timestamp=cutoff,
        cutoff_interval_id=480000 + index,
        context_db=context,
        context_mask=mask,
        context_interval_ids=np.arange(479977, 480001, dtype=np.int64) + index,
        context_source_timestamps=cutoff - np.arange(23, -1, -1) * np.timedelta64(1, "h"),
        target_interval_ids=np.asarray([480001, 480003, 480006], dtype=np.int64) + index,
        target_source_timestamps=targets,
        target_db=np.asarray([-70.0, -69.0, -68.0]),
        target_mask=np.ones(3, dtype=bool),
        target_qc_status=("OBSERVED_SOURCE_PRODUCT",) * 3,
        source_archive_sha256="a" * 64,
        past_members=("validation.csv",),
        target_members=("validation.csv",),
    )


class _FakePipeline:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[np.ndarray] = []
        self.fail_on_call = fail_on_call

    def predict_quantiles(
        self, inputs: np.ndarray, **kwargs: object
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self.calls.append(inputs.copy())
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected interruption")
        assert kwargs == {
            "prediction_length": 6,
            "quantile_levels": CHRONOS_QUANTILES.tolist(),
            "batch_size": 8,
            "context_length": 24,
            "cross_learning": False,
            "limit_prediction_length": False,
        }
        predictions = []
        for sample in inputs:
            values = np.empty((4, 6, 5), dtype=np.float64)
            for variate in range(4):
                for step in range(6):
                    values[variate, step] = variate * 100 + step * 10 + np.arange(5)
            predictions.append(values)
        return predictions, [item[..., 2] for item in predictions]


def _config(tmp_path: Path) -> tuple[Path, Path, Path]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    files = {}
    for name, value in {
        ".gitattributes": "fixture lfs attributes\n",
        "README.md": "license: apache-2.0\n",
        "config.json": "{}\n",
        "model.safetensors": "fixture weights\n",
    }.items():
        path = snapshot / name
        path.write_text(value, encoding="utf-8")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {
        "schema_version": "1.0",
        "study_id": "aeon3_geb_2024_hourly_sv_v1",
        "phase": "train_validation_frozen_chronos2_zero_shot",
        "status": "PROPOSED_FOR_INDEPENDENT_PREFIT_REVIEW",
        "source_sha256": "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde",
        "calibration_access": "PROHIBITED_IN_THIS_PHASE",
        "test_access": "PROHIBITED_IN_THIS_PHASE",
        "model": {
            "repo_id": "amazon/chronos-2",
            "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
            "license": "Apache-2.0",
            "package": "chronos-forecasting",
            "package_version": "2.3.2",
            "package_wheel_sha256": "0f0d9a1972f252d6cf584b9fa749bd1b389b3c1cf05a889c4130cb10652c2117",
            "package_sdist_sha256": "910b0891310b74598a937bb9fe8447ea16393bcc2e1ad8d328c1f8099f909201",
            "snapshot_files_sha256": files,
        },
        "input": {
            "layout": "multivariate_4x24",
            "frequencies_hz": [38000, 125000, 200000, 455000],
            "missing_values": "nan_native_observation_mask",
            "target_variate_index": 0,
        },
        "prediction_length": 6,
        "horizon_zero_based_indices": [0, 2, 5],
        "quantiles": CHRONOS_QUANTILES.tolist(),
        "quantile_monotonicity": "sort_each_horizon_five_outputs",
        "cross_learning": False,
        "rows_per_resume_shard": 2,
        "pipeline_series_batch_size": 8,
        "context_length": 24,
        "peak_process_rss_limit_bytes": 22 * 1024**3,
        "peak_gpu_reserved_limit_bytes": 10 * 1024**3,
        "max_gpu_seconds": 7200,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    review = {
        "disposition": "APPROVE_CHRONOS2_TRAIN_VALIDATION_ZERO_SHOT_ONLY",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "implementation_code_sha256": _code_sha256(),
        "model_revision": config["model"]["revision"],
        "source_archive_sha256": config["source_sha256"],
        "calibration_access": "PROHIBITED",
        "test_access": "PROHIBITED",
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return config_path, review_path, snapshot


def test_context_uses_four_variates_and_native_nan_missing_mask() -> None:
    rows = [_row(0, missing_channel=True), _row(1)]
    context = prepare_multivariate_context(rows)
    assert context.shape == (2, 4, 24)
    assert np.isnan(context[0, 2, 5])
    assert context[0, 0, 0] == rows[0].context_db[0, 0]
    assert np.isfinite(context[:, 0]).all()
    assert rows[0].context_db[5, 2] == 12345.0


def test_zero_shot_extracts_fixed_38khz_steps_and_protocol_score(tmp_path: Path) -> None:
    config, review, snapshot = _config(tmp_path)
    rows = [_row(index, missing_channel=index == 0) for index in range(24)]
    pipeline = _FakePipeline()
    result = execute_zero_shot(
        rows,
        output=tmp_path / "run",
        config_path=config,
        review_path=review,
        model_snapshot=snapshot,
        cohort_sha256="b" * 64,
        pipeline=pipeline,
        device="cuda",
    )
    assert len(pipeline.calls) == 12
    assert pipeline.calls[0].shape == (2, 4, 24)
    assert result["status"] == "COMPLETED_TRAIN_VALIDATION_ZERO_SHOT_NOT_FINAL_EVALUATION"
    assert result["metrics"]["eligible_days_per_horizon"] == [1, 1, 1]
    with np.load(tmp_path / "run" / "validation-predictions.npz", allow_pickle=False) as saved:
        assert saved["row_ids"].tolist() == [row.row_id for row in rows]
        assert saved["quantiles_db"].shape == (24, 3, 5)
        np.testing.assert_array_equal(
            saved["quantiles_db"][0],
            np.asarray([[0, 1, 2, 3, 4], [20, 21, 22, 23, 24], [50, 51, 52, 53, 54]]),
        )


def test_interrupted_run_resumes_only_missing_hash_bound_shards(tmp_path: Path) -> None:
    config, review, snapshot = _config(tmp_path)
    rows = [_row(index) for index in range(24)]
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="injected interruption"):
        execute_zero_shot(
            rows,
            output=output,
            config_path=config,
            review_path=review,
            model_snapshot=snapshot,
            cohort_sha256="b" * 64,
            pipeline=_FakePipeline(fail_on_call=2),
            device="cuda",
        )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "IN_PROGRESS"
    assert list(manifest["completed_shards"]) == ["batch-000000.npz"]
    first_elapsed = manifest["cumulative_inference_seconds"]
    first_rss = manifest["peak_process_tree_rss_bytes"]
    assert first_elapsed > 0
    assert first_rss > 0
    resumed = _FakePipeline()
    result = execute_zero_shot(
        rows,
        output=output,
        config_path=config,
        review_path=review,
        model_snapshot=snapshot,
        cohort_sha256="b" * 64,
        pipeline=resumed,
        device="cuda",
    )
    assert result["status"].startswith("COMPLETED")
    assert len(resumed.calls) == 11
    assert result["cumulative_inference_seconds"] >= first_elapsed
    assert result["elapsed_inference_seconds"] == result["cumulative_inference_seconds"]
    assert result["peak_process_tree_rss_bytes"] >= first_rss
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "IN_PROGRESS"
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    recovered = _FakePipeline()
    result = execute_zero_shot(
        rows,
        output=output,
        config_path=config,
        review_path=review,
        model_snapshot=snapshot,
        cohort_sha256="b" * 64,
        pipeline=recovered,
        device="cuda",
    )
    assert result["status"].startswith("COMPLETED")
    assert recovered.calls == []


def test_resume_rejects_mutated_context_even_when_row_id_is_unchanged(tmp_path: Path) -> None:
    config, review, snapshot = _config(tmp_path)
    rows = [_row(index) for index in range(24)]
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="injected interruption"):
        execute_zero_shot(
            rows,
            output=output,
            config_path=config,
            review_path=review,
            model_snapshot=snapshot,
            cohort_sha256="b" * 64,
            pipeline=_FakePipeline(fail_on_call=2),
            device="cuda",
        )
    rows[0].context_db[0, 0] += 1.0
    resumed = _FakePipeline()
    with pytest.raises(ValueError, match="resume binding"):
        execute_zero_shot(
            rows,
            output=output,
            config_path=config,
            review_path=review,
            model_snapshot=snapshot,
            cohort_sha256="b" * 64,
            pipeline=resumed,
            device="cuda",
        )
    assert resumed.calls == []


def test_resume_rejects_reordered_completed_shard_keys(tmp_path: Path) -> None:
    config, review, snapshot = _config(tmp_path)
    rows = [_row(index) for index in range(24)]
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="injected interruption"):
        execute_zero_shot(
            rows,
            output=output,
            config_path=config,
            review_path=review,
            model_snapshot=snapshot,
            cohort_sha256="b" * 64,
            pipeline=_FakePipeline(fail_on_call=3),
            device="cuda",
        )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_shards"] = dict(reversed(list(manifest["completed_shards"].items())))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical prefix"):
        execute_zero_shot(
            rows,
            output=output,
            config_path=config,
            review_path=review,
            model_snapshot=snapshot,
            cohort_sha256="b" * 64,
            pipeline=_FakePipeline(),
            device="cuda",
        )


def test_resume_enforces_cumulative_inference_cap_before_next_batch(tmp_path: Path) -> None:
    config, review, snapshot = _config(tmp_path)
    rows = [_row(index) for index in range(24)]
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="injected interruption"):
        execute_zero_shot(
            rows,
            output=output,
            config_path=config,
            review_path=review,
            model_snapshot=snapshot,
            cohort_sha256="b" * 64,
            pipeline=_FakePipeline(fail_on_call=2),
            device="cuda",
        )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cumulative_inference_seconds"] = 7200.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    resumed = _FakePipeline()
    with pytest.raises(TimeoutError, match="cumulative"):
        execute_zero_shot(
            rows,
            output=output,
            config_path=config,
            review_path=review,
            model_snapshot=snapshot,
            cohort_sha256="b" * 64,
            pipeline=resumed,
            device="cuda",
        )
    assert resumed.calls == []


def test_snapshot_hash_or_review_mismatch_blocks_before_pipeline_call(tmp_path: Path) -> None:
    config, review, snapshot = _config(tmp_path)
    (snapshot / "model.safetensors").write_text("tampered", encoding="utf-8")
    pipeline = _FakePipeline()
    with pytest.raises(ValueError, match="snapshot digest"):
        execute_zero_shot(
            [_row(index) for index in range(24)],
            output=tmp_path / "run",
            config_path=config,
            review_path=review,
            model_snapshot=snapshot,
            cohort_sha256="b" * 64,
            pipeline=pipeline,
            device="cuda",
        )
    assert not pipeline.calls
