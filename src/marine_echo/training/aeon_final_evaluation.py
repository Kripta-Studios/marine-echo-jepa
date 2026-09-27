"""Hash-gated AEON interval calibration and retrospective TEST scoring.

This runner consumes forecast-only artifacts produced by separately reviewed
model adapters. It never selects a model and it validates every freeze before
constructing the real CAL/TEST reader.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from marine_echo.evaluation.aeon import (
    apply_interval_widening,
    calibrate_interval_widening,
    daily_pinball,
    paired_48h_bootstrap,
)
from marine_echo.training import aeon_corpus, aeon_evaluation_reader, aeon_windows
from marine_echo.training.aeon_evaluation_reader import AeonEvaluationReader
from marine_echo.training.aeon_forecast_adapters import adapter_composite_sha256
from marine_echo.training.aeon_windows import AeonHourlyWindow

_STUDY = "aeon3_geb_2024_hourly_sv_v1"
_SOURCE_SHA256 = "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde"
_VALIDATION_ROW_SHA256 = "9ce5ed6d60c4082efd3f342b3ee09ddadc5cf6286be6d6a3e953ab0f6166a99f"
_CORRECTED_RESCORE_SHA256 = "32cea9f8141fbad220a3e47d9e840039ad23b4f8e893efbcfb90f52944214c46"
_RESCORE_OUTCOME_REVIEW_SHA256 = "16bb9ef931f5e2d8f8a3322fa609c5cb1c769f99f5e4a89a5ef519a0fb52f44f"
_METADATA_OUTCOME_REVIEW_SHA256 = "9d9c6ca4580d6a4ead4117a00a1a872710bf6e323eb17d491837e24c1011bebe"
_SELECTION_RULE_REVIEW_SHA256 = "dd3dcbdf5e93d42e021abbaf4bf310bec0b40ea52d53411802e00770840127b2"
_NEUTRAL_COMPARISON_SHA256 = "f4a0f4771868866ff7c330f4033c5ca9da7c6c36cdb19a1758cf2f383146757b"
_NEUTRAL_COMPARISON_REVIEW_SHA256 = (
    "53a3a164915eea4ab3bc8851c096bbc42821cb05330eb4c2829039df83df3e07"
)
_DEVELOPMENT_SELECTION_RECORD_REVIEW_SHA256 = (
    "527bf68313cac2354ced78d8cf2673153a92e2f2d08845451ee9afaccfab47c8"
)
_SELECTED_MODEL_SPECS: dict[str, tuple[str, str, tuple[tuple[str, str], ...]]] = {
    "core_direct_equal_three_seed_ensemble": (
        "CORE_CONVENTIONAL_SELECTION",
        "core_neural_ensemble",
        (
            ("direct_seed7", "a616f9160f359e54a6da5d87dc3b7a46edfee46e449e2944af7a69be8ae25c95"),
            ("direct_seed13", "eb6ca62f4c68024b43ea8b2102a0754d646623fd91cbf1ad31fb72218faf48dc"),
            ("direct_seed23", "ee3ae09ebf05bc41ec762e946a1d0046be33fff4ea47472a992893a294054761"),
        ),
    ),
    "core_ema_equal_three_seed_ensemble": (
        "CORE_JEPA_SELECTION",
        "core_neural_ensemble",
        (
            ("ema_jepa_seed7", "f1fc41860e21cb2c47868488b051c68cd951487d717fc0fe033e63b8dfcc664c"),
            ("ema_jepa_seed13", "59cb3d1e6d79ca2d9807f24c5e816d0e5c89fad0e274e703d6e9da5b44024633"),
            ("ema_jepa_seed23", "4a7c63931a31a48f60323b944110df9292856f39cb822ce0b891371a10a16a15"),
        ),
    ),
    "post_hoc_lightgbm": (
        "POST_HOC_DEVELOPMENT_SELECTION",
        "lightgbm",
        (
            ("lightgbm_model", "164389731cd5d2e0694d1fb5228196196e7701fc73269e79ef84d85520fb168b"),
            ("lightgbm_recipe", "3feee4089ce790c66adb189ff82ad4e006c350822a23aaa494e86afaef884eea"),
        ),
    ),
}
_HEX = set("0123456789abcdef")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_TRUSTED_REVIEWER_SESSION = "/root/aeon_reviewer"
_LIGHTGBM_RECIPE_SHA256 = "3feee4089ce790c66adb189ff82ad4e006c350822a23aaa494e86afaef884eea"
_CALIBRATION_OUTCOME_REVIEW_SHA256 = (
    "7b61385b29836be53d0e2c5a0b5e3f5db7e2dcb8983e534d17711ed0b44c1428"
)


class _WindowReader(Protocol):
    def iter_windows(self) -> Any: ...


class _FixtureForecaster(Protocol):
    def __call__(self, rows: list[AeonHourlyWindow]) -> dict[str, np.ndarray]: ...


def artifact_sha256(path: Path) -> str:
    """Return the byte-level SHA-256 used by every runner gate."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reader_composite_sha256() -> str:
    """Bind the reader plus the corpus parser and window/issuance implementation."""
    digest = hashlib.sha256()
    for module in (aeon_evaluation_reader, aeon_corpus, aeon_windows):
        module_file = module.__file__
        if module_file is None:
            raise RuntimeError("AEON reader dependency lacks a filesystem source path.")
        path = Path(module_file).resolve(strict=True)
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _json(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    path = path.resolve(strict=True)
    actual = artifact_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"AEON artifact digest differs: {path.name}.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"AEON JSON contract is not an object: {path.name}.")
    return value, actual


def _config(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _json(path)
    expected = {
        "schema_version": "1.0",
        "study_id": _STUDY,
        "minimum_eligible_dates": {"calibration": 12, "test": 20},
        "minimum_anchors_per_date": 18,
        "horizons_hours": [1, 3, 6],
        "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "incremental_loss_gate": 0.05,
        "per_horizon_regression_guard": 0.10,
    }
    if value != expected:
        raise ValueError("AEON final-evaluation config differs from the frozen protocol.")
    return value, digest


def _validate_adapter_options(model_id: str, adapter: str, options: object) -> None:
    if adapter == "SYNTHETIC_FIXTURE":
        if options != {}:
            raise ValueError("Synthetic AEON adapter options must be empty.")
        return
    if not isinstance(options, dict):
        raise TypeError("AEON selected-model adapter options are malformed.")
    expected_family = {
        "core_direct_equal_three_seed_ensemble": "direct",
        "core_ema_equal_three_seed_ensemble": "ema_jepa",
    }.get(model_id)
    if expected_family is not None:
        if set(options) != {"family", "device"} or options.get("family") != expected_family:
            raise ValueError("AEON neural adapter options differ from the reviewed model family.")
        device = options.get("device")
        if device not in ("cpu", "cuda"):
            raise ValueError("AEON neural adapter device must be explicit CPU or CUDA.")
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ValueError("AEON neural adapter requested unavailable CUDA.")
        return
    if model_id == "post_hoc_lightgbm":
        if set(options) != {"recipe_sha256"} or options.get("recipe_sha256") != (
            _LIGHTGBM_RECIPE_SHA256
        ):
            raise ValueError("AEON LightGBM options differ from the reviewed recipe.")
        return
    raise ValueError("AEON adapter options refer to an unreviewed model identity.")


def _selection(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _json(path)
    models = value.get("models")
    if (
        set(value) != {
            "schema_version", "status", "study_id", "test_access",
            "selection_rule_review_sha256",
            "neutral_comparison_sha256", "neutral_comparison_outcome_review_sha256",
            "development_selection_record_review_sha256",
            "adapter_composite_sha256",
            "corrected_validation_rescore_sha256",
            "corrected_validation_outcome_review_sha256", "validation_row_sha256", "models",
        }
        or
        value.get("schema_version") != "1.0"
        or value.get("status") != "FROZEN_AEON_MODEL_SELECTION"
        or value.get("study_id") != _STUDY
        or value.get("test_access") != "PROHIBITED"
        or value.get("selection_rule_review_sha256") != _SELECTION_RULE_REVIEW_SHA256
        or value.get("neutral_comparison_sha256") != _NEUTRAL_COMPARISON_SHA256
        or value.get("neutral_comparison_outcome_review_sha256")
        != _NEUTRAL_COMPARISON_REVIEW_SHA256
        or value.get("development_selection_record_review_sha256")
        != _DEVELOPMENT_SELECTION_RECORD_REVIEW_SHA256
        or value.get("adapter_composite_sha256") != adapter_composite_sha256()
        or value.get("validation_row_sha256") != _VALIDATION_ROW_SHA256
        or value.get("corrected_validation_rescore_sha256") != _CORRECTED_RESCORE_SHA256
        or value.get("corrected_validation_outcome_review_sha256") != _RESCORE_OUTCOME_REVIEW_SHA256
        or not isinstance(models, list)
        or len(models) < 2
    ):
        raise ValueError("AEON model-selection freeze is incomplete or unreviewed.")
    ids: set[str] = set()
    roles: list[str] = []
    for model in models:
        if isinstance(model, dict) and model.get("adapter") == "hybrid_raw_latent_hgb_ensemble":
            raise ValueError(
                "AEON raw-plus-latent hybrid selection is unsupported; freeze its exact three "
                "head models and three representation checkpoints before evaluation."
            )
        if (
            not isinstance(model, dict)
            or set(model) != {
                "model_id", "role", "selection_classification", "adapter",
                "component_artifacts", "ensemble_weights", "adapter_options",
            }
            or not isinstance(model.get("model_id"), str)
            or not model["model_id"]
            or _MODEL_ID.fullmatch(model["model_id"]) is None
            or model["model_id"] in ids
            or model.get("role") not in ("baseline", "candidate")
            or model.get("selection_classification") not in (
                "CORE_CONVENTIONAL_SELECTION", "CORE_JEPA_SELECTION",
                "POST_HOC_DEVELOPMENT_SELECTION",
            )
            or model.get("adapter") not in {
                "conventional", "core_neural_ensemble", "lightgbm",
                "forward_ema_ensemble", "chronos2", "SYNTHETIC_FIXTURE",
            }
            or not isinstance(model.get("component_artifacts"), list)
            or not model["component_artifacts"]
            or any(
                not isinstance(item, dict)
                or set(item) != {"artifact_id", "sha256"}
                or not isinstance(item.get("artifact_id"), str)
                or not _digest(item.get("sha256"))
                for item in model["component_artifacts"]
            )
            or not isinstance(model.get("ensemble_weights"), list)
            or not all(isinstance(weight, (int, float)) for weight in model["ensemble_weights"])
        ):
            raise ValueError("AEON selected-model identity or artifact binding is invalid.")
        artifact_ids = [item["artifact_id"] for item in model["component_artifacts"]]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("AEON selected model repeats a component artifact identity.")
        expected_weights = (
            [1 / 3, 1 / 3, 1 / 3]
            if model["adapter"] in ("core_neural_ensemble", "forward_ema_ensemble")
            else [1.0]
        )
        if model["ensemble_weights"] != expected_weights:
            raise ValueError("AEON selected-model ensemble weights differ from the frozen rule.")
        _validate_adapter_options(model["model_id"], model["adapter"], model["adapter_options"])
        ids.add(model["model_id"])
        roles.append(model["role"])
    core_conventional = [
        model for model in models
        if model["selection_classification"] == "CORE_CONVENTIONAL_SELECTION"
    ]
    core_jepa = [
        model for model in models if model["selection_classification"] == "CORE_JEPA_SELECTION"
    ]
    post_hoc = [
        model for model in models
        if model["selection_classification"] == "POST_HOC_DEVELOPMENT_SELECTION"
    ]
    if (
        len(core_conventional) != 1
        or core_conventional[0]["role"] != "baseline"
        or not core_jepa
        or any(model["role"] != "candidate" for model in core_jepa + post_hoc)
        or roles.count("baseline") != 1
    ):
        raise ValueError(
            "AEON selection needs exactly one core conventional baseline, at least one "
            "core JEPA candidate, and optional post-hoc candidates."
        )
    if not all(model["adapter"] == "SYNTHETIC_FIXTURE" for model in models):
        selected_specs = {
            model["model_id"]: (
                model["selection_classification"],
                model["adapter"],
                tuple(
                    (artifact["artifact_id"], artifact["sha256"])
                    for artifact in model["component_artifacts"]
                ),
            )
            for model in models
        }
        if selected_specs != _SELECTED_MODEL_SPECS:
            raise ValueError(
                "AEON selected component artifacts differ from the reviewed selection rule."
            )
    return value, digest


def _forecast_plan(
    path: Path, partition: str, selection: dict[str, Any], selection_sha256: str,
    candidate_sha256: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """Validate a pre-access adapter plan containing no issued-row forecasts."""
    value, digest = _json(path)
    entries = value.get("models")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("AEON TEST forecast adapter plan model list is malformed.")
    typed_entries = cast(list[dict[str, Any]], entries)
    expected = [
        (
            model["model_id"], model["role"], model["selection_classification"],
            model["adapter"],
            [(item["artifact_id"], item["sha256"]) for item in model["component_artifacts"]],
            model["ensemble_weights"],
            model["adapter_options"],
        )
        for model in selection["models"]
    ]
    actual = [
        (
            entry.get("model_id"), entry.get("role"), entry.get("selection_classification"),
            entry.get("adapter"),
            [
                (item.get("artifact_id"), item.get("sha256"))
                for item in entry.get("component_artifacts", [])
                if isinstance(item, dict)
            ],
            entry.get("ensemble_weights"),
            entry.get("options"),
        )
        for entry in typed_entries
    ]
    allowed = {
        "conventional", "core_neural_ensemble", "lightgbm",
        "forward_ema_ensemble", "chronos2", "SYNTHETIC_FIXTURE",
    }
    if (
        value.get("schema_version") != "1.0"
        or value.get("status") != f"FROZEN_AEON_{partition.upper()}_FORECAST_ADAPTER_PLAN"
        or value.get("study_id") != _STUDY
        or value.get("partition") != partition
        or value.get("selection_freeze_sha256") != selection_sha256
        or value.get("candidate_contract_sha256") != candidate_sha256
        or value.get("adapter_composite_sha256") != adapter_composite_sha256()
        or actual != expected
        or any(entry.get("adapter") not in allowed for entry in typed_entries)
    ):
        raise ValueError("AEON forecast adapter plan differs from the frozen selection.")
    return typed_entries, digest


def _test_forecast_plan(
    path: Path, selection: dict[str, Any], selection_sha256: str, candidate_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    """Compatibility wrapper for the frozen TEST adapter-plan validator."""
    return _forecast_plan(path, "test", selection, selection_sha256, candidate_sha256)


def _plan_artifact(directory: Path, value: dict[str, Any]) -> Any:
    from marine_echo.training.aeon_forecast_adapters import FrozenArtifact

    if set(value) != {"artifact_id", "path", "sha256"}:
        raise ValueError("AEON adapter-plan artifact reference is malformed.")
    path = Path(value["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("AEON adapter-plan artifact escapes its directory.")
    resolved_directory = directory.resolve(strict=True)
    resolved = (resolved_directory / path).resolve(strict=True)
    if not resolved.is_relative_to(resolved_directory):
        raise ValueError("AEON adapter-plan artifact escapes its directory.")
    return FrozenArtifact(resolved, value["sha256"])


def _preflight_forecast_plan(
    entries: list[dict[str, Any]], directory: Path, *, fixture: bool,
) -> None:
    """Validate all model bytes and options before opening a numeric partition."""
    if fixture:
        if any(entry.get("adapter") != "SYNTHETIC_FIXTURE" for entry in entries):
            raise ValueError("Synthetic AEON review contains a real adapter plan.")
        return
    from marine_echo.training.aeon_forecast_adapters import preflight_chronos_snapshot

    for entry in entries:
        adapter = entry["adapter"]
        options = entry.get("options")
        components = entry.get("component_artifacts")
        if not isinstance(options, dict) or not isinstance(components, list):
            raise TypeError("AEON adapter plan lacks options or component artifacts.")
        _validate_adapter_options(entry["model_id"], adapter, options)
        artifacts = [_plan_artifact(directory, item) for item in components]
        if any(artifact_sha256(item.path) != item.sha256 for item in artifacts):
            raise ValueError("AEON adapter-plan component digest differs.")
        family = options.get("family")
        if adapter == "conventional":
            valid = len(artifacts) == 1 and isinstance(family, str)
        elif adapter == "core_neural_ensemble":
            valid = len(artifacts) == 3 and isinstance(family, str)
        elif adapter == "lightgbm":
            valid = (
                len(artifacts) == 2
                and options.get("recipe_sha256") == components[1].get("sha256")
            )
        elif adapter == "forward_ema_ensemble":
            valid = len(artifacts) == 3 and isinstance(options.get("config_sha256"), str)
        elif adapter == "chronos2" and not artifacts:
            snapshot = Path(options.get("snapshot", ""))
            if snapshot.is_absolute() or ".." in snapshot.parts:
                raise ValueError("Chronos snapshot escapes the adapter-plan directory.")
            preflight_chronos_snapshot(
                directory / snapshot, options.get("snapshot_files_sha256")
            )
            valid = True
        else:
            valid = False
        if not valid:
            raise ValueError(f"Unsupported or malformed AEON adapter plan: {adapter}.")


def _execute_forecast_plan(
    entries: list[dict[str, Any]], rows: list[AeonHourlyWindow], directory: Path,
    *, fixture: bool, fixture_forecaster: _FixtureForecaster | None,
) -> dict[str, np.ndarray]:
    if fixture_forecaster is not None:
        if not fixture or any(entry.get("adapter") != "SYNTHETIC_FIXTURE" for entry in entries):
            raise ValueError("Injected AEON forecasting is permitted for synthetic fixtures only.")
        result = fixture_forecaster(rows)
    else:
        if fixture:
            raise ValueError("Synthetic AEON TEST execution requires its fixture forecaster.")
        from marine_echo.training import aeon_forecast_adapters
        from marine_echo.training.aeon_forecast_adapters import (
            adapt_chronos2,
            adapt_conventional,
            adapt_core_neural_ensemble,
            adapt_forward_ensemble,
            adapt_lightgbm,
        )

        adapter_sha256 = aeon_forecast_adapters.adapter_composite_sha256()

        result = {}
        for entry in entries:
            if adapter_sha256 != adapter_composite_sha256():
                raise ValueError("AEON frozen adapter code digest differs at execution.")
            adapter = entry["adapter"]
            options = entry.get("options")
            components = entry.get("component_artifacts")
            if not isinstance(options, dict) or not isinstance(components, list):
                raise TypeError("AEON adapter plan lacks options or component artifacts.")
            artifacts = [_plan_artifact(directory, item) for item in components]
            family = options.get("family")
            recipe_sha256 = options.get("recipe_sha256")
            config_sha256 = options.get("config_sha256")
            if adapter == "conventional" and len(artifacts) == 1:
                if not isinstance(family, str):
                    raise ValueError("AEON conventional adapter family is malformed.")
                prediction = adapt_conventional(
                    rows, artifacts[0], family=family
                )
            elif adapter == "core_neural_ensemble":
                if not isinstance(family, str):
                    raise ValueError("AEON core-neural adapter family is malformed.")
                prediction = adapt_core_neural_ensemble(
                    rows, artifacts, family=family, device=options.get("device", "cpu")
                )
            elif adapter == "lightgbm" and len(artifacts) == 2:
                if (
                    not isinstance(recipe_sha256, str)
                    or recipe_sha256 != components[1].get("sha256")
                ):
                    raise ValueError("AEON LightGBM recipe digest is malformed.")
                prediction = adapt_lightgbm(
                    rows, artifacts[0], recipe_sha256=recipe_sha256
                )
            elif adapter == "forward_ema_ensemble":
                if not isinstance(config_sha256, str):
                    raise ValueError("AEON forward config digest is malformed.")
                prediction = adapt_forward_ensemble(
                    rows, artifacts, config_sha256=config_sha256,
                    device=options.get("device", "cpu"),
                )
            elif adapter == "chronos2" and not artifacts:
                snapshot = Path(options.get("snapshot", ""))
                if snapshot.is_absolute() or ".." in snapshot.parts:
                    raise ValueError("Chronos snapshot escapes the adapter-plan directory.")
                prediction = adapt_chronos2(
                    rows, snapshot=(directory / snapshot),
                    snapshot_files_sha256=options.get("snapshot_files_sha256"),
                    device=options.get("device", "cuda"),
                )
            else:
                raise ValueError(f"Unsupported or malformed AEON adapter plan: {adapter}.")
            result[entry["model_id"]] = prediction
    expected_ids = [entry["model_id"] for entry in entries]
    if list(result) != expected_ids:
        raise ValueError("AEON forecast adapter output model order differs from the frozen plan.")
    for prediction in result.values():
        if (
            prediction.shape != (len(rows), 3, 5)
            or not np.isfinite(prediction).all()
            or (np.diff(prediction, axis=-1) < 0).any()
        ):
            raise ValueError("AEON forecast adapter returned invalid quantiles.")
    return result


def _review(
    path: Path,
    expected_sha256: str,
    partition: str,
    bindings: dict[str, str],
) -> tuple[dict[str, Any], bool]:
    value, _ = _json(path, expected_sha256)
    fixture = value.get("data_kind") == "SYNTHETIC_FIXTURE"
    expected_status = {
        ("calibration", True): "APPROVED_AEON_CALIBRATION_RUNNER_FIXTURE",
        ("test", True): "APPROVED_AEON_RETROSPECTIVE_TEST_RUNNER_FIXTURE",
        ("calibration", False): "APPROVED_AEON_CALIBRATION_ACCESS",
        ("test", False): "APPROVED_AEON_RETROSPECTIVE_TEST_ACCESS",
    }[(partition, fixture)]
    code_bindings = {
        "runner_code_sha256": artifact_sha256(Path(__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_composite_sha256": reader_composite_sha256(),
        "adapter_composite_sha256": adapter_composite_sha256(),
    }
    expected_access = {
        ("calibration", True): "CALIBRATION_FIXTURE_ONLY",
        ("test", True): "RETROSPECTIVE_TEST_FIXTURE_ONLY",
        ("calibration", False): "CALIBRATION_NUMERIC_ACCESS_APPROVED",
        ("test", False): "RETROSPECTIVE_TEST_NUMERIC_ACCESS_APPROVED",
    }[(partition, fixture)]
    if (
        value.get("status") != expected_status
        or value.get("partition") != partition
        or value.get("reviewer_session") != _TRUSTED_REVIEWER_SESSION
        or value.get("partition_access") != expected_access
        or any(value.get(key) != digest for key, digest in bindings.items())
        or any(value.get(key) != digest for key, digest in code_bindings.items())
    ):
        raise ValueError("AEON runner review bindings differ from the frozen artifacts.")
    return value, fixture


def _reader_review_gate(
    path: Path, expected_sha256: str, partition: str, fixture: bool,
) -> None:
    value, _ = _json(path, expected_sha256)
    expected_access = {
        ("calibration", True): "CALIBRATION_FIXTURE_ONLY",
        ("test", True): "RETROSPECTIVE_TEST_FIXTURE_ONLY",
        ("calibration", False): "CALIBRATION_NUMERIC_ACCESS_APPROVED",
        ("test", False): "RETROSPECTIVE_TEST_NUMERIC_ACCESS_APPROVED",
    }[(partition, fixture)]
    if (
        value.get("partition") != partition
        or value.get("partition_access") != expected_access
        or value.get("reviewer_session") != _TRUSTED_REVIEWER_SESSION
    ):
        raise ValueError("AEON reader review lacks trusted partition-specific approval.")


def _reader(
    *, archive: Path, review_path: Path, review_sha256: str, partition: str,
    fixture: bool, fixture_reader: _WindowReader | None,
) -> _WindowReader:
    if fixture_reader is not None:
        if not fixture:
            raise ValueError("An injected AEON reader is permitted for synthetic fixtures only.")
        return fixture_reader
    return AeonEvaluationReader(
        archive, review_path=review_path, review_sha256=review_sha256,
        partition=partition, fixture_only=fixture,
    )


def _rows(reader: _WindowReader, partition: str) -> list[AeonHourlyWindow]:
    rows = list(reader.iter_windows())
    ids = [row.row_id for row in rows]
    cutoffs = [row.cutoff_source_timestamp for row in rows]
    if (
        not rows
        or any(row.partition != partition for row in rows)
        or len(set(ids)) != len(ids)
        or any(right <= left for left, right in itertools.pairwise(cutoffs))
    ):
        raise ValueError("AEON issued rows are empty, duplicated, reordered or cross-partition.")
    return rows


def _targets(rows: list[AeonHourlyWindow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([row.target_db for row in rows]),
        np.stack([row.target_mask for row in rows]),
        np.stack([row.target_source_timestamps for row in rows]),
    )


def _require_eligible_floor(metrics: dict[str, object], minimum: int, partition: str) -> None:
    days = metrics.get("eligible_days_per_horizon")
    if (
        not isinstance(days, list)
        or len(days) != 3
        or not all(isinstance(value, int) for value in days)
        or min(days) < minimum
    ):
        raise ValueError(
            f"AEON {partition} has fewer than {minimum} eligible dates per horizon."
        )


def _write_once(output: Path, filename: str, result: dict[str, Any]) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        target = output / filename
        if not target.is_file() or json.loads(target.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing AEON final-evaluation output differs.")
        return result
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        (stage / filename).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    except BaseException:
        if stage.exists():
            for child in stage.iterdir():
                child.unlink()
            stage.rmdir()
        raise
    return result


def execute_calibration(
    *, archive: Path, reader_review_path: Path, reader_review_sha256: str,
    runner_review_path: Path, runner_review_sha256: str, config_path: Path,
    selection_freeze_path: Path, forecast_manifest_path: Path, output: Path,
    fixture_reader: _WindowReader | None = None,
    fixture_forecaster: _FixtureForecaster | None = None,
) -> dict[str, Any]:
    """Fit only nonnegative CAL interval widening for every frozen model."""
    _, config_sha = _config(config_path)
    selection, selection_sha = _selection(selection_freeze_path)
    forecast_plan, manifest_sha = _forecast_plan(
        forecast_manifest_path, "calibration", selection, selection_sha, None
    )
    _, fixture = _review(runner_review_path, runner_review_sha256, "calibration", {
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "reader_review_sha256": reader_review_sha256,
    })
    if all(model["adapter"] == "SYNTHETIC_FIXTURE" for model in selection["models"]) != fixture:
        raise ValueError("AEON selection fixture mode differs from the reviewed data access mode.")
    _reader_review_gate(
        reader_review_path, reader_review_sha256, "calibration", fixture,
    )
    _preflight_forecast_plan(forecast_plan, forecast_manifest_path.parent, fixture=fixture)
    reader = _reader(
        archive=archive, review_path=reader_review_path, review_sha256=reader_review_sha256,
        partition="calibration", fixture=fixture, fixture_reader=fixture_reader,
    )
    rows = _rows(reader, "calibration")
    generated = _execute_forecast_plan(
        forecast_plan, rows, forecast_manifest_path.parent,
        fixture=fixture, fixture_forecaster=fixture_forecaster,
    )
    truth, observed, times = _targets(rows)
    result: dict[str, Any] = {
        "status": "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING",
        "study_id": _STUDY,
        "partition": "calibration",
        "test_access": "PROHIBITED",
        "source_archive_sha256": _SOURCE_SHA256,
        "selection_freeze_sha256": selection_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "runner_review_sha256": runner_review_sha256,
        "runner_code_sha256": artifact_sha256(Path(__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_composite_sha256": reader_composite_sha256(),
        "adapter_composite_sha256": adapter_composite_sha256(),
        "reader_review_sha256": reader_review_sha256,
        "issued_row_ids": [row.row_id for row in rows],
        "models": {},
    }
    for model in selection["models"]:
        model_id = model["model_id"]
        prediction = generated[model_id]
        calibration = calibrate_interval_widening(
            truth, prediction, observed, times, partition="calibration"
        )
        _require_eligible_floor(calibration, 12, "calibration")
        widened = apply_interval_widening(prediction, np.asarray(calibration["adjustment_db"]))
        result["models"][model_id] = {
            **calibration,
            "selection_classification": model["selection_classification"],
            "raw_interval_metrics": daily_pinball(truth, prediction, observed, times),
            "widened_interval_metrics": daily_pinball(truth, widened, observed, times),
        }
    return _write_once(output, "calibration.json", result)


def _candidate(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _json(path)
    ids = value.get("candidate_cutoff_interval_ids")
    rows = value.get("candidate_rows")
    if (
        value.get("status") != "METADATA_CANDIDATE_UNIVERSE_PENDING_INDEPENDENT_REVIEW"
        or value.get("classification") != "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED"
        or value.get("numeric_test_outcome_access") != "PROHIBITED"
        or value.get("source_archive_sha256") != _SOURCE_SHA256
        or value.get("partition_start_source_date_inclusive") != "2025-01-06"
        or value.get("partition_end_source_date_exclusive") != "2025-03-01"
        or value.get("actual_issued_rows") != "UNKNOWN_NUMERIC_QC_NOT_OPENED"
        or value.get("actual_scored_rows") != "UNKNOWN_NUMERIC_QC_NOT_OPENED"
        or not isinstance(ids, list)
        or not ids
        or len(ids) != len(set(ids))
        or not all(isinstance(item, int) for item in ids)
        or not isinstance(rows, list)
        or len(rows) != len(ids)
        or [row.get("cutoff_interval_id") for row in rows if isinstance(row, dict)] != ids
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("cutoff_source_timestamp"), str)
            or not _digest(row.get("row_id"))
            or row.get("numeric_issuance_status") != "UNKNOWN"
            or row.get("target_scoring_status") != "UNKNOWN"
            for row in rows
        )
    ):
        raise ValueError("AEON metadata-only TEST candidate contract is invalid.")
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("AEON metadata-only TEST candidate row IDs are duplicated.")
    return value, digest


def _pretest_freeze(
    path: Path, *, candidate_sha: str, selection_sha: str, calibration_sha: str,
    config_sha: str, forecast_manifest_sha: str,
) -> tuple[dict[str, Any], str]:
    value, digest = _json(path)
    expected = {
        "schema_version": "1.0",
        "status": "FROZEN_AEON_RETROSPECTIVE_TEST_PREACCESS",
        "study_id": _STUDY,
        "source_archive_sha256": _SOURCE_SHA256,
        "metadata_candidate_report_sha256": candidate_sha,
        "metadata_candidate_review_sha256": _METADATA_OUTCOME_REVIEW_SHA256,
        "selection_freeze_sha256": selection_sha,
        "calibration_artifact_sha256": calibration_sha,
        "calibration_outcome_review_sha256": _CALIBRATION_OUTCOME_REVIEW_SHA256,
        "config_sha256": config_sha,
        "forecast_manifest_sha256": forecast_manifest_sha,
        "runner_code_sha256": artifact_sha256(Path(__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_composite_sha256": reader_composite_sha256(),
        "adapter_composite_sha256": adapter_composite_sha256(),
        "issued_row_rule": "EXACT_24_PRIOR_INTERVAL_IDS_OBSERVED_38KHZ",
        "primary_metric": "RAW_FIVE_QUANTILE_ELIGIBLE_TARGET_DATE_PINBALL",
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "test_access": "PROHIBITED_PENDING_INDEPENDENT_APPROVAL",
    }
    if value != expected:
        raise ValueError("AEON pretest freeze differs from the one-way reviewed dependency graph.")
    return value, digest


def _write_test_bundle(
    output: Path, rows: list[AeonHourlyWindow], forecasts: dict[str, np.ndarray],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Atomically preserve truth-free forecasts and the retrospective score."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError("AEON retrospective TEST output already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        materialized = []
        expected_ids = np.asarray([row.row_id for row in rows])
        for model_id, prediction in forecasts.items():
            path = stage / f"{model_id}-forecast.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, row_ids=expected_ids, quantiles_db=prediction)
            digest = artifact_sha256(path)
            result["models"][model_id]["forecast_artifact_sha256"] = digest
            materialized.append({
                "model_id": model_id, "artifact": path.name, "artifact_sha256": digest,
            })
        manifest = {
            "status": "MATERIALIZED_AEON_TEST_FORECASTS_AFTER_SINGLE_REVIEWED_READ",
            "forecast_plan_sha256": result["forecast_manifest_sha256"],
            "reader_review_sha256": result["reader_review_sha256"],
            "models": materialized,
        }
        (stage / "materialized-forecast-manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (stage / "test-score.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    except BaseException:
        if stage.exists():
            for child in stage.iterdir():
                child.unlink()
            stage.rmdir()
        raise
    return result


def _comparison_gates(
    *, baseline_primary: float, candidate_primary: float,
    baseline_horizons: np.ndarray, candidate_horizons: np.ndarray,
    paired_interval: list[float],
) -> dict[str, bool]:
    point_pass = candidate_primary <= 0.95 * baseline_primary
    interval_pass = paired_interval[1] < 0.0
    horizon_pass = bool(np.all(candidate_horizons <= 1.10 * baseline_horizons))
    return {
        "passes_prespecified_five_percent_point_improvement": point_pass,
        "passes_paired_95_percent_interval_strictly_favoring_candidate": interval_pass,
        "passes_incremental_loss_gate": point_pass and interval_pass,
        "passes_per_horizon_ten_percent_regression_guard": horizon_pass,
        "passes_full_unnarrowed_promotion_rule": point_pass and interval_pass and horizon_pass,
    }


def _validate_calibration_artifact(
    value: dict[str, Any], selection: dict[str, Any], selection_sha: str, config_sha: str,
) -> None:
    models = value.get("models")
    expected_model_ids = [model["model_id"] for model in selection["models"]]
    if (
        set(value) != {
            "status", "study_id", "partition", "test_access", "source_archive_sha256",
            "selection_freeze_sha256", "forecast_manifest_sha256", "config_sha256",
            "runner_review_sha256", "runner_code_sha256", "evaluation_code_sha256",
            "reader_composite_sha256", "adapter_composite_sha256", "reader_review_sha256",
            "issued_row_ids", "models",
        }
        or value.get("status") != "COMPLETED_AEON_CALIBRATION_INTERVAL_WIDENING"
        or value.get("study_id") != _STUDY
        or value.get("partition") != "calibration"
        or value.get("test_access") != "PROHIBITED"
        or value.get("source_archive_sha256") != _SOURCE_SHA256
        or value.get("selection_freeze_sha256") != selection_sha
        or value.get("config_sha256") != config_sha
        or not _digest(value.get("forecast_manifest_sha256"))
        or not _digest(value.get("runner_review_sha256"))
        or value.get("runner_code_sha256") != artifact_sha256(Path(__file__))
        or value.get("evaluation_code_sha256")
        != artifact_sha256(Path(daily_pinball.__code__.co_filename))
        or value.get("reader_composite_sha256") != reader_composite_sha256()
        or value.get("adapter_composite_sha256") != adapter_composite_sha256()
        or not _digest(value.get("reader_review_sha256"))
        or not isinstance(value.get("issued_row_ids"), list)
        or not value["issued_row_ids"]
        or any(not _digest(row_id) for row_id in value["issued_row_ids"])
        or len(value["issued_row_ids"]) != len(set(value["issued_row_ids"]))
        or not isinstance(models, dict)
        or list(models) != expected_model_ids
    ):
        raise ValueError("AEON calibration artifact lineage or selected model set differs.")
    classifications = {
        model["model_id"]: model["selection_classification"] for model in selection["models"]
    }
    for model_id, model in models.items():
        if (
            not isinstance(model, dict)
            or set(model) != {
                "adjustment_db", "eligible_days_per_horizon",
                "eligible_rows_per_horizon", "all_scored_days_per_horizon",
                "eligible_source_dates_by_horizon", "selection_classification",
                "raw_interval_metrics", "widened_interval_metrics",
            }
            or model.get("selection_classification") != classifications[model_id]
            or not isinstance(model.get("raw_interval_metrics"), dict)
            or not isinstance(model.get("widened_interval_metrics"), dict)
        ):
            raise ValueError("AEON calibration model schema differs.")
        adjustment = np.asarray(model.get("adjustment_db"), dtype=np.float64)
        days = model.get("eligible_days_per_horizon")
        eligible_rows = model.get("eligible_rows_per_horizon")
        all_days = model.get("all_scored_days_per_horizon")
        source_dates = model.get("eligible_source_dates_by_horizon")
        if (
            adjustment.shape != (3,)
            or not np.isfinite(adjustment).all()
            or (adjustment < 0).any()
            or not isinstance(days, list)
            or len(days) != 3
            or any(not isinstance(day, int) or day < 12 for day in days)
            or not isinstance(all_days, list)
            or len(all_days) != 3
            or any(not isinstance(day, int) or day < 12 for day in all_days)
            or not isinstance(source_dates, list)
            or len(source_dates) != 3
            or any(not isinstance(items, list) or len(items) < 12 for items in source_dates)
            or not isinstance(eligible_rows, list)
            or len(eligible_rows) != 3
            or any(not isinstance(count, int) or count < 1 for count in eligible_rows)
        ):
            raise ValueError("AEON calibration adjustment or eligibility evidence is invalid.")


def execute_retrospective_test(
    *, archive: Path, reader_review_path: Path, reader_review_sha256: str,
    runner_review_path: Path, runner_review_sha256: str, config_path: Path,
    selection_freeze_path: Path, candidate_contract_path: Path,
    calibration_artifact_path: Path, pretest_freeze_path: Path,
    forecast_manifest_path: Path, output: Path,
    fixture_reader: _WindowReader | None = None,
    fixture_forecaster: _FixtureForecaster | None = None,
) -> dict[str, Any]:
    """Open TEST once, forecast issued rows in-process, then score those same rows."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError("AEON retrospective TEST output already exists.")
    _, config_sha = _config(config_path)
    selection, selection_sha = _selection(selection_freeze_path)
    calibration, calibration_sha = _json(calibration_artifact_path)
    _validate_calibration_artifact(calibration, selection, selection_sha, config_sha)
    candidate, candidate_sha = _candidate(candidate_contract_path)
    forecast_plan, manifest_sha = _test_forecast_plan(
        forecast_manifest_path, selection, selection_sha, candidate_sha
    )
    _, pretest_sha = _pretest_freeze(
        pretest_freeze_path, candidate_sha=candidate_sha, selection_sha=selection_sha,
        calibration_sha=calibration_sha, config_sha=config_sha,
        forecast_manifest_sha=manifest_sha,
    )
    _, fixture = _review(runner_review_path, runner_review_sha256, "test", {
        "selection_freeze_sha256": selection_sha,
        "config_sha256": config_sha,
        "calibration_artifact_sha256": calibration_sha,
        "pretest_freeze_sha256": pretest_sha,
        "forecast_manifest_sha256": manifest_sha,
        "reader_review_sha256": reader_review_sha256,
    })
    if all(model["adapter"] == "SYNTHETIC_FIXTURE" for model in selection["models"]) != fixture:
        raise ValueError("AEON selection fixture mode differs from the reviewed data access mode.")
    _reader_review_gate(
        reader_review_path, reader_review_sha256, "test", fixture,
    )
    _preflight_forecast_plan(forecast_plan, forecast_manifest_path.parent, fixture=fixture)
    reader = _reader(
        archive=archive, review_path=reader_review_path, review_sha256=reader_review_sha256,
        partition="test", fixture=fixture, fixture_reader=fixture_reader,
    )
    rows = _rows(reader, "test")
    candidate_pairs = [
        (row["cutoff_interval_id"], row["cutoff_source_timestamp"], row["row_id"])
        for row in candidate["candidate_rows"]
    ]
    frozen = set(candidate_pairs)
    issued = {
        (row.cutoff_interval_id, str(row.cutoff_source_timestamp), row.row_id) for row in rows
    }
    if not issued <= frozen:
        raise ValueError("AEON issued row falls outside the metadata-only candidate universe.")
    generated = _execute_forecast_plan(
        forecast_plan, rows, forecast_manifest_path.parent,
        fixture=fixture, fixture_forecaster=fixture_forecaster,
    )
    truth, observed, times = _targets(rows)
    model_results: dict[str, Any] = {}
    raw: dict[str, np.ndarray] = {}
    baseline_id = next(model["model_id"] for model in selection["models"] if model["role"] == "baseline")
    for model in selection["models"]:
        model_id = model["model_id"]
        prediction = generated[model_id]
        model_calibration = calibration["models"].get(model_id)
        if not isinstance(model_calibration, dict):
            raise TypeError("AEON calibration is missing a frozen selected model.")
        adjustment = np.asarray(model_calibration.get("adjustment_db"), dtype=np.float64)
        widened = apply_interval_widening(prediction, adjustment)
        raw_metrics = daily_pinball(truth, prediction, observed, times)
        _require_eligible_floor(raw_metrics, 20, "TEST")
        raw[model_id] = prediction
        model_results[model_id] = {
            "selection_classification": model["selection_classification"],
            "raw_metrics": raw_metrics,
            "widened_interval_metrics": daily_pinball(truth, widened, observed, times),
        }
    baseline_metrics = model_results[baseline_id]["raw_metrics"]
    comparisons: dict[str, Any] = {}
    for model in selection["models"]:
        model_id = model["model_id"]
        if model_id == baseline_id:
            continue
        boot = paired_48h_bootstrap(truth, raw[baseline_id], raw[model_id], observed, times)
        base_h = np.asarray(baseline_metrics["daily_mean_pinball_db_per_horizon"])
        candidate_h = np.asarray(model_results[model_id]["raw_metrics"]["daily_mean_pinball_db_per_horizon"])
        base_primary = float(baseline_metrics["primary_daily_mean_pinball_db"])
        candidate_primary = float(model_results[model_id]["raw_metrics"]["primary_daily_mean_pinball_db"])
        relative_horizon = [
            float(candidate_value / base_value - 1.0) if base_value > 0 else None
            for base_value, candidate_value in zip(base_h, candidate_h, strict=True)
        ]
        daily_differences = []
        base_daily = baseline_metrics["daily_pinball_db_by_horizon_date_quantile"]
        candidate_daily = model_results[model_id]["raw_metrics"][
            "daily_pinball_db_by_horizon_date_quantile"
        ]
        for horizon, (base_rows, candidate_rows) in zip(
            (1, 3, 6), zip(base_daily, candidate_daily, strict=True), strict=True
        ):
            if [row["source_date"] for row in base_rows] != [
                row["source_date"] for row in candidate_rows
            ]:
                raise ValueError("AEON compared models differ in eligible TEST date support.")
            daily_differences.append({
                "horizon_hours": horizon,
                "candidate_minus_baseline_pinball_db": [
                    {
                        "source_date": base_row["source_date"],
                        "difference_db": float(np.mean(candidate_row["quantile_losses_db"]) - np.mean(base_row["quantile_losses_db"])),
                    }
                    for base_row, candidate_row in zip(base_rows, candidate_rows, strict=True)
                ],
            })
        bounds = np.asarray(np.quantile(boot, [0.025, 0.975]), dtype=np.float64).reshape(2)
        interval = [float(bounds[0]), float(bounds[1])]
        gates = _comparison_gates(
            baseline_primary=base_primary, candidate_primary=candidate_primary,
            baseline_horizons=base_h, candidate_horizons=candidate_h,
            paired_interval=interval,
        )
        comparisons[model_id] = {
            "baseline_model_id": baseline_id,
            "selection_classification": model["selection_classification"],
            "primary_relative_loss_change": (
                candidate_primary / base_primary - 1.0 if base_primary > 0 else None
            ),
            "relative_loss_change_per_horizon": relative_horizon,
            **gates,
            "daily_paired_differences": daily_differences,
            "paired_95_percent_interval_db": interval,
            "bootstrap_candidate_minus_baseline_db": boot.tolist(),
        }
    result: dict[str, Any] = {
        "status": "COMPLETED_AEON_RETROSPECTIVE_TEST_SCORING",
        "classification": "RETROSPECTIVE_EVALUATION_NOT_SEALED",
        "study_id": _STUDY,
        "partition": "test",
        "selection_freeze_sha256": selection_sha,
        "candidate_contract_sha256": candidate_sha,
        "calibration_artifact_sha256": calibration_sha,
        "pretest_freeze_sha256": pretest_sha,
        "forecast_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "runner_review_sha256": runner_review_sha256,
        "runner_code_sha256": artifact_sha256(Path(__file__)),
        "evaluation_code_sha256": artifact_sha256(Path(daily_pinball.__code__.co_filename)),
        "reader_composite_sha256": reader_composite_sha256(),
        "adapter_composite_sha256": adapter_composite_sha256(),
        "reader_review_sha256": reader_review_sha256,
        "issued_row_ids": [row.row_id for row in rows],
        "scored_target_interval_ids_by_row": [
            [int(value) if row.target_mask[index] else None
             for index, value in enumerate(row.target_interval_ids)] for row in rows
        ],
        "target_qc_status_by_row": [list(row.target_qc_status) for row in rows],
        "candidate_not_issued": [
            {"cutoff_interval_id": interval, "cutoff_source_timestamp": timestamp,
             "row_id": row_id,
             "reason": "FAILED_FROZEN_24_PRIOR_38KHZ_ISSUANCE_RULE"}
            for interval, timestamp, row_id in candidate_pairs
            if (interval, timestamp, row_id) not in issued
        ],
        "models": model_results,
        "comparisons": comparisons,
    }
    return _write_test_bundle(output, rows, raw, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "score-test"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reader-review", type=Path, required=True)
    parser.add_argument("--reader-review-sha256", required=True)
    parser.add_argument("--runner-review", type=Path, required=True)
    parser.add_argument("--runner-review-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path, required=True)
    parser.add_argument("--forecast-manifest", type=Path, required=True)
    parser.add_argument("--candidate-contract", type=Path)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--pretest-freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common = {
        "archive": args.archive, "reader_review_path": args.reader_review,
        "reader_review_sha256": args.reader_review_sha256,
        "runner_review_path": args.runner_review,
        "runner_review_sha256": args.runner_review_sha256,
        "config_path": args.config, "selection_freeze_path": args.selection_freeze,
        "forecast_manifest_path": args.forecast_manifest, "output": args.output,
    }
    if args.stage == "calibrate":
        if (
            args.candidate_contract is not None
            or args.calibration_artifact is not None
            or args.pretest_freeze is not None
        ):
            parser.error("calibrate does not accept TEST-only artifacts")
        result = execute_calibration(**common)
    else:
        if (
            args.candidate_contract is None
            or args.calibration_artifact is None
            or args.pretest_freeze is None
        ):
            parser.error(
                "score-test requires --candidate-contract, --calibration-artifact and "
                "--pretest-freeze"
            )
        result = execute_retrospective_test(
            **common, candidate_contract_path=args.candidate_contract,
            calibration_artifact_path=args.calibration_artifact,
            pretest_freeze_path=args.pretest_freeze,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
