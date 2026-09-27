"""Reviewed TRAIN-only frozen-pretrain AEON raw/latent hybrid development."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import joblib  # type: ignore[import-untyped]
import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.models.aeon_ssl import AeonTemporalSSL
from marine_echo.training.aeon_campaign import (
    QUANTILES,
    _campaign_code_sha256,
    _config_gate,
    _past_features,
    load_cohort,
)
from marine_echo.training.aeon_development import (
    _context_tensors,
    _normalizer,
    _save_predictions,
    _sha256,
)
from marine_echo.training.aeon_rescore import _artifact, _load_prediction, _row_digest, _same_rows
from marine_echo.training.aeon_windows import AeonHourlyWindow


HYBRID_SLOTS = (
    "ema_pretrain_raw_latent_hgb_seed7",
    "ema_pretrain_raw_latent_hgb_seed13",
    "ema_pretrain_raw_latent_hgb_seed23",
    "shared_sigreg_pretrain_raw_latent_hgb_seed7",
    "shared_sigreg_pretrain_raw_latent_hgb_seed13",
    "shared_sigreg_pretrain_raw_latent_hgb_seed23",
    "random_encoder_ema_raw_latent_hgb_seed7",
    "random_encoder_shared_sigreg_raw_latent_hgb_seed7",
    "shuffled_target_ema_raw_latent_hgb_seed7",
    "shuffled_target_shared_sigreg_raw_latent_hgb_seed7",
)


def _source_spec(
    slot_id: str,
) -> tuple[str, Literal["ema", "shared_sigreg"], int, Literal["pretrain", "supervised"]]:
    if slot_id not in HYBRID_SLOTS:
        raise ValueError("AEON hybrid slot differs from the finite reviewed campaign.")
    seed = int(slot_id.rsplit("seed", 1)[1])
    mode: Literal["ema", "shared_sigreg"] = (
        "shared_sigreg" if "shared_sigreg" in slot_id else "ema"
    )
    if slot_id.startswith("random_encoder_"):
        return f"random_encoder_{mode}_seed7", mode, seed, "supervised"
    if slot_id.startswith("shuffled_target_"):
        return f"temporally_shuffled_pretrain_target_{mode}_seed7", mode, seed, "pretrain"
    return f"{'ema_jepa' if mode == 'ema' else mode}_seed{seed}", mode, seed, "pretrain"


def _pretrain_encoder(
    checkpoint: Path,
    mode: Literal["ema", "shared_sigreg"],
    config: dict[str, Any],
    scaler: tuple[float, float, float, float],
    *,
    source_sha256: str,
    protocol_sha256: str,
    family: str | None = None,
    seed: int = 7,
    phase: Literal["pretrain", "supervised"] = "pretrain",
) -> AeonTemporalSSL:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_family = family or ("ema_jepa" if mode == "ema" else "shared_sigreg")
    if (
        not isinstance(saved, dict)
        or saved.get("phase") != phase
        or saved.get("step") != 1500
        or saved.get("family") != expected_family
        or saved.get("seed") != seed
        or saved.get("source_sha256") != source_sha256
        or saved.get("protocol_sha256") != protocol_sha256
        or tuple(saved.get("scaler_fit_only", ())) != scaler
    ):
        raise ValueError("AEON hybrid requires the exact final pretrain or frozen-random TRAIN checkpoint.")
    neural = config["neural"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AeonTemporalSSL(
            mode=mode,
            width=neural["encoder_width"],
            layers=neural["encoder_layers"],
            regularizer_weight=neural[
                "ema_sigreg_weight" if mode == "ema" else "shared_sigreg_weight"
            ],
        )
    initial = {
        name: value.clone() for name, value in model.state_dict().items()
        if name.startswith(("encoder.", "predictor.", "teacher."))
    }
    model.load_state_dict(saved["model_state_dict"], strict=True)
    if phase == "supervised" and any(
        not torch.equal(model.state_dict()[name], value) for name, value in initial.items()
    ):
        raise ValueError("AEON hybrid random encoder differs from frozen seed initialization.")
    model.requires_grad_(False)
    model.eval()
    return model


def _latent_features(
    model: AeonTemporalSSL,
    rows: list[AeonHourlyWindow],
    scaler: tuple[float, float, float, float],
) -> np.ndarray:
    values, mask = _context_tensors(rows, scaler)
    parts = []
    with torch.no_grad():
        for first in range(0, len(rows), 64):
            parts.append(model.encoder(values[first : first + 64], mask[first : first + 64]).numpy())
    result = np.concatenate(parts).astype(np.float64)
    if result.shape != (len(rows), model.head.in_features) or not np.isfinite(result).all():
        raise ValueError("AEON hybrid frozen encoder features are invalid.")
    return result


def _fit_quantile_heads(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    assess_x: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[list[HistGradientBoostingRegressor]]]:
    if (
        train_x.ndim != 2
        or assess_x.ndim != 2
        or train_x.shape[1] != assess_x.shape[1]
        or train_y.shape != (len(train_x), 3)
        or train_mask.shape != train_y.shape
        or not np.isfinite(train_x).all()
        or not np.isfinite(assess_x).all()
        or not np.isfinite(train_y[train_mask]).all()
    ):
        raise ValueError("AEON hybrid TRAIN/head feature or label shape differs.")
    conventional = config["conventional"]
    forecast = np.empty((len(assess_x), 3, 5), dtype=np.float64)
    models: list[list[HistGradientBoostingRegressor]] = []
    with threadpool_limits(limits=conventional["max_threads"]):
        for horizon in range(3):
            valid = train_mask[:, horizon]
            if not valid.any():
                raise ValueError("AEON hybrid TRAIN lacks a horizon target.")
            heads = []
            for index, quantile in enumerate(QUANTILES):
                head = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=float(quantile),
                    max_iter=conventional["hist_gradient_boosting_max_iter"],
                    max_depth=conventional["hist_gradient_boosting_max_depth"],
                    learning_rate=conventional["hist_gradient_boosting_learning_rate"],
                    random_state=conventional["random_state"],
                )
                head.fit(train_x[valid], train_y[valid, horizon])
                forecast[:, horizon, index] = head.predict(assess_x)
                heads.append(head)
                if psutil.Process().memory_info().rss >= 22 * 1024**3:
                    raise MemoryError("AEON hybrid reached the 22 GiB process RAM limit.")
            models.append(heads)
    forecast.sort(axis=-1)
    return forecast, models


def _hybrid_gate(hybrid: dict[str, Any], core: dict[str, Any]) -> None:
    _config_gate(core)
    head = hybrid.get("head", {})
    conventional = core["conventional"]
    if (
        hybrid.get("phase") != "train_validation_frozen_pretrain_hybrid_development"
        or hybrid.get("status") != "PROPOSED_FOR_INDEPENDENT_PREFIT_REVIEW"
        or hybrid.get("study_id") != core["study_id"]
        or hybrid.get("protocol_sha256") != core["protocol_sha256"]
        or hybrid.get("source_sha256") != core["source_sha256"]
        or hybrid.get("fit_partition") != "train"
        or hybrid.get("assessment_partition") != "validation"
        or hybrid.get("calibration_access") != "PROHIBITED_IN_THIS_PHASE"
        or hybrid.get("test_access") != "PROHIBITED_IN_THIS_PHASE"
        or hybrid.get("seeds") != [7, 13, 23]
        or hybrid.get("slots") != list(HYBRID_SLOTS)
        or hybrid.get("latent_checkpoint_phase") != "pretrain"
        or hybrid.get("latent_checkpoint_step") != 1500
        or hybrid.get("latent_encoder_frozen") is not True
        or hybrid.get("horizon_interval_steps") != [1, 3, 6]
        or hybrid.get("quantiles") != QUANTILES.tolist()
        or head.get("family") != "hist_gradient_boosting_quantile"
        or head.get("heads_per_slot") != 15
        or head.get("max_iter") != conventional["hist_gradient_boosting_max_iter"]
        or head.get("max_depth") != conventional["hist_gradient_boosting_max_depth"]
        or head.get("learning_rate") != conventional["hist_gradient_boosting_learning_rate"]
        or head.get("random_state") != conventional["random_state"]
        or head.get("max_threads") != conventional["max_threads"]
        or hybrid.get("raw_features") != "frozen_core_campaign_past_features_18_dimensions"
        or hybrid.get("latent_features")
        != "frozen_encoder_128_dimensions_with_train_only_normalizer_and_past_context"
    ):
        raise ValueError("AEON hybrid differs from the fixed TRAIN/validation head contract.")


def run_hybrids(
    archive: Path,
    split_review: Path,
    core_config_path: Path,
    hybrid_config_path: Path,
    campaign_output: Path,
    rescore_report: Path,
    hybrid_review_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Fit two fixed hybrids only after exact independent prefit review."""
    core_sha256 = _sha256(core_config_path)
    core = json.loads(core_config_path.read_text(encoding="utf-8"))
    hybrid_sha256 = _sha256(hybrid_config_path)
    hybrid = json.loads(hybrid_config_path.read_text(encoding="utf-8"))
    _hybrid_gate(hybrid, core)
    campaign_output = campaign_output.resolve(strict=True)
    manifest_path = campaign_output / "manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rescore_sha256 = _sha256(rescore_report)
    rescore = json.loads(rescore_report.read_text(encoding="utf-8"))
    split_sha256 = _sha256(split_review)
    review_sha256 = _sha256(hybrid_review_path)
    review = json.loads(hybrid_review_path.read_text(encoding="utf-8"))
    code_sha256 = _sha256(Path(__file__))
    if (
        hybrid.get("core_campaign_config_sha256") != core_sha256
        or manifest.get("status") != "COMPLETED_TRAIN_VALIDATION_CAMPAIGN_NOT_FINAL_EVALUATION"
        or manifest.get("config_sha256") != core_sha256
        or manifest.get("code_sha256") != _campaign_code_sha256()
        or rescore.get("campaign_manifest_sha256") != manifest_sha256
        or rescore.get("status")
        != "PROTOCOL_VALIDATION_RESCORE_PENDING_INDEPENDENT_SELECTION_REVIEW"
        or review.get("status") != "APPROVED_AEON_TRAIN_VALIDATION_HYBRID"
        or review.get("hybrid_code_sha256") != code_sha256
        or review.get("hybrid_config_sha256") != hybrid_sha256
        or review.get("campaign_manifest_sha256") != manifest_sha256
        or review.get("rescore_report_sha256") != rescore_sha256
        or review.get("split_review_sha256") != split_sha256
        or review.get("cohort_sha256") != hybrid.get("cohort_sha256")
        or review.get("test_access") != "PROHIBITED"
    ):
        raise ValueError("AEON hybrid lacks an exact independent prefit approval.")
    fit, assess, cohort_sha256, _ = load_cohort(archive, split_review)
    if cohort_sha256 != hybrid["cohort_sha256"] or cohort_sha256 != manifest["cohort_sha256"]:
        raise ValueError("AEON hybrid source cohort differs from approved campaign.")
    entries = manifest["slots"]
    raw_dir = campaign_output / "hist_gradient_boosting"
    raw_record = json.loads(_artifact(
        raw_dir, "slot.json", entries["hist_gradient_boosting"]["slot_sha256"]
    ).read_text(encoding="utf-8"))
    if (
        raw_record.get("run_id") != "hist_gradient_boosting"
        or raw_record.get("family") != "hist_gradient_boosting"
        or rescore.get("slots", {}).get("hist_gradient_boosting", {}).get("prediction_sha256")
        != raw_record.get("prediction_sha256")
    ):
        raise ValueError("AEON raw-only B3 differs from the independently rescored reference.")
    raw_prediction_path = _artifact(
        raw_dir, raw_record.get("prediction_path"), raw_record.get("prediction_sha256")
    )
    raw_model_path = _artifact(raw_dir, raw_record.get("model_path"), raw_record.get("model_sha256"))
    raw_rows, raw_forecast = _load_prediction(raw_prediction_path)
    raw_x_fit, raw_x_assess = _past_features(fit), _past_features(assess)
    if raw_x_fit.shape[1] != 18 or raw_x_assess.shape[1] != 18:
        raise ValueError("AEON raw-only control is not the frozen 18-d feature extractor.")
    raw_model = joblib.load(raw_model_path)
    raw_heads = raw_model.get("models") if isinstance(raw_model, dict) else None
    if not isinstance(raw_model, dict) or not isinstance(raw_heads, list) or raw_model.get("family") != "hist_gradient_boosting" or len(raw_heads) != 3 or any(
        len(heads) != 5 for heads in raw_heads
    ):
        raise ValueError("AEON raw-only HGB model is not the matched 15-head reference.")
    reproduced_raw = np.empty_like(raw_forecast)
    for horizon, heads in enumerate(raw_heads):
        for index, head in enumerate(heads):
            if (
                head.loss != "quantile"
                or head.quantile != float(QUANTILES[index])
                or head.max_iter != hybrid["head"]["max_iter"]
                or head.max_depth != hybrid["head"]["max_depth"]
                or head.learning_rate != hybrid["head"]["learning_rate"]
                or head.random_state != hybrid["head"]["random_state"]
                or head.n_features_in_ != 18
            ):
                raise ValueError("AEON raw-only HGB head differs from the matched recipe.")
            reproduced_raw[:, horizon, index] = head.predict(raw_x_assess)
    reproduced_raw.sort(axis=-1)
    if not np.array_equal(reproduced_raw, raw_forecast):
        raise ValueError("AEON raw-only saved forecast differs from matched HGB heads.")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("AEON hybrid output already exists; never refit implicitly.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        check_path = stage / "cohort-check.npz"
        _save_predictions(check_path, assess, raw_forecast)
        expected_rows, _ = _load_prediction(check_path)
        check_path.unlink()
        if not _same_rows(raw_rows, expected_rows):
            raise ValueError("AEON raw-only saved rows differ from reviewed validation cohort.")
        truth = np.stack([row.target_db for row in assess])
        observed = np.stack([row.target_mask for row in assess])
        source_times = np.stack([row.target_source_timestamps for row in assess])
        raw_score = daily_pinball(truth, raw_forecast, observed, source_times)
        if raw_score != rescore["slots"]["hist_gradient_boosting"]["protocol_validation_metrics"]:
            raise ValueError("AEON raw-only B3 protocol score differs from independent rescore.")
        report: dict[str, Any] = {
            "status": "COMPLETED_TRAIN_VALIDATION_HYBRID_DEVELOPMENT_PENDING_SELECTION_REVIEW",
            "classification": "DEVELOPMENT_NOT_FINAL_EVALUATION",
            "test_access": "PROHIBITED",
            "source_archive_sha256": hybrid["source_sha256"],
            "cohort_sha256": cohort_sha256,
            "core_manifest_sha256": manifest_sha256,
            "core_rescore_sha256": rescore_sha256,
            "core_config_sha256": core_sha256,
            "hybrid_config_sha256": hybrid_sha256,
            "hybrid_code_sha256": code_sha256,
            "review_sha256": review_sha256,
            "evaluation_code_sha256": _sha256(Path(daily_pinball.__code__.co_filename)),
            "validation_row_sha256": _row_digest(raw_rows),
            "train_rows": len(fit),
            "validation_issued_rows": len(assess),
            "selection": "NONE_PENDING_INDEPENDENT_REVIEW",
            "slots": {
                "raw_only_hgb": {
                    "source_slot": "hist_gradient_boosting",
                    "source_slot_sha256": entries["hist_gradient_boosting"]["slot_sha256"],
                    "model_sha256": raw_record["model_sha256"],
                    "prediction_sha256": raw_record["prediction_sha256"],
                    "protocol_validation_metrics": raw_score,
                }
            },
        }
        scaler = _normalizer(fit)
        for slot_id in HYBRID_SLOTS:
            source_id, mode, seed, phase = _source_spec(slot_id)
            source_dir = campaign_output / source_id
            source_record = json.loads(_artifact(
                source_dir, "slot.json", entries[source_id]["slot_sha256"]
            ).read_text(encoding="utf-8"))
            checkpoints = [
                item for item in source_record.get("checkpoints", [])
                if item.get("phase") == phase and item.get("step") == 1500
            ]
            if (
                len(checkpoints) != 1
                or source_record.get("run_id") != source_id
                or source_record.get("family") != source_id.rsplit("_seed", 1)[0]
                or source_record.get("seed") != seed
            ):
                raise ValueError("AEON hybrid final pretrain checkpoint is absent or repeated.")
            item = checkpoints[0]
            checkpoint = _artifact(source_dir, item.get("path"), item.get("sha256"))
            model = _pretrain_encoder(
                checkpoint, mode, core, scaler,
                source_sha256=hybrid["source_sha256"],
                protocol_sha256=hybrid["protocol_sha256"],
                family=source_record.get("family"), seed=seed, phase=phase,
            )
            latent_fit = _latent_features(model, fit, scaler)
            latent_assess = _latent_features(model, assess, scaler)
            x_fit = np.concatenate((raw_x_fit, latent_fit), axis=1)
            x_assess = np.concatenate((raw_x_assess, latent_assess), axis=1)
            if x_fit.shape[1] != 146 or x_assess.shape[1] != 146:
                raise ValueError("AEON hybrid feature width differs from 18 raw plus 128 latent.")
            forecast, heads = _fit_quantile_heads(x_fit, train_y=np.stack([row.target_db for row in fit]),
                                                  train_mask=np.stack([row.target_mask for row in fit]),
                                                  assess_x=x_assess, config=core)
            slot_dir = stage / slot_id
            slot_dir.mkdir()
            model_path = slot_dir / "model.joblib"
            joblib.dump({"family": slot_id, "models": heads}, model_path)
            prediction_path = slot_dir / "validation-predictions.npz"
            _save_predictions(prediction_path, assess, forecast)
            saved_rows, saved_forecast = _load_prediction(prediction_path)
            if not _same_rows(saved_rows, raw_rows) or not np.array_equal(saved_forecast, forecast):
                raise ValueError("AEON hybrid saved predictions differ from matched validation rows.")
            report["slots"][slot_id] = {
                "source_pretrain_run_id": source_id,
                "representation_checkpoint_phase": phase,
                "representation_checkpoint_sha256": item["sha256"],
                "representation_seed": seed,
                "model_path": f"{slot_id}/model.joblib",
                "model_sha256": _sha256(model_path),
                "prediction_path": f"{slot_id}/validation-predictions.npz",
                "prediction_sha256": _sha256(prediction_path),
                "feature_dimensions": {"raw": 18, "frozen_pretrain_latent": 128},
                "protocol_validation_metrics": daily_pinball(
                    truth, forecast, observed, source_times
                ),
            }
        (stage / "hybrid-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    except BaseException:
        # Keep a failed stage for inspection; never expose it as a completed run.
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "archive", "split-review", "core-config", "hybrid-config", "campaign-output",
        "rescore-report", "hybrid-review", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_hybrids(
        arguments.archive, arguments.split_review, arguments.core_config,
        arguments.hybrid_config, arguments.campaign_output, arguments.rescore_report,
        arguments.hybrid_review, arguments.output,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
