"""Hash-bound real TRAIN-development cohort for AZFP response-code forecasts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from marine_echo.training.raw_response_windows import build_raw_window

INDEX_SHA256 = "c6754334465b32e15bdfdcc9799b75408775ed38c19c3e7a93ea6c49dde869d1"
SUPPORT_SHA256 = "95cbb5ceb0b5bc462c7dd01a3057ed75f37a133c9b7acc70850abe37c3302aaa"
ROW_SHA256 = {
    "fit": "8ef760ebf72c86ce0f3528a0709e15c2ea2e577c1d32bb7d63fa6a46fe34339d",
    "assessment": "99f9b9d8b0307516e9e75fbe12d4b228738ef022c2fb5eede757c5c2e3f5077b",
}
PERIODS = {
    "fit": ("2020-02-17", "2020-04-01"),
    "assessment": ("2020-04-01", "2020-04-15"),
}
REFERENCE_CONFIG = "f90a1a2681898ad2c6dabb5eded4c7fa4c5f5f154aefd10b79dbc82629c74860"
FROZEN_RUN_CONFIG: dict[str, Any] = {
    "run_id": "raw_response_development_deterministic_20260927",
    "output_path": "outputs/raw-response-development-v1-deterministic-20260927",
    "device": "cuda",
    "deterministic_cuda": True,
    "cublas_workspace_config": ":4096:8",
    "allow_tf32": False,
    "ridge_alpha": 1.0,
    "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
    "direct_width": 48,
    "direct_layers": 2,
    "direct_heads": 4,
    "seed": 7,
    "batch_size": 8,
    "updates": 128,
    "warmup_updates": 16,
    "checkpoints": [64, 128],
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "adamw_betas": [0.9, 0.999],
    "adamw_eps": 1e-8,
    "gradient_clip_norm": 1.0,
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class RawCohort:
    partition: str
    cutoffs: NDArray[np.datetime64]
    context: NDArray[np.float32]
    context_mask: NDArray[np.bool_]
    targets: NDArray[np.float64]
    target_mask: NDArray[np.bool_]
    target_times: NDArray[np.datetime64]
    row_sha256: str
    index_sha256: str


def _read_partition(root: Path, partition: str, index: dict) -> RawCohort:
    start, end = PERIODS[partition]
    first = np.datetime64(start, "ns")
    last = np.datetime64(end, "ns")
    days = [day for day in index["days"] if first <= np.datetime64(day) < last]
    if len(days) != int((last - first) / np.timedelta64(1, "D")):
        raise ValueError("Frozen partition day count differs")
    columns: dict[str, list[NDArray]] = {
        key: []
        for key in (
            "bin_start",
            "profile_code_sum",
            "profile_code_count",
            "target_code_sum",
            "target_code_count",
            "observed_ping_count",
            "valid_target_ping_count",
            "configuration_id",
        )
    }
    for day in days:
        entry = index["days"][day]
        manifest_path = root / str(entry["manifest_path"])
        if file_sha256(manifest_path) != entry["manifest_sha256"]:
            raise ValueError(f"Raw response manifest changed: {day}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_path = root / str(manifest["shard_path"])
        if (
            file_sha256(shard_path) != entry["shard_sha256"]
            or manifest["shard_sha256"] != entry["shard_sha256"]
        ):
            raise ValueError(f"Raw response shard changed: {day}")
        with np.load(shard_path, allow_pickle=False) as shard:
            if set(columns) - set(shard.files):
                raise ValueError(f"Required response arrays missing: {day}")
            if not np.array_equal(
                shard["bin_start"],
                np.arange(
                    np.datetime64(day),
                    np.datetime64(day) + np.timedelta64(1, "D"),
                    np.timedelta64(15, "m"),
                ),
            ):
                raise ValueError(f"Quarter-hour grid changed: {day}")
            if not np.array_equal(shard["bin_end"], shard["bin_start"] + np.timedelta64(15, "m")):
                raise ValueError(f"Quarter-hour ends changed: {day}")
            for key, parts in columns.items():
                parts.append(shard[key].copy())
    arrays = {key: np.concatenate(parts) for key, parts in columns.items()}
    if not np.array_equal(arrays["bin_start"], np.arange(first, last, np.timedelta64(15, "m"))):
        raise ValueError("Partition contains missing or duplicated quarter-hours")
    if not np.array_equal(arrays["target_code_count"], arrays["valid_target_ping_count"] * 180):
        raise ValueError("Target response count differs from complete-valid pings")
    if (arrays["profile_code_count"] < 0).any() or (
        arrays["profile_code_count"] > arrays["observed_ping_count"][:, None, None] * 3
    ).any():
        raise ValueError("Impossible profile group counts")
    rows_path = root / f"evidence/v2/raw-response-support/development_{partition}_rows.json"
    if file_sha256(rows_path) != ROW_SHA256[partition]:
        raise ValueError("Reviewed issuance rows changed")
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if len(rows) != len(arrays["bin_start"]) // 4:
        raise ValueError("Issued-hour grid differs")
    cutoffs = []
    context = []
    masks = []
    targets = []
    target_masks = []
    target_times = []
    for hour_index, row in enumerate(rows):
        cutoff = first + hour_index * np.timedelta64(1, "h")
        if np.datetime64(row["cutoff"].removesuffix("Z"), "ns") != cutoff:
            raise ValueError("Reviewed cutoff order changed")
        if not row["issued"]:
            continue
        slot = hour_index * 4
        past = slice(slot - 96, slot)
        if slot < 96 or np.any(
            arrays["configuration_id"][past][arrays["observed_ping_count"][past] > 0]
            != REFERENCE_CONFIG
        ):
            raise ValueError("Past acquisition configuration differs")
        valid_hourly = arrays["valid_target_ping_count"][past].reshape(24, 4).sum(axis=1)
        if valid_hourly[-1] < 120 or (valid_hourly >= 120).sum() < 18:
            raise ValueError("Issued context fails frozen valid-ping gates")
        window = build_raw_window(
            first,
            arrays["profile_code_sum"],
            arrays["profile_code_count"],
            arrays["target_code_sum"],
            arrays["target_code_count"],
            cutoff_slot=slot,
        )
        eligible = np.array([bool(row["horizons"][str(h)]["eligible"]) for h in (1, 3, 6)])
        if np.any(eligible & ~window.target_observed):
            raise ValueError("Reviewed eligible row has no observed target")
        for hindex, horizon in enumerate((1, 3, 6)):
            scored = row["horizons"][str(horizon)]
            if eligible[hindex]:
                if (
                    np.datetime64(str(scored["target_start"]).removesuffix("Z"), "ns")
                    != window.target_times[hindex]
                ):
                    raise ValueError("Reviewed target time differs")
                hslot = slot + 4 * (horizon - 1)
                observed = int(arrays["observed_ping_count"][hslot : hslot + 4].sum())
                valid = int(arrays["valid_target_ping_count"][hslot : hslot + 4].sum())
                if (
                    valid != scored["valid"]
                    or observed != scored["observed"]
                    or valid < 120
                    or observed <= 0
                    or valid / observed < 0.95
                ):
                    raise ValueError("Reviewed target QC differs")
                if np.any(
                    arrays["configuration_id"][hslot : hslot + 4][
                        arrays["observed_ping_count"][hslot : hslot + 4] > 0
                    ]
                    != REFERENCE_CONFIG
                ):
                    raise ValueError("Target acquisition configuration differs")
        cutoffs.append(window.cutoff)
        context.append(
            np.where(window.context_mask, window.context_codes, np.nan).astype(np.float32)
        )
        masks.append(window.context_mask)
        targets.append(np.where(eligible, window.targets, np.nan))
        target_masks.append(eligible)
        target_times.append(window.target_times)
    return RawCohort(
        partition,
        np.asarray(cutoffs),
        np.stack(context),
        np.stack(masks),
        np.stack(targets),
        np.stack(target_masks),
        np.stack(target_times),
        ROW_SHA256[partition],
        INDEX_SHA256,
    )


def load_raw_cohorts(root: Path, *, review_path: Path) -> tuple[RawCohort, RawCohort]:
    """Read the exact approved TRAIN cohort only after the distinct pre-fit gate."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    code_files = {
        "window_code_sha256": root / "src/marine_echo/training/raw_response_windows.py",
        "cohort_code_sha256": root / "src/marine_echo/training/raw_response_cohort.py",
        "executor_code_sha256": root / "src/marine_echo/training/raw_response_development.py",
        "runner_code_sha256": root / "tools/v2_run_raw_development.py",
        "model_code_sha256": root / "src/marine_echo/models/compact.py",
    }
    if (
        review.get("status") != "APPROVED_RAW_RESPONSE_PREFIT"
        or review.get("study_id") != "raw_response_development_v1"
        or review.get("reviewer_session") != "/root/v2_reviewer"
        or review.get("index_sha256") != INDEX_SHA256
        or review.get("support_sha256") != SUPPORT_SHA256
        or review.get("row_sha256") != ROW_SHA256
        or review.get("run_config") != FROZEN_RUN_CONFIG
        or any(review.get(key) != file_sha256(path) for key, path in code_files.items())
    ):
        raise ValueError("Exact independent raw-response pre-fit approval is required")
    index_path = root / "evidence/v2/raw-response-development-reviewed/index.json"
    support_path = root / "evidence/v2/raw-response-support/eligibility_raw_response.json"
    if file_sha256(index_path) != INDEX_SHA256 or file_sha256(support_path) != SUPPORT_SHA256:
        raise ValueError("Reviewed source or support changed")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("study_id") != "raw_response_development_v1" or len(index.get("days", {})) != 58:
        raise ValueError("Raw-response source identity changed")
    fit = _read_partition(root, "fit", index)
    assessment = _read_partition(root, "assessment", index)
    return fit, assessment
