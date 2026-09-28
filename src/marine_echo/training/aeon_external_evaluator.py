"""Synthetic-testable materialization and scoring for frozen AEON external transfer.

Numeric access is intentionally absent from this module. A separate reviewed
runner must preflight source, candidate, model, and code bytes before any ZIP
acoustic cell is opened. The functions here cannot fit or select a model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from marine_echo.evaluation.aeon import QUANTILES, daily_pinball, eligible_source_dates
from marine_echo.training.aeon_corpus import AeonHourlySlot
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _candidate_source_time(value: object) -> np.datetime64:
    if not isinstance(value, str):
        raise TypeError("AEON external Stage 1 candidate cutoff/source time differs.")
    try:
        instant = np.datetime64(value, "us")
    except ValueError as exc:
        raise ValueError("AEON external Stage 1 candidate cutoff/source time differs.") from exc
    if np.isnat(instant) or value not in (
        str(instant), str(instant.astype("datetime64[s]")),
    ):
        raise ValueError("AEON external Stage 1 candidate cutoff/source time differs.")
    return instant


def _candidate_identity(candidate: Mapping[str, Any], archive_sha256: str) -> int:
    cutoff = candidate.get("cutoff_interval_id")
    if not isinstance(cutoff, int) or cutoff < 0:
        raise ValueError("AEON external candidate identity has invalid cutoff.")
    expected_id = hashlib.sha256(
        f"aeon-external-full-depth:{archive_sha256}:{cutoff}:24:1,3,6".encode("ascii")
    ).hexdigest()
    if (
        candidate.get("row_id") != expected_id
        or candidate.get("target_interval_ids") != [cutoff + 1, cutoff + 3, cutoff + 6]
        or candidate.get("actual_issued_status") != "UNKNOWN"
        or candidate.get("target_scoring_status") != "UNKNOWN"
    ):
        raise ValueError("AEON external candidate identity differs from Stage 1.")
    return cutoff


def materialize_candidates(
    slots: Sequence[AeonHourlySlot], candidates: Sequence[Mapping[str, Any]],
    archive_sha256: str,
) -> tuple[list[AeonHourlyWindow], dict[str, list[str]]]:
    """Issue from 24 past numeric-QC rows only; never filter on future availability."""
    if not slots or len(archive_sha256) != 64:
        raise ValueError("AEON external slots/candidates/source binding is empty.")
    by_id = {slot.interval_id: slot for slot in slots}
    if len(by_id) != len(slots) or any(slot.archive_sha256 != archive_sha256 for slot in slots):
        raise ValueError("AEON external slots duplicate IDs or mix source archives.")
    windows: list[AeonHourlyWindow] = []
    not_issued: dict[str, list[str]] = {}
    previous_cutoff = -1
    for candidate in candidates:
        cutoff_id = _candidate_identity(candidate, archive_sha256)
        if cutoff_id <= previous_cutoff:
            raise ValueError("AEON external Stage 1 candidate cutoff order or uniqueness differs.")
        previous_cutoff = cutoff_id
        cutoff = by_id.get(cutoff_id)
        if cutoff is None or (
            _candidate_source_time(candidate.get("cutoff_source_timestamp"))
            != cutoff.source_timestamp
            or candidate.get("cutoff_source_date") != str(
                cutoff.source_timestamp.astype("datetime64[D]")
            )
        ):
            raise ValueError("AEON external Stage 1 candidate cutoff/source time differs.")
        context = [by_id.get(value) for value in range(cutoff_id - 23, cutoff_id + 1)]
        if any(slot is None for slot in context):
            raise ValueError("AEON external Stage 1 candidate lacks a predecessor interval.")
        past = [slot for slot in context if slot is not None]
        if any(
            not np.timedelta64(55, "m") <= right.source_timestamp - left.source_timestamp
            <= np.timedelta64(65, "m")
            for left, right in pairwise(past)
        ):
            raise ValueError("AEON external Stage 1 candidate past source-time rule differs.")
        invalid = sorted({
            "PAST_38_" + slot.qc_status[0]
            for slot in past if not bool(slot.observed_mask[0])
        })
        if invalid:
            not_issued[str(candidate["row_id"])] = invalid
            continue
        values = np.stack([slot.sv_db for slot in past]).astype(np.float64, copy=True)
        mask = np.stack([slot.observed_mask for slot in past]).astype(bool, copy=True)
        if (
            values.shape != (24, 4) or mask.shape != (24, 4)
            or not mask[:, 0].all() or not np.isfinite(values[mask]).all()
        ):
            raise ValueError("AEON external past context mask or values are invalid.")
        values[~mask] = np.nan
        target_ids = [cutoff_id + step for step in (1, 3, 6)]
        target_values = np.full(3, np.nan, dtype=np.float64)
        target_mask = np.zeros(3, dtype=bool)
        target_times = np.full(3, np.datetime64("NaT", "us"), dtype="datetime64[us]")
        target_status: list[str] = []
        target_members: set[str] = set()
        for index, (step, target_id) in enumerate(zip((1, 3, 6), target_ids, strict=True)):
            future = by_id.get(target_id)
            if future is None:
                target_status.append("MISSING_INTERVAL")
                continue
            target_times[index] = future.source_timestamp
            target_members.update(future.member_names)
            delta = future.source_timestamp - cutoff.source_timestamp
            if not np.timedelta64(55 * step, "m") <= delta <= np.timedelta64(65 * step, "m"):
                target_status.append("SOURCE_TIME_DISCONTINUITY")
                continue
            target_status.append(future.qc_status[0])
            if future.observed_mask[0]:
                target_values[index] = future.sv_db[0]
                target_mask[index] = True
        windows.append(AeonHourlyWindow(
            row_id=str(candidate["row_id"]),
            partition="external_transfer",
            cutoff_source_timestamp=cutoff.source_timestamp,
            cutoff_interval_id=cutoff_id,
            context_db=values,
            context_mask=mask,
            context_interval_ids=np.asarray([slot.interval_id for slot in past], dtype=np.int64),
            context_source_timestamps=np.asarray(
                [slot.source_timestamp for slot in past], dtype="datetime64[us]"
            ),
            target_interval_ids=np.asarray(target_ids, dtype=np.int64),
            target_source_timestamps=target_times,
            target_db=target_values,
            target_mask=target_mask,
            target_qc_status=tuple(target_status),  # type: ignore[arg-type]
            source_archive_sha256=archive_sha256,
            past_members=tuple(sorted({name for slot in past for name in slot.member_names})),
            target_members=tuple(sorted(target_members)),
        ))
    if len(windows) + len(not_issued) != len(candidates):
        raise AssertionError("Every AEON external Stage 1 candidate needs an issuance decision.")
    return windows, not_issued


def adapter_compatible_rows(rows: Sequence[AeonHourlyWindow]) -> list[AeonHourlyWindow]:
    """Create transient adapter inputs; persisted rows retain external identity."""
    if any(row.partition != "external_transfer" for row in rows):
        raise ValueError("AEON external adapter input identity differs.")
    return [replace(row, partition="test") for row in rows]


def _paired_two_date_bootstrap(
    truth: NDArray[np.float64], observed: NDArray[np.bool_],
    times: NDArray[np.datetime64], direct: NDArray[np.float64],
    ema: NDArray[np.float64], *, draws: int = 2000, seed: int = 20260928,
) -> NDArray[np.float64]:
    if draws != 2000 or seed != 20260928:
        raise ValueError("AEON external bootstrap draws/seed differ from frozen contract.")
    dates = times.astype("datetime64[D]")
    residual_direct = truth[..., None] - direct
    residual_ema = truth[..., None] - ema
    loss_direct = np.maximum(
        QUANTILES * residual_direct, (QUANTILES - 1) * residual_direct
    ).mean(axis=-1)
    loss_ema = np.maximum(QUANTILES * residual_ema, (QUANTILES - 1) * residual_ema).mean(
        axis=-1
    )
    eligible_by_horizon = []
    for horizon in range(3):
        valid_dates, counts = np.unique(dates[observed[:, horizon], horizon], return_counts=True)
        eligible = valid_dates[counts >= 18]
        if not len(eligible):
            raise ValueError("AEON external paired bootstrap lacks an eligible source date.")
        eligible_by_horizon.append(eligible)
    first = min(days[0] for days in eligible_by_horizon)
    last = max(days[-1] for days in eligible_by_horizon)
    all_dates = np.arange(first, last + np.timedelta64(1, "D"))
    differences = np.full((len(all_dates), 3), np.nan)
    for horizon in range(3):
        eligible_set = set(eligible_by_horizon[horizon])
        for index, day in enumerate(all_dates):
            if day not in eligible_set:
                continue
            mask = observed[:, horizon] & (dates[:, horizon] == day)
            differences[index, horizon] = float(
                (loss_ema[mask, horizon] - loss_direct[mask, horizon]).mean()
            )
    blocks = [block for start in range(0, len(all_dates), 2)
              if np.isfinite(block := differences[start:start + 2]).any()]
    if not blocks:
        raise ValueError("AEON external bootstrap has no paired eligible date blocks.")
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        for _ in range(10_000):
            choices = rng.integers(0, len(blocks), size=len(blocks))
            sample = np.concatenate([blocks[int(index)] for index in choices], axis=0)
            if not np.isnan(sample).all(axis=0).any():
                break
        else:
            raise ValueError("AEON external bootstrap cannot cover all horizons.")
        result[draw] = float(np.nanmean(sample, axis=0).mean())
    return result


def score_external_predictions(
    truth: NDArray[np.float64], observed: NDArray[np.bool_],
    target_times: NDArray[np.datetime64], direct: NDArray[np.float64],
    ema: NDArray[np.float64],
    *, primary_gate: bool = True,
) -> dict[str, Any]:
    """Compare both frozen models on the same daily source-date support."""
    if not len(truth):
        raise ValueError("AEON external score needs at least one issued row.")
    if (
        truth.ndim != 2 or truth.shape[1] != 3
        or observed.shape != truth.shape or observed.dtype.kind != "b"
        or target_times.shape != truth.shape or target_times.dtype.kind != "M"
        or direct.shape != (*truth.shape, 5) or ema.shape != direct.shape
        or not np.isfinite(direct).all() or not np.isfinite(ema).all()
        or (np.diff(direct, axis=-1) < 0).any()
        or (np.diff(ema, axis=-1) < 0).any()
        or not np.isfinite(truth[observed]).all()
    ):
        raise ValueError("AEON external score truth/forecast geometry differs.")
    eligible_by_horizon = eligible_source_dates(target_times, observed)
    if any(count == 0 for count in eligible_by_horizon.values()):
        return {
            "study_partition": "external_transfer",
            "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
            "issued_rows": len(truth),
            "scored_rows_per_horizon": observed.sum(axis=0).astype(int).tolist(),
            "eligible_dates_per_horizon": list(eligible_by_horizon.values()),
            "cohort_eligible": False,
            "direct_raw_metrics": "NOT_COMPUTABLE_NO_ELIGIBLE_DATE_IN_EVERY_HORIZON",
            "ema_raw_metrics": "NOT_COMPUTABLE_NO_ELIGIBLE_DATE_IN_EVERY_HORIZON",
            "relative_loss_reduction": None,
            "paired_bootstrap_status": "NOT_RUN_COHORT_INELIGIBLE",
            "jepa_value_gate": "INELIGIBLE_NOT_A_NEGATIVE_TRANSFER_RESULT" if primary_gate
            else "DESCRIPTIVE_ONLY_NOT_PRIMARY_GATE",
        }
    direct_metrics = daily_pinball(truth, direct, observed, target_times)
    ema_metrics = daily_pinball(truth, ema, observed, target_times)
    eligible_days = cast(list[int], direct_metrics["eligible_days_per_horizon"])
    if eligible_days != ema_metrics["eligible_days_per_horizon"]:
        raise AssertionError("AEON external models differ on source-date support.")
    eligible = all(count >= 90 for count in eligible_days)
    direct_loss = float(cast(float, direct_metrics["primary_daily_mean_pinball_db"]))
    ema_loss = float(cast(float, ema_metrics["primary_daily_mean_pinball_db"]))
    relative = (direct_loss - ema_loss) / direct_loss if direct_loss > 0 else None
    report: dict[str, Any] = {
        "study_partition": "external_transfer",
        "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
        "issued_rows": len(truth),
        "eligible_dates_per_horizon": eligible_days,
        "cohort_eligible": eligible,
        "direct_raw_metrics": direct_metrics,
        "ema_raw_metrics": ema_metrics,
        "relative_loss_reduction": relative,
    }
    if not eligible:
        report["paired_bootstrap_status"] = "NOT_RUN_COHORT_INELIGIBLE"
        report["jepa_value_gate"] = (
            "INELIGIBLE_NOT_A_NEGATIVE_TRANSFER_RESULT" if primary_gate
            else "DESCRIPTIVE_ONLY_NOT_PRIMARY_GATE"
        )
        return report
    draws = _paired_two_date_bootstrap(truth, observed, target_times, direct, ema)
    interval = np.quantile(draws, [0.025, 0.975]).astype(float).tolist()
    baseline_horizon = cast(list[float], direct_metrics["daily_mean_pinball_db_per_horizon"])
    candidate_horizon = cast(list[float], ema_metrics["daily_mean_pinball_db_per_horizon"])
    horizon_relative = [
        (float(candidate) - float(base)) / float(base)
        for base, candidate in zip(baseline_horizon, candidate_horizon, strict=True)
    ]
    passed = bool(
        relative is not None and relative >= 0.05
        and interval[1] < 0 and all(value <= 0.10 for value in horizon_relative)
    )
    report.update({
        "paired_bootstrap_status": "COMPLETED_2000_DRAWS_SEED_20260928",
        "paired_95pct_difference_interval_db": interval,
        "per_horizon_relative_loss_difference": horizon_relative,
        "jepa_value_gate": (
            "PASSED_RETROSPECTIVE_EXTERNAL_TRANSFER" if passed
            else "FAILED_ELIGIBLE_EXECUTED_TRANSFER"
        ) if primary_gate else "DESCRIPTIVE_ONLY_NOT_PRIMARY_GATE",
    })
    return report
