"""Independent standard-library test oracles; integrate only after project review."""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

QUANTILES = ("0.05", "0.25", "0.5", "0.75", "0.95")
HORIZONS = (1, 3, 6)


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number, not a boolean or string.")
    if not math.isfinite(value):
        raise ValueError("NaN and infinity are forbidden.")
    return float(value)


def linear_mean_db(values: Iterable[float]) -> float:
    """Average linear scattering values, returning dB with a stable log-sum."""
    numbers = [_finite(value) for value in values]
    if not numbers:
        raise ValueError("No supported observations; do not replace missing data by zero.")
    peak = max(numbers)
    linear = math.fsum(10.0 ** ((value - peak) / 10.0) for value in numbers)
    return peak + 10.0 * math.log10(linear / len(numbers))


def pinball(quantile: float, observed: float, predicted: float) -> float:
    q, y, f = (_finite(value) for value in (quantile, observed, predicted))
    if not 0.0 < q < 1.0:
        raise ValueError("Quantile must lie strictly between zero and one.")
    error = y - f
    return (q - float(error < 0.0)) * error


def parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamp must be a timezone-aware ISO8601 string.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp lacks an explicit timezone.")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def target_intervals(cutoff: str) -> list[tuple[str, str]]:
    t = parse_timestamp(cutoff)
    return [
        (_iso(t + timedelta(hours=h - 1)), _iso(t + timedelta(hours=h)))
        for h in HORIZONS
    ]


def validate_forecast(data: dict[str, Any]) -> None:
    """Semantic checks additional to, not replacing, forecast.schema.json."""
    if data.get("schema_version") != "1.0":
        raise ValueError("Unknown schema version.")
    cutoff = parse_timestamp(data["prediction_cutoff"])
    status = data["status"]
    if status not in ("ok", "abstained"):
        raise ValueError("Unknown prediction status.")
    if not 0 <= _finite(data["support_fraction"]) <= 1:
        raise ValueError("Support fraction outside [0,1].")
    if data["data_kind"] not in ("public_real", "synthetic_fixture"):
        raise ValueError("Unknown data kind.")
    if (data["data_kind"] == "synthetic_fixture") != (data["mode"] == "synthetic_fixture"):
        raise ValueError("Synthetic fixtures cannot masquerade as real inference/replay.")
    if data["units"] == "dB re 1 m^-1" and data["calibrated"] is not True:
        raise ValueError("Physical units need verified calibration provenance.")
    for name in ("checkpoint_sha256", "preprocessing_sha256", "split_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", data[name]):
            raise ValueError(f"Invalid digest: {name}")
    if not isinstance(data["reasons"], list):
        raise ValueError("Reasons must be a list.")
    if status == "abstained" and not data["reasons"]:
        raise ValueError("Abstention needs a reason.")
    last = data["last_observation_at"]
    age = data["observation_age_hours"]
    if last is None:
        if status != "abstained" or age is not None:
            raise ValueError("No observations requires explicit abstention and unknown age.")
    else:
        last_time = parse_timestamp(last)
        expected_age = (cutoff - last_time).total_seconds() / 3600
        if expected_age < 0 or not math.isclose(_finite(age), expected_age, abs_tol=1e-6):
            raise ValueError("Future observation or inconsistent observation age.")
    forecasts = data["forecasts"]
    if len(forecasts) != 3 or sorted(x["horizon_hours"] for x in forecasts) != list(HORIZONS):
        raise ValueError("Expected each of the 1/3/6-hour horizons exactly once.")
    expected_windows = dict(zip(HORIZONS, target_intervals(data["prediction_cutoff"])))
    for forecast in forecasts:
        start, end = expected_windows[forecast["horizon_hours"]]
        if parse_timestamp(forecast["target_start"]) != parse_timestamp(start):
            raise ValueError("Target start disagrees with the frozen interval.")
        if parse_timestamp(forecast["target_end"]) != parse_timestamp(end):
            raise ValueError("Target end disagrees with the frozen interval.")
        quantiles = forecast["quantiles"]
        if set(quantiles) != set(QUANTILES):
            raise ValueError("Wrong quantile keys.")
        values = [quantiles[key] for key in QUANTILES]
        if status == "abstained":
            if any(value is not None for value in values):
                raise ValueError("An abstention must not fabricate numerical predictions.")
        else:
            numbers = [_finite(value) for value in values]
            if numbers != sorted(numbers):
                raise ValueError("Quantiles cross.")


def validate_task_dag(tasks: list[dict[str, Any]]) -> None:
    ids = [task["id"] for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate task IDs.")
    graph = {task["id"]: task["depends_on"] for task in tasks}
    for task, dependencies in graph.items():
        if any(dependency not in graph for dependency in dependencies):
            raise ValueError(f"Unknown dependency for {task}.")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task: str) -> None:
        if task in visiting:
            raise ValueError("Task dependency cycle.")
        if task in visited:
            return
        visiting.add(task)
        for dependency in graph[task]:
            visit(dependency)
        visiting.remove(task)
        visited.add(task)

    for task in graph:
        visit(task)
