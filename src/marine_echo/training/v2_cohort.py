"""Verify every native TRAIN development issue against the independent support ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marine_echo.training.v2_executor import _sha256
from marine_echo.training.v2_native import iter_native_hourly_windows
from marine_echo.training.v2_native_reader import NativeDevelopmentStream
from marine_echo.training.v2_stream import HourlyWindow


def verify_support_rows(rows: list[HourlyWindow], ledger: list[dict[str, Any]]) -> None:
    """Require identical issued cutoffs and per-horizon scoring masks."""
    issued = [entry for entry in ledger if entry.get("issued") is True]
    if len(rows) != len(issued) or len({str(row.cutoff) for row in rows}) != len(rows):
        raise ValueError("Native support row count or cutoff identities differ.")
    by_cutoff = {str(np.datetime64(entry["cutoff"], "h")): entry for entry in issued}
    if len(by_cutoff) != len(issued):
        raise ValueError("Native support row ledger repeats an issued cutoff.")
    for row in rows:
        entry = by_cutoff.get(str(np.datetime64(row.cutoff, "h")))
        if entry is None:
            raise ValueError("Native support row cutoff differs from streamed cohort.")
        if set(entry.get("horizons", {})) != {"1", "3", "6"}:
            raise ValueError("Native support row horizons differ from the frozen schedule.")
        for index, horizon in enumerate((1, 3, 6)):
            target = entry["horizons"][str(horizon)]
            fraction = target.get("fraction")
            if (
                np.datetime64(target["target_start"], "h")
                != np.datetime64(row.target_interval_start[index], "h")
                or target.get("index_eligible") != bool(row.target_mask[index])
                or target.get("fraction_eligible") != bool(row.target_detection_mask[index])
                or (
                    row.target_detection_mask[index]
                    and (
                        fraction is None
                        or not np.isclose(
                            float(fraction),
                            row.target_detection_fraction[index],
                            rtol=1e-9,
                            atol=1e-12,
                        )
                    )
                )
            ):
                raise ValueError("Native support row target masks or fraction differ.")


@dataclass(frozen=True)
class NativeDevelopmentCohort:
    fit: list[HourlyWindow]
    assessment: list[HourlyWindow]
    native_index_sha256: str
    support_report_sha256: str
    fit_support_rows_sha256: str
    assessment_support_rows_sha256: str


def prepare_native_development_cohort(
    root: Path,
    *,
    native_index_sha256: str,
    support_report_sha256: str,
) -> NativeDevelopmentCohort:
    """Load all 58 declared TRAIN days only after independent support adequacy."""
    root = root.resolve(strict=True)
    support_path = root / "evidence/v2/native-support/eligibility_v2.json"
    if _sha256(support_path) != support_report_sha256:
        raise ValueError("Native support report digest differs.")
    report = json.loads(support_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "NATIVE_SUPPORT_ADEQUATE_PENDING_REVIEW"
        or report.get("model_scores_computed") is not False
        or report.get("non_train_acoustic_payloads_read") is not False
    ):
        raise ValueError("Native support is ineligible; D1 target search and fitting must stop.")
    entries = report.get("partitions", {})
    expected_calendar = {
        "development_fit": ("2020-02-17", "2020-04-01"),
        "development_assessment": ("2020-04-01", "2020-04-15"),
    }
    ledgers = {}
    for name, (start, end) in expected_calendar.items():
        part = entries.get(name, {})
        relative = f"evidence/v2/native-support/{name}_rows.json"
        path = root / relative
        if (
            part.get("start") != start
            or part.get("end") != end
            or Path(str(part.get("row_path", "")).replace("\\", "/")).as_posix() != relative
            or _sha256(path) != part.get("rows_sha256")
            or part.get("adequate_all_horizons") is not True
        ):
            raise ValueError("Native support partition or row ledger differs.")
        ledgers[name] = json.loads(path.read_text(encoding="utf-8"))
    index_relative = "evidence/v2/native-development/index.json"
    artifact_hashes = report.get("artifact_sha256", {})
    registered_index = artifact_hashes.get(index_relative.replace("/", "\\"))
    index_path = root / index_relative
    if registered_index != native_index_sha256 or _sha256(index_path) != native_index_sha256:
        raise ValueError("Native support index digest differs from the reader input.")
    stream = NativeDevelopmentStream(
        root,
        index_path,
        index_sha256=native_index_sha256,
        start_day="2020-02-17",
        end_day_exclusive="2020-04-15",
    )
    expected_days = {
        str(day)
        for day in np.arange(
            np.datetime64("2020-02-17", "D"),
            np.datetime64("2020-04-15", "D"),
            dtype="datetime64[D]",
        )
    }
    if set(stream.days) != expected_days:
        raise ValueError("Native development requires the complete 58-day TRAIN calendar.")
    fit: list[HourlyWindow] = []
    assessment: list[HourlyWindow] = []
    for row in iter_native_hourly_windows(stream, partition="train"):
        cutoff = row.cutoff
        if np.datetime64("2020-02-18") <= cutoff <= np.datetime64("2020-03-31T18"):
            fit.append(row)
        elif np.datetime64("2020-04-02") <= cutoff <= np.datetime64("2020-04-14T18"):
            assessment.append(row)
    verify_support_rows(fit, ledgers["development_fit"])
    verify_support_rows(assessment, ledgers["development_assessment"])
    return NativeDevelopmentCohort(
        fit=fit,
        assessment=assessment,
        native_index_sha256=native_index_sha256,
        support_report_sha256=support_report_sha256,
        fit_support_rows_sha256=entries["development_fit"]["rows_sha256"],
        assessment_support_rows_sha256=entries["development_assessment"]["rows_sha256"],
    )
