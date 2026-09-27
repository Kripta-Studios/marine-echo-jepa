"""Past-only issuance and future-score support for conditioned AZFP code means."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def audit_raw_partition(
    hours: list[dict[str, Any]],
    *,
    reference_configuration: str,
    block_origin: str | None = None,
) -> dict[str, Any]:
    """Audit one chronological partition without reading response-code values."""
    if not reference_configuration or not hours:
        raise ValueError("Reference configuration and hourly records required")
    if any(not str(hour["time"]).endswith("Z") for hour in hours):
        raise ValueError("Explicit UTC hourly timestamps required")
    times = np.array([np.datetime64(hour["time"][:-1], "h") for hour in hours])
    if np.isnat(times).any() or not np.all(np.diff(times) == np.timedelta64(1, "h")):
        raise ValueError("Contiguous strictly hourly records required")
    origin_text = block_origin[:-1] if block_origin and block_origin.endswith("Z") else block_origin
    origin = np.datetime64(origin_text or str(times[0]), "h")
    if np.isnat(origin):
        raise ValueError("Fixed block origin required")
    for hour in hours:
        observed = hour["observed"]
        valid = hour["valid"]
        zero_affected = hour["zero_affected_pings"]
        nonfinite = hour["nonfinite_pings"]
        if (
            not all(isinstance(x, (int, np.integer)) for x in (observed, valid, zero_affected, nonfinite))
            or min(observed, valid, zero_affected, nonfinite) < 0
            or valid > observed
            or zero_affected > observed - valid
            or nonfinite > observed - valid
            or hour["zero_samples"] < 0
            or hour["nonfinite_samples"] < 0
        ):
            raise ValueError("Impossible raw-response QC counts")
    rows = []
    issue_exclusions: Counter[str] = Counter()
    for i, hour in enumerate(hours):
        row: dict[str, Any] = {"cutoff": hour["time"], "issued": False, "horizons": {}}
        if i < 24:
            reason = "partition_boundary"
        else:
            past = hours[i - 24 : i]
            observed_configs = {x["configuration"] for x in past if x["observed"] > 0}
            if observed_configs != {reference_configuration}:
                reason = "past_configuration"
            elif past[-1]["valid"] < 120:
                reason = "past_last_hour_valid_pings"
            elif sum(x["valid"] >= 120 for x in past) < 18:
                reason = "past_context_valid_pings"
            else:
                reason = None
                row["issued"] = True
                row["configuration"] = reference_configuration
        row["issue_exclusion"] = reason
        if reason:
            issue_exclusions[reason] += 1
        if not row["issued"]:
            rows.append(row)
            continue
        for horizon in (1, 3, 6):
            j = i + horizon - 1
            if j >= len(hours):
                row["horizons"][str(horizon)] = {
                    "target_start": None,
                    "eligible": False,
                    "reason": "partition_boundary",
                    "block_48h": None,
                }
                continue
            target = hours[j]
            observed = target["observed"]
            valid = target["valid"]
            fraction = valid / observed if observed else None
            if observed == 0:
                score_reason = "missing_target"
            elif target["configuration"] != reference_configuration:
                score_reason = "target_configuration"
            elif valid < 120:
                score_reason = "insufficient_valid_pings"
            elif fraction is None or fraction < 0.95:
                score_reason = "below_valid_fraction_floor"
            else:
                score_reason = None
            block = int((times[j] - origin) / np.timedelta64(48, "h"))
            row["horizons"][str(horizon)] = {
                "target_start": target["time"],
                "observed": observed,
                "valid": valid,
                "valid_fraction": fraction,
                "scheduled_coverage": observed / 240,
                "missing_scheduled": max(240 - observed, 0),
                "excess_observed": max(observed - 240, 0),
                "zero_affected_pings": target["zero_affected_pings"],
                "zero_samples": target["zero_samples"],
                "nonfinite_pings": target["nonfinite_pings"],
                "nonfinite_samples": target["nonfinite_samples"],
                "configuration": target["configuration"],
                "eligible": score_reason is None,
                "reason": score_reason,
                "block_48h": block,
            }
        rows.append(row)
    summaries = {}
    for horizon_key in ("1", "3", "6"):
        issued = [row for row in rows if row["issued"]]
        eligible = [row for row in issued if row["horizons"][horizon_key]["eligible"]]
        summaries[horizon_key] = {
            "issued_rows": len(issued),
            "eligible_rows": len(eligible),
            "target_days": len({row["horizons"][horizon_key]["target_start"][:10] for row in eligible}),
            "cutoff_days": len({row["cutoff"][:10] for row in eligible}),
            "populated_48h_blocks": len({row["horizons"][horizon_key]["block_48h"] for row in eligible}),
            "score_exclusions": dict(Counter(
                row["horizons"][horizon_key]["reason"]
                for row in issued if row["horizons"][horizon_key]["reason"] is not None
            )),
            "target_days_list": sorted({row["horizons"][horizon_key]["target_start"][:10] for row in eligible}),
        }
    return {
        "rows": rows,
        "issue_exclusions": dict(issue_exclusions),
        "horizons": summaries,
        "hourly_observed_pings": sum(x["observed"] for x in hours),
        "hourly_valid_target_pings": sum(x["valid"] for x in hours),
        "zero_affected_pings": sum(x["zero_affected_pings"] for x in hours),
        "zero_affected_hours": sum(x["zero_affected_pings"] > 0 for x in hours),
        "zero_samples": sum(x["zero_samples"] for x in hours),
        "nonfinite_pings": sum(x["nonfinite_pings"] for x in hours),
        "nonfinite_hours": sum(x["nonfinite_pings"] > 0 for x in hours),
        "nonfinite_samples": sum(x["nonfinite_samples"] for x in hours),
        "missing_scheduled_pings": sum(max(240 - x["observed"], 0) for x in hours),
        "excess_observed_pings": sum(max(x["observed"] - 240, 0) for x in hours),
    }
