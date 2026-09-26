"""Forecast schema and numerical/time invariants; raw counts are a separate contract."""

import math
from datetime import datetime, timedelta
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Explicit UTC required.")
    return parsed


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class HorizonForecast(StrictModel):
    horizon_hours: Literal[1, 3, 6]
    target_start: str
    target_end: str
    quantiles: dict[str, float | None]


class Forecast(StrictModel):
    schema_version: Literal["1.0"]
    dataset_id: str
    domain: str
    data_kind: Literal["public_real", "synthetic_fixture"]
    mode: Literal["cached_replay", "live_cpu", "synthetic_fixture"]
    prediction_cutoff: str
    last_observation_at: str | None
    observation_age_hours: Annotated[float | None, Field(ge=0)]
    range_convention: Literal["range_from_transducer_m", "verified_depth_m"]
    calibrated: bool
    units: Literal["dB re 1 m^-1", "relative_dB", "synthetic_dB"]
    support_fraction: Annotated[float, Field(ge=0, le=1)]
    status: Literal["ok", "abstained"]
    reasons: list[str]
    model_id: str
    checkpoint_sha256: Digest
    preprocessing_sha256: Digest
    split_sha256: Digest
    evidence_id: str
    scope_disclaimer: Annotated[str, Field(min_length=20)]
    forecasts: list[HorizonForecast]

    @model_validator(mode="after")
    def semantics(self) -> Self:
        cutoff = timestamp(self.prediction_cutoff)
        if (self.data_kind == "synthetic_fixture") != (self.mode == "synthetic_fixture"):
            raise ValueError("Synthetic evidence cannot masquerade as public data.")
        if self.units == "dB re 1 m^-1" and not self.calibrated:
            raise ValueError("Physical units require verified calibration.")
        if self.last_observation_at is None:
            if self.status != "abstained" or self.observation_age_hours is not None:
                raise ValueError("Missing history requires explicit abstention.")
        else:
            age = (cutoff - timestamp(self.last_observation_at)).total_seconds() / 3600
            if (
                self.observation_age_hours is None
                or age < 0
                or not math.isclose(age, self.observation_age_hours, abs_tol=1e-6)
            ):
                raise ValueError("Observation age inconsistent with cutoff.")
        if self.status == "abstained" and not self.reasons:
            raise ValueError("Abstention must explain its reason.")
        if sorted(f.horizon_hours for f in self.forecasts) != [1, 3, 6]:
            raise ValueError("Exactly three unique horizons required.")
        keys = ["0.05", "0.25", "0.5", "0.75", "0.95"]
        for item in self.forecasts:
            if timestamp(item.target_start) != cutoff + timedelta(
                hours=item.horizon_hours - 1
            ) or timestamp(item.target_end) != cutoff + timedelta(hours=item.horizon_hours):
                raise ValueError("Incorrect future interval.")
            if set(item.quantiles) != set(keys):
                raise ValueError("Wrong quantile keys.")
            values = [item.quantiles[key] for key in keys]
            if self.status == "abstained":
                if any(v is not None for v in values):
                    raise ValueError("Abstentions cannot contain numeric forecasts.")
            else:
                if any(v is None for v in values):
                    raise ValueError("Successful forecasts require all quantiles.")
                numbers = [v for v in values if v is not None]
                if numbers != sorted(numbers):
                    raise ValueError("Crossed quantiles.")
        return self
