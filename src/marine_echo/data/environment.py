"""Explicit environmental units and past-only matching; no implicit calibration approval."""

from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from math import asin, cos, isfinite, radians, sin, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray


def parse_iridium_position(line: str) -> tuple[datetime, float, float]:
    """Parse only position fields documented by the local source readme.

    Only source fix code 1 is retained for this conservative audit. Its exact
    receiver semantics still require provenance. Mixed signs are rejected.
    Internal SBPC temperature is deliberately not parsed as water temperature.
    """
    fields = line.split()
    if len(fields) < 10 or fields[0] != "1":
        raise ValueError("Iridium record does not carry an accepted GPS fix.")
    day = [int(x) for x in fields[3].split("-")]
    clock = [int(x) for x in fields[4].split(":")]
    if len(day) != 3 or len(clock) != 3:
        raise ValueError("Invalid Iridium date/time fields.")
    stamp = datetime(day[0], day[1], day[2], clock[0], clock[1], clock[2], tzinfo=UTC)
    coordinates = []
    for index in (5, 7):
        degrees, minutes = float(fields[index]), float(fields[index + 1])
        if (
            not isfinite(degrees)
            or not degrees.is_integer()
            or not isfinite(minutes)
            or abs(minutes) >= 60
            or degrees * minutes < 0
        ):
            raise ValueError("Invalid or ambiguous signed degrees/minutes.")
        coordinates.append(degrees + minutes / 60)
    _coordinates(*coordinates)
    return stamp, coordinates[0], coordinates[1]


def _aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Environmental timestamps must be timezone-aware.")


def _coordinates(latitude: float, longitude: float) -> None:
    if not isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude.")
    if not isfinite(longitude) or not -180 <= longitude <= 360:
        raise ValueError("Invalid longitude.")


@dataclass(frozen=True)
class ProfileRecord:
    observed_at: datetime
    available_at: datetime | None
    latitude: float
    longitude: float
    quality_verified: bool
    source_id: str

    def __post_init__(self) -> None:
        _aware(self.observed_at)
        if self.available_at is not None:
            _aware(self.available_at)
            if self.available_at < self.observed_at:
                raise ValueError("Availability cannot precede the environmental observation.")
        if not isinstance(self.quality_verified, bool):
            raise TypeError("Quality verification must be an explicit boolean.")
        _coordinates(self.latitude, self.longitude)
        if not self.source_id:
            raise ValueError("Environmental source identity is required.")


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _coordinates(lat1, lon1)
    _coordinates(lat2, lon2)
    a, b, c, d = map(radians, (lat1, lon1, lat2, lon2))
    hav = sin((c - a) / 2) ** 2 + cos(a) * cos(c) * sin((d - b) / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(min(1.0, max(0.0, hav))))


def match_profile(
    records: list[ProfileRecord],
    cutoff: datetime,
    latitude: float,
    longitude: float,
    max_age_hours: float,
    max_distance_km: float,
) -> ProfileRecord:
    """Select the latest eligible measured record; bounds require external scientific review.

    A retrospective daily mean is not available at the start of its observation day.
    Unknown historical availability fails closed; this helper makes no latency assumption.
    """
    _aware(cutoff)
    _coordinates(latitude, longitude)
    if not all(isfinite(x) and x > 0 for x in (max_age_hours, max_distance_km)):
        raise ValueError("Matching bounds must be positive and finite.")
    candidates = [
        r
        for r in records
        if r.quality_verified
        and r.available_at is not None
        and r.available_at <= cutoff
        and 0 <= (cutoff - r.observed_at).total_seconds() <= max_age_hours * 3600
        and distance_km(latitude, longitude, r.latitude, r.longitude) <= max_distance_km
    ]
    if not candidates:
        raise ValueError("No compatible environmental record available at issue time.")
    latest = max(r.observed_at for r in candidates)
    selected = [r for r in candidates if r.observed_at == latest]
    if len(selected) != 1:
        raise ValueError(
            "Ambiguous contemporaneous environmental records need a reviewed priority."
        )
    return selected[0]


def seawater_properties(
    depth_m: Any,
    absolute_salinity_g_kg: Any,
    conservative_temperature_c: Any,
    latitude: float,
    longitude: float,
) -> dict[str, NDArray[np.float64]]:
    """Convert measured positive-down depth/SA/CT with pinned GSW, without extrapolation.

    Pressure is sea pressure (dbar), not absolute pressure. This conversion does not
    establish spatial applicability, transducer depth, or profile availability.
    """
    _coordinates(latitude, longitude)
    depth, sa, ct = [
        np.asarray(x, dtype=np.float64)
        for x in (depth_m, absolute_salinity_g_kg, conservative_temperature_c)
    ]
    if (
        depth.ndim != 1
        or not depth.size
        or depth.shape != sa.shape
        or sa.shape != ct.shape
        or not all(np.isfinite(x).all() for x in (depth, sa, ct))
        or (depth < 0).any()
        or (np.diff(depth) <= 0).any()
        or (sa < 0).any()
        or (sa > 50).any()
        or (ct < -5).any()
        or (ct > 50).any()
    ):
        raise ValueError("Invalid physical profile dimensions, units or finite support.")
    gsw = import_module("gsw")
    if gsw.__version__ != "3.6.23":
        raise ValueError("The reviewed conversion path requires gsw==3.6.23.")
    pressure = gsw.p_from_z(-depth, latitude)
    result = {
        "pressure_dbar": pressure,
        "practical_salinity": gsw.SP_from_SA(sa, pressure, longitude, latitude),
        "in_situ_temperature_c": gsw.t_from_CT(sa, ct, pressure),
        "sound_speed_m_s": gsw.sound_speed(sa, ct, pressure),
    }
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("TEOS-10 conversion produced unsupported values.")
    return result
