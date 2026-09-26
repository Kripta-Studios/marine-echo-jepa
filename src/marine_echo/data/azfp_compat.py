"""Pinned Echopype 0.11.1 AZFP singleton-header compatibility.

Use only in a process whose AZFP opens all pass through this wrapper. The
process-local lock serializes wrapped calls; unrelated unwrapped calls do not
participate in it. No raw datagram, ping or acoustic value is modified here.
"""

from __future__ import annotations

import hashlib
import inspect
import threading
from pathlib import Path
from typing import Any

import echopype as ep  # type: ignore[import-untyped]
import numpy as np
from echopype.convert.api import SONAR_MODELS  # type: ignore[import-untyped]
from echopype.convert.parse_azfp import ParseAZFP  # type: ignore[import-untyped]

_EXPECTED_EP_VERSION = "0.11.1"
_EXPECTED_PARSER_SHA256 = "d5ac606f3372b36e940f32df2255d23fdb8848e87d14f8794cface0050275b45"
_LOCK = threading.RLock()
_FREQUENCY_FIELDS = (
    "dig_rate",
    "lock_out_index",
    "num_bins",
    "range_samples_per_bin",
    "data_type",
    "gain",
    "pulse_len",
    "board_num",
    "frequency",
)
_SCALAR_FIELDS = (
    "profile_flag",
    "serial_number",
    "burst_int",
    "ping_per_profile",
    "avg_pings",
    "ping_period",
    "phase",
    "num_chan",
    "spare_chan",
)


class ParseAZFPSingletonCompat(ParseAZFP):
    """Apply upstream's own constant-header reduction to a single ping."""

    def _check_uniqueness(self) -> None:
        if not self.unpacked_data or np.asarray(self.unpacked_data["profile_flag"]).size != 1:
            super()._check_uniqueness()
            return
        for field in _FREQUENCY_FIELDS:
            unique = np.unique(self.unpacked_data[field], axis=0)
            if unique.shape[0] != 1:
                raise ValueError(f"Header value {field} is not constant for each ping")
            self.unpacked_data[field] = unique.squeeze()
        for field in _SCALAR_FIELDS:
            unique = np.unique(self.unpacked_data[field])
            if unique.shape[0] != 1:
                raise ValueError(f"Header value {field} is not constant for each ping")
            self.unpacked_data[field] = unique.squeeze()


def _verify_pinned_upstream() -> None:
    source_name = inspect.getsourcefile(ParseAZFP)
    if ep.__version__ != _EXPECTED_EP_VERSION or source_name is None:
        raise RuntimeError("AZFP compatibility requires the pinned Echopype 0.11.1 parser.")
    with Path(source_name).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != _EXPECTED_PARSER_SHA256:
        raise RuntimeError("Pinned Echopype AZFP parser source differs from reviewed bytes.")


def open_raw_azfp_compat(path: Path, xml_path: Path) -> Any:
    """Call normal Echopype open_raw with temporary singleton normalization.

    Caller must serialize all AZFP opens in this process through this wrapper.
    The original parser binding is restored even if parsing fails.
    """
    _verify_pinned_upstream()
    with _LOCK:
        original = SONAR_MODELS["AZFP"]["parser"]
        if original is not ParseAZFP:
            raise RuntimeError("AZFP parser binding already differs; refusing nested patch.")
        SONAR_MODELS["AZFP"]["parser"] = ParseAZFPSingletonCompat
        try:
            return ep.open_raw(path, sonar_model="AZFP", xml_path=xml_path)
        finally:
            SONAR_MODELS["AZFP"]["parser"] = original
