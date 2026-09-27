"""Native candidate shards are streamed only through verified source manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training.v2_native_reader import NativeDevelopmentStream


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, str]:
    shard = root / "data/processed/v2-native-development/2020-02-17.npz"
    manifest = root / "evidence/v2/native-development/2020-02-17.json"
    shard.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    starts = np.datetime64("2020-02-17T00:00") + np.arange(96) * np.timedelta64(15, "m")
    linear = np.full((96, 4, 64), np.nan)
    linear[:, 0, 5:50] = 1e-8
    weight = np.zeros((96, 4, 64))
    weight[:, 0, 5:50] = 30.0
    raw_times = (starts[:, None] + np.arange(40) * np.timedelta64(15, "s")).reshape(-1)
    np.savez_compressed(
        shard,
        bin_start=starts,
        bin_end=starts + np.timedelta64(15, "m"),
        linear_sv=linear,
        detected_range_ping_m=weight,
        observed_ping_count=np.full(96, 40),
        expected_ping_count=np.full(96, 60),
        raw_ping_time=raw_times,
        frequency_hz=np.array([38000, 125000, 200000, 455000]),
        range_edges_m=np.arange(65) * 2.0,
        configuration_id=np.array(["a" * 64] * 96),
        native_detected_sample_count=np.full(96, 400),
    )
    manifest.write_text(
        json.dumps(
            {
                "data_kind": "REAL",
                "study_id": "v2_candidate2",
                "status": "NATIVE_TRAIN_DEVELOPMENT_PROCESSED_NOT_APPROVED_FOR_FITTING",
                "day": "2020-02-17",
                "shard_path": str(shard.relative_to(root)).replace("\\", "/"),
                "shard_sha256": _sha(shard),
                "bindings": {"native_freeze": {"path": "fixture", "sha256": "d" * 64}},
                "source_hours": [
                    {
                        "source": "source.01A",
                        "source_sha256": "b" * 64,
                        "configuration_sha256": "a" * 64,
                        "pings_in_day": 3840,
                        "first_utc": str(raw_times[0]),
                        "last_utc": str(raw_times[-1]),
                    }
                ],
                "historical_manifest_sha256": "e" * 64,
                "observed_pings": 3840,
            }
        ),
        encoding="utf-8",
    )
    index = manifest.parent / "index.json"
    index.write_text(
        json.dumps(
            {
                "data_kind": "REAL",
                "study_id": "v2_candidate2",
                "bindings": {"native_freeze": {"path": "fixture", "sha256": "d" * 64}},
                "days": {
                    "2020-02-17": {
                        "manifest_path": str(manifest.relative_to(root)).replace("\\", "/"),
                        "manifest_sha256": _sha(manifest),
                        "shard_sha256": _sha(shard),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return index, _sha(index)


def test_native_reader_verifies_index_manifest_and_shard(tmp_path: Path) -> None:
    index, digest = _fixture(tmp_path)
    row = next(
        iter(NativeDevelopmentStream(tmp_path, index, index_sha256=digest, fixture_only=True))
    )
    assert row.linear_sv.shape == (4, 64)
    assert row.detected_range_ping_m[0, 5:50].sum() == 45 * 30
    assert row.source_sha256 == ("b" * 64,)
    assert row.configuration_id == "a" * 64
    shard = tmp_path / "data/processed/v2-native-development/2020-02-17.npz"
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest"):
        next(iter(NativeDevelopmentStream(tmp_path, index, index_sha256=digest, fixture_only=True)))


def test_native_reader_requires_exact_index_digest(tmp_path: Path) -> None:
    index, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="index digest"):
        NativeDevelopmentStream(tmp_path, index, index_sha256="0" * 64, fixture_only=True)
