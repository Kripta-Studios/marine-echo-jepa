import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marine_echo.serving.api import create_app


def cached(tmp_path: Path, mutation: str = "none") -> tuple[TestClient, dict[str, object]]:
    stamp = datetime.fromisoformat("2020-02-17T12:00:00+00:00")
    digest = "a" * 64
    payload = {
        "schema_version": "1.0",
        "dataset_id": "test_real",
        "domain": "Test fixture of public-data contract",
        "data_kind": "public_real",
        "mode": "cached_replay",
        "prediction_cutoff": stamp.isoformat(),
        "last_observation_at": stamp.isoformat(),
        "observation_age_hours": 0.0,
        "range_convention": "range_from_transducer_m",
        "calibrated": True,
        "units": "dB re 1 m^-1",
        "support_fraction": 1.0,
        "status": "ok",
        "reasons": [],
        "model_id": "direct",
        "checkpoint_sha256": digest,
        "preprocessing_sha256": digest,
        "split_sha256": digest,
        "evidence_id": "unit-fixture",
        "scope_disclaimer": "Synthetic software fixture; not scientific evidence.",
        "forecasts": [
            {
                "horizon_hours": h,
                "target_start": (stamp + timedelta(hours=h - 1)).isoformat(),
                "target_end": (stamp + timedelta(hours=h)).isoformat(),
                "quantiles": {
                    "0.05": -80.0,
                    "0.25": -75.0,
                    "0.5": -70.0,
                    "0.75": -65.0,
                    "0.95": -60.0,
                },
            }
            for h in [1, 3, 6]
        ],
    }
    if mutation == "future":
        payload["future_truth"] = [1]
    elif mutation == "dataset":
        payload["dataset_id"] = "other"
    elif mutation == "cutoff":
        payload["prediction_cutoff"] = "2020-02-17T13:00:00Z"
    elif mutation == "age":
        payload["observation_age_hours"] = 1.0
    elif mutation == "hash":
        payload["checkpoint_sha256"] = "b" * 64
    elif mutation == "crossing":
        payload["forecasts"][0]["quantiles"]["0.5"] = -90.0
    elif mutation == "nan":
        payload["forecasts"][0]["quantiles"]["0.5"] = float("nan")
    body = json.dumps(payload).encode()
    (tmp_path / "forecast.json").write_bytes(body)
    catalog = {
        "release_class": "SYNTHETIC_TEST_FIXTURE",
        "datasets": [{"id": "test_real"}],
        "models": [
            {
                "id": "direct",
                "status": "AVAILABLE",
                "checkpoint_sha256": digest,
                "preprocessing_sha256": digest,
                "split_sha256": digest,
            }
        ],
        "artifacts": {
            "forecast": {
                "path": "forecast.json",
                "sha256": hashlib.sha256(body).hexdigest(),
                "kind": "forecast",
            }
        },
        "forecasts": {"test_real|direct|2020-02-17T12:00:00+00:00|0": "forecast"},
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    return TestClient(create_app(tmp_path)), {
        "dataset_id": "test_real",
        "model_id": "direct",
        "cutoff": stamp.isoformat(),
        "mode": "cached_replay",
    }


def test_valid_cached_response_passes_contract(tmp_path: Path) -> None:
    client, body = cached(tmp_path)
    assert client.post("/api/v1/forecast", json=body).status_code == 200


@pytest.mark.parametrize("mutation", ["future", "dataset", "cutoff", "age", "hash", "crossing"])
def test_cached_response_mismatch_is_409(tmp_path: Path, mutation: str) -> None:
    client, body = cached(tmp_path, mutation)
    assert client.post("/api/v1/forecast", json=body).status_code == 409


def test_nan_artifact_is_never_served(tmp_path: Path) -> None:
    client, body = cached(tmp_path, "nan")
    assert client.post("/api/v1/forecast", json=body).status_code == 503
