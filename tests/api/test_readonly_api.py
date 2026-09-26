"""Independent expected HTTP boundaries for the local artifact server."""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marine_echo.serving.api import create_app


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    observations = {
        "dataset_id": "test_real",
        "units": "raw_counts",
        "calibrated": False,
        "data_kind": "public_real",
        "frequency_hz": [38000],
        "range_convention": "sample_index",
        "rows": [
            {"event_time_utc": "2020-02-17T00:00:00Z", "counts": [[12, 13]]},
            {"event_time_utc": "2020-02-17T01:00:00Z", "counts": [[14, 15]]},
        ],
    }
    body = json.dumps(observations).encode()
    (tmp_path / "observations.json").write_bytes(body)
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "release_class": "ENGINEERING_DEMO_ONLY",
                "datasets": [{"id": "test_real", "observations_id": "obs"}],
                "models": [{"id": "direct", "status": "NOT_RUN"}],
                "experiments": [],
                "artifacts": {
                    "obs": {
                        "path": "observations.json",
                        "sha256": hashlib.sha256(body).hexdigest(),
                        "kind": "observations",
                    }
                },
            }
        )
    )
    return tmp_path


def test_health_and_read_only(catalog: Path) -> None:
    client = TestClient(create_app(catalog))
    assert client.get("/health").json()["ready"] is True
    assert client.delete("/api/v1/datasets/test_real").status_code in (404, 405)
    assert client.get("/api/v1/models").json()[0]["status"] == "NOT_RUN"


def test_past_payload_excludes_future(catalog: Path) -> None:
    client = TestClient(create_app(catalog))
    response = client.get(
        "/api/v1/datasets/test_real/observations",
        params={
            "start": "2020-02-17T00:00:00Z",
            "end": "2020-02-17T02:00:00Z",
            "cutoff": "2020-02-17T00:30:00Z",
        },
    )
    assert response.status_code == 200
    assert len(response.json()["rows"]) == 1
    assert response.json()["rows"][0]["counts"] == [[12, 13]]


@pytest.mark.parametrize(
    "query",
    [
        {"start": "2020-02-17", "end": "2020-02-18"},
        {"start": "2020-02-17T00:00:00Z", "end": "2020-03-18T00:00:00Z"},
    ],
)
def test_reject_bad_time_bounds(catalog: Path, query: dict[str, str]) -> None:
    client = TestClient(create_app(catalog))
    assert client.get("/api/v1/datasets/test_real/observations", params=query).status_code == 422


def test_corruption_fails_closed(catalog: Path) -> None:
    (catalog / "observations.json").write_text("{}")
    client = TestClient(create_app(catalog))
    assert client.get("/health").status_code == 503
    assert client.get("/api/v1/datasets").status_code == 503


def test_path_escape_rejected(catalog: Path) -> None:
    body = json.loads((catalog / "catalog.json").read_text())
    body["artifacts"]["obs"]["path"] = "../secret.json"
    (catalog / "catalog.json").write_text(json.dumps(body))
    assert TestClient(create_app(catalog)).get("/health").status_code == 503


def test_unexecuted_model_and_extra_fields_rejected(catalog: Path) -> None:
    client = TestClient(create_app(catalog))
    request = {
        "dataset_id": "test_real",
        "model_id": "direct",
        "cutoff": "2020-02-17T00:30:00Z",
        "mode": "cached_replay",
    }
    assert client.post("/api/v1/forecast", json=request).status_code == 503
    assert (
        client.post("/api/v1/forecast", json={**request, "future_truth": [12]}).status_code == 422
    )
    assert (
        client.post("/api/v1/forecast", json={**request, "cutoff": "2020-02-17"}).status_code == 422
    )


def test_reject_external_origins_and_large_body(catalog: Path) -> None:
    client = TestClient(create_app(catalog))
    assert client.get("/health", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/v1/forecast", content="x" * 20000).status_code == 413


def test_unknown_export_does_not_disclose_paths(catalog: Path) -> None:
    response = TestClient(create_app(catalog)).get("/api/v1/exports/not-registered")
    assert response.status_code == 404
    assert str(catalog) not in response.text
