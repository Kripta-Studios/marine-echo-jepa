"""Immutable local artifact API."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("An explicit UTC timestamp is required.")
    return parsed.astimezone(UTC)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset_id: str
    model_id: str
    cutoff: str
    mode: Literal["cached_replay", "live_cpu"] = "cached_replay"
    observation_age_hours: Literal[0, 1, 3, 6] = 0

    @field_validator("cutoff")
    @classmethod
    def check_cutoff(cls, value: str) -> str:
        utc(value)
        return value


class ArtifactStore:
    """Load only bounded, hash-verified JSON bytes under a fixed local root."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.catalog: dict[str, Any] = {}
        self.payloads: dict[str, bytes] = {}
        self.error: str | None = None
        try:
            catalog_path = self.root / "catalog.json"
            if catalog_path.is_symlink() or catalog_path.stat().st_size > 2_000_000:
                raise ValueError("Invalid catalog.")
            self.catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            for key, entry in self.catalog["artifacts"].items():
                relative = Path(entry["path"])
                candidate = self.root / relative
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe artifact path.")
                if not candidate.resolve().is_relative_to(self.root):
                    raise ValueError("Artifact escapes root.")
                if any(
                    p.is_symlink() for p in [candidate, *candidate.parents] if p != self.root.parent
                ):
                    raise ValueError("Linked artifact rejected.")
                if candidate.stat().st_size > 32_000_000:
                    raise ValueError("Artifact size limit exceeded.")
                payload = candidate.read_bytes()
                if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ValueError("Artifact integrity mismatch.")
                self.payloads[key] = payload
        except (OSError, ValueError, KeyError, TypeError):
            self.error = "Missing, corrupt or unsafe artifacts. Verify the release checksums and rebuild the catalog."

    def require_ready(self) -> None:
        if self.error:
            raise HTTPException(503, self.error)

    def json(self, key: str, kind: str | None = None) -> Any:
        self.require_ready()
        if key not in self.payloads or (kind and self.catalog["artifacts"][key]["kind"] != kind):
            raise HTTPException(404, "Unknown artifact ID.")
        return json.loads(self.payloads[key])


def create_app(artifact_root: Path, web_root: Path | None = None) -> FastAPI:
    store = ArtifactStore(artifact_root)
    app = FastAPI(title="Marine Echo JEPA", version="0.1.0", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def boundary(request: Request, call_next: Any) -> Response:
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Cross-origin access is disabled."}, status_code=403)
        # Count streamed bytes; do not trust Content-Length from a client.
        if request.method == "POST":
            size = 0
            chunks = []
            async for chunk in request.stream():
                size += len(chunk)
                if size > 8192:
                    return JSONResponse({"detail": "Request body too large."}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        store.require_ready()
        return {"ready": True, "version": "0.1.0", "release_class": store.catalog["release_class"]}

    @app.get("/api/v1/datasets")
    def datasets() -> Any:
        store.require_ready()
        return store.catalog["datasets"]

    @app.get("/api/v1/models")
    def models() -> Any:
        store.require_ready()
        return store.catalog["models"]

    @app.get("/api/v1/experiments")
    def experiments() -> Any:
        store.require_ready()
        return store.catalog["experiments"]

    @app.get("/api/v1/datasets/{dataset_id}/observations")
    def observations(
        dataset_id: str,
        start: str,
        end: str,
        cutoff: str | None = None,
        limit: int = Query(1024, ge=1, le=1024),
    ) -> Any:
        store.require_ready()
        dataset = next((d for d in store.catalog["datasets"] if d["id"] == dataset_id), None)
        if dataset is None:
            raise HTTPException(404, "Unknown dataset ID.")
        try:
            begin, finish = utc(start), utc(end)
            if not timedelta(0) < finish - begin <= timedelta(hours=48):
                raise ValueError("Request a positive interval of at most 48 hours.")
            if cutoff:
                finish = min(finish, utc(cutoff))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        data = store.json(dataset["observations_id"], "observations")
        selected = [row for row in data["rows"] if begin <= utc(row["event_time_utc"]) < finish]
        if len(selected) > limit:
            raise HTTPException(422, "Too many observations; request a shorter interval.")
        return {**data, "rows": selected}

    @app.post("/api/v1/forecast")
    def forecast(body: ForecastRequest) -> Any:
        store.require_ready()
        if not any(d["id"] == body.dataset_id for d in store.catalog["datasets"]):
            raise HTTPException(404, "Unknown dataset ID.")
        model = next((m for m in store.catalog["models"] if m["id"] == body.model_id), None)
        if model is None:
            raise HTTPException(404, "Unknown model ID.")
        if model["status"] != "AVAILABLE":
            raise HTTPException(
                503, model.get("reason", "Model has not completed the required scientific gates.")
            )
        if body.mode != "cached_replay":
            raise HTTPException(503, "This release supports cached replay only.")
        key = "|".join(
            [
                body.dataset_id,
                body.model_id,
                utc(body.cutoff).isoformat(),
                str(body.observation_age_hours),
            ]
        )
        entry = store.catalog.get("forecasts", {}).get(key)
        if entry is None:
            raise HTTPException(
                503, "No verified cached forecast exists at this cutoff and observation age."
            )
        return store.json(entry, "forecast")

    @app.get("/api/v1/evidence/{evidence_id}")
    def evidence(evidence_id: str) -> Any:
        return store.json(evidence_id, "evidence")

    @app.get("/api/v1/exports/{export_id}")
    def export(export_id: str) -> Response:
        store.require_ready()
        if export_id not in store.catalog.get("exports", []):
            raise HTTPException(404, "Unknown export ID.")
        store.json(export_id, "export")
        return Response(
            store.payloads[export_id],
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="marine-echo-evidence.json"'},
        )

    if web_root is not None:
        app.mount("/", StaticFiles(directory=web_root, html=True), name="web")
    return app
