# API contract v1

Bind loopback only. GET operations are read-only. Schema IDs, units and provenance are required,
not decorative frontend fields. Use FastAPI-generated OpenAPI and test the actual schema.
The attached `schemas/forecast.schema.json` is the initial interchange contract; extend only
through versioned ADRs and synchronized frontend types.

| Endpoint | Purpose | Required restrictions |
|---|---|---|
| `GET /health` | app/build/artifact readiness | no secrets, paths or GPU requirement |
| `GET /api/v1/datasets` | catalogue, license, domain, calibration state | explicit real/synthetic; no fake fleet |
| `GET /api/v1/datasets/{id}/observations` | bounded echogram and telemetry slice | max48h, <=1024 time bins, past cutoff optional |
| `GET /api/v1/models` | versions, family, scope, promotion status | rejected candidates remain visible in lab |
| `POST /api/v1/forecast` | local inference or cached replay | typed IDs, UTC cutoff, horizon enum and mode |
| `GET /api/v1/evidence/{id}` | report, run/split hashes and failures | only published catalog entries |
| `GET /api/v1/experiments` | comparison table with uncertainty | no request-time evaluation or metric recalculation |
| `GET /api/v1/exports/{id}` | approved CSV/JSON forecast export | static allowlist; safe CSV cells |

Forecast is computationally read-only even though POST is used for its structured request.
No training/deletion/deployment endpoints. Limit forecast batch to one anchor and three horizons;
cap request body and timeouts. Return 422 for invalid UTC, unknown horizon or invalid schema;
404 for unknown IDs; 409 for artifact/protocol mismatch; 503 for unavailable model. A scientifically
unsupported window yields a valid response with `status=abstained`, null values and reasons,
not made-up zeros. Never serialize NaN/Infinity into JSON.

Each response carries: schema_version, dataset_id, domain, data_kind, mode, prediction_cutoff,
last_observation_at, observation_age_hours, horizon definitions, range convention, calibrated
flag, units, quantiles, support_fraction, status, reasons, model_id, checkpoint_sha256,
preprocessing_sha256, split_sha256, evidence_id and scope_disclaimer.

Raw observation freshness must reflect actual availability or the explicit replay assumption.
Ground truth is a separate endpoint/slice revealed after the cutoff or by a visibly named
`Reveal outcome` control. No frontend state may feed revealed future values into forecast
requests. Test network payloads, not merely whether the chart hides the future.

Contract testing examples: a stale-window request; all-missing window; a missing 455kHz channel;
corrupt checkpoint hash; range-vs-depth label; cross-origin request; path traversal; a timezone
without offset; oversized date range; an HTML download masquerading as a dataset. Include
cache invalidation keyed by data+model+preprocessing hashes, cutoff and mode, not cutoff alone.
