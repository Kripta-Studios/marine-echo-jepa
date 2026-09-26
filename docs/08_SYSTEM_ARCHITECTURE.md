# System architecture

## Deliberately small stack

Python package `marine_echo`: Typer CLI, Pydantic contracts, NumPy/xarray acoustic adapters,
scikit-learn baselines, PyTorch compact models, FastAPI read-only service. Frontend: React +
TypeScript + Vite, a local Canvas/SVG echogram, and a bundled chart library if justified.
Use a supported Node LTS release and exact lockfile after preflight. Do not invent current
package version numbers. Avoid Redis, Kafka, Kubernetes, a managed database, a vector database,
an LLM API or an agent runtime in the delivered app. No chatbot is needed.

Source layout to implement:
```
src/marine_echo/
  contracts/       # units, provenance, forecast and evidence models
  data/            # download inventory, AZFP/OOI adapters, calibration and shards
  features/        # past-only numerical features and windows
  models/          # baselines, encoders, EMA-JEPA, shared-SIGReg and heads
  training/        # loops, checkpoint/resume, one-GPU ownership
  evaluation/      # splits, quantiles, bootstrap, freshness and reports
  serving/         # immutable artifact catalog, local API and CLI
web/src/
  api/ components/ pages/ charts/ styles/
tests/
  unit/ integration/ scientific/ api/ security/ e2e/
```

Storage:
```
data/raw/<source>/<release>/                 # never committed
 data/processed/<dataset_hash>/<day>/        # transactional shards
 data/manifests/<dataset_hash>.json
runs/<run_id>/config.json metrics.json checkpoint.pt manifest.json
reports/<protocol_id>/predictions.parquet summary.json report.md
release/demo/artifacts/                     # small provenance-labelled slice
```
Whitespace before example paths is decorative; actual paths have no leading spaces.
JSON manifests and Parquet summaries suffice. SQLite may index local artifacts if measured
need arises; keep it read-only while serving, rebuildable from manifests.

## Separation and immutable services

The web API cannot launch training, shell commands or arbitrary downloads. It accepts only
validated dataset/model IDs and bounded time ranges, never filesystem paths or remote URLs.
No upload UI in P0. Training writes new run directories; the app reads a frozen release
catalog. The catalog is atomically published only after checksums and schema validation.
Artifact IDs map to allowlisted files below a configured root; resolve-and-containment tests
must reject traversal and symlinks escaping the root.

Frontend production build is served by FastAPI from local static assets. One command starts
the meeting app on 127.0.0.1:8765. Vite is development only. All fonts, scripts, chart code and
sample data needed for the meeting are local. A map can be a simple bundled outline/local
coordinate plot with correct attribution; do not require paid map APIs or online tiles.

## Inference paths

`live_cpu`: the selected loaded model generates outputs from a validated past-only window.
`cached_replay`: immutable precomputed outputs from the same versioned model/config, explicitly
labelled; no implication of current ocean conditions. `synthetic_fixture`: UI/test-only badge
and excluded from evidence. The same typed response is used across paths with mode explicit.

Point forecasts, distribution forecasts, evidence and explanations are separate contracts.
Reason strings are deterministic summaries of observed missingness/age/quality or model
agreement, not generated causal stories. Load trusted locally produced state dicts with safe
weights-only APIs when available; no arbitrary pickle upload or remote checkpoint execution.

## Release reliability

The small demo slice must fit on the laptop and start without the raw 4.4GB archive. Include
at least three preregistered replay cases: a median-error validation case, a high-error
validation case and a deterministic randomly selected eligible case; add a test case only
after final evaluation, selected by a rule frozen before scores. Label selection provenance.
Provide terminal startup diagnostics, friendly missing-artifact errors and a one-command
health check. Capture browser screenshots as QA after the app exists; do not ship fictional
screenshots as evidence of implementation.
