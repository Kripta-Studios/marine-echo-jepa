# Definition of done and release matrix

Four separate gates:

G0 DATA: verified real source access, rights, calibration, provenance and eligible windows.
G1 ENGINEERING: reliable local app, unit/integration/security/E2E tests and release integrity.
G2 EXPERIMENT: required families/seeds/controls actually executed, fair sealed evaluation and
recomputable outcomes. A valid negative hypothesis can pass G2.
G3 INCREMENTAL VALUE: the preregistered JEPA improvement criterion passes. Do not require a
positive G3 to ship a truthful comparison MVP, but never claim it passed when only G1 did.
Commercial Marine validation is `NOT_EVALUATED` regardless of these public-data gates.

## Required release artifacts after implementation

```
release/
  README_RUN.md
  REPORT.md
  DATA_CARD.md
  MODEL_CARD.md
  LIMITATIONS.md
  MARINE_DATA_REQUEST.md
  requirements-locks/
  demo/
    web/                         # built frontend, local assets
    artifacts/                   # approved small real slice and selected weights/replay
    catalog.json
  evidence/
    protocol.json
    split_manifest.json
    training_registry.json
    final_predictions.parquet
    metrics.json
    paired_block_draws.json
    reviews/
    test_logs/
    runtime.json
  SHA256SUMS
```

The release references the full source repository commit. It need not redistribute the raw
4.4GB archive; the licensed derived slice must include provenance/attribution. All numerical
cells are generated from actual artifacts. Missing runs use null/status/reason, never placeholder
zeros, random metrics or a hand-edited 'passed' flag.

## Release checks

Clean production build and CPU inference on Windows. Full offline browser journey, fresh start
with no prior server, relative-path relocation, checksum verification and actionable errors
for missing/corrupt artifacts. Verify a copied/extracted release, not only the developer folder.
Measured goals: CPU single-anchor forecast p95 <=2s after warmup; replay navigation p95<=300ms
for bounded local data; production cold startup <=30s excluding first dependency installation.
These are acceptance targets to measure, not current observations. If live inference fails the
goal, ship explicitly labelled cached replay and mark live-inference performance blocked.

Independent reviewer signs a structured report stating what was executed versus inspected.
Return owner-facing terminal report: implemented scope, dataset/access/calibration status,
actual model/run outcomes, test counts and skips, budget used, limitations, file paths and
exact launch/reproduction commands. No 'done' based only on a roadmap or scaffold.

The handoff helpers shipped now are not these future release artifacts. Keep
`evidence/HANDOFF_QA.md` separate from future application/model evidence.
