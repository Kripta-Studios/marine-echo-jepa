# Scope and acceptance

## Required vertical slice

Data acquisition -> immutable raw manifest -> calibrated/qualified arrays -> deterministic
preprocessing -> chronological split -> baselines -> compact JEPA experiments -> calibrated
forecast artefacts -> read-only API -> interactive web app -> offline release -> meeting evidence.

## P0 — must implement and execute

- A downloader/probe with explicit access states, attribution, hashes and safe extraction.
- Primary AZFP parser through pinned Echopype; metadata/calibration gates and failure reports.
- OOI EK60 adapter as a bounded contingency, not a second large project.
- Canonical local acoustic store, QC report, station/deployment/episode identities.
- Persistence, daily-seasonal, linear/ridge and tree-quantile baselines.
- A small matched direct neural predictor, temporal EMA-JEPA, and a shared-encoder
  SIGReg/LeWorldModel-inspired alternative, each with executable CPU/CUDA smoke tests.
- A bounded real-data comparison following the experiment protocol; three seeds for the
  eligible neural families and documented computational/scientific failures.
- One-hour, three-hour and six-hour forecasts of a clearly defined backscatter index and
  range profile, with observations unavailable at prediction time excluded.
- Read-only API, typed client, evidence-driven web interface and offline meeting mode.
- Review, negative-result handling, a technical report and a portable release manifest.

## P1 — execute only after P0 is protected

- Frozen Chronos-2 baseline under the allowed licence and an explicit size/time cap.
- Controlled observation-age replay with a simple refresh rule and equal-query-count control.
- Cross-platform OOI diagnostic, separately labelled and never a substitute for primary test.

The observation-age UI is P0; claiming a measured benefit from adaptive refresh is P1. In P0,
show the predefined stale-input stress test and label the refresh threshold as a heuristic.

## Out of scope

Tuna species classification; tonnes/catch; fishing route optimization; commercial fuel savings;
closed-loop MASS feeding; drones/video; RAG chatbot; multi-tenant SaaS; OAuth; billing; Kubernetes;
reinforcement learning; learning from all public sonar archives; generative video; full JEPA-Anything
reproduction; training a large foundation model; automated Marine outreach.

## Honest outcome classes

`COMPLETE_PUBLIC_MVP`: software and bounded real-data experiments complete, whether JEPA wins or not.
`COMPLETE_WITH_NEGATIVE_JEPA`: complete product; baselines remain champion and negative results visible.
`ENGINEERING_DEMO_ONLY`: real ingest/replay works but insufficient data/calibration prevents the planned study.
`ACCESS_BLOCKED`: no permitted real acoustic corpus was acquired; synthetic fixtures may test engineering
but cannot satisfy the public-data requirement.

No external outage authorizes rebranding a fixture as real evidence. No failed improvement threshold
requires pointless retraining until a win appears. An unexecuted model is not a negative result.
