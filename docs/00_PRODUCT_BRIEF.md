# Product brief and decision

## Meeting objective

Earn a concrete follow-up: an authorized small historical export, an acoustic-domain partner,
and an agreed pilot question. Do not try to demonstrate an ocean foundation model in two weeks.

## Product

**Marine Echo JEPA — Acoustic Forecast & Freshness Lab.** A local, read-only application that
replays genuine acoustic observations, forecasts later acoustic state, compares learned models
with strong baselines, and explains the effect of stale observations.

A visitor should be able to choose a real episode, inspect the past, issue a forecast with a
visible cutoff, reveal what happened, and inspect the exact model/data/evaluation evidence.

The proposed integration target is the existing M3iGO / MSB+ / MarineView ecosystem [S01].
There is no existing integration in this project. The public-data MVP demonstrates software
and modelling capability, not business value on Marine's instruments.

## Why this dataset route

A drifting acoustic observatory is a more direct technical proxy for a buoy history than a
moving survey vessel. However, an ice-tethered Arctic observatory still differs fundamentally
from a tropical fish-aggregating device: species, acoustic calibration, environment, movement,
sampling and operational labels differ. These differences remain visible in the app.

Use the primary corpus to evaluate **within-series future prediction**. Do not manufacture
multiple independent buoys by relabelling days or files. OOI is a separate platform/domain,
not a validation set that can be silently pooled into the primary result.

## Capability ladder

1. Data access, calibration, QC and honest replay.
2. Baselines and uncertainty with a sealed chronological test.
3. Compact predictive embeddings and a fair incremental-value comparison.
4. Simulated observation-age degradation and data-refresh triggers.
5. A company-data adapter contract and a clear pilot request.

Only steps 1–4 are implemented against public data. Step 5 is a schema/demo adapter, not a
reverse-engineered proprietary parser. See `references/SOURCES.md` for source identifiers.
