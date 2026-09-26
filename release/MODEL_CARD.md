# Model card — Marine Echo JEPA research demonstrator

**Snapshot:** 2026-09-26
**Model status:** No benchmark model has been trained, selected, evaluated, or promoted.
**Release status:** Engineering-only; the forecasting P0 is incomplete.

## Summary

Marine Echo JEPA is a local public-data research demonstrator for examining acoustic replay
provenance and the planned comparison of conventional forecasting baselines with compact
predictive-representation candidates. The current application can display a one-day real AZFP
diagnostic as uncalibrated raw counts. It does not provide a validated physical-unit forecast.

The protocol names persistence, seasonal, ridge, histogram-gradient-boosting, direct neural,
EMA-JEPA, and shared-SIGReg model families. Those are planned comparison families, not evidence
that a benchmark training run exists. Benchmark runs completed: **0**. Model smoke implementation
is in progress and does not count as benchmark evidence.

## Intended use

- Inspect local source provenance and a bounded raw-count diagnostic replay.
- Exercise read-only evidence screens, controlled replay interactions, and export paths during
  engineering review.
- Prepare a future within-deployment acoustic forecasting study after the data, calibration,
  eligibility, protocol, and review gates pass.

## Out-of-scope use

Do not use this demonstrator for operational forecasts, instrument intervention, buoy control,
fleet decisions, fish or species identification, biomass, catch, fuel-saving estimates, or claims
about Marine Instruments products or production integration. The current data and code do not
support those uses.

## Inputs and data lineage

The replay source is a single bounded day from the public downward-looking Arctic AZFP record,
aggregated into 96 trailing 15-minute bins. Displayed values are raw digitizer counts across
sample-index bands. Physical range and depth have not been established. See
[DATA_CARD.md](DATA_CARD.md) for source hashes, integrity scope, aggregation and calibration
limits.

Primary physical-unit calibration is blocked: verified seawater environmental inputs and
instrument-matched calibration provenance are missing, and the manual/archive serials conflict.
No train/validation/calibration/test model artifact has been selected from this diagnostic.

## Outputs and uncertainty

No benchmark forecast output is available. The comparison screen must remain visibly blocked or
not executed until actual supported model artifacts are supplied. No prediction interval, median,
loss, accuracy, or latency result is claimed. Empirical uncertainty calibration has not been
performed. Revealed later observations are raw replay data shown after the user action; they are
not model predictions.

## Evaluation and value claim

The experiment protocol is specified in `docs/06_EXPERIMENT_PROTOCOL.md`, but the active run
protocol is not yet frozen following independent review. No final-test result is reported.
Therefore:

- JEPA incremental value: **NOT_EVALUATED**.
- Relative value versus conventional baselines: **NOT_EVALUATED**.
- Commercial Marine validation: **NOT_EVALUATED**.

These states are incomplete evidence, not a negative JEPA finding. A learned acoustic predictor,
if later validated, would remain a predictor of an acoustic index and would not be a causal
biological model.

## Known risks and update conditions

Calibration, full eligible chronological coverage, canonical quality-controlled inputs, frozen
model comparisons, uncertainty evaluation, independent scientific review and offline release
checks remain open. Update this card only from versioned data/run artifacts and approved review
evidence. Do not replace unavailable outcomes with zeros, smooth placeholders or synthetic values.


## Later evidence update

The internet search found and downloaded MOSAiC environmental profiles (DOI 10.18739/A21J9790B). Their spatial/temporal applicability and calibration procedure remain unapproved; see ADR0003. Earlier missing-input statements describe the initial supplied files, not the absence of public environmental data. Model software tests and a 100-update CUDA resource profile passed using synthetic fixtures, not benchmark data. All three configured native agents subsequently returned usage-limit errors. Final independent scientific/release review and Sol review of these Luna-authored drafts remain incomplete.
