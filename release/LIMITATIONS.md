# Limitations and incomplete gates

**Snapshot:** 2026-09-26
**Disposition:** The evidence supports an engineering-only diagnostic. P0 is incomplete; this is
not a completed MVP and not a negative JEPA result.

## Data and calibration

- The primary source is a real Arctic research deployment, not an operational fishing buoy or a
  Marine Instruments dataset. The catalog’s deployment span does not prove complete, usable
  coverage for every day.
- The present real-data replay is a single-day diagnostic from 2020-02-17: 96 trailing
  15-minute bins, 24 `.01B` chunks, and 5,564 pings. It does not meet the protocol’s minimum
  chronological benchmark eligibility.
- Replay values are raw digitizer counts aggregated over sample indices. The sample index is not
  a verified physical range or depth. Do not label these data as dB, Sv, calibrated backscatter,
  or metres.
- The primary AZFP environmental calibration is blocked. Verified seawater temperature,
  salinity, pressure and absorption provenance are absent; the configured XML sound speed alone
  is insufficient. Internal sonar thermistor readings do not establish seawater temperature.
- The platform manual says the downward instrument is serial 55169, while the archive XML/DPL
  identify 55170. The manual is platform context only until instrument-specific provenance is
  resolved. Its legacy operational details are not redistributed.

## Fallback and resources

- OOI is a separate upward-looking EK60 feasibility check, not a replacement primary dataset.
  Only one 52,436,316-byte raw file was acquired and parsed. Its indicative calibration-code
  path and output round-trip do not amount to independent field calibration or benchmark
  eligibility.
- A rough 90-day OOI acquisition estimate is about 63.6 GiB, exceeding the configured 40 GiB
  raw-data ceiling. The fallback is `RESOURCE_PLAN_BLOCKED`; no broad OOI download or dataset
  switch has occurred.
- Free disk was measured at about 62.6 GiB before extraction and 49.3 GiB afterward, below the
  100 GiB full-pipeline planning threshold. Process-tree peak RAM was not measured. No paid
  infrastructure was used; spend is USD 0.

## Models and evaluation

- Benchmark runs completed: **0**. Core model smoke implementation remains in progress; a smoke
  test or partial code path is not a benchmark result.
- The active protocol still needs independent R1/R2 review and freeze. The 90-day overall, 12-day
  calibration, and 20-day test eligibility requirements have not been established on canonical
  quality-controlled data.
- There is no selected conventional baseline, completed direct neural comparison, completed JEPA
  candidate comparison, empirical interval calibration, paired confidence interval, or final
  test report in this engineering snapshot.
- JEPA value is **NOT_EVALUATED**. Missing experiments are not a negative result; no improvement
  or failure claim is warranted.

## Application and transfer

- The five-screen local interface is wired to source and raw replay evidence. Calibrated forecast
  output is unavailable. Observation-age masking is a replay simulation; its forecast error,
  interval width, refresh benefit, and operational cost have not been measured.
- Final offline journey, clean-start, relocation, release-hash, CPU-inference and independent R3
  review evidence must be reported by the leader before release. This documentation does not
  assert those gates passed.
- Commercial Marine validation is **NOT_EVALUATED**. These public data cannot establish tuna
  biomass, species, catch, fuel savings, or Marine production integration. No customer data were
  supplied and no outreach was made.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use and
[MARINE_DATA_REQUEST.md](MARINE_DATA_REQUEST.md) for the internal evidence checklist that would
address the main blockers.


## Later evidence update

The internet search found and downloaded MOSAiC environmental profiles (DOI 10.18739/A21J9790B). Their spatial/temporal applicability and calibration procedure remain unapproved; see ADR0003. Earlier missing-input statements describe the initial supplied files, not the absence of public environmental data. Model software tests and a 100-update CUDA resource profile passed using synthetic fixtures, not benchmark data. All three configured native agents subsequently returned usage-limit errors. Final independent scientific/release review and Sol review of these Luna-authored drafts remain incomplete.
