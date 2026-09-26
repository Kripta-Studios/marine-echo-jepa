# Marine Echo JEPA — engineering-only meeting brief

**Evidence snapshot:** 2026-09-26
**Current release class:** `ENGINEERING_DEMO_ONLY`
**P0 status:** INCOMPLETE — no benchmark runs have completed.

> The demonstrator replays a real Arctic AZFP diagnostic in raw digitizer counts. Primary
> physical calibration is blocked, forecast experiments have not run, and neither JEPA value
> nor commercial Marine validation has been evaluated. This is not a negative JEPA result.

## Opening statement

“We have verified and safely extracted the locally supplied PANGAEA archive and can replay one
real day of downward-looking AZFP observations. The replay is deliberately shown as raw counts
over sample indices because the available files do not establish the seawater inputs needed for
physical calibration. No benchmark forecast has been run, so this demonstration says nothing
about JEPA value or commercial performance.”

## Meeting walk-through

1. **Establish provenance.** Show the source bar for PANGAEA 949811, its CC BY 4.0 attribution,
   local SHA-256/ZIP CRC status, and the persistent scope notice. The source archive and full raw
   instrument files remain in the local research workspace; the meeting documents do not include
   the raw archive or instrument manual.
2. **Replay the diagnostic.** Select the real 2020-02-17 diagnostic window and a channel. The
   recorded derivative contains 96 trailing 15-minute bins built from 24 `.01B` chunks and
   5,564 pings. Each bin contains four channels by 64 sample-index bands of mean raw digitizer
   counts. Sample index is not range in metres or verified depth.
3. **Show the calibration boundary.** The XML/DPL identify AZFP 55170, while the platform manual
   lists a different downward unit, 55169. Verified seawater temperature, salinity, pressure,
   and absorption provenance are not available. Keep the displayed unit as counts; do not label
   it dB, Sv, or a physical backscatter index.
4. **Keep forecast and outcome separate.** The comparison view reports that calibrated forecasts
   are unavailable. The replay may reveal later observed raw counts only after the reveal action;
   that is not a model forecast. The observation-age control offers predefined 0/1/3/6-hour
   masks as a labelled replay simulation and has no measured error or interval-width result. The
   displayed three-hour refresh threshold is a demonstration heuristic, not an optimized schedule.
5. **Close on evidence and the next decision.** Show the run registry as not executed, the
   protected-test state as not opened, and the evidence export. Ask for authoritative
   instrument-matched calibration inputs or an owner-approved resource plan before considering
   more data acquisition. This is an internal meeting handout; no customer outreach or data
   request has been sent.

## Evidence and open gates

| Area | Verified state at this snapshot |
|---|---|
| Primary source | Five owner-supplied PANGAEA files inventoried locally. The 4,676,515,350-byte ZIP has local SHA-256 `bc9cf56bd92e7dd1c83cfca6d296585a832e13e27635c04f14ff864bf2831ab9`; a full ZIP CRC scan found no bad member. |
| Extraction | 3,744 members and 10,169,592,780 declared expanded bytes extracted with per-member CRC/SHA checks. Two XML and two DPL files were preserved. |
| AZFP physical calibration | **BLOCKED.** Environmental calibration inputs and instrument-matched provenance are incomplete; the manual/archive serial discrepancy remains unresolved. No dB, Sv, or metre-range result is claimed. |
| Real replay | One bounded public-data diagnostic for 2020-02-17, explicitly `raw_counts`, uncalibrated, with a `sample_index` range convention. It is not the benchmark dataset or a model result. |
| OOI fallback | One 52,436,316-byte raw file was acquired for feasibility. The pinned Echopype calibration-code path used indicative fields from that file and an xarray round-trip equality check passed; this is not independent field calibration or an eligible benchmark. A rough 90-day estimate is about 63.6 GiB, above the 40 GiB raw-data cap. **RESOURCE_PLAN_BLOCKED.** The active primary dataset was not changed. |
| Forecast experiments | **0 benchmark runs completed.** Model smoke implementation is in progress; it is not counted as a benchmark run. |
| JEPA incremental value | **NOT_EVALUATED.** No positive or negative scientific conclusion is available. |
| Commercial Marine validation | **NOT_EVALUATED.** Public Arctic research data cannot establish tuna biomass, species, catch, fuel savings, or Marine production integration. |
| Spend | USD 0 paid infrastructure; no cloud resource was provisioned. |

The protocol document describes the intended experiment, but the active run protocol still
requires independent R1/R2 freeze. No final-test outcome is reported here. Full P0 completion,
offline release verification, and independent release review remain outstanding. The correct
meeting description is an engineering-only diagnostic, not a completed forecasting MVP.

## Next evidence needed

- Obtain authoritative seawater/environmental calibration inputs with provenance appropriate to
  the AZFP serial in the archive, plus configuration or calibration records that resolve the
  manual serial conflict.
- If primary physical calibration remains unavailable and OOI is still considered, obtain an
  owner-approved acquisition, disk, and retention plan that addresses the 40 GiB raw-data limit.
- Complete the frozen data eligibility, baseline, matched neural/JEPA experiments, calibration,
  and independent reviews before describing model performance.
- Collect separately authorized customer data and acceptance criteria before evaluating any
  Marine operational or commercial claim.

### Source attribution

De La Torre, Pedro R.; Berge, Jørgen; Granskog, Mats A.; Katlein, Christian; Divine, Dmitry V.;
Raphael, Ian; Geoffroy, Maxime; Vogedes, Daniel; Itkin, Polona; Daase, Malin; Zolich, Artur;
Cottier, Finlo (2022). *Data from downward looking Acoustic zooplankton and fish profiler (AZFP)
deployed on drifting sea ice in the Arctic during MOSAiC expedition.* PANGAEA.
[https://doi.org/10.1594/PANGAEA.949811](https://doi.org/10.1594/PANGAEA.949811). License: CC BY 4.0.


## Later evidence update

The internet search found and downloaded MOSAiC environmental profiles (DOI 10.18739/A21J9790B). Their spatial/temporal applicability and calibration procedure remain unapproved; see ADR0003. Earlier missing-input statements describe the initial supplied files, not the absence of public environmental data. Model software tests and a 100-update CUDA resource profile passed using synthetic fixtures, not benchmark data. All three configured native agents subsequently returned usage-limit errors. Final independent scientific/release review and Sol review of these Luna-authored drafts remain incomplete.
