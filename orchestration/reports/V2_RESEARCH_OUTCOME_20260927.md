# Prospective v2 research outcome — 27 September 2026

The owner authorized the prospective v2 data/protocol amendment in
`continuation_v2/CODEX_FINISH_V2.md`. This is a new study. V1's unchanged
90-day eligibility threshold, negative support findings, blocked 25-slot
registry, and historical r2 release remain intact.

## Observation-aligned calibrated target

The first conservative two-metre grid candidate under ADR 0005 had zero
eligible hourly targets under its frozen detection rule. ADR 0006 fixed a
native range-length 38 kHz, 10–100 m calibrated Sv candidate and a 10%
detection floor before the 58-day TRAIN-development processing. Its fit segment
had 21/22/21 eligible target days at 1/3/6 hours. The fixed April assessment
segment had zero at all horizons because the maximum detection fraction was
below 10%. The independent reviewer reconstructed the support report and
accepted this negative data-eligibility finding. ADR 0006 terminates this
calibrated-target search. No calibrated core fit or final evaluation was
scientifically authorized.

## Separate raw response engineering study

ADR 0007 defined a transformed complete-positive AZFP 38 kHz instrument
response-code mean. Its fit segment contained 42 target days per horizon; its
fixed April TRAIN-development assessment contained 13 target days and seven
48-hour blocks per horizon. This is a code-valued instrument response, not
calibrated Sv, acoustic backscatter in physical units, fish, biomass, or catch.

The first real ridge/direct execution produced a ridge prediction file and
direct checkpoints but failed the frozen deterministic replay gate with a
maximum absolute weight difference of 2.09e-5. It was not promoted. A
prospective, independently approved deterministic CUDA remedy used a new run
identity and passed exact step-64-to-128 replay. The successful run issued
212 predictions for each model and scored 176/140/135 rows at 1/3/6 hours
on identical fixed April development cohorts. The distinct reviewer recomputed
every saved-row metric and loaded the saved step-64 and step-128 checkpoints.

| Horizon | Ridge daily mean pinball, code | Direct daily mean pinball, code | Ridge median MAE, code | Direct median MAE, code |
| --- | ---: | ---: | ---: | ---: |
| 1 h | 12.2312 | 64.9060 | 37.5834 | 210.5434 |
| 3 h | 19.9990 | 64.3886 | 62.5379 | 207.4076 |
| 6 h | 30.0450 | 65.9337 | 97.4132 | 215.9236 |

Ridge outperformed this direct neural model in the fixed raw-code development
comparison at all three horizons. This is no JEPA result: no JEPA model ran.
The calibrated v2 25-slot campaign remains NOT_RUN, and the April rows are
TRAIN development, not a sealed holdout or final evaluation. The public-source
experiment does not validate operations or a Marine Instruments product claim.

The successful result SHA-256 is
`4237f375ba129fce95be952325e56cf5bc9615a5d89b2ed9ba91dfc8771cb49d`.
The step-128 direct checkpoint SHA-256 is
`713e4dba5bf94f8f29abd2df60c674b68a9b6b89dbaa3f7ce64fc386c1848e49`.
Exact prediction hashes and the independent approval are in
`orchestration/reviews/V2_RAW_RESPONSE_DEVELOPMENT_RESULT_20260927.json`.
The two real runs did not consume calibrated core slots. Peak sampled process
RSS was 2,068,160,512 bytes and peak GPU reserved memory was 102,760,448
bytes, within the local limits. Paid infrastructure spending was USD 0.

## Release classification

The separate v2 offline package is an engineering research release. It shows
all reviewed issued rows, metrics, checkpoint provenance, and explicit gates.
The normal calibrated forecasting MVP and JEPA-value gates are incomplete.
The historical v1 preview and r2 package are not promoted or rewritten by it.
