# ADR 0008: prospective AEON hourly Sv study

Date: 2026-09-27. Status: PROPOSED, awaiting independent prefit review.
Study identity: `aeon3_geb_2024_hourly_sv_v1`. This is neither historical v1 nor the
MOSAiC v2 calibrated candidate or raw-response development experiment.

## Source decision

Use version 2 of the [AEON AZFP Integrated Sv products](https://figshare.com/articles/dataset/AZFP/29247113), specifically Figshare file 61937281,
`AEON3 GEB Mar2024-Mar2025 AZFP Sv.zip`, 44,727,557 bytes, publisher MD5
`083769f09573164ea8f43ce273971ec0`, local SHA-256
`4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde`.
The fixed Georges Basin lander is documented by the [UNH AEON portal](https://eos.unh.edu/aeon/data/aeon-data-portal).
The ZIP has 38, 125, 200 and 455 kHz processed CSVs with the source filename
identifier 55144; its binding to a physical instrument serial is not independently
verified. It is 340,711,670 bytes expanded. It and the 23,400-byte source processing
readme were acquired within the local resource cap and verified against the publisher MD5.
This source was selected by fixed location, longer contiguous nominal coverage, four
native channels, a publisher-described Sv calibration/processing path, and small public
transfer size, before inspecting any acoustic outcome values or scores. The 2021 AEON3
archive was a metadata pilot and is not joined to the selected deployment.

The readme says Echoview applied AZFP per-unit calibration coefficients, TVG and
frequency-dependent absorption and then noise removal, median filtering, manual
exclusions and surface correction. The exact 55144 coefficient files, `.ecs` settings,
processing version and raw-source hashes are not yet independently verified. The
published variable is called `Sv_mean`, in calibrated volume-backscattering-strength
units by source description. We will label it **source-reported conditioned Sv**,
not independently field-calibrated truth. The 15-degree off-vertical beam and
source-defined depth reference mean this is not a direct fish density or biomass.

## Observation and target

The primary target is the single 38 kHz row `Sv_mean` from each published
`60minFullDepth.csv` interval, whose metadata define the fixed layer 0–200 m.
It is the Echoview mean Sv over the source-conditioned analysis domain
nominally bounded 0–200 m;
do not recompute a mean of dB numbers, invent raw-ping support, replace absent
rows, or use `NASC` as Sv. `Height_mean` and exclusion geometry are reported as
quality descriptors, not interpreted as verified raw-ping coverage. A row is
target-observed only when exactly one record exists for its source `Interval`,
its fixed layer is 0–200 m, `Ping_E - Ping_S + 1 == 150` for the nominal complete
source interval, and `Sv_mean` is finite and not one of Echoview's documented
special no-data/threshold export values (`-999`, `999`, `-9999`, `-9.9e37`,
`9.9e37`). The [Echoview special-value reference](https://support.echoview.com/WebHelp/Reference/File_Formats/Export_File_Formats/Special_Export_Values.htm)
is the source for those exclusions; do not trim other numerical extremes using
held-out outcomes. Duplicates, absent rows, nonfinite/special values, partial
source intervals, and geometry changes are distinct failures. The 150-ping rule
is fixed from source metadata: it is the modal complete 60-minute integration
interval; 12 known partial intervals occur at file/deployment boundaries.
Complete pings do not establish complete valid acoustic-sample support after
the publisher's exclusions. Do not interpret noise-thresholded or excluded
returns as zeros. `60minPartition` 5 m profiles are not model inputs or
targets in this study; any profile experiment requires its own amendment.

The main task predicts the next, third and sixth *scheduled source hourly
products* from the preceding 24 consecutive source `Interval` IDs. The source
date/time fields describe a measurement summary near each hour but the public
readme does not establish UTC or a product-availability latency. Preserve source
timestamps verbatim; do not silently label them UTC or claim live issuance.
Horizon labels are nominal 1/3/6 source-hour steps, not exact wall-clock UTC
intervals. Validate monotone timestamps and roughly hourly spacing from metadata,
and reject discontinuities rather than bridging them. Inputs may include only
prior source products and prior missingness; future `Sv_mean`, validity, pings,
height, surface line and processing exclusions are never model inputs.

The fixed model input is a 24 × 4 tensor of prior published `60minFullDepth`
`Sv_mean` values at 38/125/200/455 kHz, plus a 24 × 4 observed/complete mask.
No later feature or depth-profile selection is permitted from model scores.
For development issuance, require all 24 predecessor interval IDs present,
strictly ordered, with a complete, target-observed 38 kHz product at each
predecessor. Other-channel missingness or partial intervals are masked using
only their own past metadata. Do not use
future target availability to issue a prediction. Score only a predeclared
future interval with a target-observed row; preserve all issued predictions
and their non-scoring reasons. Evaluate all compared families on identical
target-observed issued rows and also report coverage over every issued row.
There is no unsupported claim of 80% raw target-ping support: this is a
different, publisher-aggregated observand from the v1 target.

## Temporal allocation and gates

Freeze the 360 source-date-field calendar days from 2024-03-06 through
2025-02-28 inclusive before numeric outcome processing. Allocate dates by
calendar, independent of missingness or Sv: TRAIN 2024-03-06 to 2024-10-07
(216 days), validation 2024-10-08 to 2024-11-30 (54), interval calibration
2024-12-01 to 2025-01-05 (36), retrospective test 2025-01-06 to
2025-02-28 (54). A window's full context and last target must remain inside
its partition; enforce interval-ID and derived-row disjointness. Immutable monthly
source CSV files can straddle a calendar split, so their file hashes may recur
as provenance in adjacent partitions. Never compute a file-level statistic,
normalizer, target, feature, or pretraining sample across the partition boundary;
filter by source date and interval ID before any fitting or aggregation. Audit
the exact boundary rows and raw interval identities for overlap or future-context
leakage. A shared source container is not permission to share source observations.
No source-day windowing based on a post hoc good-data segment.

Eligibility before confirmatory comparison requires at least 90 source dates
with 18 scored hourly anchors for *each* 1/3/6-step horizon overall, at least
60 such TRAIN dates, 20 validation dates, 12 calibration dates and 20 test
dates. Count dates and prediction coverage by partition and horizon. These
floors reflect minimum repeated daily coverage and the protocol's daily
comparison unit; they are not a relaxation of v1's failed 90-day gate.
If these floors fail, report ineligibility and stop the confirmatory campaign.
Metadata-only inventory found 360 dates with 38 kHz `60minFullDepth` rows;
356 have 24 interval IDs, two have 23, one 21 and one 5. This is record
presence only, not scored-day or valid-Sv evidence.

Use source-date-blocked daily pinball loss over five fixed quantiles
0.05/0.25/0.5/0.75/0.95 and three horizons as primary. Report median MAE,
quantile coverage/width, per-horizon and per-day differences, abstentions,
source-site/single-deployment limits and three-seed variability. Select model
configuration on validation only. Calibration widens a frozen 90% interval
only; no model choice there. Pretraining, normalizers, imputers and residual
statistics use train only. Keep the finite conventional, matched direct
neural, EMA-JEPA and shared-SIGReg families, three neural seeds (7, 13, 23),
random/shuffled controls and equal input/step budgets from the existing
campaign contract, adapted to 24 hourly steps. Execute a real baseline and
real direct-neural development run before discretionary UI/release work.

Before opening retrospective test acoustic outcomes, bind source hashes,
split, feature/target code, exact cohort, selected checkpoints, baselines,
calibration, metrics and bootstrap procedure in a separately reviewed freeze.
If prior exposure cannot be excluded, call the result retrospective evaluation,
not a sealed or external generalization claim. Preserve v1/v2 failures,
registries and releases. No paid resources, cloud training, or external contact.

## Reconsideration triggers

Independent review must resolve whether source processing/calibration lineage
supports the intended physical-unit label, whether unknown timezone permits
source-date comparison, whether `Interval` is stable across month boundaries,
and whether conditioning/manual exclusions bias the target population. Any
material target or split repair must be prospective, versioned and reviewed
before fitting. No favorable validation/test outcome may choose another source,
depth band, interval mapping or threshold.
