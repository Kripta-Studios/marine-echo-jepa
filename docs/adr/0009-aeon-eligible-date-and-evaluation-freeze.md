# ADR 0009: AEON eligible-date scoring and two-stage evaluation freeze

Date: 2026-09-27. Status: PROPOSED FOR INDEPENDENT REVIEW.
Study: `aeon3_geb_2024_hourly_sv_v1`. This prospective interpretation amends
how ADR 0008 is executed after the TRAIN/validation campaign. It does not
modify ADR 0008's file or hash, retroactively approve v1, or reopen a test.

## Validation aggregation correction

ADR 0008 prespecified at least 18 scored hourly anchors per source date for
each horizon. The first campaign executor persisted every issued prediction,
but its displayed daily pinball average included source dates with fewer than
18 scored anchors. Those original values remain immutable
`NON_PROTOCOL_DIAGNOSTIC` and cannot rank or select models. Training schedules,
checkpoints and row predictions were independent of that aggregation.

Rescore every completed final-endpoint validation prediction with the same
reviewed pure evaluator. For each horizon, group scored rows by the **target's
source-reported date**, as the campaign already did. The source clock timezone
is unknown; do not call these UTC days. Include a date in that horizon's
primary score only when it has at least 18 issued, observed target rows.
Average pinball over the five fixed quantiles per row, rows per eligible date,
eligible dates equally per horizon, and then three horizons equally. Report
each horizon/quantile and eligible date separately, all issued and scored
coverage, eligible date IDs/counts/hashes, median MAE, raw interval coverage
and width. Use the exact same per-horizon date support for all compared
families. A cutoff-date regrouping can be secondary only and cannot alter
selection. No model is retrained or refit because its corrected score changes.

An independent reviewer must verify all 17 saved final-endpoint files, shared
row identity, target truth/masks/source times, hashes and deterministic
recomputed scores before any family selection or calibration numeric access.

## Calibration and retrospective test boundary

The calibration partition adjusts only the frozen comparison models' 90%
interval endpoints using nonnegative empirical residual widening, separately
per model and horizon. Fit each
adjustment on calibration source dates meeting the same 18-anchor floor, with
at least 12 eligible dates per horizon. Retain the raw five quantiles; the
primary pinball comparison uses these raw quantiles. Report raw and widened
interval coverage/width separately. Calibration cannot change
model family, seed, architecture, threshold, target or metric.

Before opening retrospective test acoustic values, freeze two layers:

1. Metadata-only candidate source interval IDs and partition date bounds,
   source archive/config/checkpoint hashes, selected models, input/target
   reader and QC code, issued-row rule, metrics, bootstrap, and calibration
   adjustment. Metadata may inspect interval IDs, source dates/times, pings,
   fixed layer and channel presence, but no test `Sv_mean` outcomes.
2. After independent approval of that candidate universe and code, read test
   acoustic values once. Apply the frozen rule: issue only where all 24 prior
   38 kHz products are present, complete, finite and non-special; other past
   channels may be masked. Future target availability affects scoring only,
   never issuance. Materialize and hash actual issued/scored IDs and every
   abstention/QC reason without filtering on favorable errors.

This resolves ADR 0008's phrase “exact cohort before test” as exact
metadata-only *candidate universe* plus exact deterministic one-time
materialization after the freeze. Requiring final numerically QC-qualified
issued IDs before test access would itself inspect test acoustic values. The
result is retrospective evaluation, not a sealed or external holdout; prior
exposure history remains visible.

Use 48-hour contiguous source-date blocks, 2,000 seeded paired bootstrap
draws and identical model support. Report the prespecified 5% incremental
loss gate, paired 95% interval, per-horizon regression guard and daily paired
differences, regardless of outcome. No outcome-selected repair can regain a
confirmatory label without new untouched data.
