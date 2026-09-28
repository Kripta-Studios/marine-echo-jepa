# ADR 0012: AEON conditioned-product external transfer

Date: 2026-09-28. Status: PROPOSED FOR DISTINCT PREACCESS REVIEW.
Study: `aeon_external_transfer_20260928_v1`. The machine-readable contract is
`configs/aeon_external_transfer.json`. This study starts after the one-time,
reviewed AEON3 Georges Basin 2024–25 retrospective TEST. It cannot turn that
TEST into a sealed holdout, revise its result, or revise historical MOSAiC v1/v2
eligibility and release findings.

## Question and source identity

The narrow question is whether the *already frozen* three-seed EMA-JEPA
forecaster transfers better than the *already frozen* three-seed direct-neural
forecaster to a second publisher-conditioned AEON AZFP product. The primary
cohort is AEON2 Eastern Coastal Shelf, April 2024–April 2025, a contemporaneous
cross-site deployment. The secondary cohort is AEON3 Georges Basin,
February 2023–February 2024, a prior-year same-site transfer described
separately. These cohorts are never pooled, used to select one another, or
used to tune a checkpoint, normalization, score, QC rule or interval width.
The primary cohort was identified after the original TEST outcome was known:
this is post-hoc-initiated external transfer, not a sealed or preregistered
confirmation.

The source is [UNH AEON's site/deployment portal](https://eos.unh.edu/aeon/data/aeon-data-portal)
and the [publisher's Figshare AZFP article, version 2](https://figshare.com/articles/dataset/AZFP/29247113),
which lists CC BY 4.0. The downloaded official file IDs, article version,
publisher MD5, local SHA-256, exact ZIP member prefix, monthly member count
and central-directory inventory digests are fixed in the JSON contract.
The ZIPs have 13 monthly `60minFullDepth` CSVs at each of 38, 125, 200 and
455 kHz. Their common header includes `Date_M`, `Time_M`, `Layer`, `Ping_S`,
`Ping_E`, `Sv_mean` and the exclusion/processing fields. At freeze drafting,
only the publisher metadata, ZIP central directories and CSV headers had
been inspected; no external CSV data rows or `Sv_mean` values had been read.
Both candidate filenames say AZFP serial `55146`, which the publisher readme
defines as `SerialNumber`; the original AEON3 2024–25 filenames say `55144`.
These are not two proven instrument-independent replications. The archived
CSV alone does not bind exact raw files, calibration certificates, firmware
configuration or Echoview settings by hash.

The publisher readme describes per-unit calibration coefficients imported
from firmware configuration, standard TVG/absorption settings, background
noise removal, median filtering, manual exclusions and a tilted-beam surface
correction. The studied quantity remains the publisher's conditioned
`Sv_mean` in its reported dB unit, not an independently calibrated absolute
backscatter or biological abundance. The source clock's timezone and product
availability latency are unresolved. AEON2 site depth is about 231 m,
whereas the original AEON3 site is about 189 m. The nominal 0–200 m layer
therefore covers different parts of the water column at the two sites; any
positive transfer result applies to these source-defined layers only.

## Frozen observation and evaluation rules

Select only 38 kHz `60minFullDepth` rows with the source-defined `Layer`
covering exactly 0–200 m and the original study's 150-ping complete-interval
rule. Exclude the same special values and nonfinite `Sv_mean` as in the
original reader. The 24 preceding observed four-frequency source intervals
and past-only masks supply each input; the 1, 3 and 6 source-interval steps
and five quantiles are unchanged. No future target is filled, no missing
hour is interpolated and no new calendar-clock interpretation is imposed.
The fixed ping rule may make an external archive ineligible; it must not be
changed after reading external outcomes. Report mismatched completeness and
missingness as observed, and stop if the source schema cannot implement the
same definition.

Use only the original reviewed checkpoint components and original AEON3 TRAIN
scaler, each bound by the existing selection freeze. The direct and EMA
ensembles each average their three fixed seeds equally, then use the same
quantile rearrangement as original TEST. No external fitting, fine-tuning,
site-wise dB offset, normalization, interval recalibration, model selection
or checkpoint substitution is permitted. This deliberately makes instrument
and site shift part of the transfer challenge. A diagnostic based on external
labels must be labelled post-access descriptive and cannot repair this run.

Stage 1 freezes metadata-only candidates whose 24 predecessor interval IDs
exist, whose predecessor 38 kHz metadata meet the fixed geometry, ping and
source-time rules, and whose `cutoff + 6` remains within the archive's interval
range. A missing future horizon row does not exclude a candidate. These are
not actual issued or scored rows.
Stage 2 issues only candidate rows whose preceding 24 intervals have valid
finite, non-special 38 kHz `Sv_mean`; invalid other past channels become masks.
Future target availability or value never filters issuance. Preserve every
candidate that fails the past-value rule with its non-issuance reason, and
preserve all issued rows whether or not their targets can be scored. Score on identical
eligible anchors for both models, require at least 18 scored anchors per
source date and horizon, and require at least 90 eligible scored source dates
for each horizon *within each archive* before claiming that archive meets
the cohort eligibility gate. Ninety is unchanged; no count from another
study or horizon is borrowed. Report each date and horizon, exclusions,
coverage and interval width as well as raw five-quantile daily mean pinball
in dB. The primary relative reduction is `(direct loss - EMA loss)/direct
loss` on common AEON2 support. Form consecutive two-source-date blocks of
paired eligible daily EMA-minus-direct losses, retaining their three horizon
columns and missing-date masks; sample these blocks with replacement for
2,000 draws using the new study's fixed seed 20260928. The old evaluation
function hard-codes seed 20260926 and cannot silently be reused with this new
seed. The primary JEPA-value gate requires at
least 5% relative reduction, a paired 95% interval strictly favoring EMA,
and no horizon with more than 10% regression. The secondary cohort has the
same score and eligibility definitions but is descriptive, never a fallback
for a failed primary gate. A failed eligible *executed* comparison is a
negative transfer result; a source failing schema/support or an unexecuted
comparison is not.

## Staged access and review

Stage 0 verifies publisher identity, license, hashes, ZIP inventory, exact
24-column schema header and this contract; it does not open external CSV data
rows. A distinct reviewer must approve Stage 1 code and exact allowed
metadata columns before a row-level metadata scan. A CSV parser necessarily
streams past the bytes of every field. Stage 1 may retain only `Date_M`,
`Time_M`, `Interval`, `Layer`, `Layer_depth_min`, `Layer_depth_max`, `Ping_S`
and `Ping_E`. The CSV tokenizer may transiently materialize disallowed cell
strings to project the eight allowed fields; the raw row is discarded
immediately. Stage 1 must never semantically interpret, numerically convert,
branch on, aggregate, log, persist or expose `Sv_mean` or any other
disallowed field value, including `Process_ID`.
It must reject any header other than the exact frozen 24-column sequence.
The metadata-only candidate inventory must be hash-bound, show all dates and
failure reasons, and be independently reviewed. The full reader, frozen
component hashes, candidate join, predictor and metric/bootstrap code require
a separate distinct pre-numeric review before Stage 2 opens any external
acoustic values. Stage 2 then performs one-pass zero-shot inference and
evaluation for both fixed cohorts, preserving all predictions, metrics,
logs and errors. A final distinct outcome review checks source identity,
row lineage, hashes, common support and recomputes scores before any claim
is promoted into the app or release. The leader may implement and integrate
but cannot approve its own access or result.

No new v1 census, any-band audit, cloud purchase, credential switch or
company contact is needed for this public-data comparison. Local process RAM
and GPU memory remain under the repository limits; there is one GPU owner.
