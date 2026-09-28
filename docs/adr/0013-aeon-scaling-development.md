# ADR 0013: post-hoc AEON scaling development

Date: 2026-09-28. Status: PROPOSED, requiring a distinct independent prefit
review before numeric execution. Study identity:
`aeon3_geb_2024_hourly_sv_scale_30k_development_v1`.

The 3,000-update core campaign and its retrospective and external-transfer
outcomes remain historical evidence. This amendment asks whether a longer
schedule improves validation on the **same** 2024–25 AEON3 TRAIN/validation
cohort. It does not reinterpret the already inspected retrospective TEST as a
new holdout, and it grants no CAL or TEST access. The source target remains
38 kHz source-reported conditioned `Sv_mean` for the 0–200 m full-depth source
product. Inputs remain the past 24 source intervals at 38/125/200/455 kHz and
their masks; forecast steps remain 1/3/6 and quantiles 0.05/0.25/0.5/0.75/0.95.

## Thirty-thousand-update schedule

Stage 1 trains direct seed 7 for 30,000 supervised AdamW updates and EMA-JEPA
seed 7 for 15,000 TRAIN-only representation updates followed by 15,000
supervised updates. Each uses the original width-128, 3-layer encoder,
batch size 64, learning rate 0.0003, weight decay 0.0001, gradient clip 1,
TRAIN-only normalizer, equal source windows and original loss functions.
Only a single local training process may run. Process RSS must remain below
22 GiB and CUDA reserved memory below 10 GiB. CPU fallback must be labelled.

The runner saves a model checkpoint and all issued validation predictions at
every 2,500 supervised updates. The **final endpoint only** is used for
comparison and any subsequent inference. Earlier checkpoints form a retained
learning-curve diagnostic, never a pool for selecting a more favorable model.
This avoids giving direct (12 supervised checks) more selection opportunities
than EMA-JEPA (6 supervised checks). The old campaign's raw all-scored-date
metric included dates with fewer than 18 scored target anchors and is not a
protocol score. Every new checkpoint is therefore rescored with the reviewed
`evaluation.aeon.daily_pinball` rule: at least 18 observed target anchors per
source-reported date per horizon, row -> date -> horizon equal averaging. The
exact eligible-date lists and SHA-256 hashes are retained, and support must
match between checkpoints and families. Original raw scores, if retained,
are explicitly labelled non-protocol diagnostics and cannot drive decisions.

Compare each corrected final seed-7 result with its original 3,000-update
**same-family seed-7** corrected final-endpoint validation result from the
independently reviewed rescore report. Stage 2 runs both families for seeds
13 and 23 only if at least one stage-1 family has a final primary loss at
least 1% lower than that comparator. Otherwise stage 2 is explicitly not run.
This fixed screen conserves compute; it is not an independent significance
test. A 50,000-update extension is not part of this amendment and requires a
new reviewed schedule after 30k evidence.

The new runner requires an exact independent prefit approval binding source,
split review, cohort, code, config, this ADR, original campaign manifest,
corrected 3k rescore report and its outcome review hashes. It stages each slot
atomically, hashes predictions,
checkpoints and slot records, and validates hashes on resume. The original
campaign loader can open only TRAIN and validation source dates. The original
core outputs and frozen protocol are never mutated by this study.

## Expanded-data candidate: earlier same-site deployment

The downloaded publisher AEON3 GEB February 2023–February 2024 archive
(Figshare file 61937275, SHA-256
`b6d8380ed986c91d565ea7d669761ff8f66da8f2b5cd3e794fbf71c040bf1d37`)
has 0–200 m 38 kHz full-depth source products and 38/125/200/455 kHz
channels. Its 8,507 issued windows were previously used as a descriptive
external-transfer evaluation and their outcomes are already inspected. If
reused as TRAIN, they cease to be external evaluation data for every derived
model. The 2024–25 archive's existing TRAIN yields 4,965 windows; the simple
sum is 13,472 candidate windows, 2.71 times the old fit count. The actual
joint TRAIN count needs new QC and cohort validation. Five times the old
count is unsupported by these two archives alone.

An expanded-data study needs a **separate implementation and review**. Read
the prior archive only through a new hash-bound TRAIN reader applying the
same 0–200 m geometry, 150-ping, finite/non-special-Sv, timestamp and past-only
issuance rules. Bind each source archive and interval ID in row identity;
never bridge a 24-hour context or 1/3/6-step target across deployments.
Audit unit and frequency metadata, instrument filename identifiers (prior
`55146`, current `55144`), processing lineage and channel masks before
pooling. Assign the prior 2023–24 deployment and only the original 2024–25
TRAIN dates to expanded TRAIN. Keep original 2024–25 validation dates for
development; leave CAL/TEST unavailable to training. Fit normalizers and
SSL pairs solely on this expanded TRAIN. Report source-specific counts and
losses to expose domain shift. This is post-hoc training because the prior
archive's transfer outcomes were already examined; any future evaluation
requires a fresh, untouched comparable source and a separate reviewed freeze.

The expanded-data reader, new split/cohort digest, exact prefit review and
training runs are **not implemented or authorized** by the 30k runner. Storage
space on `E:` is sufficient for the known compressed AEON archives, but no
claim is made about uninspected raw or additional deployments. Historical
v1/v2 eligibility failures, thresholds, blocked registry and r2 remain intact.
