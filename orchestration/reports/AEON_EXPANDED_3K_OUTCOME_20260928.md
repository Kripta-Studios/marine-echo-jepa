# AEON expanded-TRAIN 3k development outcome

This separate post-hoc study reclassified the already published prior-year
AEON3 Georges Basin deployment as TRAIN alongside the original 2024–25 TRAIN
partition. The original 2024–25 validation partition remained unchanged. A
distinct reviewer approved numeric cohort access and, after measuring its QC,
approved the two frozen seed-7 fits. The same reviewer independently rebuilt
and replayed the completed artifacts; the exact verdict is recorded in
`orchestration/reviews/AEON_EXPANDED_3K_OUTCOME_REVIEW_20260928.json`.

The joint TRAIN cohort has 13,472 windows: 8,507 prior-year plus 4,965
original TRAIN windows, 2.7134 times the original TRAIN size. Validation has
1,219 unchanged windows. All prior-year 8,507 metadata candidates were issued
by numeric QC; source-specific target support and row identities are preserved
in `evidence/aeon_scale/expanded_cohort_20260928.json`. There is no TRAIN/VAL
raw interval overlap. This is a second same-site deployment, not 3× or 5× data
and not an external holdout for the newly trained models.

| Family, seed 7 | Original-only corrected 3k daily pinball | Expanded-TRAIN corrected 3k daily pinball | Observed change in loss |
| --- | ---: | ---: | ---: |
| Direct neural | 0.651154 dB | 0.640615 dB | 1.62% lower |
| EMA-JEPA | 0.648805 dB | 0.654855 dB | 0.93% higher |

Both families used exactly 3,000 optimizer updates and the frozen final
endpoint. Direct trained for 3,000 supervised updates. EMA-JEPA trained for
1,500 TRAIN-only, source-homogeneous SSL updates followed by 1,500 supervised
updates. The evaluator used the corrected at-least-18-anchor daily pinball on
the same 50 validation dates per horizon, with 1,194, 1,192 and 1,189 eligible
rows. Each slot saved a final checkpoint and finite, monotone, nonconstant
1,219 × 3 × 5 prediction array. The reviewer reproduced the joint TRAIN
scaler, deterministic source sampling, 12 checkpoint hashes, all 9 scheduled
validation artifacts, final model and prediction hashes, scores and support.

The direct model shows a modest single-seed development gain. The EMA-JEPA
model does not improve on its original-only comparator and its expanded score
is 2.22% worse than expanded direct. Its final pretraining target effective-rank
fraction is 4.33%, with a predictor/target RMS ratio of about 28.92. It is not
an exact zero-variance collapse, but the low-rank and scale mismatch does not
support a useful JEPA representation claim.

This intervention changes the deployment as well as row count. Serial,
calendar and potentially processing differ, so the direct score difference is
not a causal estimate of the effect of more data. Validation has been
repeatedly inspected during development; neither these scores nor the prior
historical descriptive external-transfer result evaluate the derived models
on a sealed or independent holdout. The historical result remains preserved,
but its source is now TRAIN for this distinct study. No CAL/TEST, physical
calibration, SOTA, operational or biological claim follows.

Direct peak process RSS was 1,937,559,552 bytes and PyTorch GPU reserved
memory 25,165,824 bytes. EMA-JEPA peaked at 2,086,113,280 bytes RSS and
27,262,976 bytes reserved. The two-slot training process exited zero and stayed
below local resource caps.
