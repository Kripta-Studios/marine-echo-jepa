# AEON scaling and expanded-TRAIN investigation, 28 September 2026

This report records what was actually acquired, trained, checked and observed
in two new, separately reviewed, post-hoc AEON development studies. It does
not amend the historical MOSAiC v1/v2 eligibility failures, blocked registry,
historical r2, original AEON 3k campaign or prior-year descriptive transfer
result. The new models were assessed on repeatedly inspected validation data,
not a sealed or independent holdout.

## Physical target, data and eligibility

The response is the publisher's **conditioned 38-kHz hourly `Sv_mean` product**
in dB re 1 m^-1 at nominal 0–200 m in the AEON3 Georges Basin archive. It is
not an independently calibrated backscatter measurement, species, biomass,
catch, fuel savings or a causal intervention effect. The original source has
four acoustic channels and 24 preceding hourly source products as model input;
the model forecasts 38-kHz response quantiles at +1, +3 and +6 source
intervals. Clock timezone, availability latency, calibration coefficients and
publisher processing/exclusions are not independently verified.

The original 2024–25 publisher ZIP, file 61937281 (serial identifier 55144),
yielded 4,965 TRAIN and 1,219 validation windows under the reviewed split.
The previously downloaded same-site 2023–24 ZIP, file 61937275 (serial 55146),
yielded 8,507 numeric-QC-issued prior-year windows. An independent review
reconstructed all source/row/support digests and zero TRAIN/validation raw
interval overlap before the prior-year cohort was reclassified as TRAIN for
the expanded study. That reclassification retires its use as an external test
for the new derived models; the historical descriptive transfer result itself
remains unchanged.

Four other official Figshare version-2 AZFP archives were downloaded to `E:`
(file IDs 61937263, 61937266, 61937272 and 61937278; 81,879,095 compressed
bytes). Publisher sizes/MD5 and local archive hashes were verified. A
metadata-only scan found 25,409 new 38-kHz FullDepth rows but **zero** rows
with the exact original 0–200 m target geometry. Their products use 0–220,
0–225 or 0–230 m; some also differ in channel or ping mode. They cannot be
called 5× comparable data under the current physical target, and their
acoustic values were not opened for this study. A new reviewed target and
representation protocol would be needed before using them.

## Architecture and experiment controls

The direct neural comparator is a three-layer, width-128 MLP encoder over
24 × 4 observed values and matching masks, followed by a 3-horizon ×
5-quantile head. Its supervised objective is quantile pinball loss; output
quantiles are ordered. EMA-JEPA uses the same encoder/head for supervised
forecasting, plus a two-layer representation predictor during TRAIN-only
pretraining. Its online student sees the first 18 of the 24 preceding products;
its EMA teacher (momentum 0.996) sees the last six **also preceding** products.
The predictor matches the teacher representation with smooth L1 loss plus a
0.03-weight sliced Epps–Pulley regularizer. No future truth or validation
window is used for SSL fitting. Both studies use AdamW at fixed 3e-4,
batch size 64, weight decay 1e-4 and gradient clipping 1.0. The original
same-family corrected 3k seed-7 comparisons are direct 0.6511541518561285
and EMA-JEPA 0.6488046342025475 dB daily mean pinball.

The evaluator requires at least 18 observed target anchors per source date
and horizon. The unchanged 1,219-row validation partition contributes 1,194,
1,192 and 1,189 eligible rows across 50 source dates at +1, +3 and +6. The
frozen **final checkpoint** is the decision endpoint; intermediate validation
checks are diagnostic and cannot be chosen retrospectively. All forecast
arrays contain 1,219 × 3 × 5 finite, ordered, nonconstant predictions.

## Study A: more updates on unchanged data

ADR 0013 and `configs/aeon_scale_30k.json` froze direct seed 7 at 30,000
supervised updates and EMA-JEPA seed 7 at 15,000 TRAIN-only SSL plus 15,000
supervised updates. Both real runs completed with final checkpoint and
prediction files under `E:\marine-echo-jepa-scale\aeon30k_stage1`.

| Family | Original corrected 3k | 30k final | Change in validation loss |
| --- | ---: | ---: | ---: |
| Direct, seed 7 | 0.651154 dB | 1.178379 dB | 80.97% worse |
| EMA-JEPA, seed 7 | 0.648805 dB | 0.913424 dB | 40.79% worse |

Direct's diagnostic score rose from 0.650889 dB at 2,500 updates to 1.178379
dB at 30,000. EMA-JEPA's supervised-phase score rose from 0.664914 dB at
2,500 to 0.913424 dB at 15,000. These curves show progressive deterioration
on this validation partition at the fixed learning rate; they do not prove a
unique cause. The predeclared Stage-2 rule required at least 1% improvement
over the same-family original 3k endpoint in either family. The executable
gate returned `STAGE_2_NOT_AUTHORIZED_BY_PREDECLARED_RULE`; seeds 13 and 23
were **not run**. A 50k continuation is not supported or authorized by this
reviewed result. The early, near-baseline checkpoints were not selected.

The independent reviewer replayed all 18 scheduled validation artifacts,
checked all 24 checkpoint hashes, final model/prediction hashes, corrected
support and metrics. EMA-JEPA's pretraining representation was not exactly
constant, but its target effective-rank fraction was about 2.87% and the
predictor/target RMS ratio was about 16.27, indicating severe low-rank/scale
mismatch. Peak process RSS was 1.887 GB direct and 2.030 GB EMA-JEPA;
PyTorch GPU reserved memory peaked at 25.2 and 27.3 MB. Direct supervised
training took about 17.5 minutes; EMA-JEPA pretraining plus supervised
training took about 25.4 minutes. Extra RAM/VRAM alone would not reverse the
observed deterioration.

## Study B: larger same-site TRAIN cohort at the original 3k budget

ADR 0014 and `configs/aeon_expanded_3k.json` froze a new development study
using the 8,507 prior-year and 4,965 current TRAIN windows together: **13,472
TRAIN windows, 2.7134× the original count**, not 3× or 5×. The 1,219-window
validation partition stayed unchanged. The scaler was fitted on joint TRAIN
only. Supervised sampling was from natural pooled TRAIN rows; EMA-JEPA SSL
batches were source-homogeneous, chosen according to source row proportions.
The exact sampled prior/current counts were 121,177/70,823 for direct
supervision; 60,799/35,201 for EMA supervision; and 61,952/34,048 for EMA
pretraining (968 prior and 532 current batches). The independent reviewer
reconstructed these counts and the scaler exactly.

Direct seed 7 completed 3,000 supervised updates. EMA-JEPA seed 7 completed
1,500 SSL plus 1,500 supervised updates. Both used the same fixed final
endpoint and saved real checkpoints/predictions under
`E:\marine-echo-jepa-scale\aeon_expanded_3k`.

| Family | Original-only corrected 3k | Expanded-TRAIN corrected 3k | Observed change in loss |
| --- | ---: | ---: | ---: |
| Direct, seed 7 | 0.651154 dB | 0.640615 dB | 1.62% lower |
| EMA-JEPA, seed 7 | 0.648805 dB | 0.654855 dB | 0.93% higher |

The reviewer rebuilt 13,472 TRAIN and 1,219 validation rows from both exact
ZIPs; verified the joint TRAIN scaler, 12 checkpoint hashes, nine scheduled
validation artifacts, final prediction/model hashes and corrected metric/date
support; and confirmed finite, ordered, nonconstant predictions. Direct's
small single-seed development improvement is observed, not a causal effect of
row count: deployment, instrument serial, calendar and potentially processing
also changed. EMA-JEPA did not benefit and was 2.22% worse than expanded
direct. Its final-pretraining target effective-rank fraction was about 4.33%
and predictor/target RMS ratio about 28.92. There was no exact zero-variance
collapse, but these diagnostics do not support useful JEPA representations.

## Software checks, release and remaining limits

The new corpus/experiment runners, exact cohort gates and result artifacts
have distinct preaccess, prefit and outcome reviews. The active investigation
ran one GPU training process at a time, below the local 10 GiB GPU-reserved
and 22 GiB process-RAM limits; it used no cloud or paid credentials. The
experiment integration suite passed 21 tests. The reviewed app supplement
passed 25 Python tests, four React tests, Ruff, TypeScript typecheck, ESLint
and a production web build. The first source review requested bundling real
final binaries; that fix was independently re-reviewed and approved. The new
offline package contains eight final prediction/checkpoint binaries, exact
review/config/protocol/manifest/slot evidence and no raw acoustic archives or
intermediate checkpoints.

The new, separate release candidate
`release/aeon-offline-scale-expanded-20260928-r1.zip` was built with 107
hash-verified files (SHA-256
`b8c7874e438522e6a923f3d602b93dc3e1f5a0e60e68ef573a069e920a4f9d54`).
Its sidecar, complete directory inventory and separately extracted copy
passed checksum verification. Subsequent exact-output review requested a
README provenance-path correction; r1 remains unapproved. A corrected r2
passed offline install and API checks but failed mobile fit and also remains
unapproved. The independently reviewed CSS repair was committed as `f975d2b`.
The new r3 archive SHA-256 is
`5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57`.
It contains 107 verified assets and passed fresh relocated offline install,
API, exact displayed metric and historical replay checks, and desktop/mobile
Chromium tests with no page errors or external browser requests. The same
regression fails the preserved r2 mobile fit. The distinct exact-byte reviewer
subsequently approved this unchanged r3 archive for an offline research release
(severity NONE), verifying provenance and offline/API behavior and inspecting
the coordinator browser evidence. The external approval is
`orchestration/reviews/AEON_SCALE_EXPANDED_OFFLINE_R3_REVIEW_20260929.json`;
the immutable archive retains its honest review-pending build-time metadata.
Detailed evidence is in
`orchestration/reports/AEON_SCALE_EXPANDED_OFFLINE_R3_VALIDATION_20260929.md`.
Earlier r2/r4 releases and historical experiment evidence have not been overwritten.

The results establish a **negative same-data 30k scaling outcome** and a
**small direct-only expanded-TRAIN development observation**. They do not
establish JEPA value, state-of-the-art performance, independent generalization,
physical calibration, a deployable live forecasting service or business
validation. No unexecuted seed or model is described as a negative result.
