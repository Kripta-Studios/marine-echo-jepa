# Execution status

## Prospective v2 continuation - active 27 September 2026

Owner authorization for the documented v2 amendment is recorded in the continuation
prompt. The leader owns data/design; `/root/v2_builder` owns real executors in the
separate `impl/v2-execution` worktree; `/root/v2_reviewer` independently reviews
`continuation_v2/REVIEW_V2.md`. V1 evidence below remains historical and unchanged.

ADR 0005's conservative two-metre-grid candidate failed its frozen 10% detected
support rule: zero eligible hourly targets on original TRAIN. Its row report
remains under `evidence/v2/support/`. The distinct reviewer approved the native
range-length Candidate 2 contract in ADR 0006, its processing pilot and the
fixed 58-day TRAIN development processing. All 58 days, 17 February through
14 April, completed with 259083 observed pings and hash-bound native shards.
The audited index SHA256 is
`58b9fc7825b804511b92726d707589db9c818159fb3081d47ed37b8676533b0a`.
Summed worker time was 1259.904 seconds, not a wall-clock measurement; maximum
sampled single-process RSS was 0.266 GiB. The independently reviewed support
method ran successfully. Candidate 2 has 21/22/21 eligible fit target days at
1/3/6 hours, but zero eligible April assessment target days at every horizon:
its maximum observed detection fraction is below the frozen 10% floor.
`evidence/v2/native-support/eligibility_v2.json` records the complete counts,
exclusions and row hashes. The distinct reviewer independently reconstructed
all 1392 hours and approved this negative support result, blocking Candidate 2
fits. ADR 0006 stops further D1 calibrated-target search. A separate
complete-positive AZFP response-code engineering target in ADR 0007 is approved
prospectively. Its separate hash-bound TRAIN development processing completed
all 58 fixed days with 259083 observed and target-valid pings and no ambiguous
zero or nonfinite code exclusions in the fixed segment. The reviewer independently
accepted the exact processing and support results. Fit target days are 42/42/42;
April assessment target days are 13/13/13, with all seven fixed 48-hour blocks
populated at each horizon. These meet ADR 0007's engineering support gates only.
The first hash-bound real ridge/direct engineering execution passed pre-fit review
and loaded only the fixed TRAIN-development cohort. It wrote a ridge assessment
prediction file and direct checkpoints at 64 and 128 updates, then failed the
frozen 64-to-128 weight-equivalence gate (maximum absolute difference 2.09e-5).
No direct prediction or result was promoted from that attempt. Its output and
failure evidence remain preserved. The distinct reviewer approved a deterministic
CUDA remedy under a new run identity. That single real TRAIN-development ridge
and direct run exited zero, wrote 212 issued-row prediction files for both models,
and restored an exact 64-to-128 weight replay with 128 updates. The distinct
reviewer independently recomputed the saved prediction-row metrics and loaded
both checkpoints, approving the real development result. Ridge daily mean
pinball code was 12.231/19.999/30.045 at 1/3/6 hours versus direct neural
64.906/64.389/65.934. Direct was worse at every horizon. The reviewed
checkpoint hashes are in the result review. No final evaluation ran. This route
cannot complete the calibrated-acoustic v2 MVP or consume its 25 campaign slots.
V1 census and any-band audit were not rerun. The builder's real calibrated-v2
campaign backend is integrated as code and fixtures, with no real D1 fit. The
distinct reviewer approved the final scheduler/restart, hybrid best-checkpoint,
and reused-slot lineage implementation at commit `23b180b`. Active restarted
processes are protected by immutable per-generation leases. This is code
approval only; calibrated Candidate 2's zero April support still bars the
25-slot campaign and JEPA runs. The separate raw-code offline research app
renders all 212 reviewed cutoffs (636 cutoff-by-horizon rows) and the verified
direct checkpoint hash. Its eight browser journeys passed on the built app and
again on the relocated archive. The 20,421,117-byte archive SHA-256 is
`20df1e7c7292f09a16dea0b641b82da12efff4c2660d2a9159fa599f53b2d2e6`.
Offline install/start, clean/corrupt/restored hash verification and local API
checks passed. The distinct reviewer rejected that first archive because the
Experiment Lab linked a draft protocol as "Frozen protocol". The archive and
review are preserved. A corrected separate r2 archive uses "Protocol and
evidence"; its SHA-256 is
`8c3eee64695c4ef3007ac6af9ab449434ef87e2c0bd4fbe5ab1641a59666b5ef`.
The r2 archive passed fresh relocation, checksum corruption detection, offline
starts and eight browser journeys. The distinct reviewer approved that exact
archive as an offline research engineering release after independently checking
all 336 payload hashes, packaged source, the corrected draft-protocol link,
unchanged scientific artifacts and preserved v1/r2 history. Its review is an
external immutable-archive sidecar at
`orchestration/reviews/V2_RESEARCH_RELEASE_FINAL_20260927.json`.

The existing GPU environment passed a CUDA kernel smoke with PyTorch 2.11.0+cu128;
this is not a training-memory measurement. Data-environment PyTorch is absent and
has not been installed. Free disk was 29.38 GiB: use existing source bytes and
streamed shards; no speculative bulk download or duplicate raw corpus. Original
handoff verification still fails at the recorded `.gitattributes` scaffold mismatch.
Current evidence: `evidence/v2/`, machine state: `orchestration/run_ledger.json:v2`.

## Preserved v1 checkpoint

P0 INCOMPLETE - D1 DATA INELIGIBLE UNDER THE REVIEWED METHOD; D2 BLOCKED.

All 100 original TRAIN days were processed successfully using the independently reviewed
factory/regional calibration and QC method. The singleton parser repair is reviewed; all
16 overlapping successful v1 days match v2 arrays exactly. V1 failure evidence is preserved.

Independent recomputation accepted both necessary-support results:
- Original 38 kHz, 10-100 m target: zero supported TRAIN anchor dates; even granting all
  67 non-TRAIN calendar days eligibility gives 67 overall, below the required 90.
- Every fixed nonnegative-weight band on the existing 64-cell 38 kHz grid: the generous
  per-horizon best-cell relaxation yields 9 anchors on 6 TRAIN dates, hence at most
  73 overall days. No band, frequency, QC or support rule was changed.

These are data-eligibility results, not negative JEPA results. Calibration/test eligible
days remain NOT_EVALUATED. All 25 benchmark slots are BLOCKED, with null metrics and zero
updates. No training, validation selection, interval calibration or final evaluation ran.
R0/R1 corpus promotion is blocked; R2 and normal R3 release are ineligible.

The matching serial 55170 factory certificate was recovered from swapped public registry
attachments and checked against XML. Environmental profiles and geometry were evaluated.
The fixed regional/depth/slant method is accepted for TRAIN development only, not full
field calibration. Its conditional slant sensitivity reached 0.162029 dB and 0.608893 m
at 38 kHz. The earlier global sensitivity failure (1.080721 dB > 1 dB) remains preserved.

The documented OOI fallback has official deployment geometry and embedded raw acquisition
coefficients, but lacks sufficient calibration/processed-product lineage and an approved
bounded acquisition plan. Bounded investigation is complete; do not start a new dataset
project or relax the protocol to force a result.

Exposure: existing audited MOSAiC replays are TRAIN-only. All 100 census manifests contain
2,135 source records and zero out-of-day pings. No held-out acoustic payload was processed
in this continuation. Full-period retrospective environmental metadata was inspected, and
the historical OOI diagnostic file was exposed. Earlier human exposure is not exhaustively
known: the intended holdout is NOT_ESTABLISHED as sealed.

The native builder and distinct independent reviewer recovered availability. Coordinator
owned calibration/integration in main; builder used the separate core worktree; reviewer
remained read-only. Historical quota failures remain recorded. No paid credentials/cloud.
No GPU owner or marine trainer is active; unrelated user processes remain untouched.

Latest checks: 229 app/data/scientific/security/API passes, 11 model passes with two opt-in
GPU skips, 27 focused any-band/map passes, and successful lint/format/type checks. These
are software checks. Full census wall time was 4667.160 s; sampled worker-tree RAM peaked
at 0.319389 GiB. Any-band audit took 11.125 s and 0.174793 GiB. Infrastructure cost is USD 0.

The actual scientific results are integrated into outputs/continuation-preview-20260927.
Both original and copied-path previews passed five offline browser tests (navigation p95
197.787 ms and 186.135 ms). The distinct reviewer accepted the exact artifact and visual/browser evidence
within working-preview scope only. No new engineering diagnostic ZIP substitutes for
the requested normal release. Historical release/meeting-20260926-r2.zip and its extracted
payload are unchanged (SHA256 df02ff06cbbb05d967f8c157423c0c76b2acf66ddb16f1a1fcfbe36a710a5cc3).

Authoritative evidence: evidence/continuation/train_census_v2_execution.json,
target_only_bound.json, train_support_map.json, any_band_bound.json and the matching
orchestration/reviews/*RESULT_20260927.json records. See the continuation report for
source searches, assumptions, tests and historical evidence.

Next scientific action requires an independently justified prospective protocol/data
resolution within owner scope. The current Campaign command correctly exits 2 and lists
its scientific and real-executor blockers. It cannot approve or open the final test.

Final detail: orchestration/reports/CONTINUATION_OUTCOME_20260927.md.

## Prospective source scout — 2026-09-27

The Hugging Face source preflight downloaded and hash-verified four bounded metadata/index
files (13,297,604 bytes total). No inspected Hub candidate supplies the required calibrated
longitudinal 38 kHz Sv target; no acoustic payload was admitted or training initiated.
The Norwegian cabled EK60 observatory is a promising but unacquired lead whose catalog
routes data access through email and lacks a public file manifest and calibration lineage.
Evidence and acquisition decision: `orchestration/reports/HF_SOURCE_SCOUT_20260927.md`
and `evidence/continuation/hf_source_scout_20260927.json`.

Subsequent source search found the public AEON AZFP integrated Sv collection. Its readme
and one 11.3 MB deployment ZIP were downloaded and publisher-hash verified. The 38 kHz
hourly CSVs have records on 174 dates, including 172 dates with 24 interval IDs, but
valid-target support, calibration lineage and the 15-minute-contract mismatch remain
unresolved. This is a prospective candidate, not an eligibility or model result.

The longer AEON3 March 2024–February 2025 archive was subsequently downloaded and
publisher-hash verified as the proposed new-study source. Metadata-only inspection found
360 date fields, including 356 with 24 hourly interval IDs and all four AZFP channels.
ADR 0008 proposed a distinct hourly source-product study at this source-selection
checkpoint. Its subsequent review and development execution are recorded below.

## AEON prospective study execution — 2026-09-27

The optional frozen Chronos-2 TRAIN/validation baseline now has a separate hash-bound,
restart-safe implementation on `impl/aeon-chronos`. Its official multivariate input choice,
native NaN missingness path, exact model/package digests and resource caps are recorded in
`configs/aeon_chronos.json` and the implementation report. The exact model snapshot is cached
and verified. Inference is `BLOCKED_BEFORE_INFERENCE`: `chronos-forecasting==2.3.2` is absent,
the independent prefit review is not yet recorded, and the coordinator has not assigned the
single GPU for inference. No calibration or test acoustic outcome was opened.

ADR 0008 and its row-level split amendment received distinct independent approval for
TRAIN/validation development only in `orchestration/reviews/AEON_SPLIT_AMENDMENT_20260927.json`.
The source-reported conditioned 38 kHz hourly FullDepth `Sv_mean` is a different
observand and study from the failed v1 and MOSAiC v2 work. Calibration and test acoustic
outcomes remain unopened pending a separate pretest freeze; neither is a sealed holdout.

A real persistence baseline and direct neural development run executed on AEON TRAIN
and validation rows. After the loader was hardened, the 128-update CPU run was repeated
under the inspectable code at
`outputs/aeon3_geb_2024_hourly_sv_v1/first_development_seed7_codebound/run.json`.
Its 4,965 TRAIN windows and 1,219 validation issuances yielded 1,217 scored rows per
horizon. Baseline daily mean pinball was 0.6943/0.9143/1.0226 dB at 1/3/6 source-hour
steps; direct was 0.6877/0.6994/0.7226 dB. The code-bound run saved all issued-row
predictions and a genuine direct checkpoint. A distinct reviewer recomputed the cohort,
scores and checkpoint predictions and approved the bounded development artifact in
`orchestration/reviews/AEON_DEVELOPMENT_SSL_REVIEW_20260927.json`. These are
development outcomes, not final evaluation or JEPA evidence. The first run remains
preserved with its historical code hash; no result was silently replaced.

The larger 128-wide, three-layer, three-seed AEON core campaign was specified in
`configs/aeon_campaign.json` for independent prefit code/config review. At that
checkpoint no AEON JEPA, calibration, or test run had executed.
Machine state is `orchestration/aeon_run_ledger.json`; the v1 blocked ledger is unchanged.

The 17-slot TRAIN/validation campaign later received exact independent prefit approval
in `orchestration/reviews/AEON_CAMPAIGN_PREFIT_20260927.json` and is now running serially
under the local GPU/RAM caps. Its conventional and three direct-neural slots have
completed; JEPA and controls remain in progress. A review addendum found that the
campaign's saved all-scored-day validation metric includes low-anchor days. Preserve
those values as `NON_PROTOCOL_DIAGNOSTIC` and use none for model selection. Every final
row prediction will be rescored under the preregistered >=18-anchor/day/horizon rule
before any selection or CAL/TEST access. See
`orchestration/reviews/AEON_CAMPAIGN_PREFIT_METRIC_CORRECTION_20260927.json`.

The full 17-slot core campaign subsequently completed with exit status zero,
real final-endpoint model artifacts and 1,219 issued validation predictions per
slot. The reviewed deterministic rescore retained the original printed values as
`NON_PROTOCOL_DIAGNOSTIC` and recomputed the frozen 18-anchor/day metric on 50
eligible source dates per horizon (1,194/1,192/1,189 scored rows). The distinct
reviewer independently recomputed every score and checked all 17 checkpoint,
model, prediction and row hashes in
`orchestration/reviews/AEON_VALIDATION_RESCORE_OUTCOME_REVIEW_20260927.json`.
The best individual direct seed scored 0.644897 dB daily mean pinball; EMA-JEPA
seeds scored 0.648805/0.647989/0.647978 dB and shared-SIGReg seeds
0.652222/0.657561/0.663477 dB. These are TRAIN/validation development
results, with no family selection or final evaluation yet. Hybrid fitting awaits
its separate exact prefit review. TRAIN-only frozen-random representation
diagnostics are executing under refreshed code approval. CAL/TEST acoustic
outcomes remain unopened.

A September 2026 primary-source architecture survey and explicitly post-hoc
development extension are recorded in
`orchestration/reports/SOTA_ARCHITECTURE_REVIEW_20260927.md`. That extension
considers a fixed supervised tree, pinned Chronos-2 zero-shot comparator and
future-block EMA-JEPA with an anti-collapse variance floor. It does not turn
the completed core validation run into preregistered confirmation. New models
remain unfitted pending separate code/config review and the one-trainer rule.

The ten-slot hybrid campaign later exited zero and passed independent outcome
review. Its 150 downstream tree heads reproduce the saved 1,219-row predictions
bit-exactly. The raw-only B3 score is 0.669379 dB; the best learned EMA hybrid
scores 0.660870 dB, whereas its matched frozen-random-feature hybrid scores
0.654412 dB. The three-seed direct ensemble scores 0.639062 dB, ahead of
EMA-JEPA (0.642170 dB) and shared-SIGReg (0.648643 dB). The independent
review in `orchestration/reviews/AEON_HYBRID_OUTCOME_REVIEW_20260927.json`
records a negative incremental JEPA-value finding for the core development
comparison. A TRAIN-only random representation audit found seed-7 effective
ranks 3.97 (EMA) and 3.53 (shared) versus 6.37 for the frozen random encoder;
its separate review is
`orchestration/reviews/AEON_RANDOM_DIAGNOSTIC_OUTCOME_REVIEW_20260927.json`.
These are development findings, not retrospective TEST outcomes.

The post-hoc LightGBM challenger received independent prefit approval. Its
first invocation exited 1 before any head fit because an exclusive-write row
probe used an already-created temporary file. The failure and reviewed repair
are documented in
`orchestration/reports/AEON_SOTA_SUPERVISED_FIRST_ATTEMPT_FAILURE_20260927.md`.
The serialized retry completed and passed distinct outcome review: 4,965 TRAIN
rows, 1,219 issued validation forecasts, and corrected daily mean pinball
0.638459724465 dB across 50 eligible source dates per horizon. This is about
0.000602 dB below the direct-neural three-seed ensemble, a tiny post-hoc
development difference with no significance or SOTA claim. The exact model and
prediction files were reloaded and all 15 heads' predictions reproduced
bit-exactly by the reviewer; see
`orchestration/reviews/AEON_SOTA_SUPERVISED_OUTCOME_REVIEW_20260927.json`.
Chronos-2's exact public checkpoint is locally cached and its runner implemented;
inference awaits distinct prefit review. The forward-EMA JEPA adaptation with
fixed VICReg-style anti-collapse penalties is implemented and awaits prefit
review. CAL/TEST acoustic outcomes remain unopened pending final selection and
freeze.
