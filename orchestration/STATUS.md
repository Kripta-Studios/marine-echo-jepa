# Execution status

## AEON external conditioned-product transfer - active 28 September 2026

The reviewed AEON3 2024–25 retrospective result and r4 offline release remain
unchanged. A separate post-hoc-initiated transfer study has downloaded and
publisher-MD5/local-SHA-256-verified two public Figshare version-2 AZFP
archives: primary AEON2 East Coast Shelf April 2024–April 2025 and secondary
AEON3 Georges Basin February 2023–February 2024. The external candidate
archives have the same source-reported 38 kHz `60minFullDepth` field and four
channels. Both candidate filenames report serial 55146, so they are not two
proven instrument-independent replications. Site depth and source-product
conditioning limit physical comparability. ADR 0012 and
`configs/aeon_external_transfer.json` freeze the proposed zero-shot source,
target, model, support and analysis rules. A distinct reviewer approved the
Stage 0 design and source/header identities, then approved the exact Stage 1
metadata runner. Its one real two-source scan exited zero. The independently
reviewed metadata output has zero AEON2 candidates because all 8,682 38 kHz hourly
rows use source-defined 0–230 m geometry, while the frozen target is 0–200 m.
AEON3 prior-year has 8,507 metadata-only candidates on 357 source dates;
actual issuance, numerical support and scores remain unknown. The distinct
reviewer independently reconstructed both complete metadata reports and
approved this outcome. The AEON2 primary is metadata-ineligible, its JEPA
gate is NOT_EVALUATED, and its acoustic values must stay closed. The secondary
may only run as a descriptive same-site/prior-year transfer. The distinct
Stage 2 review approved one secondary-only numeric attempt. It passed source,
report and checkpoint preflight but exited 1 during candidate materialization:
whole-second source timestamps differed as text from the evaluator's fixed
microsecond rendering of the same instant. The secondary CSV rows were opened,
but no prediction, metric or output directory was produced. AEON2 acoustic
rows were not opened. The failed attempt remains recorded. The timestamp-instant
fix passed 55 focused tests and a distinct remedial review authorized one
secondary-only retry. That exact retry exited zero and saved 8,507 issued rows,
two equal-three-seed model forecast files and metrics. Its report counts 353
eligible source dates at each horizon. Reported raw daily mean pinball is
0.670178 dB for direct and 0.678922 dB for EMA-JEPA, a -1.3046% relative
loss reduction; its paired 95% EMA-minus-direct interval is
[0.002253, 0.015516] dB. The distinct reviewer reconstructed all 8,846
source slots, issuance, QC, targets, metrics and bootstrap, and replayed both
8,507 by 3 by 5 forecast arrays bit-exact from all six frozen checkpoints.
The outcome review SHA-256 is
`ec1ad3ecc437d28f47c53063d6abc6c8f9efe08c36a0f39dec4bda8ef9ad255d`.
This is a reviewed descriptive negative prior-year same-site comparison and cannot substitute
for the metadata-ineligible AEON2 primary. Machine state:
`orchestration/aeon_external_run_ledger.json`.

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

The optional frozen Chronos-2 post-hoc TRAIN/validation development baseline now has a separate hash-bound,
restart-safe implementation on `impl/aeon-chronos`. Its official multivariate input choice,
native NaN missingness path, exact model/package digests and resource caps are recorded in
`configs/aeon_chronos.json` and the implementation report. The exact model snapshot is cached
and verified. Inference is `BLOCKED_BEFORE_INFERENCE`: `chronos-forecasting==2.3.2` is absent,
the independent prefit review is not yet recorded, and the coordinator has not assigned the
single GPU for inference. Its AEON state-of-the-art claim is `NOT_ESTABLISHED`. No calibration
or test acoustic outcome was opened.

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
Chronos-2's exact public checkpoint and zero-shot runner passed distinct prefit
review; its frozen zero-shot TRAIN/validation inference completed on CUDA with
exit status zero and 1,219 real validation forecasts. Its corrected primary
pinball is 0.693518 dB, behind the direct-neural and LightGBM development
results. Distinct outcome review verified all 77 shard hashes and bit-exact
assembly, reproduced the full corrected metric and one pinned-model CUDA shard,
and approved neutral post-hoc comparison only. No SOTA or incremental JEPA
value follows from this outcome.
The forward-EMA JEPA adaptation with fixed VICReg-style anti-collapse penalties
completed all five serial TRAIN/validation slots on CUDA with exit status zero,
real checkpoints and forecasts. Distinct outcome review recomputed all five
metrics and checkpoint predictions and approved post-hoc comparison only. The
three-seed forward ensemble scores 0.653315 dB versus direct 0.639062 dB,
worse overall and at all horizons. The fixed variance penalty raised TRAIN
teacher rank but did not establish incremental JEPA value. No selection or
final inference has been performed.
CAL/TEST acoustic outcomes remain unopened pending final selection and freeze.

The separately reviewed metadata-only TEST scanner completed one real pass over
the source archive and inventoried 1,216 candidate cutoff interval IDs. Its
saved report leaves actual issued and scored rows explicitly unknown until the
one-time numerical QC/materialization boundary. Distinct outcome review
reconstructed all candidates and 76 excluded cutoffs/reasons from metadata,
then froze the inventory. The 49 source dates with at least 18 metadata
candidates are only an upper bound on numerical eligibility. No TEST
`Sv_mean` values were parsed.

A draft final CAL/TEST scorer has synthetic tests but received independent
`REQUEST_CHANGES` before any numeric access. Its first issuance/forecast-order
defect was repaired through a one-pass reader, and a second review still
requires exact selected checkpoint-to-plan bindings, composite adapter code
hashes, mandatory core JEPA comparison, a fail-closed hybrid choice, exact
Chronos snapshot contents and correctly separated value/promotion gates.
No CAL or TEST numeric run has started.

The independently approved five-source neutral validation comparator executed
exit zero and saved all individual/control and equal-weight three-seed family
scores on identical support, plus paired 48-hour bootstrap diagnostics. A
distinct reviewer reproduced all 40 candidates, 49 bound files, every daily
score and all fixed-seed bootstrap intervals exactly. It selected no model.
The lowest development
scores are post-hoc LightGBM 0.638460 dB, core direct ensemble 0.639062 dB,
and core EMA-JEPA ensemble 0.642170 dB; the tiny LightGBM difference is not a
SOTA or inferential claim. Final evaluator/adapter review still requests exact
artifact and transitive-code bindings before CAL or TEST access.

ADR 0011 records the resulting reviewed development choices: direct-neural
three-seed core conventional, EMA-JEPA three-seed core comparison, and LightGBM
as the exploratory post-hoc operational candidate. A distinct reviewer checked
all six neural checkpoint and LightGBM model/recipe hashes and approved the
record; the machine-readable component freeze remains pending. The separate
AEON development-only app/API package passed independent review with the five
approved model outcomes, 25 historical blocked entries and zero AEON cached
forecasts. It is not the final research release.

The AEON portable offline research wrapper is integrated on main and passed its
focused tests. Its isolated smoke ZIP was verified after fresh extraction,
offline dependency installation, loopback API serving and headless Chromium
page inspection. Distinct source review approved this exact development-only
wrapper (review SHA-256
`a7e99f8337195aa676073820a3234a0aaa3de8d76cce8af1b828b234f26c9001`).
It remains development only, with no AEON cached forecasts. Earlier final
evaluator reviews found real-access and lineage weaknesses; their exact
rejected source states are preserved. The repaired runner now forecasts from
frozen components in process, binds the trusted reviewer and passes full-file
Ruff. One stronger
supervised architecture was trialled on TRAIN/validation only as a separate
post-hoc exploratory experiment under local GPU limits. Its fixed first seed
completed 3,000 updates and scored 0.929989 dB on the same eligible validation
dates, worse than the reviewed direct ensemble's 0.639062 dB. The fixed
continuation gate failed, so its other two seeds were not run. The checkpoint,
predictions and report are preserved under `outputs/aeon_patchtst_exploratory_20260927_r2/`;
distinct outcome review reproduced the score, row identity and checkpoint
forecasts and approved the negative exploratory record only (SHA-256
`232f453e1584472e1dcc1ea1d8812c76104a40d4cd9645f19720b25e0423adec`).
The evaluator's one-pass CAL repair, pre-reader model checks and separate
independent source approval are integrated.

The first CAL access command safely failed at the runner-review hash gate
before constructing its reader because a reviewer record transcribed one
config digest incorrectly. The original reviews remain in history; distinct
superseding V2 source, selection, reader and runner records corrected that
digest and passed every pre-reader gate. The exact real CAL command then
completed exit zero and wrote `outputs/aeon3_geb_2024_hourly_sv_v1/calibration/calibration.json`
(SHA-256 `a6e35bc5fa5c27b2b0669e7fb9fe7f8ff1bdd69952193418bbbe130e46438c94`).
It has 810 issued rows and 34 eligible target-source dates at each horizon,
above the frozen 12-date CAL floor. Frozen direct, EMA-JEPA and LightGBM raw
daily pinball scores are 0.653051, 0.633284 and 0.639519 dB on CAL; these
values do not change model selection. Their nonnegative 90% interval widening
was fitted. Distinct outcome review reloaded every frozen model, reconstructed
all 810 rows, all nine adjustments and the complete raw/widened metrics, and
approved the exact CAL artifact for a later pretest freeze (SHA-256
`7b61385b29836be53d0e2c5a0b5e3f5db7e2dcb8983e534d17711ed0b44c1428`).
At that checkpoint TEST numeric values remained unopened while the separate
source/access graph was repaired and reviewed.

The separate pretest source, freeze, reader and runner reviews then approved one
exact retrospective TEST execution. That command completed exit zero and saved
the single-read score plus three forecast files under
`outputs/aeon3_geb_2024_hourly_sv_v1/retrospective_test/`. The score JSON SHA-256
is `23f9d350528e9f99b35949c6281c39d209dff7f7db44a992d241d64ccde8bd89`.
All 1,216 metadata candidates became issued rows; actual eligible target-source
dates are 49/50/51 at 1/3/6 hours, above the frozen 20-date floor. Independently
reviewed raw primary daily pinball is 0.563387 dB for direct, 0.533954 dB for
EMA-JEPA and 0.542202 dB for LightGBM. The frozen EMA-JEPA comparison improved
the direct loss by 5.224%, with a paired 95% candidate-minus-direct interval
[-0.045456, -0.014497] dB and no per-horizon guard breach. It passes the full
within-study retrospective rule. LightGBM misses the 5% point criterion.
The distinct reviewer reconstructed issuance/QC, all model forecasts bit
exactly from checkpoints, metrics, widening, 2,000 bootstrap draws and gates,
approving the exact outcome (review SHA-256
`75a241f00d34f8d373ccbd4c5b2e2c35397e2c0ff0a20a6e112ce4953f13489a`).
This positive one-deployment forecast-family result is retrospective, not
sealed or SOTA; post-hoc family selection, a validation non-result and the
random-hybrid control limit JEPA mechanism attribution. The CAL-only app integration separately passed
independent review, retaining historical blocked rows and zero AEON cached
forecasts.

The separate AEON final source and offline package are now independently
approved. The exact reviewed release is
`release/aeon-offline-retrospective-candidate-20260927-r4.zip`, SHA-256
`619fabfae8c880a2079d4fdc3399d1f4de248c4575d97c59d9a617e1b4063d63`.
Its preserved builder name is an archive identity; the external exact-byte
review `orchestration/reviews/AEON_OFFLINE_PACKAGE_R4_REVIEW_20260927.json`
(SHA-256 `fb0c2cd75e6823eed2226bb614b391597293b1e083aa165110d1fb995196787d`)
approves it as an offline research release. The rejected first source claim
review and r3 header contradiction remain recorded, not silently overwritten.
Fresh relocation verified all 78 packaged files, installed 13 wheels without
network access, served the reviewed study and bounded replay on loopback, and
passed a Chromium page, model-switch and pagination smoke with no page errors.
The package contains 3,648 saved model-row forecasts over 1,216 cutoffs,
retains all 25 historical blocked registry rows, and has no live AEON
forecasting or raw acoustic archive. Its build-time metadata still says
release review pending; the independent external record approves the immutable
bytes without rebuilding them. This is a retrospective non-sealed research
release, not a production or business-validation gate.
