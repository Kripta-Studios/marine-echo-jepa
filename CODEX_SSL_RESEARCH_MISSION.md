# Marine-JEPA: model-first self-supervised research mission

Date: 29 September 2026
Status: proposed owner instruction; activated when the owner supplies the accompanying start prompt.
Deliverable: trained reusable acoustic encoders, trained forecast readouts, independent evaluation and a scientific model package. NOT another app or meeting handoff.

## 1. New objective and precedence

The owner explicitly wants an impressive self-supervised model and a credible state-of-the-art research attempt, not another diagnostic application. This mission supersedes previous planning instructions saying `NO_FURTHER_FITTING_BEFORE_MEETING`, `DIAGNOSTIC_ONLY`, or `MEETING_READINESS_ONLY` for NEW research runs. Those were prioritization decisions, not permanent scientific constraints. Do not ask again merely to authorize the local training described here once the owner supplies this mission.

It does not supersede filesystem restrictions, review policies, credentials, cloud-payment approvals, immutable historical protocols, or source-specific restrictions on historical studies. Use a separately versioned research study and record its data-access authorization before opening additional numerical payloads.

Continue the existing checkout:
`C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\marine-echo-jepa`

Use a new branch such as `research/marine-jepa-vnext`; inspect the actual Git state first. Do not reset to 2455fd3 or erase later work. Reuse the environments, source archives, experiment utilities, tests and proven data parsing. Do not create another repository. Freeze web/release implementation except for changes indispensable to model execution. All authored code, reports, comments, identifiers and commits remain English.

Preserve the approved r3 archive and historical scientific evidence. The recorded r3 digest is:
`5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57`.

## 2. Accepted findings, not another preliminary investigation

Read the latest diagnostic and its distinct review once and turn their findings into design requirements:
- Direct-1500 outperformed the historical EMA recipe on exposed validation; that does not retire self-supervised research.
- A flattened, position-specific MLP plus permanently masked suffix removes loss gradients from some input columns during historical EMA pretraining.
- The EMA suffix teacher has much lower centered variation than the predictor; regularizing predictor outputs alone did not establish useful latent geometry.
- Shared-SIGReg also showed low rank, so suffix starvation is not a sufficient explanation for all failures.
- Extending the old recipe to 30k updates reduced sampled training error and worsened validation. Repeating that unchanged run is not the assignment.
- The prior-year AEON3 archive is TRAIN for expanded models and descendants, including preprocessing descendants.
- Historical forecast-head initialization and shuffled-control batches were not completely matched.
- The previous forward-EMA MLP variant already ran and lost on development. Do not rerun it under a new name.
- 64 observations cannot yield a centered covariance rank greater than 63; do not invent a rank acceptance threshold ignoring sample size.

Do not spend this phase redoing the completed census, checkpoint audit, full historical preservation inventory after every minor edit, or meeting materials. Test NEW model behavior and TRAIN it.

## 3. Research target

Working name: Marine-JEPA (an internal project label, not a verified unique publication name).

Learn a reusable representation of acoustic observations and their temporal evolution across deployments, frequencies and native measurement geometries. Test whether it improves forecasting and robustness when the encoder is frozen or adapted with limited new deployment data.

Primary task: predict each deployment's OWN published native 38-kHz integrated acoustic product at 1, 3 and 6 source intervals, conditioned on its measurement metadata. Preserve the old 0–200 m task as a separate historical reference; do not relabel other layers as 0–200 m.

The ambition is a strong transferable acoustic pretrained model. Architecture names, parameter counts, reconstructed plots and software tests do not establish SOTA. A new benchmark without comparable published results supports `best among evaluated methods on benchmark X`, not an unqualified field-wide SOTA claim. Pursue publishable comparisons rather than substituting either inflated claims or a demo.

## 4. Two research candidates, not an unbounded architecture menu

### Candidate A: source-grounded CF-JEPA adaptation

Retrieve and pin the actual official `WDSLab/CF-JEPA` source. Inspect its encoder, objectives, crop sampling, causal inference and online/EMA readout choices. Do not blindly execute its setup shell script. Retain applicable licenses and isolate optional dependencies from the production environment.

Implement a source-faithful temporal baseline before adding marine-specific components. The author implementation uses a shared multi-scale temporal convolutional encoder, forward target crops at multiple horizons and distinct online/EMA readout behavior. This is materially different from the historical flattened MLP adaptation. Document every deviation introduced for masking, irregular availability and metadata.

Run its small TRAIN-only integration test and one fixed real-data development fit. A paper reproduction on an official small forecasting setup is a useful additional correctness/benchmark check within budget, not permission to claim the paper's reported scores as local results.

### Candidate B: observation-aware LeWM-inspired predictive encoder

Use one shared temporal patch encoder for context and future blocks, plus a horizon/query-conditioned latent predictor. Share input processing across time positions and channels as appropriate. Encode frequencies, interval duration, geometry type and bounds, observation masks and time offsets explicitly.

Start with a compact temporal token model, not an image model applied to screenshots:
- initial model class: shared patch projection plus temporal convolution/attention;
- proposed width 192, four to six temporal blocks, four or six attention heads where applicable;
- initial total target 2–8 million trainable parameters, measured after construction;
- base patch length 4 source intervals, with padding and mask handling tested;
- predefined history comparison 24 versus 96 source intervals, with ALL matched baselines receiving the same history;
- latent width 64 or 128 selected on TRAIN/development only, not inferred from an arbitrary desire for high rank;
- token budget at most 512 per sample for the initial implementation; factorize time/frequency/range attention only if actual profiles require it.

This is a design proposal, not a reproduced architecture or a guarantee that 8M parameters outperform 1M. Match a directly supervised version of the same backbone. The original MLP remains a historical and computational floor, not the only supervised competitor.

## 5. Data: native measurement semantics unlock additional archives

The report says four additional official AEON archives already exist on E:, with 25,409 38-kHz FullDepth rows at nominal 0–220, 0–225 or 0–230 m. Existing source IDs include 61937263, 61937266, 61937272 and 61937278. Verify actual inventories and identities; do not redownload unnecessarily or treat all rows as usable windows.

The original and prior-year AEON3 archives (IDs 61937281 and 61937275) are existing development/training resources. Preserve their earlier experiment lineage.

New native-product data contract:
`value, observed_mask, censoring_or_qc, frequency_hz, interval_start, interval_end, clock_status, geometry_kind, upper_bound, lower_bound, orientation, processing_id_or_unknown, instrument_mode_or_unknown, deployment_id_for_grouping`.

- Native 0–230 m and native 0–200 m are distinct measurements, not interchangeable labels.
- Train the model to predict the future of the actual product specified by a query. Give all comparisons the same metadata and observations.
- Do not multiply a dB value by 200/230, invent a cropped average, or claim the model has recovered unobserved depth structure.
- Metadata conditioning does not prove comparability or remove instrument/processing domain shift. Test these empirically and report the limitation.
- Do not concatenate across deployments, long gaps, clock discontinuities or configuration changes.
- Do not fit scalers or SSL objectives on held-out payloads. Context-only normalization, if used, must have its own causal test and same baseline treatment. Never normalize a test deployment using all of its future observations.
- Valid weak values, censored observations and missing values are distinct. Use source quality flags rather than recycling MOSAiC's detection or 90-day rules.
- Unknown clock timezone does not prevent relative-source-interval forecasting; do not fabricate UTC or real-time product availability.
- Require source-product provenance, not a new hardware calibration expedition before every experiment. Forecasts remain conditioned-product forecasts unless absolute calibration is independently established.

Richer profile path: inspect already acquired archive inventories for genuine layer-resolved or higher-cadence exports. Preserve numerical profiles and frequency/range coordinates when they actually exist. Provider-supported additional processed profiles may be acquired within the existing access/storage budget. No profile availability or independent calibration is assumed. Lack of profiles must not stall the native integrated-product campaign. Do not call the scalar-only version a learned full water-column model.

## 6. Reserve data before building a new selection loop

The recent repeated use of old validation cannot be undone. Use it for debugging and historical comparisons, never as the sole basis of the new model claim.

Before scoring new models, create a source-level exposure ledger from actual history:
1. All sources used by an ancestor's weights or preprocessing are training-exposed for that ancestor.
2. Separate metadata-only inventory from previous numerical inspection and outcome-guided decisions.
3. Select train/validation/final-test deployment groups using metadata and a deterministic policy BEFORE new numerical scores.
4. Prefer an entire non-AEON3 site as held-out test and a different non-AEON3 deployment for validation, subject to actual source availability. If there are only two sites, use whole-deployment separation and disclose the narrower claim.
5. Compatible unexposed archives may support held-out evaluation without sharing the old 0–200 m geometry, because this is a NEW native-product task. This does not rescue the old AEON2 experiment.
6. Keep within-deployment context/target blocks wholly inside their declared partitions, or specify a deployment-history evaluation convention and prove no fitting uses assessment targets. Use conservative disjoint raw-interval blocks initially.
7. Reserve enough later contiguous data for an adaptation experiment only if separately partitioned: an adaptation prefix, calibration where needed, and a later test suffix. An adapted model is no longer zero-shot.

If clean whole-site evaluation is unavailable, run a documented deployment-held-out study. Do not claim it is sealed when exposure is uncertain. Do not silently choose a new favorable test set after scoring. Independent reviewer approves exact split and sample policy before the first selection run.

## 7. Self-supervision specification

Pretraining is genuinely task-independent of the scalar forecast readout. It must produce an encoder checkpoint that can be frozen and reused; end-to-end quantile improvement alone is insufficient evidence of representation value.

For a TRAIN-only trajectory, form context C ending at cutoff t and genuinely later nonoverlapping blocks Y_h. Sample multiple crop origins within eligible TRAIN so roles do not permanently starve a subset of shared input processing. Target crops at horizons 1/3/6 must preserve actual interval semantics. No target from validation/test enters pretraining.

Let `z_c = E(C, metadata)` and `z_h = E(Y_h, metadata)`. Predict `zhat_h = P(z_c, horizon, query_metadata)`. Candidate B starts with the LeWM-inspired objective:
`L = average_h MSE(zhat_h, z_h) + lambda * SIGReg(encoded_observations)`.

Candidate B has no EMA or target detachment in this shared-encoder route. SIGReg acts on genuinely encoded observations from context and target blocks, not only on predictor outputs. Preserve upstream normalization/projection semantics deliberately; do not place a normalization that mathematically forces the distribution incompatible with the regularizer and then compensate by increasing lambda. Record how correlated tokens and batch sampling affect the regularizer's effective sample set.

Candidate A follows the pinned author EMA objective and crop design unless an explicit, reviewed adaptation is necessary. Do not conflate the two recipes or transplant one loss's guarantees to the other.

Start with a single future-prediction objective. Cross-frequency masking and multi-scale blocks are controlled additions, not six loss terms introduced simultaneously. For optional masked-channel reconstruction distinguish offline imputation from causal forecasting and prevent access to the missing channel through preprocessing.

A latent predictor trained only on passive observations is not an action-conditioned biological simulator. Do not invent actions from depth or time metadata.

## 8. Strong matched comparisons and scientific endpoints

Mandatory comparisons:
- persistence and appropriate source-period seasonal reference;
- strong LightGBM quantile forecasts;
- directly supervised temporal backbone matched to the new candidate;
- source-faithful CF-JEPA baseline;
- LeWM-inspired candidate;
- a pinned existing pretrained forecaster such as Chronos-2, with the SAME permitted historical information and explicit zero-shot/fine-tuned status;
- at least one non-JEPA SSL reference, preferably a source-pinned TS2Vec-style encoder or a masked-reconstruction model sharing the candidate backbone.

Historical poor one-seed PatchTST results do not establish that every competitively trained patch model is weak. Conversely, do not repeat it by default just to generate another losing baseline.

Two levels of fairness must be reported separately:
1. equal downstream supervision and data with/without pretraining (extra pretraining compute reported);
2. comparable total measured compute and tuning budget for the full recipes.

Do not call equal optimizer steps equal compute. Match initial encoder/head tensors where the comparison intends to isolate the loss. Match correct/permuted target batches and target multisets; only pairing changes. Log or deterministically bind the actual sample sequence.

Representation outputs:
- frozen encoder + the same lightweight readout for each SSL/random/supervised feature comparator;
- full fine-tuning results separately;
- 1/7/30-day adaptation-prefix learning curve if cohort support permits, with the same numerical targets given to scratch baselines;
- controlled past-channel/observation dropout on the SAME held-out targets, no invented new truth;
- no claims of saved human annotation from numeric targets already present automatically.

Primary forecasting score: five-quantile pinball, equal weighting over eligible source dates, horizons and evaluated deployments. Report all native-product dB scores per deployment. Predefine an additional cross-deployment skill score relative to the same frozen persistence baseline for EACH deployment, with a denominator rule fixed on development. Do not silently compare differently aggregated old and new scores.

Evaluation precision:
- shared support across comparisons plus explicit coverage/abstention reporting;
- at least three seeds for final learned candidates and matched direct reference;
- separate seed variance and contiguous-time block uncertainty;
- resample deployments where defensible and otherwise label intervals conditional on the small set of observed deployments;
- never treat hourly rows as independent deployments;
- no quantile calibration on final outcomes, and no guarantee of coverage under unseen-site shift;
- numerical replay tolerances chosen for the runtime in advance, not mandatory cross-device bit identity.

Suggested research success target (to freeze before final evaluation): >=5% primary loss reduction versus the strongest development-selected non-JEPA reference, favorable paired uncertainty, and no catastrophic regression concealed by averaging. This is a proposed target, NOT the definition of SOTA or an expected result. Also require evidence that learned representations outperform relevant random/permuted controls and offer transfer or downstream sample-efficiency value.

## 9. Finite but substantive execution budget

A tiny 6,000-update loss-location screen is not the entire new research mission. Equally, do not run a boundless parameter search until a favorable outcome appears.

Default research envelope to freeze against measured throughput:
- one GPU owner, peak reserved/allocated <10 GiB and process RAM <22 GiB;
- CPU/float32 loss checks, then optional mixed precision only after correctness is established;
- local GPU runtime envelope of 96 aggregate GPU-hours for this new campaign, including baselines and controls, with durable resumable commands;
- at most eight seed-7 candidate/configuration screens total across the two main SSL routes;
- at most two SSL finalists and the matched direct reference expanded to seeds 7/13/23;
- at most 50,000 pretraining updates and 5,000 supervised/readout updates per trajectory as ceilings, not mandatory endpoints;
- each run uses a predefined warmup/decay schedule, checkpoint cadence and validation-based early stopping/selection; the old final-endpoint-only policy does not automatically apply to this new study;
- equal checkpoint-selection opportunity and tuning accounting for competing methods;
- operational profiling and correctness smokes do not count as completed scientific runs;
- one bounded hypothesis revision is permitted after a failed development screen, documented before rerunning. No arbitrary extra sweeps.

Use GPU time only where useful; small heads and tree fits may run on CPU. An observed resource failure permits batch/token engineering fixes without changing final assessment rules. The $500 historical compute ceiling remains; cloud provisioning and paid credentials require explicit owner authorization. Do not silently purchase services. Keep a $100 reserve if paid compute is separately authorized.

The model's measured fit cost, available data and validation behavior govern use of the ceilings. A validated candidate need not grow just to consume VRAM. The goal is the strongest supported model, not the largest model.

## 10. TDD and reviews that enable actual execution

Regression tests must cover:
- source-geometry identity through tokenization, query and scoring;
- no artificial 0–200 m label from a larger aggregate;
- nonoverlapping context/future-target intervals and deployment boundaries;
- perturbing future assessment values never changes model inputs;
- gradients reach the shared encoder from all intended observation roles;
- correct EMA detachment versus shared-encoder gradient flow;
- encoder regularization and checkpointed regularizer state;
- finite losses, collapsed/random/planted-signal checks, shape/mask correctness;
- shared initialization and identical batching for paired controls;
- train-only scaling, held-out archive exclusion and no preprocessing contamination;
- checkpoint/RNG/optimizer/sampler/scheduler resume and prediction replay;
- learned readouts never access assessment targets;
- metrics can be independently reconstructed from saved row-level outputs.

Do not require every temporal positional embedding to receive a gradient on every example; test the intended shared weight coverage across a declared batch/crop schedule. Do not maximize rank by rule regardless of downstream usefulness.

Use three substantive reviews: prefit data/code/experiment contract; results and checkpoint reconstruction; claim/model-package review. Routine fixes and runs under the approved finite contract do not require new owner confirmation for every step. A failed representation gate stops THAT candidate or triggers the one approved design revision; it must not redirect the entire task to meeting preparation.

Use the owner's currently verified model routing. The most recent diagnostic verified GPT-6.1 Sol/high implementation and a distinct GPT-6.1 Sol/high reviewer. Inspect local configuration rather than silently reverting to an older requested route. Leader may implement; implementer cannot approve its own work as an independent review. Reviewer execution should use an allowed private scratch workspace where possible. Never bypass command denials or invent independence when execution was inaccessible.

## 11. Deliverables and definition of completion

Required model artifacts, not website work:
1. pretrained encoder weights/config plus training source/split/objective hashes;
2. latent predictor and fine-tuned/frozen-readout checkpoints;
3. standard inference and evaluation commands tested on a relocated model-only package;
4. complete benchmark rows, seeds, baselines, controls and tuning/resource accounting;
5. transfer/adaptation/robustness results with independent data provenance;
6. model card and scientific report stating positive, negative and undecided questions;
7. independent reconstruction record distinguishing executed from inspected evidence.

Expose reusable interfaces such as `encode(observations, metadata)` and `forecast(context, query)`; exact API can follow repository conventions. Prefer safe weight-only formats and no embedded executable model objects. A browser is not required to use or evaluate the model.

A negative finite study is a legitimate research outcome, but it is NOT fulfillment of the user's desired SOTA model. Report this distinction without substituting an app release. If performance is negative, document the highest-information next hypothesis rather than announcing that the general objective is impossible. Actual new trained models and real comparisons remain required wherever data/access allow them.

Begin now with the new source-native contract, reserved split and shared temporal implementation; reuse accepted diagnostics, implement, review and TRAIN. Do not answer with a plan and stop.
