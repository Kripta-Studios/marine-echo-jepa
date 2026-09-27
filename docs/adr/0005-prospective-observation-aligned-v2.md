# Prospective observation-aligned study v2

Date: 2026-09-27. Status: proposed for independent review, not data approval.

The owner explicitly authorizes prospective changes in `continuation_v2/CODEX_FINISH_V2.md`.
V1 remains INELIGIBLE_UNDER_ORIGINAL_PROTOCOL. Its 90-day gate, support rules, failed
assays, blocked 25-slot registry and historical r2 are immutable historical evidence.

The preserved XML specifies one unaveraged ping every 15 seconds. This is not an
hourly burst schedule. Existing TRAIN manifests contain 421819 observed pings versus
576000 scheduled pings and one acquisition configuration across 2135 source records.
Noise-rejection flags overlap padding and earlier invalidity; their sum is not a
disjoint causal attribution. Native sample diagnostics must separate these states.

Exactly one candidate is defined before computing its support summary. No frequency,
range or model-score search is authorized by this decision. Retain 38 kHz and 10–100 m
slant range because these have the existing instrument/environment/geometry review.
All four frequencies are inventoried, but only 38 kHz is a numerical model input.

The candidate is the **sampled detected-cell backscatter mean and detection fraction**.
Its population is the fixed 45 two-metre range cells, at observed scheduled acquisitions
inside each fixed future hour. The same frozen calibration, per-ping background
subtraction, 3 dB SNR rule, proximal exclusion, saturation and tilt rules apply. A
detected cell must pass the existing conservative native-to-grid overlap rule.
Compute sum(linear_sv * valid_ping_count) / sum(valid_ping_count), then
10 log10(value / 1 m^-1). The linear quantity is the arithmetic mean of
noise-corrected factory/regional-assumption calibrated sv in m^-1 among detected
observed ping-cell pairs. Its claim label is **detection-conditioned sampled Sv**.
This is a mean among detected sampled cells, not full-band acoustic state, continuous
hourly mean, biomass, or an estimate corrected for censoring. Strong-echo selection is
intrinsic and must be visible with every result.

Detection fraction is sum(valid_ping_count)/(45 * observed_ping_count). Scheduled
acquisition coverage is observed_ping_count/expected_ping_count, with excess pings
reported separately. Neither replaces v1's support definition. Missing scheduled
acquisitions, finite returns below the detection rule, invalid measurements and
missing target values remain distinct; none is zero backscatter.

Prospective numerical floors: each future hour needs at least 120 observed pings
(half of the nominal 240 acquisitions), at least 10% detected grid-cell/ping pairs,
and one known configuration. These floors exclude very sparse measurements; they
are not an independence count, representativeness claim, or precision guarantee.
All failures and horizon-specific target availability are reported. Detection
fraction itself is a joint target on hours with adequate acquisition, including
zero detections; the conditional backscatter target is missing in that case.
The 10% threshold explicitly selects stronger-return regimes; even successful
prediction cannot describe the total 10–100 m acoustic state.

One known configuration means every quarter-hour containing observed pings in the
target has the same non-null configuration hash, with no mixed-configuration flag.
An empty quarter-hour remains missing acquisition, not a new configuration. The
same rule applies to observed context bins; windows crossing an actual change are
excluded. A configuration flag due solely to an acquisition gap does not imply a
hardware change: use the manifest IDs and per-bin observed counts explicitly.

Keep hourly issuance, 24-hour context and the fixed target intervals [t,t+1h),
[t+2h,t+3h), [t+5h,t+6h). Context contains trailing 15-minute observed profiles,
masks, detected fractions, acquisition counts and past elapsed ages. Missing values
remain masked; no interpolation. Predictors never receive future times, counts,
masks or actual validity. Forecast availability depends only on the past: the last
hour must have at least 120 acquired pings and a defined detected-cell index, and
at least 12 of the preceding 24 hours must have acquired pings. Full context and
targets must be inside their chronological partition, with no shared raw support.
Replay assumes zero latency at bin end; actual telemetry availability is unknown.
Every model predicts the same past-eligible issued row IDs, including those whose
future index is unavailable. Future eligibility determines only the scoring mask.
All issued rows remain in coverage/abstention denominators; future validity never
determines whether a predictor is called.

Before any fitting, freeze joint supervision as masked mean quantile pinball on the
index normalized by TRAIN-only mean/standard deviation, plus 1.0 times masked mean
squared error on detection fraction in [0,1]. Compute each component on its own
valid cohort, uniformly over available horizons; an absent component is skipped,
and a batch with neither component fails. No masked target becomes numerical truth.
Report detection-fraction MAE and MSE separately from primary daily mean pinball in
dB; fraction loss is not added to the primary reported research metric. Conventional
methods predict the same fraction using a matched regression/persistence head.
No profile supervision is fitted in this development slice; profile heads, if
present in the existing architecture, are unused and receive no claimed validation.
JEPA representation pretraining remains TRAIN-only; its nonzero temporal objective
is followed by frozen-encoder supervised probes with the same joint supervision.

Keep the original train/validation/calibration/test calendar boundaries. First
development vertical slice uses only original TRAIN: fit February 17–March 31,
development assessment April 1–April 14, with whole-window containment within each
range. This calendar choice precedes candidate support and model score access.
These runs are DEVELOPMENT, not final evaluation or independent campaign seeds.
The first development run is ridge alpha=1 with TRAIN residual quantiles and a
matched existing compact direct encoder, seed 7, AdamW learning rate 0.0003,
weight decay 0.0001, gradient clip 1, batch size 16, 128 updates, 16 warmup updates,
and checkpoints at 64 and 128. No development score selects or extends this run.
Resume verification compares interruption at 64 with uninterrupted execution using
the same batch sequence. This is an engineering smoke on real data. Its scores
must not influence any campaign configuration, model-family choice, feature,
target, preprocessing, training duration or candidate-selection decision. It is
not uncounted tuning. These 128 updates do not consume or replace a final core slot.

Study adequacy is descriptive/exploratory: at least 20 TRAIN days and five days in
each other partition for any reported horizon, with day and fixed 48-hour block
counts disclosed. Fewer than ten populated evaluation blocks makes an inferential
superiority claim unavailable; report descriptive paired errors instead. These
counts do not establish power. Retain 48-hour block resampling, 2000 draws, and
the historical 5% research hurdle when inferential evidence is eligible. No sealed
holdout is claimed: RETROSPECTIVE_TEMPORAL_HOLDOUT is the evaluation label.

Existing physical approval covers TRAIN development only. New support summaries
and first vertical slice require distinct V2-DATA/V2-PROTOCOL development approval.
Full-period physical qualification, finite campaign mapping, pre-evaluation freeze,
evaluation and release each need their own explicit independent scope decision.
