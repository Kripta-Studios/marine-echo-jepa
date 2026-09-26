# Experiment protocol — freeze before accessing test outcomes

## Questions and separate claims

H1: Does temporal self-supervised pretraining improve future acoustic-index prediction
relative to the strongest validation-selected conventional model?
H2: Does it outperform the same neural encoder trained directly on observed future targets?
H3: Is improvement attributable to temporal representation learning rather than an ensemble,
extra covariates, leakage, a decoder, or more tuning?

These are within-deployment public-data questions. No fish species, biomass, tropical FAD,
fuel-saving, catch, or Marine Instruments performance claim follows from a positive answer.
Future numerical targets are automatically observed: do not advertise manual-label savings
against numerical forecasting baselines. A 10%/25% target-availability experiment is optional
and must be called a restricted-training-target experiment, not manual annotation reduction.

## Calendar split and eligibility

Use all complete UTC calendar days between catalog deployment start/end, before inspecting
outcomes. Allocate chronological calendar ranges 60% train, 15% validation, 10% calibration,
15% test, rounding boundaries once and persisting exact timestamps. Never compute boundaries
from which observations happen to look clean. Metadata-only inventory and deterministic QC
are permitted; do not visualize held-out outcomes or scores before freeze.

For each partition, require the full 24-hour context and final six-hour target extent to lie
inside that partition. This leaves approximately 30 hours between the last usable anchor of
one partition and the first of the next; explicitly test that no raw timestamp is shared.
A filename, day, or derived window must never be assigned independently by a random split.
All pretraining, scaling, imputation statistics, target thresholds, feature selection and
regularizer diagnostic references use training only. Calibration is not validation.

Minimum eligibility: 90 usable days overall, 12 calibration days and 20 test days, plus at
least 80% target support per included sample. A small example that fails these gates remains
a loader/demo diagnostic. Report day counts before and after QC. One deployment is still one
deployment, regardless of how many windows or bootstrap draws it produces.

Use hourly prediction anchors for the primary evaluation. Context is 96 trailing 15-minute
bins. Three outcomes are the one-hour intervals ending at +1, +3 and +6 hours. Target intervals
are [t,t+1h), [t+2h,t+3h), [t+5h,t+6h); changing their definition requires a new protocol version.

## Primary metric and frozen model selection

Primary metric: mean pinball loss in dB of the 38 kHz log-linear acoustic index, averaged
uniformly over quantiles [.05,.25,.5,.75,.95], horizons [1,3,6], then equally over eligible UTC
days. Pinball(q,y,f) = (q - 1[y < f]) * (y - f). Report each horizon and quantile separately.

Secondary: median-prediction MAE in dB; valid-bin profile MAE; 90% interval coverage and width;
valid prediction coverage/abstention; seed variability; CPU latency; GPU/RAM and train cost.
Do not replace the primary metric because another metric looks better. Do not mix a scalar
interval with a depth-wise profile interval. Count all eligible rows, including abstentions;
compare errors on a shared support and report the resulting coverage explicitly.

Validation selects the best conventional family B0–B4, the best JEPA family M1/M2 and any
hybrid choice. All families get the declared same inputs and recorded tuning budget.
B4 is a matched-capacity supervised neural model, not an intentionally weak small network.
Report all families, not just a winner. If an ensemble is used, add an equivalent-seed neural
ensemble and, where meaningful, a tree ensemble with comparable fitting budget.

## Run allocation

Seeds: 7, 13, 23 for B4, M1 and M2. Seed-7 development precedes replication. A maximum of two
validation-only configurations per learned family is allowed; each configuration has at most
3,000 updates per training phase. After family configuration selection, retrain its three
seeds from scratch under that fixed configuration. Record any reused seed-7 endpoint rather
than counting it as an independent fourth run. The hybrid head receives train-only embeddings.
Tree hyperparameter search is capped at eight declared configurations per family. Optional
Chronos-2 is frozen and gets at most two GPU-hours; no optional model can delay P0 completion.

Run random-encoder and temporally shuffled-target controls at seed 7 for both M1/M2 after
configuration selection, preserving preprocessing, decoder and step budgets. Shuffle targets
inside train, with a minimum temporal separation, never across splits. Record that single-seed
controls support only bounded diagnostic attribution. Do not silently skip control failures.

## Calibration and uncertainty

Each baseline must output a distribution: residual quantiles fitted using training/validation
as appropriate, or direct quantile heads. Reserve calibration to adjust only the frozen 90%
interval: implement a documented nonnegative residual-quantile widening rule. Report raw and
adjusted coverage. Do not rewrite the five model quantiles using test observations.

This is a serially dependent nonstationary deployment. Label intervals as empirically
calibrated. Do not promise distribution-free future coverage or independent-sample conformal
coverage. Report calibration day count and an interval-width diagnostic. Quantile crossing
must be prevented or resolved by a frozen monotone rearrangement applied equally to models.

## Test seal and scientific criterion

Before evaluating test predictions, independently review and freeze:
- exact source, split, config and checkpoint hashes;
- selected comparison, supported channels/range and horizon definitions;
- seeds, ensembles, calibration rule and statistical procedure;
- row-eligibility policy and all failures/abstentions;
- a protocol JSON signed by a local SHA-256 digest (integrity, not third-party certification).

Materialize the final test result once. Retain immutable predictions and all row errors.
A deterministic repeated evaluation to check exact reproducibility is permitted, but no
selection or tuning from its result. A bug discovered after opening test outcomes invalidates
that confirmatory use; preserve it and classify a repaired run as post-hoc diagnostic unless
new untouched future data is available. Never call a familiar test set sealed again.

Proposed incremental-value gate: at least 5% reduction in primary loss versus the strongest
frozen conventional reference, with a paired block-bootstrap 95% interval strictly favouring
the candidate. Use 48-hour contiguous blocks and 2,000 seeded resamples, identical draws for
all models, stratified only by preregistered rules. With ~20 test days the effective sample is
small: state this limitation; the interval is conditional on this deployment, not universal
statistical proof. Also report paired daily differences, seed results and horizon-specific
regressions. No horizon may deteriorate by more than 10% relative primary loss without an
explicitly narrowed output claim. Keep the original full gate outcome visible.

A negative value gate can coexist with a complete MVP: serve the best supported conventional
model and keep JEPA in the comparison lab. Missing data/runs or broken evaluation are blocked
or incomplete evidence, not scientific negative results.

## Observation freshness study

After core selection, replay stale observations at ages 1/3/6 hours and channel/block dropout
using fixed masks and seeds. Ground truth remains unchanged and is not available to decisions.
Show how interval width and error vary. A refresh heuristic may use age, missingness and
training/validation-calibrated disagreement, never the actual future residual. Optional
refresh evaluation compares equal query counts with a fixed schedule; do not claim satellite
cost savings, energy savings or counterfactual biological control from this simulation.

## Endpoint and support implementation notes

Use the bin-boundary rules in docs/03_ACOUSTIC_DATA_CONTRACT.md and testL06. The80% primary
target support gate applies to the fixed38kHz analysis band, not to all bins of every high
frequency. Frequency-specific unavailable ranges remain masked; never discard an entire
deployment merely because a high-frequency channel cannot observe the full128m grid.
