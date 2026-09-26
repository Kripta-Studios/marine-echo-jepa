# Models and training implementation

## Bounded hypothesis

A compact model trained to predict future acoustic representations may improve future-state
prediction or resilience to stale/missing observations relative to direct predictors. This is a
hypothesis, not a condition the implementation must force to become true.

No fish annotations are required to create future acoustic targets. This is also true for direct
forecasting baselines: do not sell an artificial “label saving” when labels are generated from
observed future sensor values.

## Shared information budget

All compared models see the same permitted 24-hour history, masks, calendar features and optional
past telemetry. No future GPS or future measured environmental variables. Set training/selection
budgets before evaluation. Keep identical train/validation/calibration/test manifests.

## Baselines

B0: last-valid-profile/index persistence; abstain beyond the registered age limit.
B1: same-hour previous-day profile/index, using only available past measurements.
B2: ridge forecast over fixed lags, trend, variability and mask/age features.
B3: histogram-gradient-boosting quantile regression over the same engineered history. Train
one head per horizon/quantile where required; regularize and select only within the fixed grid.
B4: small direct temporal neural model using the same encoder capacity as the JEPA candidate,
trained directly on future scalar/profile targets. This is the essential matched neural control.

Optional B5: frozen Chronos-2 [S10], only after licence/revision/size verification, no tuning on
test, no downloading unrelated foundation checkpoints. Limit this arm to two GPU-hours initially.

## Core encoder

Input example `[B,96,4,64]`: 24 h at 15-minute bins, four frequencies, 64 range bins, plus masks.
Tokenize four time bins by eight range bins, mixing the frequency values and masks through a
small learned projection. This produces 24 x 8 = 192 context tokens. Use dimension 96,
three Transformer blocks, four heads and explicit relative time/range encodings. The default
parameter target is below 1.5M; the hard policy cap is 3M unless a pre-test ADR justifies a change.

There is no benefit to pretending a 2B-parameter video model fits the data. Fixed-frequency channels
are acceptable for D1; cross-instrument generality is out of scope until a frequency-aware adapter
is validated. Never relabel 125 kHz as 120 kHz to make an OOI comparison work.

## M1 — temporal EMA-JEPA

Encode only the context available at cutoff. A predictor with horizon queries produces eight
range tokens for each future one-hour interval. An EMA target encoder encodes the corresponding
four future 15-minute frames, yielding the target tokens. Stop gradients through this teacher.
Minimize masked prediction loss; add a modest validated variance/covariance regularizer as needed.
Masking must respect valid observations and not turn missing data into zero-valued signal.

Context attention may be bidirectional **within the past window**. It must not include target
frames. Future blocks do not overlap the context; shared source indices are audited explicitly.
Use floating-point diagnostics for target variance, effective rank and prediction scale.

## M2 — LeWorldModel-inspired shared encoder

Use the same input/token/predictor budget, but one shared encoder receiving gradients from both
sides, with predictive loss plus SIGReg. Reuse an attributed, pinned tested implementation from
the official LeJEPA/LeWorldModel lineage [S07, S08]. Verify the gradient path in a test. Do not
retain `.detach()` on the target and call the result end-to-end LeWorldModel. No EMA here.

The original LeWorldModel is action-conditioned and visual. This adaptation forecasts passive
acoustic observations and must be labelled **LeWorldModel-inspired**, not a literal reproduction
or evidence of action-conditioned control. Horizon/calendar/context tokens are not physical actions.

## Downstream forecasts

For each representation, evaluate a frozen ridge diagnostic and a tree-quantile hybrid using
raw baseline features plus latent/current-future summaries. Primary comparison: hybrid M1/M2
versus B3/B4 with the same information and validation protocol. An optional small profile decoder
is trained on training data only and clearly identified as supervised on observed future profiles.

The user-facing scalar head predicts ordered quantiles 0.05, 0.25, 0.50, 0.75, 0.95 at each horizon.
Profile predictions may be point forecasts; do not display scalar uncertainty as a spatial confidence
map. Record incoherence between separately predicted index and profile; derive the displayed
profile summary consistently or label separate heads, rather than falsely making them identical.

## Budgets and guards

Default AdamW, gradient clipping, FP32 reference, BF16 only after hardware/kernel checks, batch
16 with measured increase to 32 if safe. Maximum 3,000 optimizer updates per main training phase,
validation-only early stopping, and a run time cap. Record actual tokens/updates/epochs.
SIGReg statistics use adequate independent windows in a real batch; gradient accumulation does
not increase the number of samples inside a distribution regularizer automatically.

Micro-overfit on training-only small data is a pipeline check, not a scientific result. Test zero
inputs, all-missing masks, constant data, one bad channel, resume equivalence and non-finite loss.
If a family repeatedly collapses under the registered recipe, record the result rather than launch
an unlimited hyperparameter search.

## Controls and ensemble fairness

Run random frozen encoder and shuffled-future pretraining controls at bounded cost. A shuffled
future control must break temporal association while retaining the same data split and shapes.
Do not demand that an artificial control be worse as an implementation test; measure it honestly.
If a three-seed JEPA ensemble is presented, include an equal-size direct-model ensemble and its
cost; do not attribute the whole ensemble gain to JEPA.

## Initial optimizer recipes and model-selection budget

Machine-readable defaults and the only two initial per-family development configurations are
in configs/experiments.json. AdamW starts at3e-4, weight_decay1e-4, betas(.9,.95), clip1.0,
150 warmup updates, validation/checkpoint each250 updates and a3000-update phase cap. The
direct model compares3e-4 versus1e-3; EMA-JEPA compares representation regularizer weights
.03/.10; shared-SIGReg compares.04/.10. These are proposed recipes, not tuned results.
Freeze the exact imported SIGReg normalization and coefficient convention before the run;
weights are not comparable across implementations with different batch/loss scaling.

Train windows may be sampled randomly only within train. For regularizer minibatches, prefer
non-overlapping30-hour raw supports and report actual overlap rather than claiming independent
ocean realizations. Keep an explicit sampling fallback if too few valid supports exist.
Gradient accumulation does not change those per-batch statistics.

Avoid hundreds of tree models for full profiles. Use a train-only multi-output ridge or small
supervised decoder for the256-bin profile, alongside scalar quantile heads. Each displayed
profile has its own model identifier; it is not a spatial uncertainty map for the scalar head.
The profile is mandatory to visualize, but its secondary quality cannot replace the primary
scalar forecast selection metric.
