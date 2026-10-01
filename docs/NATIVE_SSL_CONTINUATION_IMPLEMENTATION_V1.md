# Native SSL continuation implementation

This continuation preserves `research/marine-jepa-vnext` at and after
`2d51c43dbb624f980c3a964d51bff6d413bde1f7`. The supplied `dd012f0` is a historical
reference. No reset, completed-job retraining, app change or release work was
performed. The owner's new research authorization is recorded in ADR0025.

## Research state

The original campaign still has 40 completed neural endpoints of 43 planned,
with four additional reference methods. Numerical reconstruction still covers
the independently reviewed original 28 methods. The five newer completed run
scores remain provisional. Neither software tests nor a metadata registration
complete a numerical reconstruction.

The independently reconstructed DEV comparison remains LightGBM 0.505274 dB,
mean CF strong frozen readout 0.628424 dB and mean CF full fine-tuning 0.558910 dB.
These are means of seed scores. They are not prediction-ensemble scores, a
reserved-site result or evidence of SOTA. The original cross-backbone scratch
contrast does not isolate SSL.

The three missing endpoints retain their existing identities, recipes and
seed-23 SSL parent. This continuation does not create substitute approvals or
change their executor, 3000/2000/3000 update limits or Band accounting.

## New executable control extension

The CF_MATCHED_CONTROLS extension has two methods at seeds 7/13/23: the original
CF backbone trained directly with its forecasting readout, and the same backbone
frozen at fresh random initialization with that readout. Their nominal budgets
are 3000 and 2000 supervised updates respectively. There is no additional CF
pretraining, latent objective, predictor training or EMA update in either arm.

The implementation calls the original CF model factory, preserves the encoder
branch consumed by the trained forecasting endpoints, and obtains the original
fresh QueryHead initialized at seed+100000. It reuses the original TRAIN-only
channel/target scalers, masked pinball, supervised index sequence, clipping,
warmup/cosine schedule and scheduled daily DEV selection helpers. The frozen
arm keeps encoder parameters and buffers unchanged. It saves distinct control
artifact kinds so supervised or random encoders cannot be presented as SSL
pretrained weights. Reusable inference loads safe tensor state without invoking
the training initializer or consulting provenance paths.

Full-H96 forecast extraction is retained in every CF arm. The existing sampled
pretraining-crop versus inference difference remains a limitation. Existing
shared masked/shuffled controls do not establish CF objective specificity.
Native measurement metadata and query geometry remain part of the inputs;
0–230 m cannot be relabelled as 0–200 m.

The new resource executor requires a genuine distinct CF prefit with exact
source/data/config/runtime bindings before creating an attempt or child process.
It supervises only its own process tree, journals ownership before launch,
charges failed and resumed attempt lifetimes, preserves unresolved cleanup and
rejects completed-cell retraining. The separate extension cap is 12 full-owned
GPU-hours inside 96 aggregate hours, with 12 aggregate hours reserved for
evaluation. Original Band limits remain unchanged. The manifest is prospective
registration, not approval or a fit record.

## Comparison preparation

The added seed-summary function accepts the exact original 43-neural/47-method
identities and DEV role only. It retains all three fixed seeds in each group,
reports sample SD, and makes an unavailable seed explicitly unassessable. It
never substitutes extension cells, selects a best seed, drops missing records,
decodes predictions or describes a mean of scores as an ensemble.

The existing V5 comparison and V6 inventory preparers remain unchanged. Their
production manifests are still unavailable until all three original missing
endpoints genuinely complete. No duplicate historical scaler files were
manufactured and no inventory guard was weakened.

## Gates and evidence

`orchestration/native_ssl_continuation_state_v1.json` records the closeout.
ROOT's integrated command passed 123 checks in 19.68 seconds with exit 0;
the prospective-review preparer passed seven additional checks in 0.25 seconds.
The integrated command included all four previously ROOT-only optimizer and
resume cases. The initial integrated run's three failures and subsequent fixes
remain in `evidence/ssl-cf-controls-root-v1/integrated-correctness-01.*`.
The six actual configuration registrations and six prospective prefit requests
were written with exit 0. They contain no independent approval. Builder evidence
and source proof preserve all 26 dependency bindings and the exact four-file
delivery. Actual receipts live in `evidence/ssl-cf-controls-root-v1/`.

CPU fixtures are explicitly synthetic correctness evidence. Builder author
checks and coordinator checks do not constitute independent scientific review.
No new real fit, prediction reconstruction or reserved numeric read occurred.
No finalist is selected from incomplete DEV evidence or reserved scores. Existing
final-source and prefix contracts remain in force, including H96, prefix-contained
labels, four-day warmup and the predefined common suffix separation.

The existing independent-review route was automatically rejected before
inspection. The preserved receipts, supported feedback/admin/service remediation
and exact remaining dependency are recorded in
`docs/NATIVE_SSL_REVIEW_SERVICE_DEPENDENCY_V1.md`. This continuation did not retry
that route, change permissions, create another reviewer or count author checks
as approval. Restored legitimate review execution must precede genuine
prefit/source/access and numerical review. The full research objective remains
incomplete.
