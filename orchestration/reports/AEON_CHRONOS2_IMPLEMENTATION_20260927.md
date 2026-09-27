# AEON Chronos-2 zero-shot implementation

Date: 2026-09-27. Classification: post-hoc TRAIN/validation development, not a final
evaluation. Chronos-2 was added after the finite core campaign and validation selection;
its state-of-the-art standing in this AEON study is `NOT_ESTABLISHED`.

## Frozen application choice

The optional baseline uses the official `amazon/chronos-2` model at Hugging Face
revision `29ec3766d36d6f73f0696f85560a422f50e8498c`. The model repository and
`chronos-forecasting` package identify Apache-2.0 licensing. The exact model card,
configuration and weight digests are recorded in `configs/aeon_chronos.json`, along
with the PyPI 2.3.2 wheel and source-distribution SHA-256 values.

Each eligible AEON validation issuance is passed through the official multivariate
array API as four target variates by 24 prior source intervals. The variate order is
38, 125, 200 and 455 kHz. The issuance contract already requires all 24 prior 38 kHz
products. Missing past products in the other three channels remain `NaN`; Chronos-2's
native preprocessing converts those values to its observation mask. No learned or
hand-written imputation is introduced. Cross-learning across issuance rows is disabled.

The frozen model predicts six source-interval steps and five model-native quantiles.
Only the 38 kHz variate is evaluated, at zero-based steps 0, 2 and 5, corresponding to
the preregistered 1/3/6 source-interval horizons. The five outputs receive the same
fixed monotone rearrangement used by the campaign contract. No TRAIN labels, validation
targets, calibration outcomes or test outcomes enter inference. TRAIN is read only as
part of the already reviewed cohort reconstruction and digest; the model remains zero-shot.

## Executor and safety properties

`src/marine_echo/training/aeon_chronos.py` accepts only the reviewed TRAIN/validation
reader and exact standard validation cohort. It refuses a changed config, source, model revision,
snapshot file digest, independent-review config/code binding, row digest or code digest. Every batch
is written as an atomic shard. A restart validates completed shard hashes and row IDs,
recovers a fully written orphan shard, and computes only missing batches. The final NPZ
uses the existing AEON prediction-row schema and the corrected evaluator requiring at
least 18 scored target-source-date anchors per horizon.

The external shard size is 16 issuance rows and the official pipeline series batch size
is 64, accounting for four variates per issuance. The executor checks a 22 GiB process-tree
RSS cap, a 10 GiB peak reserved GPU cap and a two-GPU-hour elapsed cap after every batch.
Elapsed inference time and peak resource maxima persist cumulatively across restarts. Final
assembly accepts only the canonical batch-name sequence and revalidates every shard's schema,
row IDs, finite values and quantile order. No training or parameter update API is called.
Both the normal CLI and direct executor calls require the exact split-review, corrected-rescore,
rescore-outcome-review, cohort and standard validation-row hashes. Arbitrary rows cannot reach
pipeline loading. The corrected-rescore row digest `9ce5ed6d...` binds the standard scoring-row
schema; the separate full input digest `ea672898...` additionally binds all context values,
masks, timestamps and provenance used by Chronos-2.

## Current execution state

The exact model snapshot is present in the shared Hugging Face cache and `hf cache verify`
checked all four repository files. Real inference has not run. The Python environment does
not currently contain `chronos-forecasting==2.3.2`; shared dependency and lockfile changes
belong to the coordinator. An independent prefit review artifact bound to the final config
and coordinated single-GPU ownership are also required before the real command can start.
Calibration and retrospective test acoustic outcomes remain prohibited.
