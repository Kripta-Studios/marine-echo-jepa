# Completed first native shared-temporal SSL screen

Recorded30 September2026. This is a completed real TRAIN/development screen,
not the final comparative model report. App/release work remains frozen.

The separate native-data contract and deployment reservation were independently
approved before numerical model selection. TRAIN has18,593 issued windows,
18,312 SSL-eligible; development has7,593 issued windows. Whole AEON2 site
deployments remain reserved and their numerical values unopened. Native product
geometry remains200/220/225/230m as supplied; development forecasts0–225m.

The shared model uses per-channel native metadata conditioning, patch4 shared
projection and four temporal Transformer blocks, width192 and latent64. The same
encoder receives gradients from context and future TRAIN crops. Its objective
combines horizon/query-conditioned latent prediction with encoded-context/future
SIGReg, without flattened whole-history MLP encoding, unit-normalized latent
outputs or a relocated predictor-only regularizer. This is LeWorldModel-inspired
acoustic adaptation, not an unqualified reproduction of that paper.

The independently approved fixed seed7/history96 job completed exit0 after6000
SSL updates plus four fresh500-update selection probes. Original TRAIN-only
scalers and exact source/config/review/data hashes are stored with the artifacts.
The final selected pretrained encoder is step4500; earlier interim reports of
step1500 are preserved progress evidence and are not the final selection.

| Development-only endpoint | Daily pinball (dB) |
| --- | ---: |
| Selected shared SSL, short frozen selection probe |0.8246468454704822|
| Persistence |0.9891759561557131|
| Seasonal24 |1.031738466565871|

These values use the predeclared five quantiles, horizons1/3/6 and equal
date/horizon/deployment weighting, with at least18 daily anchors. They do not
establish representation-transfer advantage, JEPA value or SOTA: strong2000-update
frozen readouts, matched supervised/non-JEPA controls, finalist seeds and held-out
assessment remain required.

The completed shared run owns3371.063 synchronized device-runtime seconds,
including probes and evaluation. Peak allocated199,167,488bytes;
reserved247,463,936bytes; process RSS5,176,950,784bytes. The coordinator charges
its more conservative full owned walltime separately in the run ledger.

Real artifacts are in
`outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0/`:

- `selected_encoder.pt`: SSL-pretrained encoder and TRAIN scaler lineage,
  SHA256`a9f332f080201dccd99bccc25ba658b4136ba760df08a3b29f6c4f1bdc164141`.
- `inference.pt`: that encoder with the selected supervised short-probe head,
  SHA256`b085583862f9b69e153bc041d9a07277bea70cac8e891ccaa859facc3fddd4d8`.
- Complete resumable optimizer/RNG/checkpoint states, per-candidate probes,
  `membership.json`, `scalers.json`, raw development predictions and metrics.

Actual CPU replay passed on64 real development prefixes using only `x`,
`observed`, `metadata` and `query`; no future/assessment targets were inputs.
The reusable API produced64-dimensional finite, nonconstant embeddings and
forecasts with maximum saved-forecast replay error0.00000762939453125dB.
Evidence: `evidence/ssl-research-v1/shared-pretrained-cpu-replay.json`.

The initial unindexed CUDA wrapper attempt failed before optimizer updates and
is preserved. An unchanged approved direct CLI retry using`cuda:0` completed.
Subsequent CF first-backward deterministic-kernel and LightGBM first-booster-save
failures are also preserved; neither is reported as a completed comparison.
Platform repairs require fresh exact independent review before new fits.

Absolute clock/interval edges, censoring/calibration and several instrument
properties remain unresolved as disclosed by the native contract. These public
experiments predict acoustic measurements; they establish no tuna biomass,
species, catch, fuel-saving, causal intervention or Marine production claim.

The finite programme remains active. No final-test result or SOTA declaration
is authorized by this report. Historical evidence and approved r3 stay intact.
