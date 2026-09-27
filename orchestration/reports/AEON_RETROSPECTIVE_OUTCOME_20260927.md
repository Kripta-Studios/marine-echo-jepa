# AEON retrospective acoustic forecast outcome — 27 September 2026

This is a new, separately reviewed public-data study on the AEON3 Georges Basin conditioned hourly AZFP product. The historical MOSAiC v1 90-day eligibility failure, 25 blocked slots and v2 raw-code engineering release remain unchanged. The AEON result is retrospective and one-deployment only; no sealed holdout or independent calibration of absolute backscatter has been established.

## Measurement and comparison

The target is source-reported 38 kHz FullDepth hourly `Sv_mean` at 1, 3 and 6 nominal source-interval horizons. Each forecast uses 24 earlier four-frequency source intervals and observed masks. The source clock timezone, product availability latency, publisher conditioning and instrument calibration chain are not fully resolved. Missing target intervals are excluded according to recorded QC, never filled with future truth. The minimum eligible source date has 18 observed anchors; TEST requires 20 eligible dates per horizon.

The frozen core conventional reference is an equal-weight three-seed direct neural ensemble. The frozen core SSL comparator is an equal-weight three-seed EMA-JEPA ensemble with TRAIN-only representation pretraining. Both use the same 128-width encoder class and 1/3/6-horizon five-quantile heads. A fixed-lag LightGBM quantile model is a separately labelled post-hoc development challenger. The full TRAIN/validation campaign, random/learned hybrid controls, forward-EMA ablation, 120M-parameter frozen Chronos-2 comparator and a bounded PatchTST-style challenger are detailed in [the SOTA architecture review](SOTA_ARCHITECTURE_REVIEW_20260927.md). Their development outcomes were frozen before CAL/TEST access.

CAL used 810 issued rows and 34 eligible source dates per horizon solely to fit nonnegative widening of the fixed 90% intervals. The one-time retrospective TEST issued all 1,216 metadata candidates and produced 49/50/51 eligible source dates, with 1,175/1,189/1,204 eligible scored rows at horizons 1/3/6. The primary raw five-quantile daily mean pinball loss on common support is:

| Frozen model | TEST daily pinball, dB | Relative to direct | Frozen full promotion rule |
|---|---:|---:|---|
| Direct neural, three seeds | 0.563387 | reference | reference |
| EMA-JEPA, three seeds | 0.533954 | 5.2243% lower loss | PASS within-study retrospective comparison |
| LightGBM, post-hoc | 0.542202 | 3.7602% lower loss | FAIL 5% point requirement |

The EMA-JEPA-minus-direct paired 48-hour-block, 2,000-draw bootstrap 95% interval is [-0.045456, -0.014497] dB. Its loss is lower at all three horizons, and the frozen 5% point, strictly favorable paired interval and per-horizon regression rules all pass. LightGBM's paired interval [-0.041936, -0.003336] dB excludes zero but does not satisfy the frozen 5% point/full rule. CAL-widened retrospective 90% interval coverage exceeded 90% at every horizon for all three models; widening did not select the model or replace the raw primary score.

The [distinct TEST outcome review](../reviews/AEON_RETROSPECTIVE_TEST_OUTCOME_REVIEW_20260927.json), SHA-256 `75a241f00d34f8d373ccbd4c5b2e2c35397e2c0ff0a20a6e112ce4953f13489a`, independently reconstructed every issued candidate and QC decision, regenerated all three TEST forecast tensors bit exactly from the six neural checkpoints and LightGBM model/recipe, and reproduced all metrics, widening and bootstrap draws. The reviewed score JSON is `outputs/aeon3_geb_2024_hourly_sv_v1/retrospective_test/test-score.json`, SHA-256 `23f9d350528e9f99b35949c6281c39d209dff7f7db44a992d241d64ccde8bd89`. The [machine run ledger](../aeon_run_ledger.json) binds the pretest freeze, CAL artifact, source and approvals.

## Interpretation and gates

The selected EMA-JEPA forecast family passes the fixed **within-study retrospective** performance gate against the selected direct ensemble. This is positive evidence for that comparison, but not proof that learned JEPA representations caused the gain: the corrected core validation score slightly favored direct (0.639062 versus EMA 0.642170 dB), the family choice was made after development outcomes, effective rank was low, and a frozen-random hybrid beat learned hybrids on development. The stronger forward-latent/variance adaptation and the 120M Chronos-2 zero-shot model were both worse than direct on development; the single PatchTST-style seed was markedly worse and failed its predefined continuation gate. The study therefore does not establish AEON or marine-acoustic SOTA, external generalization, or universal SSL superiority.

The software and experiment gates have real executed evidence; the narrow JEPA-value comparison passes retrospectively with the above attribution limits. Biological, operational and business-validation gates remain unproven. This product is not a species, biomass, catch, causal intervention or fuel-savings predictor. Release packaging and its independent offline review are tracked separately in `orchestration/STATUS.md`.
