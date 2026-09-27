# September 2026 forecasting and JEPA evidence for the AEON study

Status: source-grounded research and executed post-hoc TRAIN/validation development extension. It does not change the reviewed 17-slot core campaign, historical v1/v2 evidence, calibration boundary, or retrospective TEST rule. No model below is called state of the art on AEON without AEON evaluation.

## Task fit

The current target is the source-reported, conditioned 38 kHz full-depth hourly `Sv_mean` at 1, 3 and 6 nominal source-hour horizons. There are 4 historical frequency channels and masks, 24 prior source intervals, about 4,965 TRAIN and 1,219 issued validation windows from one instrument deployment. The exported source-clock timezone and exact instrument calibration lineage remain unresolved. Benchmark results on electricity, traffic, or fish-species classification do not establish performance or calibrated acoustic accuracy here. Overlapping hourly windows are not independent ocean samples; compare equal eligible source dates, per-horizon errors, coverage, and paired 48-hour blocks.

## Primary-source survey

| Candidate | Source claim and actual task | AEON fit and decision |
|---|---|---|
| CF-JEPA | [Paper](https://arxiv.org/abs/2606.07031) and [author code](https://github.com/WDSLab/CF-JEPA), June 2026: crop-based forward latent prediction at several horizons; authors route forecasting through the EMA encoder and report strong benchmark results. | Best architectural lead for a forward acoustic objective. The present AEON SSL predicts held-out **past** context; extend with future blocks formed strictly within TRAIN. Attribution must say AEON adaptation, not CF-JEPA reproduction. Audit target-context separation and effective rank. |
| VICReg / VISReg | [VICReg](https://arxiv.org/abs/2105.04906) explicitly penalizes low per-dimension variance; [VISReg](https://arxiv.org/abs/2606.02572), June 2026, reports that SIGReg-style sketching can have weak collapse gradients and adds scale control. VISReg's evidence is image SSL. | Motivates a fixed variance-floor ablation for AEON's collapsed EMA representations; do not transplant its image benchmark claim. A single reviewed coefficient and no validation sweep. |
| LeJEPA / SIGReg | [Paper](https://arxiv.org/abs/2511.08544) and [author code](https://github.com/galilai-group/lejepa) propose a shared-encoder SIGReg objective. | Current shared-SIGReg arm is a related acoustic adaptation; its TRAIN effective rank is only 3.5–5.2/128, despite adequate scale. Preserve those diagnostics and compare frozen-random controls before assigning value. |
| Distributed JEPA / HEPA | [Distributed JEPA](https://arxiv.org/abs/2609.17029), posted 15 September 2026, combines masked latent prediction with covariance and temporal-variance penalties for heterogeneous energy-series forecasting; its authors report effective ranks 185–235 and resilience to degraded inputs. [HEPA](https://arxiv.org/abs/2605.11130) uses horizon-conditioned future-latent prediction for event-oriented time series. | These are relevant anti-collapse and future-target design precedents, but neither is a validated marine hourly `Sv_mean` forecaster. AEON's fixed forward-EMA/VICReg adaptation tests the shared principle without claiming reproduction; its reviewed ensemble score is 0.653315 dB versus 0.639062 dB for matched direct neural, so improved latent rank alone did not improve this forecast. |
| Chronos-2 | [Amazon paper](https://arxiv.org/abs/2510.15821), [official repository](https://github.com/amazon-science/chronos-forecasting), and [model card](https://huggingface.co/amazon/chronos-2): 120M encoder-only probabilistic model supporting univariate/multivariate zero-shot forecasting. Its card reports leading public zero-shot benchmark results, not AEON results. | Add one frozen zero-shot arm on exactly the issued VAL rows, 24 past source steps and 1/3/6 horizons, five fixed quantiles. Exact HF revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, Apache-2.0, 119,477,664 F32 parameters, 477,930,472 repository bytes. No future covariates or local tuning. Limit initial inference to two GPU-hours and stay below local memory caps. |
| TimesFM 3.0 / 2.5 | [Google official repository](https://github.com/google-research/timesfm): 3.0 adds native multivariate forecasting and authors report top benchmark ranks; its pretrained weights are noncommercial. 2.5 weights are Apache-2.0, 200M class, but older univariate architecture. | Do not bundle 3.0 weights in a commercial-capable offline release. Chronos-2 is the smaller, permissively licensed multivariate zero-shot baseline. No TimesFM download in this extension. |
| PatchTST / iTransformer / TimeMixer | [PatchTST](https://arxiv.org/abs/2211.14730) patches and channel-independent masked pretraining; [iTransformer](https://arxiv.org/abs/2310.06625) treats variables as tokens; [TimeMixer](https://arxiv.org/abs/2405.14616) mixes multiple temporal scales. Their benchmark claims concern different data scales/tasks. | At only 24 history steps and four channels, a larger transformer is not automatically apt. A compact fixed-lag boosted-tree arm is a more informative local supervised challenger than architecture-size escalation alone. |
| EchoST-SSL | [IEEE Sensors Journal article](https://ieeexplore.ieee.org/abstract/document/11655455/), September 2026, uses spatial/temporal masking and dual-frequency contrast on echosounder pings for label-efficient representation processing. | Acoustic-domain relevance, but not a direct hourly conditioned-Sv forecasting benchmark or transferable pretrained AEON checkpoint. Do not call its classification gains forecast SOTA. |
| Other marine-acoustic learning | A [2024 fisheries-acoustics DINO study](https://doi.org/10.1016/j.ecoinf.2024.102878) learns multifrequency echosounder features for classification/regression. A [2025 marine-soundscape forecast study](https://doi.org/10.1016/j.ecoinf.2025.103189) compares NeuralProphet, N-HiTS and TiDE for passive sound-pressure-level bands over one to seven days. | Neither establishes an hourly future forecast benchmark for this source-conditioned active-echo `Sv_mean` product. Their task/labels cannot supply an AEON forecast checkpoint or biological ground truth. |

## Application boundary and sequence

1. Complete the fixed core, independently rescore its immutable validation predictions with the 18-anchor/day rule, and review the result. Execute the already-proposed frozen pretrain raw-plus-latent hybrids and their controls after distinct prefit review. No altered core checkpoints.
2. In a separately labelled **post-hoc development extension**, compare one fixed enhanced supervised tree (full 24-step past lags/masks and existing 18 summaries; TRAIN-only fit, five quantiles × three horizons) and one pinned, frozen Chronos-2 zero-shot arm to the exact B3/direct/JEPA cohort. These add model choices to development and cannot be presented as preregistered confirmation.
3. Test one fixed forward-EMA JEPA adaptation: context is 24 past intervals; EMA targets are nonoverlapping future 1/3/6 interval blocks drawn only from TRAIN; explicit variance floor targets observed low effective rank. Use the same 128-width encoder and total updates as the core, with a capacity/update-matched direct comparison, three seeds if assigning JEPA incremental value, and diagnostic random/shuffled controls. Freeze data pairing, loss coefficients, budgets, control rules and evaluator through an ADR and distinct prefit review before fitting. If this would exceed the finite resource budget, retain the negative/uncertain core result rather than launching an open search.
4. Final selection still uses only protocol-eligible validation dates. Calibration may only widen fixed 90% intervals. Retrospective TEST acoustic values remain closed until metadata-only candidate universe, final code/model selection, and independent pretest freeze are complete. Report selection-induced optimism, provenance limits and inference compute for every arm.

The 17-slot run's initially printed validation aggregates average some dates below the frozen 18-anchor floor and are `NON_PROTOCOL_DIAGNOSTIC`. Its corrected rescore, not those initial values, governs development comparisons. The source-reported Sv product is not independently validated absolute backscatter or biomass.

## Executed application to AEON

All values below are independently reviewed TRAIN/validation development
results on the same 1,219 issued rows, 50 eligible source dates per horizon and
the corrected ADR 0009 primary score. They are not a sealed or final TEST
result. The fixed 17-slot core campaign produced a direct-neural three-seed
ensemble at 0.639062 dB, EMA-JEPA at 0.642170 dB and shared-SIGReg at
0.648643 dB. Its matched random-feature hybrid scored 0.654412 dB, ahead of
every learned raw-plus-latent hybrid, so core incremental JEPA value is negative
on development evidence.

The fixed post-hoc LightGBM 210-past-feature quantile challenger produced real
predictions and a saved model at 0.638460 dB, only 0.000602 dB better than
direct. The paired 48-hour-block validation interval for the LightGBM-minus-
direct daily loss difference is [-0.009765, 0.008764] dB, crossing zero; this
does not establish a meaningful gain or SOTA. The fixed forward-EMA JEPA
adaptation raised TRAIN target effective rank to 11.93–13.45/128 but its
three-seed ensemble scored 0.653315 dB, worse than direct overall and at all
horizons. The pinned Chronos-2 120M frozen zero-shot comparator ran on all
1,219 validation rows and scored 0.693518 dB. These outcomes show that
architecture claims on other benchmarks did not transfer automatically to
this one-site, 24-step conditioned acoustic target. The separate neutral
five-source comparison was independently reconstructed across all 40
candidates and selected no model. ADR 0011 records the subsequently reviewed
development choices: the direct three-seed ensemble as the core conventional
reference, EMA-JEPA three-seed ensemble as the core JEPA comparator, and
LightGBM as a post-hoc exploratory candidate. The machine-readable component
freeze and CAL/TEST access gates are still under independent review; neither
partition's numeric values have been opened.

A further single supervised architecture trial is being undertaken on the same
TRAIN/validation support under the expanded local compute authorization. It
remains exploratory and cannot be imported into the already reviewed selection
without a new explicit comparison and independent selection review.
