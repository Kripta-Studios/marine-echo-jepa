# AEON checkpoint and objective diagnostic at 991368d

Task: bounded saved-state diagnostic, 2026-09-29. Branch: `diag/checkpoint-objective-991368d`; base: `991368d6a64c7189d803828a15a39828a00e8ea0`. This is technical diagnostic evidence on already exposed VAL. It neither selects a historical checkpoint nor changes a scientific conclusion or release.

The owner authorized only this session under GPT-6.1 Sol high and prohibited subagents. No subagents were spawned; repository model pins were not used. The runtime does not independently expose a verifiable session model identifier. Owner scope overrides the historical start prompt, task graph and role prerequisites. AGENTS, start/task/source/status documents, SOL_BUILDER, scope/DoD, protocol and AEON ADRs were read. The historical task graph contains earlier campaign statuses and is not a new execution authorization.

Only the diagnostic tool, its tests, this technical report and the assigned evidence directory are authored. Root owns STATUS, its run ledger, final scientific report and adjacent integration. Historical source, checkpoints, predictions, protocol, reviews, ledgers and releases remain unchanged. No training CLI, optimizer step, new experiment, nine-trajectory campaign, unchanged-route extension, CAL/TEST numeric access, network access, installation, paid work or release build occurred.

## Execution and verification

The main existing `../marine-echo-jepa/.venv/Scripts/python.exe` is CPython 3.12.13 with PyTorch 2.11.0+cu128, CUDA build 12.8 and CUDA available. The documented alternative `../EVOCON_JEPA_Codex_Handoff/e-jepa-ttc/.venv/Scripts/python.exe` also imported its existing CUDA runtime; it was not needed. All numerical diagnostics ran on CPU, with one executing model diagnostic at a time. The supplied audit ran serially in a child process while its coordinator waited.

| Actual command | Exit | Result |
|---|---:|---|
| `python tools/verify_handoff.py` | 1 | Known original-handoff `.editorconfig` hash mismatch, retained in `handoff.log` |
| `python tools/runtime_preflight.py --output evidence/aeon-checkpoint-diagnostic-20260929/preflight.json` | 0 | Runtime metadata; `--torch` omitted because that smoke performs an optimizer step |
| Focused new-test collection before implementation | 2 | Expected missing-module RED; `red.log` |
| Initial focused model/evaluator suite | 0 | 27 passed; `green-initial.log` |
| Supplied ZIP script against verified canonical current source | 0 | 18 synthetic CPU gradient probes and 3 paired initialization checks |
| `python tools/aeon_checkpoint_diagnostic.py --output evidence/aeon-checkpoint-diagnostic-20260929/run-01` | 1 | Completed nine VAL replays, initialization and saved-state probes; then 30k trajectory field-name failure |
| `pytest ... -k 30k` before trajectory repair | 1 | Real regression reproduces `KeyError`; `red-30k.log` |
| Repaired focused model/evaluator suite | 0 | 31 passed; `green-final.log` |
| `python tools/aeon_checkpoint_diagnostic.py --output evidence/aeon-checkpoint-diagnostic-20260929/run-02 --reuse-saved-diagnostics evidence/aeon-checkpoint-diagnostic-20260929/run-01` | 0 | Completed saved-trajectory/error/sampling work; reused completed forward/backward evidence without rerunning those probes |

Final post-change validation again passed 31 tests in 58.02 seconds, exit 0. Ruff lint and format check each exited 0. Independent artifact verification checked 36 run-02 output hashes and all 133 consumed input hashes, with no mismatch. `command-results.json` records actual statuses and environment.

The focused suite is `tests/unit/test_aeon_checkpoint_diagnostic.py tests/unit/test_aeon_ssl.py tests/unit/test_aeon_evaluation.py tests/integration/test_aeon_rescore.py`, invoked with `-q -p no:cacheprovider`. New tests cover exact temporal mapping, teacher/head gradient separation, model/input/buffer isolation, fixed endpoint absence/corruption/ambiguity/path handling, exact three-seed ensembles, deterministic TRAIN-only selection, historical input mutation detection, corrected 18-anchor support, prior-year lineage and source-bound shuffled sampling/RNG reset. The final validation logs and command-result records are in the evidence root. Optimizer training smoke, API/browser/offline/release tests are NOT_RUN for this diagnostic scope; the reviewed release is preserved.

Run 02 took 126.414 seconds after imports; its process peak was 0.794 GiB and GPU allocation was zero. Run 01 process peak was NOT_RECORDED because its final summary was not reached; no replacement memory claim is made. `e9a2c5749ddc59d23dcbd3644d5968c535029d72df3711bcd1cfe1bfa4182a07` is the exact tool SHA-256 used for run 02. An exact run-02 runtime source copy is retained as `run-02-tool-source.py`. The committed tool uses LF per existing repository attributes; its SHA-256 is `0bca4f4c1558a167209cce17db16df4cd820062ebdac938340b9811bacc3c251`. `source-byte-normalization.json` proves the only difference is CRLF -> LF and the parsed Python AST is identical. The run-01 source snapshot was reconstructed immediately after failure by removing the sole subsequently added accessor; it is labelled recovered source, not a source hash captured at process start.

## Predetermined endpoint replay

Original direct supervised-1500 and supervised-3000, and original EMA pretrain-1500 plus supervised-1500, were recovered for seeds 7,13,23. Campaign manifest -> slot SHA-256 -> phase/step checkpoint index -> checkpoint bytes were checked. Deserialized family, seed, phase, step and source identity were checked; expanded states additionally bind cohort/config and both archives. EMA pretrain lineage was verified for each original supervised component. There were no missing requested checkpoints or endpoint prediction artifacts and no substitutions.

All 1,219 VAL row identities/truth/support were validated against the reconstructed original cohort. Each horizon has 1,217 observed rows; corrected eligibility retains 1,194 / 1,192 / 1,189 rows and 50 source dates. Dates are source-reported with unspecified timezone. The current corrected evaluator uses the frozen five quantiles, three horizons, at least 18 scored anchors per source date, then equal date/horizon aggregation.

| Fixed endpoint | Seed | Corrected primary pinball, dB |
|---|---:|---:|
| direct_supervised1500 | 7 | 0.645004383 |
| direct_supervised1500 | 13 | 0.640577332 |
| direct_supervised1500 | 23 | 0.643001680 |
| direct_supervised3000 | 7 | 0.651154148 |
| direct_supervised3000 | 13 | 0.647964425 |
| direct_supervised3000 | 23 | 0.644896702 |
| ema_jepa_supervised1500 | 7 | 0.648804633 |
| ema_jepa_supervised1500 | 13 | 0.647988986 |
| ema_jepa_supervised1500 | 23 | 0.647977879 |
| direct_supervised1500 | equal 7/13/23 prediction mean | 0.639577016 |
| direct_supervised3000 | equal 7/13/23 prediction mean | 0.639061735 |
| ema_jepa_supervised1500 | equal 7/13/23 prediction mean | 0.642170456 |

CPU versus saved CUDA prediction fidelity is not bit exact: max absolute difference across the nine components is 1.52587890625e-05 dB; per-component RMS differences are 2.15289713029e-06 to 2.4651625167e-06 dB. New predictions are saved under `run-02/replays/`; original bytes are retained. Individual results and equal-three-seed prediction ensembles remain separate.

Every individual direct seed scores better at 1500 than 3000 supervised steps; however, the direct 3000 ensemble slightly outperforms the direct 1500 ensemble. Direct 1500 also scores better than EMA 1500 per seed and as an ensemble on this exposed VAL. These predetermined comparisons are post-hoc diagnostics, not prospective selection. Direct 1500 is not equal total-update budget to EMA 1500 SSL + 1500 supervised. Historical selected endpoints remain unchanged.

## Saved trajectories and fixed TRAIN errors

All 54 scheduled saved validation checks for original direct/EMA, 30k direct/EMA and expanded direct/EMA were hash-validated and rescored. No historical minimum was selected. Full per-date/quantile/horizon scores are in `saved-trajectories.json`. Saved per-step TRAIN/SSL loss curves are NOT_SAVED in the historical slot/checkpoint contracts; this tool does not reconstruct optimizer losses from forecast errors.

Each fixed TRAIN forecast diagnostic uses exactly 512 chronological evenly spaced rows from that campaign TRAIN, at the predetermined endpoints below. These sparse samples do not satisfy a full daily support census, so the TRAIN column is explicitly row-weighted descriptive pinball, not the corrected VAL primary. Twenty saved endpoints were inferred without fitting; all selected indices/row IDs/source memberships and hashes are retained in `fixed-train-forecast-errors.json`.

| Campaign/run | Fixed supervised steps | TRAIN sample pinball first -> last | Corrected saved VAL first -> last |
|---|---|---:|---:|
| original / direct_seed7 | 1500 -> 3000 | 0.536609 -> 0.516958 | 0.645004 -> 0.651154 |
| original / direct_seed13 | 1500 -> 3000 | 0.534283 -> 0.513234 | 0.640577 -> 0.647964 |
| original / direct_seed23 | 1500 -> 3000 | 0.534971 -> 0.526655 | 0.643002 -> 0.644897 |
| original / ema_jepa_seed7 | 500 -> 1500 | 0.564171 -> 0.539489 | 0.656115 -> 0.648805 |
| original / ema_jepa_seed13 | 500 -> 1500 | 0.565749 -> 0.534395 | 0.662371 -> 0.647989 |
| original / ema_jepa_seed23 | 500 -> 1500 | 0.567054 -> 0.536646 | 0.667455 -> 0.647978 |
| 30k / direct_seed7 | 2500 -> 30000 | 0.523975 -> 0.179783 | 0.650889 -> 1.178379 |
| 30k / ema_jepa_seed7 | 2500 -> 15000 | 0.525353 -> 0.354225 | 0.664914 -> 0.913424 |
| expanded / direct_seed7 | 500 -> 3000 | 0.540034 -> 0.512141 | 0.654451 -> 0.640615 |
| expanded / ema_jepa_seed7 | 500 -> 1500 | 0.548847 -> 0.530460 | 0.663361 -> 0.654855 |

The large 30k TRAIN improvement alongside worsening VAL is overfitting-consistent evidence. Original direct seeds show a smaller version across 1500/3000; ensemble behavior prevents a simplistic endpoint conclusion. The record does not isolate optimization drift, calendar/serial distribution shift, learned sensitivity to source context, or objective/view mismatch. Missing optimization loss curves prevent claiming a particular optimizer trajectory. Expanded direct improves both fixed TRAIN error and final VAL relative to its early scheduled endpoint; expanded EMA is a mixed development observation. None of these observations rescues or rewrites a scientific claim.

## ZIP source audit, initialization and gradient support

All five supplied SHA256SUMS entries were verified before script execution. Extraction was restricted to new evidence, with duplicate/size/path checks. Both current model source blobs match the supplied Git identities: `aeon_ssl.py` = `7bd8a94cde303d968df1e541af47b50ce969d342`; `sigreg.py` = `ef6f77225d03c32daa8d6509401e998658f2b198`. Canonical copies from the local Git object database let the supplied exact-byte checker run without editing the historical source. Checksums establish identity; the entire script was inspected independently before execution.

Independent current-source checks also verify every encoder tensor and both forecast-head tensors for seeds 7,13,23: direct and SSL encoder tensors match; forecast-head weight and bias tensors do not. This holds for EMA and shared-SIGReg. The SSL predictor consumes RNG before its head, so a common seed does not make all initial weights paired. Full equality maps and tensor hashes are in `initialization-equality.json`.

Temporal input mapping is 24 interleaved blocks of 8 columns: four observed values followed by four masks at each position. EMA online context zeroes both values and masks at positions 18..23; its detached teacher target uses suffix positions 18..23, with prefix zeroed. The source class docstring has a historical all-24 teacher sentence, but executed `teacher_view` is suffix-only; this diagnostic follows the executable source. Shared mode uses the same online encoder for its suffix target with target gradients retained.

On the 18 initialized real-TRAIN probes (3 seeds x 3 modes x 2 batches), EMA suffix first-layer data gradients are zero, while shared and direct have suffix support. On all eight saved EMA-state batch probes, all 6,144 suffix weights have zero total data gradient; teacher and head gradient counts are zero. On all four saved shared-state batch probes, all 6,144 suffix weights have nonzero total gradient, with no SSL head gradient. Every position has explicit value/mask norms and support counts for prediction, raw regularizer, weighted regularizer and total loss. Input-value temporal support is also recorded. Teacher parameters remain frozen and absent from gradient contributions.

This is data-gradient evidence, not proof that suffix weights never change. Historical weight decay, subsequent supervised updates and other shared layers can alter weights or outputs. No optimizer or teacher update was called by this diagnostic. All regularizer operations occurred on disposable deep copies, including its incrementing global_step buffer; input values/masks, deserialized state dictionaries and original model state hashes were checked unchanged.

## Fixed saved-state objective diagnostic

Original EMA and shared seed7 SSL500/SSL1500, 30k EMA SSL15000 and expanded EMA SSL1500 were all available. Each got exactly two deterministic TRAIN batches of 64. Indices are `linspace(0,n-1,128,dtype=int64)`, first 64 then last 64. Original batches use only original TRAIN. Expanded batch 0 is 64 prior-year TRAIN windows; batch 1 is 17 prior-year and 47 current TRAIN windows. That mixed diagnostic batch is deliberately labelled and differs from the historical source-homogeneous SSL sampler. Every window itself stays within its source. Batch member lists, source hashes, timestamps, row IDs and value/mask/target hashes are in `fixed-train-memberships.json`.

Per-feature means/std, RMS, centered RMS/covariance trace, entropy rank, centered prediction error and loss/gradient decomposition are saved separately for teacher suffix target, online short context, online full context and predictor. In shared mode the field called teacher suffix target denotes the shared online suffix encoder, not an independent EMA teacher. Centered effective rank is `exp(-sum(p_i log p_i))`, with `p_i=lambda_i/sum(lambda)` for eigenvalues of column-centered covariance; zero is reported only for trace <=1e-12. With 64 rows the centered rank ceiling is 63. Rank is not mean magnitude, Gaussianity or forecasting quality.

| Saved state / batch | Teacher suffix RMS | Online full RMS | Predictor RMS | Teacher centered rank | Online full centered rank | Prediction SmoothL1 | Weighted reg. loss | Weighted reg. / prediction gradient L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original ema_jepa_seed7 SSL500 / 0 | 0.035302 | 1.143027 | 0.751342 | 5.224 | 1.564 | 0.129001 | 0.436155 | 2.249 |
| original ema_jepa_seed7 SSL500 / 1 | 0.032817 | 0.985930 | 0.744286 | 6.642 | 1.720 | 0.114130 | 0.353414 | 2.874 |
| original ema_jepa_seed7 SSL1500 / 0 | 0.047461 | 2.209931 | 1.020251 | 3.604 | 1.817 | 0.086104 | 0.266362 | 2.666 |
| original ema_jepa_seed7 SSL1500 / 1 | 0.041459 | 2.002241 | 1.181257 | 4.972 | 1.933 | 0.100452 | 0.286195 | 4.236 |
| original shared_sigreg_seed7 SSL500 / 0 | 1.051055 | 0.668682 | 0.658981 | 2.053 | 2.200 | 0.175873 | 0.428535 | 1.656 |
| original shared_sigreg_seed7 SSL500 / 1 | 0.691740 | 0.434193 | 0.480806 | 1.547 | 1.514 | 0.089540 | 0.576836 | 1.040 |
| original shared_sigreg_seed7 SSL1500 / 0 | 1.346081 | 1.086630 | 0.982957 | 2.849 | 3.123 | 0.153239 | 0.378381 | 2.353 |
| original shared_sigreg_seed7 SSL1500 / 1 | 0.840847 | 0.481613 | 0.530670 | 2.746 | 3.132 | 0.103346 | 0.507796 | 2.053 |
| 30k ema_jepa_seed7 SSL15000 / 0 | 0.072417 | 2.168272 | 1.406050 | 3.630 | 3.617 | 0.120475 | 0.198415 | 4.120 |
| 30k ema_jepa_seed7 SSL15000 / 1 | 0.060342 | 0.938578 | 0.851057 | 4.046 | 4.573 | 0.072857 | 0.245391 | 4.156 |
| expanded ema_jepa_seed7 SSL1500 / 0 | 0.036608 | 0.865697 | 0.989310 | 4.028 | 1.930 | 0.078159 | 0.265744 | 4.315 |
| expanded ema_jepa_seed7 SSL1500 / 1 | 0.037231 | 0.685782 | 1.119302 | 4.978 | 3.313 | 0.085161 | 0.252003 | 4.123 |

For original EMA SSL1500, suffix target centered RMS is 0.00805..0.01242 while predictor RMS is 1.020..1.181 and centered prediction MSE is 0.994..1.247. Weighted regularizer gradient norms exceed predictive norms by 2.67..4.24 on these batches. All saved EMA probes show ratios above 2.2, and 30k/expanded endpoints exceed 4.1. Cosines vary in sign, so components can oppose or reinforce each other. Raw losses and per-module gradients are in the JSON. These observations support an objective/view/scale mismatch concern; they do not demonstrate a causal explanation for supervised error or prove useful JEPA representations. Shared suffix scale is materially larger but still has low centered rank on these small serially dependent batches.

## Sampling and expanded exposure

The actual source composite SHA-256 exactly matches the original campaign manifest: `8054f0c95962bee496bda75559583bb0dd189324d1636a9e7667833ee51a4e83`. Therefore the requested rejection/duplicate/RNG-spill narrative is not supported for this indexed campaign. The verified source samples aligned contexts without replacement; shuffled contexts come from interval-ID residues modulo 24, also without replacement, with a random nonzero batch roll. Both reconstructed streams have 64 unique rows in every batch, zero within-batch duplicates, and visit all 4,965 contexts across 1500 batches. Shuffled matched pairs have at least 24 interval separation. Duplicate visits across batches are expected repeated exposure, not within-batch duplicates.

Shuffled pretraining consumes extra NumPy draws and changes the context sampling distribution. However, the source explicitly resets supervised sampling to `SeedSequence([seed,2])`, so aligned/shuffled supervised sequence hashes are identical (`7f6d023e5d53552a1bf9a9fa761bdcba532c165f4f3da92062253c8c7d79230f`). Hypothetical continuing-RNG hashes differ and are labelled hypothetical. Context/pair sequence hashes, uniqueness and separation are retained in `sampling-audit.json`. This is a reconstructed schedule, not a recorded historical per-batch sample log; it does not claim a pairing-only intervention.

Expanded cohort reconstruction exactly matched the saved cohort digest, row identities, raw support and actual counts: 8,507 prior-year plus 4,965 current TRAIN windows = 13,472, with zero rejected prior metadata candidates. Original-source rejected candidate counts remain NOT_AVAILABLE_FROM_ORIGINAL_READER_NO_COMPARABLE_CANDIDATE_UNIVERSE. Original input/target scaler is `[-82.9586895, 6.4819896, -87.6822701, 4.3069681]`; joint TRAIN scaler is `[-83.0875451, 5.8354158, -87.9052851, 4.1910403]`.

Independent index-only reconstruction matches all saved source draw counts: expanded direct supervised prior/current = 121,177/70,823; EMA SSL = 61,952/34,048 (968/532 source-homogeneous batches), EMA supervised = 60,799/35,201. Supervised sampling is pooled without replacement within batches. Nominal direct presentations per window decrease 38.6707 -> 14.2518; EMA each phase decreases 19.3353 -> 7.1259. These are presentation averages, not independent epochs or effective sample sizes. Volume, calendar/deployment identity, joint scaling, homogeneous SSL batches and per-window exposure change together, so the expanded result cannot be attributed solely to volume.

## Preservation, missing evidence and interpretation limits

All requested saved endpoints and representative SSL states were recovered. Missing evidence is the original per-step TRAIN/SSL optimization loss curves, a comparable original-source rejected-candidate universe, historical per-batch sample logs, and run-01 process peak. CPU/CUDA forecasts have tiny measured differences rather than bit-exact fidelity. Two batches and sparse TRAIN samples limit representation/error interpretation; no significance test or historical-best selection was performed.

Prior-year data is TRAIN for expanded models and every affected descendant. It cannot be their external evaluation. Original frozen-model transfer remains only its historical descriptive result. CAL/TEST numeric partitions were not accessed, and already exposed outcomes were not used to choose a new experiment, rescue a scientific claim or rewrite earlier results. No causal biological, biomass/species/catch/fuel or Marine production claim follows.

`historical-input-hashes.json` verifies 133 consumed inputs, including reused evidence, unchanged after the finishing process. Root owns the wider 383-file snapshot. Reviewed r3 SHA-256 remains `5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57`. The local evidence `.gitattributes` preserves exact bytes across Git checkouts. Every run-02 output has a SHA-256 in `output-hashes.json`; the outer evidence manifest also binds logs, exits, tests and this report. Run-01 failure and RED evidence are preserved. Independent review is not claimed by this single session. Software/experiment/JEPA-value/business gates and the historical release remain separate and unchanged.

Primary evidence: `evidence/aeon-checkpoint-diagnostic-20260929/run-02/`. Full command logs/exits and final test/lint evidence: `evidence/aeon-checkpoint-diagnostic-20260929/`. Root should integrate the committed allowed files after its own review; STATUS, its ledger and final scientific report were not edited here.
