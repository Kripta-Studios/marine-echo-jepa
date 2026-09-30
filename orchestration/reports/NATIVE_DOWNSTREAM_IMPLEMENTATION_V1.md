# Native downstream implementation V1

Implemented separate frozen-readout, full-finetune, and random direct endpoints. This is a software correctness handoff, not a downstream experiment result or an independent approval. All executed optimization used CPU fixtures explicitly labelled `SYNTHETIC_CORRECTNESS_ONLY`; no real TRAIN/development payload, final test, or GPU was opened/executed in this lane.

## Authored scope and integration

Only these new paths were authored for this task:

- `src/marine_echo/training/native_downstream.py`
- `tests/integration/test_native_downstream.py`
- `orchestration/reports/NATIVE_DOWNSTREAM_IMPLEMENTATION_V1.md`
- `evidence/ssl-downstream-builder-v1/**` (executed check logs, source inspection, and a main-helper CLI bootstrap)

The tests import the actual main `native_ssl`, `native_temporal`, `sigreg`, and `native_resources` modules. They add only the builder training directory to the already imported main package's search path. No main files, frozen builder files, contracts, configuration, STATUS, ledger, lockfiles, junctions, historical evidence, or release files were changed. The current amendments were read, including the final four-opportunity core-screen correction; core screen probes are not counted as downstream endpoints.

Commit: **NOT_RUN**. The earlier Git index operation returned exit128: `Unable to create .../marine-echo-jepa/.git/worktrees/marine-jepa-vnext-builder/index.lock: Permission denied`. That exact denied operation was not retried. Parent integration must copy these new bytes and record its own commit. Untracked files from the preceding SSL task remain outside this task's authored scope.

## Endpoints and reproducibility

`DownstreamConfig` is immutable. Real endpoints fix seed7/13/23, history24/96, batch64, LR0.0003, weight decay0.0001, clipping1, native daily floor18, patience4, helper warmup10% and cosine floor0.1. Frozen readout has2000 updates/cadence500; full finetune and direct have3000/cadence750. A hard5000-update per-trajectory cap also applies to correctness fixtures. Only explicit CPU `correctness_smoke` admits tiny architectures/budgets; the synthetic review must additionally bind its exact `correctness_core_config`. The public CLI exposes no synthetic shortcut.

Pretrained modes require `selected_encoder.pt`, completed parent `run.json`, parent `inference.pt`, exact membership, parent configuration and original independent prefit review. Method/history must match the selected ancestor. The full parent inference encoder tensors, config, scalers and bindings must agree with the selected artifact. Parent identities are bound by hashes, including the original review and membership. Declared TRAIN source/deployment membership is checked before numerical access; exact row/eligibility membership and train-statistic equality are checked after admitted numeric loading and before optimizer construction. TRAIN statistics are recomputed solely to verify the saved scaler lineage, and saved ancestor scalers are reused unchanged. Development never fits scalers. Direct accepts no ancestor and fits scalers on verified TRAIN only.

Every head is freshly initialized through the core seed+100000 initializer; pretraining RNG/head state is discarded. TRAIN sampling uses core `batch_indices(pool,batch,seed,"readout",step)` independently of mode/method and ancestor RNG. All native controls have bit-identical common initialization for each prescribed seed. CF uses the selected EMA encoder and128-dimensional real readout; it has a different head shape from the64-dimensional shared family and is not claimed to have identical tensors across those shapes.

Frozen readouts use eval/no-grad encoding and optimize the head only. Encoder parameters and buffers are compared against their initial selected state after every update and when accepting resume. Full finetuning enables the selected encoder and fresh head; direct enables the same shared encoder and fresh head from random initialization. Unused SSL predictors/regularizers and CF online branches remain frozen. CF full finetuning treats the previously selected EMA features as a supervised encoder; it does not update EMA or use the author's pretraining objective. This is a downstream adaptation, not full paper reproduction or SSL-only training.

Development selects the earliest checkpoint with a strictly improved native daily pinball, evaluated only at the fixed cadence. Scores come from the unchanged core evaluator: source-date mean, horizon mean, then deployment mean, with native floor18 and complete deployment/horizon coverage. Ties retain the earlier checkpoint. Patience counts non-improving evaluation checks, not optimizer steps. Each real endpoint has four planned checks; no checkpoint beyond its fixed update budget is added. Insufficient selection support fails rather than inventing a row-level fallback metric. No development loss is backpropagated.

## Prefit gate and artifact contract

Before numerical NPZ access or weights loading, require a distinct `APPROVED_DOWNSTREAM_PREFIT` review identifying implementation session `01a0ef20-7397-7ba1-a98c-f59bc38ddcc1`, explicitly allowing mode/method. Required `bindings` map resolved absolute paths to exact SHA256. `required_paths(inputs, core_config)` enumerates this module, all imported project helper sources/package initializers, pinned CF source/license when applicable, TRAIN/development NPZ and cohort JSON, split, ADR0016, separately frozen downstream protocol/config, and all selected parent artifacts/config/review. Empty, missing, stale bindings, rejected/self review, wrong cohort roles, source role overlap, incorrect ancestry, mismatched runtime config, and unsupported dimensions fail closed. Review also requires explicit `train_npz_sha256`, `dev_npz_sha256`, `split_sha256`. The protocol/config identities are mandatory bindings; there is no empty-map acceptance. The reader rejects a test-role scalar before reading numerical members.

The output directory must be new. Existing runs require explicit same-directory `latest.pt` resume; completed `run.json` outputs are protected. Resume into a new output directory is also supported for a forked continuation of the exact same identities. Source/data/config/review/ancestor hashes, device kind, scalers, scheduled selection history, sampler counters/sequences and frozen selected state must match. Reads use `torch.load(weights_only=True,map_location="cpu")`. Resume checkpoints contain model (including head/buffers), optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, complete selection/best model/counters, scalers, identities, and cumulative elapsed/resource peaks. Cadence checkpoints and `latest.pt` are complete; `complete.pt` preserves the final optimizer state before restoring the selected model.

Successful outputs:

- `inference.pt`: existing core `native_ssl_weights_only_inference_v1` schema plus downstream configuration, supervised ancestry, evidence kind and review identity. Core `predict_from_checkpoint` and downstream `load_inference().forecast` replay the same predictions. `load_inference().encode` accepts context/masks/metadata only; `forecast` adds query only. Neither API accepts assessment targets/future crops.
- `selected_encoder.pt`: frozen mode emits a byte-identical copy of its selected parent artifact; the input is untouched. Full/direct emit `native_downstream_supervised_encoder_v1` with explicit supervised ancestry and `ssl_only=False`.
- `predictions.npz`: development predictions[N,3,5], targets/masks/dates, row/deployment/archive identities, cutoff, metadata/query, context observation mask, shared assessment support, native query bounds/frequency/interval, quantiles and horizons. Sorted quantiles are monotonic. The artifact supports independent metric reconstruction without model execution.
- `metrics.json`: native daily pinball, daily/deployment/horizon detail and coverage.
- `membership.json`: exact TRAIN identities, supervised pool, all sample sequences and deterministic sequence hash.
- `run.json`: complete bindings, review/ancestor identity, selected step, actual supervised updates, sample and label presentations, total/encoder/head/optimized parameters, and separately measured additional downstream elapsed/GPU time/memory. Parent screening compute is not relabelled as downstream work.

Resources reuse the main lock/ownership checks, <10GiB allocated/reserved CUDA and RSS<22GiB. All CPU checks reported zero GPU elapsed and never initialized CUDA. Additional elapsed time covers the optimizer/resource-owned stage including evaluation/checkpoint output; scaler/data preparation occurs before that interval. CUDA ownership/limits and durable artifact output still require parent execution.

## Executed verification

All commands used `tools.exec_command(shell="cmd.exe",login=false)` under unchanged workspace-write/never permissions and the existing environment. Known PowerShell launcher failure is distinct from a denied filesystem operation. No denied operation was rerun through another mechanism. Python was invoked with `-B`, pytest with `-s -p no:cacheprovider`, Ruff with `--no-cache` for checks.

Red command (exit2, preserved `red-tests.txt`):

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q -s -p no:cacheprovider tests\integration\test_native_downstream.py
```

It failed at collection because the requested new module did not exist. Meaningful optimizer failure evidence is also preserved: `green-attempt-1.txt` (exit1: three failures due to incorrect QueryHead API) and `green-attempt-2.txt` (exit1: three failures due to incorrect evaluator key). Both were corrected against the actual read-only core interfaces. `green-attempt-3.txt` passed6 checks. Expanded tests passed27, then33 checks before the final additions.

Final focused command, same command as above (exit0): **37 passed in11.99s**, preserved `focused-pytest-final.txt`. Coverage includes:

- immutable frozen weights/buffers and nonzero head gradients; full/direct weight changes and encoder gradients;
- CF EMA/online separation, BatchNorm buffer immutability, exact CF frozen resume/replay;
- all native control initializers and independent seed7/13/23 batches;
- safe full CPU checkpoint continuation versus uninterrupted execution, including exact model/optimizer/scheduler/RNG state, sample/counter validation and frozen best-state tamper rejection;
- every required binding missing, stale protocol/config/selected ancestor, rejected/self/method/mode review, wrong cohort role/runtime config, corrupt parent identity/config/review, bogus TRAIN row/deployment/scalers/features, and test-role NPZ rejection before numerical access;
- sorted context-only inference and existing core replay, native query bounds, accurate update/label/sample/parameter/elapsed counts;
- independent pinball reconstruction from saved predictions, masked-target exclusion, unequal date/horizon/deployment weighting and insufficient floor support;
- earliest selection on ties and patience counting, existing-output protection and cross-device resume rejection.

Fixtures use real Torch weights-only and NumPy codecs over an in-memory file transport. **No durable synthetic model/prediction artifacts are claimed.** This is a separate correctness harness, not a retry of an earlier denied durable fixture write. Parent must execute durable synthetic checks under its own permissions. No optimizer cache denial occurred in these CPU checks.

Final lint/format commands, each exit0:

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff check --no-cache src\marine_echo\training\native_downstream.py tests\integration\test_native_downstream.py evidence\ssl-downstream-builder-v1\verify_sources.py
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff format --check src\marine_echo\training\native_downstream.py tests\integration\test_native_downstream.py evidence\ssl-downstream-builder-v1\verify_sources.py
```

Source/parameter inspection and CLI help, each exit0:

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B evidence\ssl-downstream-builder-v1\verify_sources.py
..\marine-echo-jepa\.venv\Scripts\python.exe -B evidence\ssl-downstream-builder-v1\verify_sources.py --runner --help
```

Their inspected logs are `source-inspection-final.json` and `cli-help.txt`. Python3.12.13, PyTorch2.11.0+cu128, CUDA initialized=false. CPU initialization only measured shared total2,457,621 / encoder2,424,448 / head5,189; CF total893,701 / EMA encoder412,800 / head18,565. Full optimized parameter counts are2,429,637 shared and431,365 CF; frozen optimized counts equal the corresponding head counts. Total CF includes its unused online encoder/predictors, which downstream training does not optimize.

Exact source hashes, including all main helpers and CF pins, are in `source-inspection-final.json`. Main core at validation: native_ssl `a5d4be8301169da6ec6d4693cb085395f1fcc94c10b0c13cf9c753975c741744`; native_temporal `85c4af6c67acb661359cde8bcf22de479e4da6a499cc7843bce0fee6c155e422`; sigreg `2cf5dc8e6fc610fbe167339a8750f5e2d9ddf9a48f416717b8310500536469a3`; native_resources `c9ca3fbd0759ae00824774ed61a08185361234430d5ed63f09fa3885545c7c38`. Parent must issue a new exact review after integration because resolved source paths change. A different core/helper hash requires new review and validation.

## Public invocation after parent integration

Run in main using its source first on PYTHONPATH and separately frozen, reviewed downstream JSON:

```text
.venv\Scripts\python.exe -m marine_echo.training.native_downstream --train TRAIN.npz --dev DEVELOPMENT.npz --train-cohort TRAIN.json --dev-cohort DEVELOPMENT.json --split SPLIT.json --adr0016 ADR0016.md --protocol DOWNSTREAM_PROTOCOL.md --config DOWNSTREAM_CONFIG.json --review DOWNSTREAM_REVIEW.json --encoder PARENT/selected_encoder.pt --ancestor-review PARENT_REVIEW.json --ancestor-config ORIGINAL_CORE_CONFIG.json --output NEW_DOWNSTREAM_RUN --device cuda
```

This is an unexecuted root-owned real-job command. `DOWNSTREAM_CONFIG.json` contains all fields from `DownstreamConfig(...).to_dict()` and is immutable after independent review. `--ancestor-config` points to the exact original core configuration bound in the parent review; omission defaults to `PARENT/config.json`. For direct use mode `direct_end_to_end`, method `direct`, and omit all three ancestor flags. Resume adds `--resume RUN/latest.pt` with identical approved inputs/config/device kind. Before integration, `verify_sources.py --runner` exposes the same CLI while importing main helpers; it does not execute any fitting merely by being imported or passed `--help`.

Remaining gates: parent byte integration/commit, durable CPU artifact tests, independent downstream protocol/config/source/ancestor review, resource-owned real TRAIN/development jobs, and scientific assessment. All are **NOT_RUN in this implementation lane**. No downstream scientific result or self-approval is asserted.
