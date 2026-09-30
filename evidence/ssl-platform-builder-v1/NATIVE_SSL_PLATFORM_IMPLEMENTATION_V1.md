# Native SSL platform repairs V1

Two bounded patches are prepared and CPU-tested. Main and historical evidence were not changed by this lane. No real corpus, GPU job, installation, credential, junction, permission change, additional agent, Git index write/retry, scientific experiment or self-approval occurred. The implementation session remains `01a0ef20-7397-7ba1-a98c-f59bc38ddcc1` under workspace-write/never.

## Files and exact integration identities

Before editing the sole permitted existing builder file, `inspect_sources.py` verified that both builder and main `src/marine_echo/models/native_temporal.py` matched the owner's exact SHA256 `85c4af6c67acb661359cde8bcf22de479e4da6a499cc7843bce0fee6c155e422`. The result is preserved in `main-before.json`. Only that builder source was modified. Main still held its before bytes at the earlier `main-after-read-only-verification.json` check. At the final handoff check, both main sources had changed externally to the exact proposed after hashes; this is recorded in `main-current-after-drift.json` and `delivery-identities-observed.json`. This lane never wrote main. Do not reapply the original forward patches to those already matching after bytes.

| Target in main | Main before SHA256 | Proposed after SHA256 |
| --- | --- | --- |
| `src/marine_echo/models/native_temporal.py` | `85c4af6c67acb661359cde8bcf22de479e4da6a499cc7843bce0fee6c155e422` | `b7d6b9de5221643871c0f21dbfa440a2c6763600ff3dff5818dfb8918e661a32` |
| `src/marine_echo/training/native_references.py` | `b692368053501ca946978a3c583d649356584e91c1e020597397e669d8aa1d7b` | `0d6583644ab1fce8d87a4194048f2176237b47deb57f0fa8494df6aa9012d584` |

Root-owned integration files are `native_temporal.patch` and `native_references.patch` in this directory. `patch-manifest.json` records their exact hashes and status `PATCHES_PREPARED_NOT_APPLIED_NOT_APPROVED`. Complete proposed source bytes are `native_temporal.after.py` and `native_references.after.py`. The former is an exact copy of the repaired builder model; the latter was authored only here, never over the existing builder/main reference source. `package_patches.py` generated unified diffs against hash-checked main bytes and asserted that restoring the three pooling calls/removing the one new utility restores the entire original model AST. Both patches passed read-only `git apply --check` against main, exit0. No patch was applied to main.

New bounded tests are `tests/unit/test_native_cf_pooling.py` and `tests/unit/test_native_booster_serialization.py`. All other task-authored files are inside `evidence/ssl-platform-builder-v1/**`. Existing runner, downstream module, reader, evaluator, contracts, configurations, lockfiles, STATUS, ledger, app/release and historical artifacts were untouched.

## CF pooling repair

Added `deterministic_adaptive_avg_pool1d(values, output_size)`. For length L and output O, bin i uses the exact integer boundaries `i*L//O` and `((i+1)*L+O-1)//O`, corresponding to floor/ceil. It computes ordinary slice means and stacks them on the last axis. This preserves output shape, overlapping bins and upsampling with O>L. The formulas agree with PyTorch's [adaptive pooling boundary implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/AdaptivePooling.h).

All three CF sites now call this utility: prediction resizing, eight-bin variance/covariance pooling, and multi-scale invariance pooling. No crop distribution, RNG call, target detachment, author coefficient, EMA behavior, shared encoder architecture, source pin or mathematical loss term was altered. The AST assertion covers every original model definition. The graph contains MeanBackward/SliceBackward/StackBackward and no pooling backward nodes. No deterministic setting is disabled or changed to warning-only. Native and slice reductions can differ in floating-point accumulation order, so equality to native kernels is numerical rather than bitwise; deterministic replay of the repaired path is bitwise on CPU.

Executed CPU parity covers float32/float64, lengths1/3/5/7/9/11/17/31, output sizes smaller/equal/larger than input, batched/unbatched and noncontiguous inputs, overlapping bins, arbitrary weighted gradients and float64 gradcheck. Utility tolerances are2e-6 for float32 and2e-14 for float64. Whole CF loss/model-gradient parity versus native CPU pooling uses2e-5 and5e-12, respectively. NumPy crop RNG continuation also agrees exactly. Repaired CF loss/backward is replayed exactly, online gradients are nonzero, and EMA targets have no gradients. Blocking `F.adaptive_avg_pool1d` during the actual CF loss still permits backward; the pre-repair test failed on that exact operation.

Root CUDA verification is **NOT_RUN here**. `root_cuda_correctness.py` requires explicit `--device cuda --root-gpu-owner`, an integrated main model matching the proposed hash, and root's existing deterministic cuBLAS environment (`CUBLAS_WORKSPACE_CONFIG=:4096:8` or`:16:8`) before Python starts. It uses the main resource ownership/limits, checks CUDA utility values/gradients against native CPU results, blocks native adaptive pooling in CF, and requires bit-identical repeated CF losses/gradients under strict deterministic algorithms. Its inputs are tiny synthetic tensors; it constructs no optimizer, reads no corpus, writes no checkpoint, and reports0 optimizer updates. Only `--help` was executed here; that did not initialize CUDA.

## LightGBM serialization repair and error interpretation

Installed LightGBM is4.6.0. Its inspected `basic.py` SHA256 is `3bb8761547ec0c5064947ab95eee3785c36defc294494d459c0db58a51a722f2`. Installed `save_model` calls `LGBM_BoosterSaveModel` with `_c_str(str(filename))`, where `_c_str` UTF-8-encodes into a narrow `ctypes.c_char_p`. `model_to_string()` instead calls the in-memory save API, decodes UTF-8 and appends pandas categorical metadata; the constructor accepts `model_str`. These APIs are documented in the official [LightGBM4.6.0 Booster reference](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Booster.html).

The reported root message, `Model file ... is not available for writes`, is LightGBM's generic C++ writer-initialization failure. [GBDT::SaveModelToFile](https://raw.githubusercontent.com/microsoft/LightGBM/v4.6.0/src/boosting/gbdt_model_text.cpp) emits it when `VirtualFileWriter::Init()` fails; it does not include an OS errno/WinError. It is not itself a Python `PermissionError` or a sandbox command denial. With the non-ASCII Windows root and UTF-8 narrow filename boundary, the observation is consistent with native Unicode filename handling failure. The evidence does not conclusively distinguish every possible original-path ACL, missing-parent, path-length or disk failure: that path was deliberately not retried, relocated or probed by a write. No claim is made that this check proved the original main output path writable.

The proposed change replaces only `model.booster_.save_model(str(path))` with `save_booster_text(model.booster_, path)`. The helper obtains `booster.model_to_string()` and writes it through Python `Path.open("x",encoding="utf-8",newline="")`; `load_booster_text(path)` reads UTF-8 through Python and constructs `lgb.Booster(model_str=text)`. A filename never reaches the native booster writer/reader. Default iteration/importance behavior matches the old default save call. All fifteen fresh horizon/quantile estimators, all fixed recipe parameters, fitting calls, predictions and sorting are preserved. AST comparison of the entire reference `run` function, with only its serialization call normalized, proves this remains the same recipe.

Permission/existence failures propagate after one attempt. There is no exception fallback, ASCII relocation, permission change, filename-native retry or overwrite. Root must use new reviewed run directories. Existing failed output paths and the previously denied filesystem operations were not retried.

Fifteen independently fitted tiny CPU quantile boosters (three horizons × five quantiles) use generic synthetic polynomial/sinusoidal inputs. They are clearly `SYNTHETIC_CORRECTNESS_ONLY`, not acoustic data or a real reference experiment. In-memory text tests preserve all12 trees and categorical metadata, verify nonconstant predictions, exercise a missing-value input, and require bit-identical replay for all fifteen. Native filename calls are blocked in these tests. Exclusive-open/no-overwrite and one-attempt permission propagation are also verified.

A single ordinary Python write to a **new** Unicode synthetic fixture path succeeded, exit0: `SYNTHETIC_CORRECTNESS_ONLY_Unicode_Á_é_booster_v1.txt` in this evidence directory. Its complete model hash is `11bdd3c08b606b3dfa992cbd227ec459d79094f13ce46f1e7b7d177c4c7e6266`; reload produced bit-identical predictions for29 synthetic rows. `unicode-fixture.txt` records the result. This proves Python text persistence/replay works at this allowed Unicode path, while leaving the original C++ failure path untouched. `run_unicode_fixture.py` refuses an existing fixture before fitting/writing; do not rerun it on the preserved path.

## Red → green evidence and commands

All commands used the existing `../marine-echo-jepa/.venv/Scripts/python.exe`, explicit `cmd.exe`, `login=false`, and unchanged sandbox permissions. No install or alternate execution/permission bypass occurred.

- `red-cf-pooling.txt`: exit1,45 failed/1 passed while utility absent.
- `red-cf-native-path.txt`: exit1,1 failed/46 deselected; actual CF called the native adaptive pool blocked by the test.
- `red-booster-text.txt`: exit1,17 failed before text helpers/call substitution existed.
- `green-platform-1.txt`: exit0,64 passed after implementation.
- `focused-pytest-final.txt`: exit0,**65 passed in7.51s** after adding the autograd graph assertion and integration-compatible recipe normalization.
- `focused-pytest-strict-final.txt`: exit0,**65 passed in5.79s** after eliminating even test-teardown determinism restoration; strict deterministic algorithms remain enabled throughout. This is the final tested version.
- `unicode-fixture.txt`: exit0, durable NEW Unicode synthetic text/prediction replay passed.
- `package-patches.txt`: exit0, source hashes and AST preservation asserted while producing patches.
- Initial `verify_delivery.py` before-only assertion exited1 because main changed externally during handoff; `delivery-identities.json` preserves that failure status. Final read-only verification accepts only the exact before or exact proposed after hashes and exited0, recording both exact after hashes in `delivery-identities-observed.json`. No stale source was overwritten or patch re-applied.
- `patch-apply-check.txt`: exit0, both diffs applicable to exact current main, no source mutation.
- `format-check-final.txt`: exit0,9 files already formatted.
- `ruff-strict-final.txt` and `format-check-strict-final.txt`: final checks exit0 after the strict fixture change.
- `ruff-correct-package-classification.txt`: exit0, all checks passed. Initial `ruff-final.txt` exited1 solely because the copied `.after.py` outside `src` was classified as a third-party import layout. The final command explicitly classifies `marine_echo` as first-party; no frozen configuration or source-import layout was rewritten to accommodate the evidence location.

Focused final command:

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q -s -p no:cacheprovider tests\unit\test_native_cf_pooling.py tests\unit\test_native_booster_serialization.py
```

Final lint/format:

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff check --no-cache --config=lint.isort.known-first-party=['marine_echo'] src\marine_echo\models\native_temporal.py tests\unit\test_native_cf_pooling.py tests\unit\test_native_booster_serialization.py evidence\ssl-platform-builder-v1
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff format --check src\marine_echo\models\native_temporal.py tests\unit\test_native_cf_pooling.py tests\unit\test_native_booster_serialization.py evidence\ssl-platform-builder-v1
```

Read-only patch applicability command:

```text
git -C ..\marine-echo-jepa apply --check ..\marine-jepa-vnext-builder\evidence\ssl-platform-builder-v1\native_temporal.patch ..\marine-jepa-vnext-builder\evidence\ssl-platform-builder-v1\native_references.patch
```

One-time already-executed durable fixture command (do not repeat its preserved path):

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B evidence\ssl-platform-builder-v1\run_unicode_fixture.py
```

Root-only unexecuted CUDA correctness invocation, after integrating exact bytes and acquiring the GPU according to the existing resource policy:

```text
.venv\Scripts\python.exe -B evidence\ssl-platform-builder-v1\root_cuda_correctness.py --device cuda --root-gpu-owner
```

The root script and these test files should be copied with the patch package for those checks. No real run is launched by preparing/copying them. Python bytecode and pytest/Ruff cache writes were disabled for executed checks. Actual durable writes that were attempted in this task succeeded; simulated permission-denial tests intercept before touching a filesystem.

## Remaining gates

Root alone integrates approved diffs, executes actual deterministic CUDA correctness, and obtains renewed distinct review bindings for every changed source before fits. This monolithic model module's new hash affects all reviews that bind it, even methods whose computation did not change; existing approvals are not asserted to cover the repair. Historical evidence/source pins remain untouched. No resumed or renewed scientific job was run here. GPU correctness, renewed independent review and any real-job recovery are **NOT_RUN/PARENT_OWNED**. Git index/commit is **NOT_RUN**; the earlier index-lock denial was not retried.
