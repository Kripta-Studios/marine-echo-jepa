# Native latent predictor delivery

The new `NativeLatentPredictor` provides context-only access to saved SSL latent objectives and selected embeddings. No trainer, model, existing inference adapter, protocol, config, ledger, lockfile or historical evidence was changed. This is a model-only implementation, not a scientific result, fit approval or release.

## Closed authored bytes

| File | SHA256 |
| --- | --- |
| src/marine_echo/inference/native_latent.py | cfdbc430d9e9a57c6b815492254119fe0682d728b1e89d2c5e5ec97ed7c14822 |
| tests/unit/test_native_latent.py | 09543753f52810ecae17cf4d7d5a9babce1a117cd5510c633861b77a7110c3a8 |
| tests/integration/test_native_latent.py | a5c16aa378d98563f5b2707df6533fb47e91ee030120fdf513588530efc9fcfc |

All other authored paths are new files under `evidence/ssl-latent-builder-v1/`. The final proof lists seventeen protected main source/lock hashes and actual imported helper/factory paths/hashes. It verifies the closed band-execution sources remain unchanged. Initial sixteen-source proof is preserved; import-alias closure expansion adds `native_resources.py` in the second baseline.

## Public API and semantics

```python
from marine_echo.inference.native_latent import NativeLatentPredictor

adapter = NativeLatentPredictor(weights, device="cpu")
embedding = adapter.encode(x, observed, metadata)
# Shared/band shared_ssl or permuted_ssl only:
future_latents = adapter.predict_latents(x, observed, metadata, query)
# CF only, without any query:
ordinal_zones = adapter.predict_cf_zones(x, observed, metadata)
```

The example is documentation, not an executed public checkpoint call. Devices are exactly `cpu` or `cuda:0`; builder checks were CPU only. `x` is raw dB[N,96,4], observed is Boolean with the same shape, metadata is native encoded[N,4,10], and the shared issued query is[N,3,10]. Inputs retain nominal60-minute intervals, native38/125/200/455-kHz order, actual increasing integrated geometry, complete observed primary prefix and context offset0. Query fields match primary geometry/frequency and source offsets1/3/6. Native0–230m is preserved, never relabeled0–200m or rescaled as a depth profile. Existing immutable input/scaler/query validators are reused unchanged.

`encode` returns float32[N,D] from the selected shared/band encoder or CF EMA encoder. `predict_latents` returns float32[N,3,D] through that shared encoder and the actual saved QueryHead. Its targets are overlapping four-interval blocks starting at native source offsets1/3/6. Permuted pairing is explicitly a control. Latent coordinates are not acoustic dB, quantile forecasts, depth profiles, species or biomass.

CF is separate: `predict_cf_zones` returns float32[N,96,3,D], axes sample/context-token/ordinal-zone/latent. It applies the three saved linear predictors to ONLINE sequence embeddings. `encode` independently uses the saved SELECTED EMA pooled encoder. The CF training loss compared normalized directions against EMA teacher representations; this API returns raw linear coordinates. Applying full issued H96 is outside the sampled forward crop-view support. Its three zones do not have fixed1/3/6 offsets. Passing a query to this method is unsupported; CF `predict_latents` fails with the zone-versus-horizon semantic reason. No rollout, adaptation or new objective was added.

Masked SSL and random/direct controls can expose their accurately labeled embeddings, but `predict_latents` rejects their untrained forward heads. Downstream artifacts are rejected even when a generator uses the SSL inference kind together with supervised ancestry/config fields. No supervised fine-tuned encoder/predictor is reinterpreted as intact SSL. The optional acoustic forecast interface was deliberately not added; saved readout/reconstruction/regularizer tensors are validated, never executed here.

## Safe loading and state

Only `native_ssl_weights_only_inference_v1` and the separate `native_band_ssl_weights_only_inference_v1` are accepted. Legacy configuration has no band architecture field; band requires its exact architecture at both top level and config. Resume/optimizer/selected-encoder/downstream/unknown kinds or extra payload fields are refused. Complete original Config fields, finite numeric settings, saved H96, explicit bounded dimensions/batch, complete finite positive TRAIN scaler state and valid copied source digest bindings are required. A trained objective must declare a positive selected pretrain step within its saved pretrain budget. This step is an existing inference-schema field, not an invented historical checkpoint field.

The loader always calls `torch.load(weights_only=True,map_location="cpu")`. Shared/band models are empty meta templates, explicitly cast to float32 before strict assignment. CF uses an exact load-only module layout with separate online/EMA/readout/three linear predictor templates, avoiding the training constructor's stochastic predictor initialization. All saved parameters/buffers must have exact keys, shapes, CPU strided layout, template dtypes and finite values; BN variances/counters and regularizer step cannot be negative. All modules are eval/frozen. Batches are capped at64 and reuse the saved TRAIN scalers. Missing secondary fills are ignored before normalization; observed values and normalized/output float32 values must remain finite.

Config, scalers, bindings and objective metadata are deep copies. Provenance paths, including `cf_source`, are never opened, imported, hashed or executed by the API. A relocated artifact works with inaccessible ancestral paths. There is no training initializer, optimizer, Config fit-policy validation, RNG reset, compilation, cache setup or global dtype change. Source-clock continuity/issuance association cannot be proven by these input arrays and remains the caller's responsibility.

## Actual verification

All checks used private SYNTHETIC_CORRECTNESS_ONLY CPU fixtures and real in-memory Torch ZIP codecs; no public arrays or checkpoint tensors were read. Two schema tests use only synthetic zero tensors in full-size meta templates to check exact production fields (legacy omits evidence_kind, band includes it). REAL-shaped identity fields in those fixtures are not scientific evidence.

The focused red run exited1: one CF branch replay failed, seventy checks were deselected, elapsed19.59 seconds. Divergent synthetic online/EMA weights and BN buffers produced mismatches in all5184 coordinates when the initial zone implementation incorrectly used EMA features. `red-cf-branch-v1.log` and `red-source-hash-v1.json` preserve the failure and before-source hash. The corrected zone method uses online sequence features while encode remains EMA.

The initial full run passed81 checks in38.80 seconds, exit0. After refactoring and eight added safety/semantic checks, the final run passed89 checks in41.77 seconds, exit0. Both logs are preserved. Ruff check and format-check passed, exit0; final six Python files are formatted. Source/import proof executed, exit0, after the final tests.

Coverage includes exact original/shared/band/CF forward replay; selected EMA versus online semantics; frozen parameters/BN buffers/no gradients; batch split and negative-stride replay; empty secondary channels/masked-fill invariance; caller-array preservation; native230 and query/metadata dependence; no future/target API arguments; strict shape/frequency/interval/geometry/primary issuance; missing/extra/nonfinite/mistyped state/config/scalers; untrained/downstream kind and family rejection; unsafe pickle globals; safe codec flags; inaccessible ancestry; constructor/inference RNG conservation; no optimizer/cache/fit policy/global dtype change; output overflow rejection; unused readout/regularizer exclusion.

Executed commands are in `green-final-v1.log`, `lint-final-v1.log` and `proof-lint-final-v1.log`. Root can repeat synthetic checks after integrating all three authored files plus evidence support:

```text
.venv\Scripts\python.exe -B -m pytest -q --import-mode=importlib -p no:cacheprovider tests/unit/test_native_latent.py tests/integration/test_native_latent.py
.venv\Scripts\python.exe -B -m ruff check --no-cache src/marine_echo/inference/native_latent.py tests/unit/test_native_latent.py tests/integration/test_native_latent.py evidence/ssl-latent-builder-v1/test_support.py evidence/ssl-latent-builder-v1/source_probe.py evidence/ssl-latent-builder-v1/verify_delivery.py
.venv\Scripts\python.exe -B -m ruff format --check --no-cache src/marine_echo/inference/native_latent.py tests/unit/test_native_latent.py tests/integration/test_native_latent.py evidence/ssl-latent-builder-v1/test_support.py evidence/ssl-latent-builder-v1/source_probe.py evidence/ssl-latent-builder-v1/verify_delivery.py
```

The source-proof helpers describe builder/main paths for this delivery; root should verify its integrated import closure against frozen bytes rather than assume identical absolute worktree provenance. Existing original acoustic adapter was inspected for schema; it was not used for replay or modified.

Root-owned durable checkpoint/relocation transport, actual trained public checkpoint replay, GPU execution, independent inference/package review and packaging are **NOT_RUN**. No restricted operation was retried and no denied cache/Git/index/path was touched. There is no commit, self-review, scientific quality claim, fitting or final numerical access in this delivery.
