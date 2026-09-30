Implemented `NativeAcousticEncoder(weights, device="cpu")` in the new
`src/marine_echo/inference/native_encoder.py`. This is an encoder-only interface
for `native_ssl_selected_encoder_v1`, with no forecast head, query, future crop,
target, optimizer, fitting, RNG reset or determinism-policy change. The new code
and all executed checks are bounded to the owner's requested scope.

The loader uses actual Torch serialization with `weights_only=True` and
`map_location="cpu"`. It reads the existing `config/scalers/encoder/bindings`
schema, accepts an absent `selected_pretrain_step`, and rejects other artifact
kinds, including downstream supervised encoder artifacts. It constructs only
the saved shared or CF backbone on the meta device, casts the empty template to
the native float32 contract without changing the caller's default dtype, checks
the exact parameter/buffer keys, shapes, dtypes and finiteness, and assigns saved
tensors with strict loading. All parameters are frozen and the encoder remains
in evaluation mode during encoding. For CF the selected checkpoint is the saved
EMA feature branch, including its frozen batch-normalization buffers.

`encode(x, observed, metadata)` returns float32 NumPy `[N,D]` in the saved batch
size. It reuses the existing saved TRAIN channel scaler helper and never fits
scalers. The complete original scaler/config state and source bindings are
available through defensive-copy metadata properties. Source bindings are
provenance; their paths are never opened or executed by this interface.

Inputs must have exactly the saved history 24 or 96, four ordered channels,
Boolean masks and a complete observed primary 38-kHz prefix. Observed raw dB and
all measurement metadata must be finite. Secondary missing fills, including
NaN/Inf, are ignored before scaling. Metadata must preserve the native encoded
38/125/200/455-kHz frequencies, nominal 60-minute intervals, integrated-product
geometry, increasing nonnegative depth bounds, unknown orientation, Boolean
known flags and context offset zero. The native bounds are passed unchanged;
the tests distinguish 0–230 m from 0–200 m. No dB geometry rescaling or profile
interpretation is introduced. Precision overflow or depth bounds that collapse
when converted to model float32 are rejected. Reversed/strided NumPy views are
supported, including a final single-row batch.

The reported `training_kind` is `ssl_pretrained` for shared, masked and CF;
`permuted_pairing_ssl_control` for permuted; `supervised_feature_encoder` for
direct; and `untrained_control` for random frozen. Method names do not confer
scientific validity, and direct/random representations are not labeled SSL.

Public usage after root integrates the new source:

```python
from marine_echo.inference.native_encoder import NativeAcousticEncoder

encoder = NativeAcousticEncoder("selected_encoder.pt", device="cpu")
features = encoder.encode(raw_db_context, observed_boolean_mask, native_metadata)
print(features.shape, encoder.method, encoder.training_kind)
```

The caller remains responsible for source-date contiguity and the association
between the supplied context and issuance cutoff: these three arrays contain
no timestamps or source interval identifiers. Exact shape, masks and context
offset checks cannot establish temporal provenance for arbitrarily mislabeled
arrays. Selected-artifact lineage is retained as declared provenance, without
re-running the scientific prefit review. Root owns verification against actual
pretrained artifacts and the separate distinct inference/downstream/package
review. CUDA inference is an optional interface device; no CUDA operation was
executed or validated in this builder task.

The new `tests/unit/test_native_encoder.py` contains 81 executed synthetic CPU
checks. All fixtures are explicitly `SYNTHETIC_CORRECTNESS_ONLY`; real Torch
checkpoint codecs run in memory. Tiny backbones exercise behavioral rejection
and replay paths; additional synthetic checks use main's default shared latent
64 and CF latent 128 backbones. Reference outputs come from actual main
backbones, rather than a copied implementation or a dummy encoder. Tests cover
both histories, all six methods, multiple native geometries, incomplete
secondary channels, exact same-batch encoding replay, geometry sensitivity,
masked-fill invariance, frozen parameters/buffers and absent gradients,
CF EMA-state/BN replay, batch partition equality, unsafe pickle rejection,
invalid dimensions/scalers/state/masks/metadata, no head/query/target dependency,
no fit/review calls, unchanged Python/NumPy/Torch RNG and determinism state,
caller float64 default dtype, float64/strided inputs and precision admission.

Commands executed through ordinary `cmd.exe`, existing environment, no installs
or permissions changes:

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q -s -p no:cacheprovider tests\unit\test_native_encoder.py
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff check --no-cache src\marine_echo\inference\native_encoder.py tests\unit\test_native_encoder.py evidence\ssl-encoder-api-builder-v1\inspect_sources.py
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff format --check --no-cache src\marine_echo\inference\native_encoder.py tests\unit\test_native_encoder.py evidence\ssl-encoder-api-builder-v1\inspect_sources.py
..\marine-echo-jepa\.venv\Scripts\python.exe -B evidence\ssl-encoder-api-builder-v1\inspect_sources.py
```

Final pytest exited 0: **81 passed in 22.90 seconds**
(`green-default-dimensions.txt`). Final Ruff check and format check exited 0
(`lint-complete.txt`, `format-check-complete.txt`). Source inspection exited 0
(`final-source-provenance.json`), with CUDA uninitialized. That JSON records the
actual imported main package/helper files and exact hashes. Tests import main
helpers read-only and load the new candidate via `importlib` because the builder
copies of the historical core are frozen/stale. Once integrated, the same test
bootstrap resolves the new file and helpers within main.

Preserved red evidence: the initial missing API produced pytest collection exit
2 (`red-tests.txt`); subsequent behavioral tests exposed four failures for
caller-default dtype and reversed inputs (`red-boundaries.txt`, exit 1).
The intermediate strided fix still failed on two singleton trailing batches
(`green-boundaries.txt`, exit 1); that issue is fixed in the delivered bytes.
The final green run includes both failures' regressions. Earlier green/lint
attempts remain untouched so the sequence is inspectable.

Delivered executable source SHA256:
`6cc07a15ea93427d6628f1b317945f6e38ff3008894a31cab2173c0b666c0f7f`.
Delivered tests SHA256:
`37e241a00c9fa87f0a886326364975783ce39bbdfd37d4158305062c59769ab6`.
Main read-only backbone SHA256:
`b7d6b9de5221643871c0f21dbfa440a2c6763600ff3dff5818dfb8918e661a32`.
Main read-only Config/Scalers runner SHA256:
`a5d4be8301169da6ec6d4693cb085395f1fcc94c10b0c13cf9c753975c741744`.
The provenance JSON also hashes SIGReg, native_resources, package initializers
and the untouched existing native_acoustic interface.

Authored paths are exclusively the new API, the new unit test and this evidence
directory (source inspector, logs, provenance JSON, executed-checks JSON and
this report, plus `verify_handoff.py` and its `handoff-verification.json` byte
inventory). The final handoff verifier exited 0 after checking JSON, logs,
report hashes and unchanged main imports; both final whole-handoff Ruff checks
also exited 0. This byte verification is not an independent review. Existing
sources and historical evidence were not changed. No
real corpus/checkpoint access, scientific fitting, optimizer, GPU execution,
agents, installation or self-approval occurred. Actual pretrained artifact
transport/replay and package integration are NOT_RUN here and remain root's
responsibility. No Git index operation or commit was attempted, honoring the
owner's prohibition on retrying the earlier denied index write.
