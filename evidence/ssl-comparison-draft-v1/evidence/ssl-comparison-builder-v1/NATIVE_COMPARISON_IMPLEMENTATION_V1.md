Implemented bounded comparison reconstruction in the new
`src/marine_echo/evaluation/native_comparison.py`, with public function
`compare_saved_predictions(manifest_path, review_path, output_path)` and a CLI.
The implementation consumes reviewed saved forecasts and support only. It
does not run inference, load acoustic corpora or model weights, select models,
fit anything, import Torch, or execute on a GPU. Main's existing
`native_product.native_scores` is imported read-only and called for every
method with daily floor 18.

Manifest schema, frozen before any assessment values are decoded:

```json
{
  "role": "development",
  "evidence_kind": "REVIEWED_SAVED_PREDICTIONS",
  "implementer_session_id": "ACTUAL_DECLARED_IMPLEMENTER_SESSION_ID",
  "root_coordinator_session_id": "ACTUAL_ROOT_COORDINATOR_SESSION_ID",
  "methods": {
    "prespecified_reference": "absolute-or-manifest-relative-reference.npz",
    "prespecified_method": "absolute-or-manifest-relative-method.npz"
  },
  "reference": "prespecified_reference",
  "bootstrap_seed": 1729,
  "bootstrap_replicates": 2000,
  "block_days": 7,
  "floor": 18
}
```

An optional `scientific_protocol_path` is resolved relative to the manifest and
must also be hash-bound. At least two named methods are required. Changing any
fixed recipe value, including its integer type, is rejected. Method names and
the single reference are prespecified by the exact manifest binding; the code
does not select a reference or tune the recipe from results.

Review schema:

```json
{
  "status": "APPROVED_COMPARISON_RECONSTRUCTION",
  "allowed_roles": ["development"],
  "evidence_kind": "REVIEWED_SAVED_PREDICTIONS",
  "reviewer_session_id": "ACTUAL_DISTINCT_REVIEWER_SESSION_ID",
  "bindings": {
    "ABSOLUTE_COMPARISON_MODULE_PATH": "SHA256",
    "ABSOLUTE_NATIVE_PRODUCT_SCORER_PATH": "SHA256",
    "ABSOLUTE_MARINE_ECHO_PACKAGE_INITIALIZER_PATH": "SHA256",
    "ABSOLUTE_EVALUATION_PACKAGE_INITIALIZER_PATH": "SHA256",
    "ABSOLUTE_MANIFEST_PATH": "SHA256",
    "ABSOLUTE_REFERENCE_NPZ_PATH": "SHA256",
    "ABSOLUTE_METHOD_NPZ_PATH": "SHA256"
  }
}
```

Every listed prediction artifact and the optional protocol must be bound.
Bindings are exact normalized absolute paths to SHA256 digests; an empty map
or missing source binding cannot authorize numeric parsing. Final assessment
requires manifest role `final_test`, review status `APPROVED_FINAL_ASSESSMENT`,
and explicit `final_test` admission in `allowed_roles`. A fitting/numeric-access
approval is not a comparison approval. Unknown roles are rejected. If the
review repeats the declared implementer/coordinator identities, they must
match the manifest exactly.

The actual builder identity is frozen in source as
`01a0ef20-7397-7ba1-a98c-f59bc38ddcc1`, verified against the ordinary runtime
`CODEX_THREAD_ID` and the owner's amendment. The reviewer must differ from this
builder, the declared implementer and the declared root coordinator. Identity
comparison is case-insensitive, and padded/blank identities are rejected.
Session declarations and hashes cannot cryptographically authenticate who
authored a supplied review JSON; root must provide a genuine distinct review
and accurate coordinator/implementer identities. This implementation provides
no self-approval or automatic review generation.

Both manifest and review are read before NPZ numeric parsing. The implementation
checks role, status, identities, evidence kind, recipe, reference, new output
path and every immutable binding before any `np.load`. Imported source hashes
are captured at import and must still match at admission. Prediction bytes are
snapshotted while hash-checking, and `np.load(BytesIO(snapshot), allow_pickle=False)`
consumes only the admitted snapshot. A later file change cannot alter the
evaluated bytes. Output uses exclusive UTF-8 `Path.open("x")`, with no overwrite,
directory creation, fallback relocation or retry. Its parent must already
exist. Permission errors propagate to the caller.

There is no correctness-smoke bypass or relaxed synthetic recipe. Tests use
`SYNTHETIC_CORRECTNESS_ONLY` in their manifest, review and fixtures, with the same
status/role/identity/source/prediction hash gates. The evidence kind must agree
between manifest and review and, if present, each NPZ's scalar evidence label.
The module uses CPU NumPy only. Synthetic fixture approvals are visibly test
objects and are not approvals of any real scientific path or data.

Required NPZ fields are exactly the shared conventional artifact contract:
`predictions[N,3,5]`, `targets[N,3]`, Boolean `observed[N,3]`, Unicode
`target_dates[N,3]`, `deployment[N]`, `row_id[N]`, and `query[N,3,10]`.
Optional source/archive/cutoff and other safe scalar/array provenance is retained
per method with original dtype, shape and values, canonically reordered if
row-aligned. No archive named by a provenance field is opened. Raw `x`, future
crop/mask, encoder or weights fields are rejected. Optional saved role labels
must match the admitted role. Conventional references do not need any extra
source/scaler/training fields. Optional nonfinite provenance values have
explicit string tokens; metrics/forecasts remain subject to strict finiteness.

Unique `(deployment,row_id)` identities define canonical lexicographic issuance
order. Duplicate pairs, different row sets, masks, target dates, native queries
or observed target values are rejected. Masked target fills can differ and are
not assessed. All forecasts must be finite, even at unobserved targets, and
quantiles must be ordered. Observed targets must be finite. This retains main's
strict rejection semantics and never shrinks to an intersection. Required
daily support keys/counts are independently reconstructed from masks/dates and
checked against every imported scorer result.

Native queries are finite integrated 38-kHz products at nominal 60 minutes,
with nonnegative increasing depth bounds, unknown orientation, Boolean known
flags and horizon offsets 1/3/6. Every method must agree exactly on the saved
queries. The report retains the original encoded queries and describes unique
native frequencies, intervals, products and bounds in native units. The tests
verify both float32 and float64 0–230 m products report 230 m, with no relabeling
as 200 m and no dB rescaling. Dates are strict source-calendar `YYYY-MM-DD`
proxies; empty dates produce no eligible daily records. No UTC interpretation
is asserted.

For uncertainty, each deployment's earliest nonempty target source-calendar
date across every horizon anchors fixed nonoverlapping seven-calendar-day
blocks. Observed nonempty dates define the blocks available for sampling,
including blocks whose daily records fail floor 18. All methods/horizons share
this assignment. A fresh `np.random.default_rng(1729)` generates 2000 replicates.
Each replicate samples the original number of deployments with replacement,
then samples the original number of observed blocks within each selected
deployment with replacement. Repeated blocks and their eligible daily records
retain multiplicity. Within each sampled deployment, eligible days are equally
weighted within each horizon, then horizons are equally weighted; sampled
deployments are equally weighted. This matches the native metric's nesting
without pooling all days or rows across deployment groups.

Intervals are the linear 2.5/97.5 percentiles of paired method-minus-reference
replicate scores. Direction is explicit and never swapped. Incomplete original
daily/horizon/deployment support produces null point differences and intervals.
If any replicate lacks a required horizon, the interval remains null; the code
does not silently condition on supported replicates. Any deployment with zero
observed calendar span also prevents an uncertainty interval, while its
otherwise defined native point score remains reported. Both cases are
`NOT_ASSESSABLE`, with a reason and actual support/replicate counts. Identical
methods with supported nonzero-span data have exact zero differences and zero
intervals. Single-block support, zero resampling variation and changing
eligible-day counts are disclosed. These conservative interval conventions
are part of the frozen algorithm; root must adopt the signed scientific
protocol before real assessment, rather than adjust them after viewing values.
Generated, unsupported and not-run replicate counts are separate. Incomplete
original support skips bootstrap generation entirely: generated 0, not-run
2000, unsupported generated replicates 0. Unsupported replicates are counted
only after an actual resampling schedule has been generated.

Output is a new strict JSON report containing exact per-method native metrics
and daily records, common issuance/support/query geometry, observed coverage,
retained optional provenance, paired differences/interval status, block
assignment, eligible-day count ranges and a deterministic resampling sequence
hash. Forecast variation is an explicit diagnostic, including median ranges,
standard deviations and per-deployment quantile ranges; constant forecasts
are not hidden. No SOTA/JEPA winner, scientific benefit, positive label, profile
or biological conclusion is generated. Deployment count is reported; site
count remains unknown because deployment groups do not establish independent
sites. The report discloses the limited groups, block/correlation assumptions,
conditional issued-row support and calendar-proxy limitations.
Report provenance directly maps each prespecified method label to its exact
prediction NPZ path and admitted byte hash.

Public commands after root integrates the new source:

```python
from marine_echo.evaluation.native_comparison import compare_saved_predictions

report = compare_saved_predictions("manifest.json", "distinct_review.json", "new_report.json")
```

```text
python -m marine_echo.evaluation.native_comparison --manifest manifest.json --review distinct_review.json --output new_report.json
```

Executed verification used the existing environment and ordinary `cmd.exe`:

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q -s -p no:cacheprovider tests\unit\test_native_comparison.py
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff check --no-cache src\marine_echo\evaluation\native_comparison.py tests\unit\test_native_comparison.py evidence\ssl-comparison-builder-v1
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff format --check --no-cache src\marine_echo\evaluation\native_comparison.py tests\unit\test_native_comparison.py evidence\ssl-comparison-builder-v1
..\marine-echo-jepa\.venv\Scripts\python.exe -B evidence\ssl-comparison-builder-v1\inspect_sources.py
```

Final focused pytest exited 0: **60 passed in 25.25 seconds**
(`green-delivery.txt`). Ruff check and format check exited 0 (`lint-delivery.txt`,
`format-check-delivery.txt`). The source inspector exited 0 and records actual
main imports, exact hashes, runtime identity and `torch_imported=false` in
`final-source-provenance.json`. The API candidate is loaded by `importlib` in builder
tests so stale builder core files cannot shadow main's read-only evaluator.
After integration, the same test bootstrap resolves the source/helpers in main.

Tests use actual NumPy compressed NPZ codecs in memory, virtual fixture paths
under the new evidence directory, and real JSON serialization. No durable
prediction NPZ file was opened or written. Independent tests reconstruct
pinball row losses and equal day/horizon/deployment scores with unequal support;
the bootstrap oracle concatenates actual daily records with repeated block
draws, independently of the implementation's weighted sum/count representation.
Other checks cover reordered rows, masked fill invariance, geometry reporting,
duplicate/missing/mismatched/nonfinite artifacts, all required review/hash/role
gates before numeric loading, constant diagnostics, insufficient floor support,
zero-span/unsupported resamples, exact identical-method zero intervals,
deterministic replay, retained optional provenance, unchanged input bytes,
reviewed snapshots despite later file mutation, and scorer support shrinkage.

Preserved red evidence: missing implementation produced pytest collection
exit 2 (`red-tests.txt`). The first implementation run had one virtual-file
fixture failure with a missing review, not a production gate failure
(`green-attempt-1.txt`, 44 passed/1 failed). Adversarial session-ID cases then
produced two meaningful failures before the identity fix
(`red-adversaries.txt`, 56 passed/2 failed): uppercase and padded self-review IDs
reached the numeric-load spy. The delivered code rejects both before parsing.
The subsequent green run and lint-driven refactor remain recorded separately.
Focused tests also preserved missing method-to-artifact provenance and missing
generated/not-run counter failures (`red-method-provenance.txt`,
`red-replicate-counts.txt`, each exit 1). Both contracts are implemented and
included in the final green run. Intermediate source/hash evidence is retained
under its original names; `final-source-provenance.json` binds delivered bytes.

Delivered source SHA256:
`a42f8cb3ddd6dd29afbcfec413060b9ac7830376a750fda24a4c6a3f782230d2`.
Delivered tests SHA256:
`52352e903db6aa8e752b4413b0e00c8faeec9cc1cd4193fa589e836a2d4f0c80`.
Read-only main native evaluator SHA256:
`fd1c643fde93494a21d560794f886056e58031f1ef6dd41ebe53662750bab314`.
The provenance JSON records package initializer hashes and the exact binding
paths; root's review must bind the integrated main path, not the builder path.

The final `verify_handoff.py` command exited 0 after checking report/log/JSON
consistency and the delivered/current main source hashes. Its
`handoff-verification.json` inventories the new deliverable bytes. Both
whole-handoff Ruff checks also exited 0 (`lint-complete.txt`,
`format-check-complete.txt`). This byte verification is not an independent
scientific review or approval.

Only the two requested new source/test paths and this new evidence directory
were authored. Earlier implementation files, main fit sources, dependencies,
contracts, STATUS, ledger and historical evidence remain untouched. No real
archives/corpora/predictions/weights, fitting, model selection, GPU, agents,
installs, credentials, permission changes, junction operations or Git index
operations occurred. Durable disk-fixture integration, genuine distinct
scientific review/protocol adoption and actual development/final reconstruction
are NOT_RUN here and remain root-owned. No commit or self-approval is claimed.
