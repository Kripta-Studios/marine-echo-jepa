# Native comparison schema correction V3

This delivery supports the existing saved-prediction schemas. Read-only inspection of main native_ssl.py lines 1039–1056 and native_downstream.py lines 798–817 confirmed that neural generators emit target_observed for forecast labels and context_observed for context. References use observed. No generator, actual artifact, fitted source, scientific recipe or historical evidence was changed. Root has not integrated or scientifically executed this comparison delivery.

The reader accepts target_observed only as a forecast-label alias when observed is absent. Each present forecast-label field must independently be Boolean [N,3]. If both fields occur, both must be valid and exactly equal. Conflicting dual masks fail before scoring, including discrepancies in masked cells. Equal dual fields are accepted with an explicit trace. context_observed never supplies forecast labels. Per-method forecast_mask_provenance records original_fields, canonical_field, canonicalized_from and equal_dual_masks. Canonicalization occurs in private loaded copies; caller arrays and source NPZ bytes stay unchanged.

Every optional array is summarized, with its name explicitly listed in summarized_optional_arrays. Descriptors contain exact original dtype, shape and SHA256 of contiguous C-order element bytes, including numeric bit patterns and Unicode code units. array_order is original_artifact_order; content_encoding is numpy_C_order_bytes. This includes target_observed, context_observed [N,96,4], metadata [N,4,10], repeated geometry, assessment_support, horizons, quantiles, cutoff and archive/source provenance whenever present. No optional array cells are dumped into JSON or silently excluded. The original NPZ hash independently identifies the container and original raw arrays. Reordering an artifact changes an order-sensitive descriptor even when canonical support and metrics remain equal.

Required scoring support remains explicit in the existing common_support representation: row/deployment identities, forecast-label mask, target dates, observed target values and complete raw encoded query. Masked target fills retain the preceding handling; they are not used for scoring or replaced on disk. Finite forecasts, finite observed targets, exact unique row sets, native geometry, independent daily support checks and strict review/source identities remain required. Native lower bound 230m is unchanged. No intersections, metric compression, fitting, model selection or numerical scientific access occurred.

The preceding ADR0015 correction remains fixed: bootstrap_seed=20260929, bootstrap_replicates=2000, block_hours=48, block_days=2, floor=18. Two nominal source-calendar-day blocks retain date gaps, paired horizons/methods and sampled multiplicities, with equal day→horizon→deployment weighting. Source-clock uncertainty remains explicit; this is not a verified UTC/absolute-time bootstrap. Seven-day/1729 manifests continue to fail before numeric loading.

Before edits, snapshot_before_schema_v3.py archived the closed protocol V2 source/tests and recorded hashes for all 56 preceding evidence files. It wrote its snapshot manifest exclusively, without redirection precreating an empty file. All earlier snapshots, red/green logs and reports are retained unchanged. The original contract remains historical; its incorrect artifact mask assumption is superseded by this schema correction.

Meaningful synthetic CPU verification used real NumPy NPZ codecs in memory with virtual fixture files, marked SYNTHETIC_CORRECTNESS_ONLY. No real prediction/corpus/weight artifacts were read. The new tests match native generator field names and dimensions, check alias replay against reference-schema metrics/support/bootstrap, malformed/conflicting dual masks before scoring, equal-dual trace, absence of label masks despite context masks, and bounded content-identifying provenance. A fixture with more than 1.5 million optional Boolean cells has less than 8,000 characters of optional provenance. Changing one context-mask cell changes its content hash while metrics/support/bootstrap stay identical. The prior 68 tests retain protocol, native geometry, finite/floor/unsupported/constant diagnostics, hierarchical scoring, common support and review admission coverage.

Executed commands used the existing main environment, explicit cmd.exe and no cache provider. Final tests imported marine_echo, marine_echo.evaluation and native_product from MAIN read-only through the existing test loader; the comparison candidate was loaded directly from this worktree. The source inspector confirms Torch was not imported and runtime builder identity matches the frozen implementing session.

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B evidence\ssl-comparison-builder-v1\snapshot_before_schema_v3.py
exit 0; 56 preceding evidence files recorded; closed source/tests archived

..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q -s -p no:cacheprovider tests\unit\test_native_comparison.py -k schema_v3 > evidence\ssl-comparison-builder-v1\red-schema-v3.txt 2>&1
exit 1; 11 failed, 3 passed, 68 deselected in 10.64s

..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q -s -p no:cacheprovider tests\unit\test_native_comparison.py > evidence\ssl-comparison-builder-v1\green-attempt-schema-v3.txt 2>&1
exit 0; 82 passed in 34.47s

..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff format --no-cache src\marine_echo\evaluation\native_comparison.py tests\unit\test_native_comparison.py evidence\ssl-comparison-builder-v1\snapshot_before_schema_v3.py > evidence\ssl-comparison-builder-v1\format-schema-v3.txt 2>&1
exit 0

..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff check --no-cache src\marine_echo\evaluation\native_comparison.py tests\unit\test_native_comparison.py evidence\ssl-comparison-builder-v1\snapshot_before_schema_v3.py > evidence\ssl-comparison-builder-v1\lint-schema-v3.txt 2>&1
exit 0; All checks passed

..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q -s -p no:cacheprovider tests\unit\test_native_comparison.py > evidence\ssl-comparison-builder-v1\green-final-schema-v3.txt 2>&1
exit 0; 82 passed in 34.17s against delivered formatted bytes

..\marine-echo-jepa\.venv\Scripts\python.exe -B evidence\ssl-comparison-builder-v1\inspect_sources.py > evidence\ssl-comparison-builder-v1\final-source-provenance-schema-v3.json
exit 0; source/import inspection only; preceding inspector bytes unchanged

..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff format --check --no-cache src\marine_echo\evaluation\native_comparison.py tests\unit\test_native_comparison.py evidence\ssl-comparison-builder-v1\snapshot_before_schema_v3.py > evidence\ssl-comparison-builder-v1\format-check-schema-v3.txt 2>&1
exit 0; 3 files already formatted
```

Delivered source SHA256: 51c5374e188bd3c7f2b00d5f1b5887f33b456c291cb0290e8b8e020ff28df020.

Delivered tests SHA256: b81927eb163364dfd89472eb95ccb4550d03eb9d0917f6179eafabf32ebec4e8.

Archived closed V2 source SHA256: 218987c6afedadef88223366d22fb61cb430c6082bc8a8518fec1d983d26cdfb. Archived closed V2 tests SHA256: a822cb7d2ab8324ebdf60a4505b7fce3d9e7412d29d08d4227cf1b44db38940f.

Unchanged MAIN native_product SHA256: fd1c643fde93494a21d560794f886056e58031f1ef6dd41ebe53662750bab314. Required source bindings and actual import paths are in final-source-provenance-schema-v3.json. The frozen ADR hash and read-only generator source hashes are recorded in handoff-verification-delivery-schema-v3.json.

Public runner after ROOT integration: python -m marine_echo.evaluation.native_comparison --manifest PATH --review PATH --output NEW_JSON_PATH. Genuine distinct reconstruction/final-assessment review must bind the integrated module, imported scoring/package sources, immutable manifest, all NPZ artifacts and optional protocol. Existing reviews are not refreshed or approved by this implementation lane. The report JSON preserves the existing raw common support, per-method metrics/diagnostics, paired differences and fixed bootstrap, adding forecast-mask field provenance and compact optional-array descriptors.

No scientific or final-test assessment, self-review, GPU operation or Git index operation occurred. Root alone integrates this corrected delivery and obtains distinct review. Durable actual-artifact verification remains root-owned. Handoff byte preservation is an engineering verification, not independent review or scientific evidence.

The initial handoff verifier exited 1 because the red traceback contains Windows-encoded Unicode path bytes and was read as UTF-8. Exact error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc1 in position 287: invalid start byte. No filesystem operation was denied. The original logs were preserved; only the new verifier was corrected to match the ASCII summary in raw bytes. Its first empty stdout artifact, handoff-verification-schema-v3.json, is retained. Final verification uses a new handoff-verification-final-schema-v3.json path. Source/tests were unchanged by this evidence-reader correction.

The first format check after that verifier-only edit exited 1 (format-check-handoff-fixed-schema-v3.txt); Ruff check and history verification exited 0. Ruff subsequently formatted only verify_handoff_schema_v3.py, then the final delivery check/format and byte verification used NEW suffixed logs/output. handoff-verification-final-schema-v3.json preserves the first successful verifier snapshot; handoff-verification-delivery-schema-v3.json is the definitive delivered-byte inventory. No numerical source/test edits occurred after the passing final pytest run.

Final handoff commands:

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff format --no-cache evidence\ssl-comparison-builder-v1\verify_handoff_schema_v3.py > evidence\ssl-comparison-builder-v1\format-handoff-delivery-schema-v3.txt 2>&1
exit 0
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff check --no-cache src\marine_echo\evaluation\native_comparison.py tests\unit\test_native_comparison.py evidence\ssl-comparison-builder-v1\snapshot_before_schema_v3.py evidence\ssl-comparison-builder-v1\verify_handoff_schema_v3.py > evidence\ssl-comparison-builder-v1\lint-handoff-delivery-schema-v3.txt 2>&1
exit 0
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff format --check --no-cache src\marine_echo\evaluation\native_comparison.py tests\unit\test_native_comparison.py evidence\ssl-comparison-builder-v1\snapshot_before_schema_v3.py evidence\ssl-comparison-builder-v1\verify_handoff_schema_v3.py > evidence\ssl-comparison-builder-v1\format-check-handoff-delivery-schema-v3.txt 2>&1
exit 0
..\marine-echo-jepa\.venv\Scripts\python.exe -B evidence\ssl-comparison-builder-v1\verify_handoff_schema_v3.py > evidence\ssl-comparison-builder-v1\handoff-verification-delivery-schema-v3.json
exit 0; 56 preceding evidence files and closed source/test snapshots verified unchanged
```
