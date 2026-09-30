# Root-only synthetic prefix checks

These checks are SYNTHETIC_CORRECTNESS_ONLY and grant no scientific approval.
Integrate the three authored files and the two evidence Python helpers into the
main checkout without altering frozen sources. Use main's existing environment.

Run from the integrated marine-echo-jepa checkout, with CPU only. Set the process
environment variable NATIVE_PREFIX_ROOT_RUNNER_CHECKS=1 explicitly. The six
optimizer tests also require that checkout name; they remain skipped in builder.

```text
.venv\Scripts\python.exe -B -m pytest -q --import-mode=importlib -p no:cacheprovider tests/unit/test_native_prefix_transfer.py tests/integration/test_native_prefix_transfer.py
.venv\Scripts\python.exe -B -m ruff check --no-cache src/marine_echo/training/native_prefix_transfer.py tests/unit/test_native_prefix_transfer.py tests/integration/test_native_prefix_transfer.py evidence/ssl-prefix-builder-v1/prefix_test_support.py evidence/ssl-prefix-builder-v1/verify_delivery.py
.venv\Scripts\python.exe -B -m ruff format --check --no-cache src/marine_echo/training/native_prefix_transfer.py tests/unit/test_native_prefix_transfer.py tests/integration/test_native_prefix_transfer.py evidence/ssl-prefix-builder-v1/prefix_test_support.py evidence/ssl-prefix-builder-v1/verify_delivery.py
```

Six root-only cases exercise actual AdamW updates in frozen/scratch modes,
four DEV choices, exact uninterrupted/resumed CPU optimizer/scheduler/RNG/model
continuation in both modes, matched labels/samples/masked-fill gradients, and
virtual protected runner completion with genuine Torch/NumPy codecs. The virtual
transport is installed before calls; it is not a retry of failed OS writes.
Preserve failures and stop that exact operation if an OS restriction occurs.

Root should additionally execute a newly named Unicode synthetic durable output
fixture using the same private receipt/NPZ schemas in its authorized workspace.
The builder did not execute that durable filesystem fixture or actual optimizers.
No scientific corpus, selected public weights, final-site values or GPU are needed.

Real fits require distinct exact APPROVED_PREFIX_TRANSFER_PREFIT plus separately
bound APPROVED_PREFIX_TRANSFER_NUMERIC_ACCESS; ADR0021 remains PROPOSED. Root must
provide genuine metadata/ancestry/cohorts/selection/supervisor and resolve its
aggregate budget. Prefix inference has its own kind; the existing zero-shot
assessment's exclusions and supported kinds remain unchanged. Later suffix use
needs a separate admitted transfer assessment, never a zero-shot reinterpretation.
