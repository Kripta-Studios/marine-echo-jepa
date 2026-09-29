# Next research decision

**NO_FURTHER_FITTING_BEFORE_MEETING**

**DRAFT / NOT_AUTHORIZED / NOT_QUEUED** — the conditional intervention below is carried forward from the approved [diagnostic decision](../../orchestration/reports/AEON_CHECKPOINT_OBJECTIVE_DIAGNOSTIC_20260929.md). Diagnostic approval does not authorize execution. A new reviewed protocol and explicit owner authorization after the meeting are required.

Move only the **0.03 regularizer from predictor outputs to online full-context representations**. Compare the current EMA objective against that single location change. The hypothesis is restored suffix encoder data-gradients and reduced centered teacher/predictor scale mismatch without sacrificing forecast quality; it is not an established causal explanation.

Copy **all initial tensors and buffers, including the forecast head**, from one common initialization. Match identical original TRAIN rows, scaler, batch indices, optimizer settings and supervised schedule. Retain the prefix student and detached suffix teacher. No expanded data, architecture change, additional hyperparameter change or checkpoint search.

One **seed-7 pair**, **1500 SSL + 1500 supervised updates per arm**, **6000 total optimizer updates**; **one local process**, **20-minute fitting wall-time cap**, peak GPU allocated/reserved memory **below 10 GiB**, process RAM **below 22 GiB**. No cloud or sweep. Use only original **4965-row TRAIN** and **1219-row repeatedly exposed development VAL**; no prior-year, CAL or TEST evaluation.

Use fixed final endpoints, never a historical minimum. Stop on resource-cap, tensor/batch-matching or nonfinite-objective failure.

Consider **separately authorized replication only if all three conditions hold** for the changed arm:

1. Nonzero suffix gradients on **both fixed TRAIN batches**.
2. At least **25% reduction** in `abs(log(predictor_centered_RMS/teacher_centered_RMS))` on **both batches**.
3. At least **1% lower final corrected VAL pinball** than the matched current-EMA arm.

Otherwise stop the intervention. Thresholds are unchanged. Passing this single-seed exploratory screen establishes neither generalization nor automatic authorization. Any replication needs its own approval. No command is queued; no fitting, diagnostic rerun, matrix, 30k/50k extension, recalibration, acquisition or new outcome is authorized by this handoff.

Keep the original retrospective positive and prior-year transfer negative results in their own scopes. Prior-year data is TRAIN for expanded models and affected descendants. The meeting continues with the approved r3 software, completed diagnostic and [pilot discussion](PILOT_DATA_REQUEST.md).
