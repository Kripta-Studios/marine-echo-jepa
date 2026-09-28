# AEON external transfer: first secondary numeric attempt

Date: 2026-09-28. Study: `aeon_external_transfer_20260928_v1`.
This is failure evidence, not a model result.

The distinct pre-numeric review at
`orchestration/reviews/AEON_EXTERNAL_STAGE2_PRENUMERIC_REVIEW_20260928.json`
(SHA-256 `7c2e80201bc48edc825b2899eee17210d8252b0296987ccf65272570266f164e`)
approved one exact secondary-only zero-shot run. The real CLI passed the frozen
source/report/checkpoint/code preflight and then exited 1 during secondary
candidate materialization, before forecasting, scoring or output creation:

```
ValueError: AEON external Stage 1 candidate cutoff/source time differs.
src/marine_echo/training/aeon_external_evaluator.py:70
```

The Stage 1 metadata inventory encodes whole-second source timestamps without
a fractional suffix. The Stage 2 evaluator converted the same instant to a
fixed microsecond-resolution string, which includes `.000000`. There are 81
such candidate timestamps in the secondary inventory; this is a code-level
serialization mismatch, not an acoustic observation or eligibility finding.
The first failing candidate has cutoff interval ID `465701`, row ID
`10b0d82a7807b404733b0f6a87d504fe4b64a7c0c69b4000cbd102a3a227aa77`,
and source timestamp `2023-02-16T04:51:57`.

The primary AEON2 numeric reader was never called; its 0–230 m layer remains
metadata-ineligible for the frozen 0–200 m target. The secondary CSV rows were
opened once under the reviewed access but no model prediction, metric,
bootstrap interval or result manifest was produced. The fixed output directory
`outputs/aeon_external_transfer_20260928_v1/zero_shot_secondary` was absent
after exit. The approved one-pass access is considered consumed. A narrow
instant-equivalence fix, focused regression, and separate distinct remedial
pre-numeric review are required before retry. No threshold, cohort, target,
model, checkpoint, scaler or score rule may change.
