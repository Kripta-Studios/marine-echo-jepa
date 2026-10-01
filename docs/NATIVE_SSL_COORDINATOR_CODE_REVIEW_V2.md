# Coordinator inspection of CF controls and transfer integration

Reviewed source: `0f8ec3c`; the full inspected HEAD is recorded in
`evidence/ssl-coordinator-review-v2/review.json`. Historical checkpoint
`dd012f0` was not restored. Date: 2026-10-01.

This is the owner's requested direct coordinator inspection, not an independent
review or scientific approval. It covers the new CF control trainer, inference,
executor and budget; comparison/scaler preparation; and the interfaces connecting
those artifacts to the existing reserved assessment and prefix implementation.
No implementation source, fitted output, approval or scientific ledger was changed.

## Findings

1. **P1: matched CF controls cannot enter the reserved assessment.**
   `native_assessment_replication.py:32` and
   `execute_bounded_native_replication_assessment.py:30` accept only the original
   three neural artifact kinds. Neither accepts
   `native_cf_control_weights_only_inference_v1`, `cf_random_frozen`, or
   `cf_direct_supervised`. The assessment rejects these in its model admission
   at line 500; its loader at line 842 also has no control adapter. Thus even
   genuinely approved, completed control fits cannot be carried into the planned
   matched reserved-site comparison. Two CPU interface probes reproduce the
   missing artifact support. Add a separately versioned typed adapter, source
   closure, control/scaler ancestry validation and CPU replay coverage. Preserve
   the distinct control names and zero-SSL provenance; relabelling them as
   `cf_jepa` is not a solution.

2. **P1: prefix adaptation cannot express the matched CF alternatives.**
   `native_prefix_desktop_transfer_v3.py:133` requires methods from the original
   encoder registry and equates CF family with `method == cf_jepa`; line 139
   explicitly excludes CF scratch. Both `cf_random_frozen/frozen_readout` and
   `direct/scratch_direct` with `family=cf` fail configuration validation. The
   parent loader additionally expects original encoder kinds and does not support
   the new CF control encoder kinds. Two CPU probes reproduce the configuration
   failures. The current trajectory is neural-only; it also supplies no fitted
   conventional-model prefix comparison. A separately versioned extension must
   preserve matched heads, allowed prefix labels, original TRAIN scalers and DEV
   selection while supporting these required alternatives. These are missing
   comparisons, not evidence that SSL wins.

3. **P1: no integrated evaluator scores the adapted checkpoints on the suffix.**
   The prefix fitter explicitly stops before suffix prediction
   (`native_prefix_desktop_transfer_v3.py:2141`) and produces
   `native_prefix_transfer_inference_v1`. Both existing assessment layers reject
   this artifact; two CPU interface probes reproduce that gap. More substantially,
   their ancestry contract requires every fitted input to be original TRAIN and
   rejects any reserved-deployment overlap (`native_assessment_replication.py:451`).
   This is correct for zero-shot evaluation and must remain intact. Adding a kind
   to an allowlist alone cannot implement adaptation assessment. A separate
   suffix evaluator must validate prefix-contained fitted ancestry and exact
   context/target disjointness, retain the common suffix, and save per-deployment
   predictions, masks, daily losses and reconstruction inputs. The reusable
   prefix predictor exists; the admitted end-to-end suffix scoring path does not.

These findings correct the earlier handoff's claim that no unfinished unblocked
engineering task remained. Synthetic implementation and verification of these
interfaces can proceed without real fitting or reserved numerical access.
Restoration of the independent review service remains a separate dependency.

## Checks and scope limits

The existing focused control suite passed **130 tests in 64.24 seconds**, exit 0,
with no skips. This includes all four ROOT CPU optimizer/resume cases, matching
initial encoder/head tensors at seeds 7/13/23, frozen buffers, saved prediction
replay, inverse transforms, budgets, admission and fixed-campaign summaries.
Fixtures are explicitly synthetic; these are not six real control fits.

The new requirement probes produced **six failures and one pass in 2.38 seconds**,
exit 1. The failures document the three missing interfaces above. The passing
probe confirms four-day context warmup, 1/7/30-day label-prefix boundaries and a
common suffix on day 41 for the tested calendar example. It does not validate
real reserved-source eligibility. Logs and commands are preserved in
`evidence/ssl-coordinator-review-v2/`; failing probes are evidence outside the
ordinary test collection and are not being represented as a green suite.

Within the inspected control path, encoder/head construction, H96 extraction,
supervised sampling, TRAIN scaling and target inverse transforms reuse the
original strong CF endpoint helpers. No additional discrepancy was found there.
Comparison preparation retains the 47-method/43-neural requirement and explicit
scaler aliases; aliases are not proof of embedded-scaler equality. No real saved
prediction or full-campaign numerical reconstruction was performed in this review.

No reviewer launch was attempted or replaced. No scientific approval was written.
No CUDA job, reserved numerical read, app, meeting, browser, release or cloud work
was performed. The scientific ledger stayed byte-identical at
`91bf4dcbedc6a603d019c28bb058549f2f39e95befc8d81de5ca6448dec2c5b0`:
aggregate 17.802355808369175/96 hours; Band 8.318836388888881/12 hours;
CF extension 0/12 hours. Scientific completion and independent review remain open.
