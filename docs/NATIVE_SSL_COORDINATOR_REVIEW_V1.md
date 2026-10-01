# Coordinator review of the pending native SSL replication jobs

The coordinator completed this review directly at the owner's request on
2026-10-01, without a subagent. It covers commit
`dd012f05dce3f18a4f20aeda3b545b0923f7274a`, the three pending downstream
proposals, their execution wrapper, shared temporal model, training paths,
and relevant inference and approval transport checks.

This is an author/coordinator review. It is not an independent scientific
approval and cannot grant the distinct prefit approval required by ADR0015
and ADR0018. No approval leaf was created, no scientific fit was launched,
and no final-test values were opened.

## Findings

1. **Medium: the preserved v2 proposal snapshots fail current source admission.**
   The default source-closure test selects
   `orchestration/native_band_replication_admission_v2.json`. All four leaves
   omit the now-required `src/marine_echo/inference/native_latent.py` binding.
   The first test run therefore returned exit 1 with 186 passes and four
   failures. The explicit v3 source-closure run returned exit 0 with four
   passes. The three currently pending proposals also passed their own full
   source admission audit. Historical v2 approvals must remain immutable;
   they cannot authorize a fresh job against current sources. Keep the
   default-test mismatch visible until a separately versioned test-selection
   change is made. This finding does not establish that historical fitted
   models used an unreviewed implementation.

2. **Blocking prerequisite: the three fresh fits lack distinct prefit approval.**
   Their approved-review destination files are absent. The actual production
   identity guard rejects a coordinator session used as the reviewer. The
   owner-authorized budget and this review do not satisfy that scientific
   gate. This is an unmet approval prerequisite, not a discovered model defect.

No additional defect was found within the inspected current proposal and
execution scope. That statement is limited to the checks below and does not
assert overall scientific completion, independent review, or SOTA.

## Completed checks

The read-only audit verified 77 unique raw-file SHA-256 bindings across the
three proposals and their parent review. It checked exact configuration and
runtime serialization, fixed recipes, source closure, TRAIN/development
cohort identities, disjoint deployment/archive roles, untouched destination
paths, and no resume or fitted ancestor for the scratch supervised retry.

Both seed-23 transfer proposals bind the actual completed SSL parent, its
original distinct review, selected encoder, inference artifact, membership,
TRAIN scalers and successful owned-process receipt. The parent membership
contains 18,593 TRAIN rows; every deployment/archive pair matches the reserved
TRAIN role. This audit hashed fitted artifacts but did not decode their tensor
payloads or raw acoustic arrays. It did not rerun the earlier 40-model tensor
inventory or independently reconstruct numerical predictions.

The frozen head has 2,000 updates and checkpoints every 500 updates; full
finetuning and scratch supervision have 3,000 updates and checkpoints every
750. The inspected implementation uses fresh seed+100000 heads, AdamW,
10% warmup followed by cosine decay to a 10% floor, gradient clipping,
and earliest strict improvement on development daily pinball. Only the
readout trains in frozen mode; the runner verifies that encoder parameters
and buffers remain unchanged. Transfer and scratch paths use the same
supervised sampler policy for a matched seed.

CPU synthetic tests exercised actual optimizer gradients, interrupted/resumed
training equivalence, frozen encoder identity, supervised encoder changes,
inference contracts, ancestry and source rejection before numerical decoding,
review transport, SSL model and latent behavior. These are correctness tests,
not training on public data or measurements of representation transfer.

Across three test commands, **260 checks passed and four historical v2 checks
failed**. The full logs are preserved, including the failed command. No
failure was hidden by replacing an approval or changing a model source.

The ledger remains closed and byte-identical at SHA-256
`91bf4dcbedc6a603d019c28bb058549f2f39e95befc8d81de5ca6448dec2c5b0`.
It accounts for 17.8023558084 aggregate GPU-hours and 8.3188363889 Band
GPU-hours; 3.6811636111 Band hours remain inside the existing limits.
This remaining budget is capacity, not a forecast that every missing run
will finish within it.

## Scientific disposition

The programme remains incomplete. This review adds no numerical result.
The independently reconstructed 28-method development comparison still
does not establish a JEPA transfer advantage or SOTA. The three proposed fits,
expanded numerical reconstruction, final freeze/access, and external transfer
evaluation retain their existing incomplete states. App/release work and
historical evidence remain preserved.

Machine-readable checks, exact bindings, commands, exit statuses and logs are
in `evidence/ssl-coordinator-review-v1/`. The review is complete within this
scope; independent approval remains a separate requirement.
