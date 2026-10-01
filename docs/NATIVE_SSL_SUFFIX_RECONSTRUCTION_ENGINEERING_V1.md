# Adapted-suffix numerical reconstruction preparation

Updated 2026-10-01 from `research/marine-jepa-vnext` at `51739b3`. The
`dd012f0` checkpoint remains a reference. Existing fits, inference implementations,
archives, approvals and scientific ledgers are preserved.

The V4 adaptation evaluator saves forecast NPZ files and daily scores, but the
historical comparison tool admits only `development` and `final_test`. An additive
`native_suffix_reconstruction_v1` module now admits the distinct `adapted_suffix`
role. It does not modify the historical comparison or zero-shot ancestry rules.

The new implementation calculates pinball loss using an independent branch-form
quantile loop and reconstructs equal source-date, horizon and deployment weighting.
It independently builds calendar blocks and reproduces the frozen seed 20260929,
2000-draw paired bootstrap using direct sampled-block addition. It does not call
the original scorer, block builder or bootstrap. It checks daily records, support,
primary scores, bootstrap sequence/counts and method-minus-reference intervals
against the saved completion using a fixed absolute floating tolerance of 1e-10.
Discrete support and labels must match exactly. Unsupported horizons or zero
calendar span retain null intervals.

Before NPZ decoding, admission requires a genuinely distinct reconstruction
review with status `APPROVED_PREFIX_SUFFIX_RECONSTRUCTION`, role `adapted_suffix`
and scope/use `saved_adapted_suffix_reconstruction`. Exact hashes bind the new
source, imported validation dependencies, reconstruction manifest, original
assessment/completion, original execution/software/access receipts, prospective
selection freeze and every saved forecast. The original receipt chain and
freeze-before-access timestamps are retained. Prefix/suffix rows and source
intervals must remain disjoint. Saved rows, masks, observed truth, dates, cutoffs,
queries and native 0–230m geometry must agree across all declared cells; support
intersection or silent numerical correction is forbidden. Adapted cells retain
their 1/7/30-day budget and adaptation label; unchanged persistence/seasonal
controls retain their zero-shot label.

Synthetic admission is confined to newly created private fixtures under
`evidence/ssl-suffix-reconstruction-v1/SYNTHETIC_CORRECTNESS_ONLY-*`. Their clearly
fictional receipt identities exercise the checks and are never scientific
approvals. No real prefix checkpoint, acoustic archive, saved real forecast,
reserved numerical value or GPU was consumed in this delivery.

The final owned CPU run passed **26 tests with no skips**. Coverage includes
independent daily reconstruction with masked NaNs and unequal day counts,
canonical row reordering, two-deployment unequal-support bootstrap draws,
identical-method zero intervals, incomplete/zero-span support, stale/self/wrong
role approvals rejected before decoding, damaged daily/paired metrics, receipt
lineage, adaptation labels, duplicate rows, cutoff mismatch, original access and
freeze checks, partition leakage and wrong native geometry. The first import
failure and a later NaN-comparison failure are retained; the latter exposed and
fixed acceptance of a reconstructed NaN against a finite expected scalar.

Receipts and exact commands are in
`evidence/ssl-transfer-integration-root-v1/suffix-reconstruction-{red-04,green-05,hardening-red-06,green-07,final-08,lint-09,format-10}.{json,log}`.
The final test supervisor enforced 600 seconds and less than 22GiB owned RAM,
including its launcher, and verified owned cleanup. Scientific ledger bytes
remained `91bf4dcbedc6a603d019c28bb058549f2f39e95befc8d81de5ca6448dec2c5b0`.
Private generated fixtures remain untracked and preserved.

This is author-validated numerical software, not a distinct scientific review,
checkpoint replay, real adaptation result or completion of the research campaign.
No standalone real invocation is currently eligible. Once genuine approvals and
actual assessment artifacts exist, a coordinator must execute this module under
an approved bounded CPU launcher and retain its full owned receipt; the module's
CLI alone does not supply whole-process resource supervision. It never loads
weights or refits/inverts scalers: those operations belong to the separately
reviewed assessment executor and subsequent declared weight replay.

Original science remains 40/43 neural endpoints, 47 required campaign methods,
28 independently reconstructed methods and five newer provisional scores.
Three Band fits and six CF control fits remain NOT_RUN; no finalists have been
frozen and reserved/prefix studies remain NOT_RUN. Aggregate usage remains
17.802355808369175/96 GPU-hours and Band 8.318836388888881/12, including failures;
78.19764419163083 aggregate hours and 3.681163611111119 Band hours remain. The CF
extension remains 0/12 inside the aggregate allowance, with at least 12 aggregate
hours reserved for evaluation. CPU correctness work is not GPU fitting.

The recorded service rejection still precedes reviewer inspection. It has not
been retried, rerouted or reclassified as scientific approval. Supported owner
feedback and administrator/service inspection are documented in
`NATIVE_SSL_REVIEW_SERVICE_DEPENDENCY_V1.md`. App, meeting, browser and release
work remain frozen.
