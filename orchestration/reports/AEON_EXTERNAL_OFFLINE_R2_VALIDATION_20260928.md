# AEON external-transfer offline candidate r2 validation

The independent source re-review approved integrated commit `e921511` for a
new candidate only (`orchestration/reviews/AEON_EXTERNAL_APP_SOURCE_REVIEW_V2_20260928.json`,
SHA-256 `6056ee55ffa2769aa7dae5d80a78c96fd740d8f584e751ef64616217c9fef0f1`).
The earlier r1 candidate remains unapproved and preserved locally. Candidate
r2 was built with `marine_echo.serving.aeon_portable --external-transfer`,
the exact reviewed retrospective TEST/CAL inputs, the new independently
reviewed secondary outcome, the production web build and locally verified
r4 offline wheelhouse. Build exit code: 0. Its separate directory is
`release/aeon-offline-external-transfer-20260928-r2`; the ZIP SHA-256 is
`f0a529ef7c3353ef880af04af3ea2106baa68cdfbfd9173d86e20c7e44b3a40d`.
The adjacent sidecar matches. Final package approval is pending a distinct
exact-archive review.

The package verifier passed all 86 files in the built directory. An independent
ZIP extraction to `outputs/aeon external r2 validation/` had 87 entries
(86 files plus checksum manifest) and passed the same 86-file verifier.
The relocated copy created a Python 3.12.13 environment from an already
installed local interpreter and installed 13 bundled packages with
`uv pip install --offline --no-index`. Its own launch script verified the
package again and served only `127.0.0.1:8768`. The local API returned the
primary `NOT_EVALUATED_METADATA_INELIGIBLE`, secondary 8,507 issued rows,
EMA relative loss reduction -0.013046330902702402 and the unchanged
historical retrospective replay (12 rows at offset 12). Root returned 200.

A Chromium smoke from the relocated server checked the external section's
zero primary candidates, 8,507 secondary issued cutoffs, direct 0.6702 dB,
EMA-JEPA 0.6789 dB, -1.30% relative reduction and absence of a cross-site
model result. The old retrospective replay also switched to EMA-JEPA and
advanced to rows 13–24 of 1,216, returning 12 rows. There were zero browser
page errors. Full-page screenshot:
`outputs/aeon-r2-browser.png`, SHA-256
`b7a32d670fe5484fe14057718d31cae9cf272c631552b9e4fe977725320ac4e7`.
The first smoke command had an ambiguous test locator matching both an outer
and inner section; the corrected locator passed without changing the app.

This package records a negative descriptive same-site prior-year JEPA
comparison. The contemporaneous primary is metadata-ineligible, with no
cross-site numeric result. Historical v1/v2 and r4 evidence remain unchanged.
No live forecasts, biological inference, production integration, cloud spend
or paid credential was involved.
