# AEON external transfer: secondary numeric outcome

Study `aeon_external_transfer_20260928_v1` is a post-hoc-initiated,
non-sealed zero-shot transfer exercise. The primary contemporaneous cross-site
AEON2 archive has zero metadata candidates: all 8,682 reported 38 kHz layers
are 0–230 m, while the preaccess target is 0–200 m. Its acoustic numeric rows
were not opened, its forecasts were not run, and its primary JEPA-value gate is
`NOT_EVALUATED_METADATA_INELIGIBLE`. The original AEON3 2024–25 retrospective
result and historical MOSAiC v1/v2 findings remain unchanged.

The distinct remedial preaccess review
`orchestration/reviews/AEON_EXTERNAL_STAGE2_RETRY_REVIEW_20260928.json`
(SHA-256 `c42904fd03ade88130ddce78372a01282ed9ead71d430e4929da8de931c92094`)
authorized one exact secondary-only retry after the preserved first-attempt
timestamp-representation failure. The retry CLI exited zero. Its manifest is
`outputs/aeon_external_transfer_20260928_v1/zero_shot_secondary/manifest.json`
(SHA-256 `f33307febf0848f8287529706ce3303213e5a1d6278d44f744c2d0367f1f8a33`).
The secondary score JSON is
`secondary_prior_year_same_site/score.json` beneath that directory (SHA-256
`da0309e50f6d7dd3e391a5bc16087f8977d44581f3419a8172e376097a8b5d04`).
The issued-row archive SHA-256 is
`071acdbb3bac028f1e4fff4ee93be419569dcca507317a16b4aa06617a9b512a`;
the direct and EMA forecast archive digests are respectively
`8337fde57943d2246323f5c808dc2745e4d57e2937418874c187f3e421b5b740`
and `0f930a3dbfc4779d2ea621b5f2a39980a8760b20649e80d435224171e66e043d`.

The runner reports that all 8,507 Stage 1 candidates were issued, with 353
eligible target-source dates at each of the 1, 3 and 6 source-interval
horizons. The frozen direct ensemble's raw daily mean pinball loss is
0.6701783099148351 dB; the frozen EMA-JEPA ensemble's is
0.6789216779097978 dB. Relative loss reduction is -0.013046330902702402.
The paired 2,000-draw 95% EMA-minus-direct interval is
[0.0022527825173525597, 0.015516133047404537] dB. On this secondary
same-site prior-year cohort, the reported comparison is negative for JEPA.
It is descriptive and cannot rescue or establish the unexecuted primary
cross-site comparison.

The distinct reviewer independently reconstructed all 8,846 source slots,
issuance, QC, targets, every metric and bootstrap draw, and replayed both
8,507 by 3 by 5 forecast arrays bit-exact from the six frozen checkpoints.
The outcome review is
`orchestration/reviews/AEON_EXTERNAL_STAGE2_OUTCOME_REVIEW_20260928.json`
(SHA-256 `ec1ad3ecc437d28f47c53063d6abc6c8f9efe08c36a0f39dec4bda8ef9ad255d`).
Its approval covers only this negative descriptive secondary comparison; the
cross-site primary remains NOT_EVALUATED.

No external archive was used for training, calibration, model selection or
site correction. The measured field is source-conditioned `Sv_mean`, with
unresolved clock zone and processing provenance; no species, biomass, catch,
fuel or causal biological conclusion follows.
