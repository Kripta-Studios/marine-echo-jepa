# ADR 0011: AEON reviewed development comparison choices

Date: 2026-09-27. Status: PROPOSED FOR INDEPENDENT SELECTION REVIEW.
Study: `aeon3_geb_2024_hourly_sv_v1`. This records a post-hoc development
selection after the core, hybrid, LightGBM, forward-JEPA and Chronos-2
validation results were opened and after the neutral five-source comparison
was independently reconstructed. It is not a prospective performance claim.
The exact machine-readable model/component freeze and final evaluator require
their own distinct approvals before numerical CAL or TEST access.

## Evidence used

The corrected same-support validation comparison is
`outputs/aeon3_geb_2024_hourly_sv_v1/validation_comparison/comparison.json`
(SHA-256 `f4a0f4771868866ff7c330f4033c5ca9da7c6c36cdb19a1758cf2f383146757b`).
The independent outcome review is
`orchestration/reviews/AEON_VALIDATION_COMPARISON_OUTCOME_REVIEW_20260927.json`
(SHA-256 `53a3a164915eea4ab3bc8851c096bbc42821cb05330eb4c2829039df83df3e07`).
The late seed-ensemble rule and its review are ADR 0010 (SHA recorded in its
review) and `AEON_DEVELOPMENT_SELECTION_RULE_REVIEW_20260927.json` (SHA-256
`dd3dcbdf5e93d42e021abbaf4bf310bec0b40ea52d53411802e00770840127b2`).
All values below use 50 eligible source dates per horizon and the fixed
>=18-anchor/day metric; source dates have an unknown clock basis.

## Frozen choices for the remaining study

| Role | Chosen reviewed family | Validation pinball dB | Exact serialized components |
|---|---|---:|---|
| Core conventional reference | Direct-neural, equal forecast average of seeds 7/13/23, followed by common quantile rearrangement | 0.6390617418 | `core_campaign/direct_seed7/checkpoint-supervised-3000.pt` `a616f9160f359e54a6da5d87dc3b7a46edfee46e449e2944af7a69be8ae25c95`; seed13 `eb6ca62f4c68024b43ea8b2102a0754d646623fd91cbf1ad31fb72218faf48dc`; seed23 `ee3ae09ebf05bc41ec762e946a1d0046be33fff4ea47472a992893a294054761`. |
| Core JEPA comparator | EMA-JEPA, equal forecast average of seeds 7/13/23, followed by the same rearrangement | 0.6421704481 | `core_campaign/ema_jepa_seed7/checkpoint-supervised-1500.pt` `f1fc41860e21cb2c47868488b051c68cd951487d717fc0fe033e63b8dfcc664c`; seed13 `59cb3d1e6d79ca2d9807f24c5e816d0e5c89fad0e274e703d6e9da5b44024633`; seed23 `4a7c63931a31a48f60323b944110df9292856f39cb822ce0b891371a10a16a15`. |
| Exploratory post-hoc operational candidate | LightGBM 15-head quantile model | 0.6384597245 | `sota_supervised_lightgbm/model.joblib` `164389731cd5d2e0694d1fb5228196196e7701fc73269e79ef84d85520fb168b`; fixed recipe SHA-256 `3feee4089ce790c66adb189ff82ad4e006c350822a23aaa494e86afaef884eea`. |

All six neural ensemble weights are exactly one third within their respective
three-component ensembles; no seed is discarded or selected by its score.
These artifact paths are relative to the AEON study output directory above.
The machine-readable freeze must additionally bind the checkpoint/recipe
files, predictor code and canonical component order by their full SHA-256
values and reject any adapter plan with substituted components.

The core conventional reference beats the core JEPA choice by 0.0031087063
dB on validation. Its paired 48-hour-block interval for EMA-minus-direct
includes zero, so neither incremental JEPA value nor a benefit is established.
The exploratory LightGBM advantage over direct is only 0.0006020173 dB
(0.094%); its paired interval [-0.0097651129, 0.0087644153] dB includes
zero. Neither a significant gain nor SOTA is established. Core hybrid
ensembles score worse than core EMA, so the unsupported hybrid final adapter
does not change this exact selection. Forward EMA and Chronos-2 remain
reported development comparators, not final selected arms. The frozen-random
hybrid control failure continues to limit JEPA attribution regardless of any
later score.

CAL may adjust only the selected models' 90% interval endpoints using the
reviewed nonnegative widening rule. Retrospective TEST acoustic values remain
closed until an independent reviewer approves the exact model/component freeze,
CAL adjustment, already frozen metadata-only candidate universe, one-pass
reader/adapters/metric/bootstrap code and pretest access graph. The metadata
inventory has 1,216 candidate cutoffs, not issued or scored rows; its review
does not establish 20 eligible numerical TEST dates. A TEST comparison, if
permitted, remains retrospective and not sealed. No model, threshold, horizon
or source target may be revised from CAL or TEST outcomes.
