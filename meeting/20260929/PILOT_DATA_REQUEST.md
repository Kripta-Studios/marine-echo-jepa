# Proposed pilot data request

This is a discussion document, not an acquisition authorization or implemented Marine integration. Request an acoustic/data owner, an operational decision owner, the current predictor/output definition and permission to examine a small time-complete sample. No company capabilities, data availability or expected savings are assumed.

## Minimum feasibility export

Propose one or two complete device/deployment sequences covering roughly seven consecutive days, including low-signal periods and transitions. This small sample tests schemas, missingness and time semantics only; it cannot establish predictive or business value. If permitted and feasible, the later predictive pilot could use the existing [transfer request's](../../docs/17_MARINE_TRANSFER_DATA_REQUEST.md) proposed 10–20 deployment histories with about 30 continuous days where possible. Counts and success rules require agreement before outcomes are opened.

| Required field group | Definition needed |
| --- | --- |
| Complete sequential history | All scheduled intervals and actual observations in order, including weak, zero, invalid, missing and retransmitted records; explicit observation/support masks, QC reasons and missing-interval ledger. A valid zero must remain distinct from invalid or unavailable. |
| Values and units | Numeric values, unit/reference, linear versus logarithmic scale, aggregation operator, weighting, denominator, support count and interval duration; thresholds, filtering and saturation flags. Preserve pre/post-filter products when permitted. |
| Frequency and geometry | Actual frequencies, channel IDs, depth/range bounds and bin edges, instrument depth/orientation, beam and range/depth convention, layer geometry and changes. Frequency proximity does not establish equivalence. |
| Identity | Pseudonymous stable device, instrument, deployment and site identifiers; deployment start/end and documented device changes. |
| Time | Acquisition timestamp, interval start/end and sequence ID, timezone/clock correction, receipt time and actual availability to the predictor/operator, revisions/backfills, schedule and latency. Never infer availability from acquisition time. |
| Processing lineage | Calibration references/date and coefficients where shareable, firmware/configuration, processing algorithm and product versions; version-change intervals and documented noise/QC procedures. |
| Rights | Written permissions for private analysis, derived results, retention/deletion, client boundaries and permitted redistribution; specify restricted fields and named data custodian. |
| Comparator and decision | Existing model/version outputs with issue and availability times, abstentions and the actual operator decision/cost definition, if recorded and shareable. |

## Two different studies

**Publisher-product replication:** preserve the supplied product's target definition and aggregation exactly. The public AEON example is source-conditioned `Sv_mean`; it does not reconstruct independent field calibration. A **0–230 m aggregate cannot be cropped to 0–200 m** without depth-resolved measurements or sufficient aggregation information. Do not rename 38 kHz as a different channel, treat an algorithmic biomass output as raw backscatter, or turn device temperature into seawater temperature.

**Future raw/depth-resolved study:** separately request legally exportable acoustic arrays/raw records, range/depth geometry, calibration/configuration and processing information sufficient to construct and audit a new target. Freeze its target, cohort, timing and split before numerical access. This study needs a new protocol and permission; the public aggregate experiment does not validate it.

Optional location/motion, telemetry/battery, environmental covariates and independently measured catch/species labels should travel in separate tables with provenance, coverage and availability times. Missing catch is not zero catch. Labels are not prerequisites for acoustic-schema feasibility and do not establish biological validity by association.

## Proposed stages and decision criteria

1. **Feasibility:** audit complete sequences, identities, masks, units, geometry, permissions, timing and versions. Success means the target and information available at each decision time can be defined without guessing. Unresolved timing or geometry blocks forecasting claims.
2. **Private read-only replay:** agree a chronological future/unseen-deployment evaluation, training-only preprocessing, existing-model comparator, fixed uncertainty/abstention rules and operational cost metric before outcomes. Success criteria are proposals to freeze jointly, not inherited public-study scores.
3. **Prospective shadow pilot:** if separately authorized, record decisions, available inputs and comparator outputs without changing fleet/device control. Evaluate forecast utility and operator relevance before considering integration.

Species, biomass, catch, fuel savings and sensing-policy benefits each require their own targets and validation. Querying a device is sensing, not evidence of a biological intervention. No production access, contact, deployment or purchase is requested by this document.
