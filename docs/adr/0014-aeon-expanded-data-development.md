# ADR 0014: post-hoc AEON3 expanded TRAIN development

Date: 2026-09-28. Status: PROPOSED, requiring distinct numeric-cohort and
prefit reviews. Study: `aeon3_geb_expanded_train_3k_development_v1`. This
amends neither the original AEON campaign nor the separately frozen 30k
schedule. The 2023–24 AEON3 outcomes have already been inspected in the
descriptive transfer study. Once used for fitting, they cease to be external
evaluation data for any model derived from this study. The original 2024–25
CAL and retrospective TEST are unavailable to fitting or selection and are
not a new sealed holdout.

## Sources, target and partition

Use only the publisher AEON3 GEB February 2023–February 2024 conditioned
`Sv_mean` ZIP (Figshare article 29247113 version 2, file 61937275,
SHA-256 `b6d8380ed986c91d565ea7d669761ff8f66da8f2b5cd3e794fbf71c040bf1d37`,
filename serial identifier `55146`) and the original AEON3 GEB March
2024–March 2025 ZIP (file 61937281,
SHA-256 `4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde`,
filename serial identifier `55144`). Publisher license is CC BY 4.0.
Neither filename serial is an independently verified physical calibration
chain. The target is the 38 kHz 0–200 m, 60-minute FullDepth publisher-
conditioned `Sv_mean` in source-reported dB re 1 m^-1; no biomass, species,
causal or independently calibrated backscatter claim follows.

Apply identical 150-ping complete-source-interval, 0–200 m geometry,
finite/non-special `Sv_mean`, four-channel past-mask, 24 preceding interval
and 1/3/6 source-interval target rules. Retain source timestamps with
unknown timezone and never bridge context, target or pretraining views
across deployments. The prior archive's independently reviewed Stage-1
metadata candidate report is bound by hash; the numeric reader must
reconstruct its candidate universe exactly before issuing windows. New
TRAIN consists of all QC-issued prior-year same-site windows plus only the
original 2024–25 TRAIN dates (2024-03-06 through 2024-10-07). Original
2024–25 validation dates (2024-10-08 through 2024-11-30) remain the sole
development assessment. Original CAL and TEST are never reader partitions.
New TRAIN row identities bind source archive SHA, publisher file ID,
filename serial identifier, license, source role and interval ID. Retain
original validation row identities for exact comparison with its corrected
3k rescore; bind all source metadata in the new cohort digest. Record source-
specific issued, scored and rejected-candidate counts and zero raw interval
or row overlap with validation. These sources gave 8,507 prior-year issued
rows and 4,965 original TRAIN rows in their historical separate studies,
but 13,472 and 2.71x are candidate arithmetic, not a new verified cohort
count. No 3x or 5x claim is authorized without this new QC.

## Sampling, fit and comparison

Use natural pooled sampling proportional to eligible TRAIN window counts,
not source-balanced oversampling. For EMA pretraining, choose a source per
batch in proportion to its window count and sample the entire batch from
that source. This prevents cross-deployment SSL views and batch regularizer
mixing while preserving natural expected source proportions. Supervised
batches sample pooled labelled TRAIN rows. Count sampled windows by source
in both phases. Fit input and target scalers solely on joint TRAIN.

Train the original width-128, 3-layer direct quantile model for 3,000
supervised optimizer updates and the matched EMA-JEPA model for 1,500
pretraining plus 1,500 supervised updates, seed 7, batch 64, AdamW
learning rate 0.0003, weight decay 0.0001, gradient clip 1, EMA teacher
momentum 0.996 and EMA regularizer weight 0.03. Save fixed final endpoints;
intermediate checks are diagnostic only. Compare each expanded seed-7 final
validation score to the **same-family seed-7 original-only 3k final** from
the independently reviewed corrected rescore, never to the 30k schedule.
This is an expanded multi-deployment TRAIN intervention with specified SSL
sampling, not a causal test of data volume alone: instrument identifier,
calendar period and possibly source processing also change. The 2024
validation primary uses target-source
dates with at least 18 scored anchors per horizon and equal quantile/date/
horizon averaging. Bind exact validation row support and eligible-date
hashes. A single validation comparison is post-hoc development, not a
generalization test or JEPA success gate.

## Access and resource gates

Before reading prior-year numeric CSV rows in this new study, a distinct
reviewer must bind the config, ADR, source ZIP hashes, metadata-report hash,
split review and all reader/cohort code hashes. Stage 1 then persists a
cohort report with actual source-specific counts, new row IDs, source
metadata and exact cohort digest. Before fitting, a second distinct prefit
review must bind that report/digest, all model/metric code hashes, original
3k corrected rescore and its outcome review. Stage 2 trains a single local
process with process RSS below 22 GiB and GPU reserved memory below 10 GiB.
No cloud or paid resource is used. An independent outcome review must
reconstruct fit provenance, final predictions and metrics before any claim.
The original 30k code, config, ADR, outputs and approvals are not modified.
