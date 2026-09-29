# ADR 0015: native acoustic self-supervised research study

Date: 2026-09-29. Status: activated by the owner's model-first instruction;
numeric materialization and fitting require distinct prefit approval.
Study: `native_acoustic_ssl_v1`. This is a separately versioned study. Historical
protocols, metrics, reviews, ledgers and the approved r3 archive are immutable.
App and release work is frozen. Meeting-only and diagnostic-only priorities do
not apply to this study. Safety and paid-resource approvals remain in force.

The target is the publisher's own hourly 38 kHz integrated FullDepth Sv at
1, 3 and 6 source intervals. Bounds, frequency, interval duration, masks,
configuration, clock uncertainty and processing uncertainty are part of the
observation and query. A native 0–230 m product remains 0–230 m. No rescaling,
depth reconstruction, absolute calibration, species or biological claim follows.

## Reservation before numerical selection

Reserve all available AEON2 deployments for final whole-site assessment. Reserve
AEON4 December 2022–December 2023 for development. Train on the two earlier
AEON4 deployments, prior-year AEON3, and only the original AEON3 TRAIN dates.
This deterministic policy uses site/deployment/time metadata, never scores.
Do not initialize from historical fitted weights or preprocessing. Start all
new models and scalers afresh. Prior-year AEON3 is training-exposed for expanded
ancestors and descendants; it cannot serve as their external test.
Original AEON3 VAL/CAL/TEST remain excluded from new fitting and selection.

The additional sources were previously inventoried through metadata only in
the recorded scaling scout. The AEON2 2024–2025 source underwent a historical
external compatibility investigation; it is reserved, but no universal sealed
claim is made. Source exposure is recorded separately from new numerical access.
The primary claim is conditioned-product transfer to the observed reserved site,
subject to these exposure limitations and the small number of deployments.

Each sample stays inside one deployment and one contiguous unchanged geometry,
processing and ping-configuration segment. Context has 96 raw intervals; the
24-interval comparison uses its suffix on identical support. Future SSL blocks
start at cutoff+1/+3/+6 and contain four intervals. They are disjoint from
context; different horizons' future blocks may overlap each other. No crossing
of gaps, clock discontinuities or configuration boundaries. Source timestamp
timezone is unknown; use relative intervals, never fabricate UTC availability.
Targets do not affect forecast issuance. Assessment masks and missing target
values affect scoring coverage only. Training crop eligibility is separate.

Complete-interval ping counts are source metadata: 240/180/150 as recorded by
the scout; AEON2's documented multiple configurations are kept as distinct
segments, with only metadata-declared counts admitted after review. Valid weak
Sv values stay observed. Publisher sentinel/nonfinite values and partial
intervals are masked; threshold settings are provenance, not a fabricated
censoring indicator. Absolute censoring status is unknown where unpublished.
Channel absence is explicit; richer profiles do not gate integrated training.

## Model and comparison contract

Candidate A follows pinned WDSLab/CF-JEPA author code, with a shared multi-scale
temporal convolutional encoder, forward crops, EMA targets, and EMA forecasting
features. Every marine adaptation is documented. Candidate B uses shared
patch4 temporal processing (width192, four blocks, latent64 initially), a
horizon/query predictor, no EMA or target detachment, MSE prediction and
SIGReg on encoded context and future observations. It does not use the
historical flattened position-specific MLP. Latents remain unconstrained,
rather than unit-normalized in conflict with Gaussian SIGReg.

Compare matched direct supervision, masked-reconstruction SSL, random frozen
features, pairing-permuted SSL, persistence, source-period seasonal reference,
LightGBM quantiles and pinned Chronos-2 zero-shot with identical history/support.
Frozen encoders plus the same lightweight quantile head are the primary
representation comparison; fine-tuning is reported separately. Paired methods
share initial encoder/head tensors, sample indices and target multisets.
Pretraining compute is extra and is measured, not equated with step counts.

## Fixed finite development and assessment policy

Initial seed7/history96/latent64 screen: 6000 SSL updates, 2000 readout updates,
batch64, AdamW peak LR0.0003, weight decay0.0001, gradient clip1, SIGReg0.03.
Use 10% linear warmup then cosine decay to 10% of peak. Pretraining checkpoints
every500; readout checkpoints every250; selection uses development daily
pinball and patience4 with identical opportunities. Each pretraining candidate
checkpoint is compared through an identically initialized train-only frozen
readout; no development gradients. Declare exact implementation policy in a
hash-bound runner config before fitting. No unchanged 30k extension.
History24 and latent128 are predefined possible screens, at most eight seed7
configuration screens across A/B, at most one documented hypothesis revision.
At most two SSL finalists and matched direct expand to seeds7/13/23. Ceilings
per trajectory: 50000 SSL, 5000 supervised/readout. Aggregate local GPU time
96 hours including controls/baselines, one GPU process, reserved/allocated
memory below10GiB and RSS below22GiB. Cloud remains disabled and local spend0.

Primary score averages five-quantile pinball equally over eligible target source
dates, horizons and deployments. Quantiles: .05/.25/.5/.75/.95. Minimum18
scored anchors per date/horizon. Use shared support and report coverage.
Skill is 1-model_loss/max(persistence_loss,0.01dB) per deployment, with the
denominator rule fixed now. Paired 48-hour contiguous-time block bootstrap
2000 draws/seed20260929; seed variance separate. Intervals are conditional on
these few deployments, not hourly rows as independent sites.
Success target: at least5% improvement over strongest development-selected
non-JEPA reference, paired95% loss-difference interval below0, and no deployment
or horizon regression above10%; useful frozen representations must beat relevant
random/permuted controls. These rules do not establish field-wide SOTA.

Freeze finalists, checkpoint/scaler hashes and inference code before final
numerical assessment. No final outcome tuning or calibration. Adaptation,
if supported, uses disjoint1/7/30-day prefixes followed by a7-day gap and a
common later suffix beginning after the largest prefix and gap; scratch models
receive identical targets. Zero-shot and adapted models are distinct.
Robustness drops past observations on the same held-out targets. All metrics
must be reconstructed from row artifacts by a distinct reviewer.

## Authorization and evidence

The owner's current instruction authorizes local training and this new native
contract. Prefit review approves split, reader, objectives, selection and hash
bindings; result reconstruction and model-package/claim review follow.
Weights, latent predictor, frozen and fine-tuned readouts, safe reusable
inference, relocation replay, comparisons, controls, model card and report are
required. Scientific failure is retained honestly; an app is not a substitute.
