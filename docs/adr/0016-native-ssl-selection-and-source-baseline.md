# ADR 0016: finite readout budget and source-faithful baseline

Date: 2026-09-30. Status: fixed before any new numerical model selection;
requires distinct prefit review together with exact executable configurations.
Study: `native_acoustic_ssl_v1`. This adds to ADR0015 without modifying its
approved data reservation, historical protocols or evidence. Train/development
materialization has completed; no numerical model score or scientific fit has
occurred when this decision is recorded. The original ADR0015 nominal schedule
was a proposal, not an executed or approved fitting recipe.

Freshly fitting a 2000-update readout at every encoder checkpoint would exceed
the mission's 5000 supervised/readout-update ceiling for a joint trajectory.
Use four matched encoder-selection opportunities instead:

- Shared, masked-reconstruction, pairing-permuted and CF SSL screens: at most
  6000 pretraining updates; encoder checkpoints every1500 updates.
- Matched direct feature screen: at most3000 supervised backbone updates;
  encoder checkpoints every750 updates.
- Each checkpoint gets a freshly and identically initialized frozen encoder
  quantile readout, at most500 TRAIN updates, selection checkpoints every250.
  All methods use the same readout sampling sequence and supervision.
- Each SSL screen therefore fits at most2000 supervised readout updates; the
  direct joint screen fits at most3000+2000=5000 supervised updates. A frozen
  random encoder requires one identical readout probe because its encoder is
  unchanged; repeating the same deterministic fit supplies no new opportunity.

Report total measured time and all probe updates. Checkpoint probes are
development assessments, not independent trials. Their head fits never update
the encoder. Warmup/cosine schedules remain10% warmup and decay to10% of peak;
AdamW LR0.0003/weight decay0.0001, gradient clip1, batch64, patience4, and the
predeclared daily metric are unchanged for the shared route.

After feature selection, run a separately accounted downstream-only2000-update
frozen readout trajectory for each selected SSL/direct/random/control encoder,
with the same head initializer, labels, batches, schedule and development
selection opportunities. Encoders remain frozen. Report this separately from
the short selection probes and from full fine-tuning. These trajectories have
their own finite ceiling and artifact ancestry; do not hide their compute or
merge them into an apparent5000-update joint fit. Final held-out comparison
requires these strong frozen readouts and matched end-to-end supervision,
three finalist seeds, controls and all mandatory conventional references.

The pinned CF-JEPA baseline must retain the actual author shared multi-scale
temporal convolution architecture, three future zones and random crop design,
normalized L1 prediction, variance/covariance/crop invariance losses, horizon
annealing, parameter-only cosine EMA and the author eval-target BatchNorm buffer
behavior. Its author128 configuration uses hidden256/output128/depth5, four
crops, crop fractions.310–.686, EMA base.992, loss coefficients
.038/.143/.401, LR.000340 and weight decay.05. Source pin:
`WDSLab/CF-JEPA@5d3d2fd1273c283fbfa03249c078619245e84033`.
Do not replace these crop zones with fixed tiny scalar-horizon blocks and call
the result source-faithful. Train trajectories can be assembled from each
sample's96 past intervals plus unique later intervals1–9, without concatenating
deployments or duplicating overlapping future-block intervals. Crop contexts
and their future zones remain disjoint. Only TRAIN trajectories enter SSL.

Native masks/metadata and quantile forecasting readouts are explicit marine
adaptations. Symmetric source training convolutions operate on allowed TRAIN
crops; assessment receives only its permitted observed prefix, so no unknown
future values enter inference. Forecasting uses EMA features as in the author
route. Source-core correctness and deviation documentation are required before
its own fit. A provisional CF implementation that changes zones or core
dimensions is not eligible for the source-faithful baseline claim.

The first eligible shared SSL job may proceed after its own exact prefit review
while the independent CF baseline extension is completed. Approval for one
route never approves a different objective/source revision. Freeze code for
each fit and save exact source/config/split hashes with its weights. Keep the
96 aggregate GPU-hour budget, maximum eight seed7 configuration screens,
three seeds7/13/23 for finalists and direct reference, one bounded hypothesis
revision, one GPU owner and all memory/payment restrictions unchanged.
