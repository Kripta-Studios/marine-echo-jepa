# Native acoustic SSL model study: closed development evidence

Status: development evidence snapshot, 2026-09-30. Held-out whole-site assessment,
prefix transfer and the band-conditioned study remain incomplete. This report
does not establish state of the art or a representation-transfer advantage.
The coordinator will retain this snapshot when writing the final model report.

## Scientific question and data

We study whether self-supervised temporal acoustic features transfer to native
integrated-product forecasting with fewer supervised labels. We train new
encoders from random initialization on the reserved TRAIN sources, then fit
fresh readouts and compare them with supervised and non-JEPA controls.

The native contract preserves measured geometry, frequency, processing status,
observed masks and source-interval provenance. A native 0–230m query represents
0–230m. We do not rename or numerically convert it to 0–200m. The source clock
has unknown timezone, and the recorded interval centres provide a nominal
60-minute calendar rather than verified UTC or exact publisher interval edges.

We reserved the AEON2 site, deployments61937278 and61937269, before new numerical
model selection. The AEON4 deployment61937266 supplies development selection.
Four earlier TRAIN sources supply18,593 issued contexts,18,312 eligible for
SSL. Development contains7,593 H96 contexts at native0–225m. Historical exposure
to compatibility metadata remains disclosed; the held-out numerical values
remain unopened at this snapshot. Every fitted ancestor must remain outside
its tests, including scaler, readout and prefix-adaptation ancestors.

Original approved r3 evidence and signed historical protocols remain intact.
The new study uses separately versioned protocols starting with ADR0015 and
the unchanged split `configs/native_ssl_split_v1.json`.

## Models and training

We adapted the authors' CF-JEPA implementation pinned at
[`WDSLab/CF-JEPA@5d3d2fd`](https://github.com/WDSLab/CF-JEPA/tree/5d3d2fd1273c283fbfa03249c078619245e84033).
The adaptation uses a multiscale temporal CNN,256-wide hidden features and
128-dimensional latents, five blocks, multiple forward crop views, three
ordinal future zones, normalized directional prediction, variance/covariance
and crop-invariance terms, and the source-defined EMA semantics. Native
acoustic windows and forecast queries make this a marine adaptation, not a
replication of the authors' complete published benchmark.

The compact shared candidate follows the shared-encoder predictive objective
and encoded-view regularization in the pinned
[`stable-worldmodel@6398811`](https://github.com/galilai-group/stable-worldmodel/tree/63988116d34cde56aea1240d5e58eb158ac67dc0).
It uses temporal patches of four intervals,192-wide features, four transformer
blocks and64 unconstrained latent coordinates. Both encoded views receive
gradients. Future-block prediction uses a query head and MSE plus0.03 SIGReg;
there is no EMA teacher, detached target or unit-normalized shared latent.
This is a temporal acoustic adaptation inspired by LeWorldModel, not its
environment or action-conditioned experimental setting.

The shared architecture's non-JEPA comparison masks25% of past observations.
The permuted-pair control preserves sampled target multisets while changing
pairing. Random frozen features receive zero encoder optimizer updates.
Supervised controls train fresh encoders and heads, or retain supervised
features under their distinct ancestry. None of these encoders uses the old
flattened MLP objective.

Screening uses6,000 SSL updates and four fresh500-update TRAIN readouts, with
development selection at scheduled checkpoints. Strong downstream comparisons
use a fresh2,000-update frozen readout or3,000 supervised updates for full
fine-tuning and scratch training. The latter expose four scheduled development
choices. Seed-controlled samples, fresh-head initialization, warmup and cosine
learning-rate decay replace the failed unchanged30,000-update recipe.

The shared encoder has2,424,448 parameters and its forecast head has5,189.
CF has412,800 encoder parameters and an18,565-parameter forecast head. Therefore
CF comparisons with shared supervised/masked controls confound objective,
encoder architecture and readout size. They cannot isolate the value of CF's
objective. Shared/control comparisons use the matched shared architecture.

## Closed seed7 development results

We score five quantiles at source offsets1/3/6. The evaluator averages eligible
source-date losses, then horizons, then deployments, requiring at least18
observations per date and horizon. It retains common issuance, observed masks,
targets and native queries, rejects nonfinite forecasts and avoids support
intersection or filling. The table reports pinball loss in dB; lower is better.

| Endpoint | Development pinball, dB |
| --- | ---: |
| LightGBM,15 quantile boosters | 0.505274 |
| Chronos2, local external pretrained comparator | 0.506916 |
| CF SSL, full fine-tuning | 0.541814 |
| CF SSL, frozen readout | 0.614422 |
| Masked SSL, full fine-tuning | 0.691806 |
| Shared scratch supervised, end-to-end | 0.765185 |
| Shared SSL, full fine-tuning | 0.772594 |
| Supervised shared features, frozen readout | 0.776118 |
| Shared SSL, frozen readout | 0.785715 |
| Masked SSL, frozen readout | 0.786842 |
| Random shared features, frozen readout | 0.812868 |
| Permuted shared features, frozen readout | 0.854690 |
| Persistence | 0.989176 |
| Seasonal24 | 1.031738 |

The distinct reviewer reconstructed all20 original endpoints,18,960 daily
losses and19 paired intervals. The reviewer also verified the bootstrap draw
sequence. Paired intervals use2,000 draws with seed20260929 and48 nominal
source-calendar hours per block, retaining gaps and draw multiplicity. The
development set contains one deployment, so these intervals describe
conditional development uncertainty rather than cross-site population effects.

The shared frozen-readout difference from random features is−0.027153dB,
95% interval[−0.062203,0.001335]. Its difference from masked SSL is−0.001127dB,
interval[−0.032181,0.029797], and from supervised frozen features is+0.009597dB,
interval[−0.017782,0.034273]. Its difference from permuted pairing is−0.068975dB,
interval[−0.092589,−0.049190]. These results show sensitivity to pairing but do
not establish the required representation advantage over strong alternatives.

Shared full fine-tuning trails masked full fine-tuning by0.080788dB,
interval[0.053367,0.108150]. CF full fine-tuning trails LightGBM by0.036540dB,
interval[0.028145,0.044476]. We do not infer SOTA from improvement over persistence.
Chronos2 has unknown external pretraining ancestry and cannot support a clean
local-ancestry held-out claim.

## Replications and the single band-conditioning revision

The executors completed CF seeds7/13/23. Frozen-readout development runner
scores are0.614422/0.668064/0.602785; full-fine-tuning scores are
0.541814/0.558057/0.576860. Shared scratch supervised seeds7/13/23 score
0.765185/0.741446/0.765869. We retain all seeds. The28-endpoint expanded
reconstruction requires admission and independent numerical verification;
these additional numbers currently have completed-run scope.

The first CF seed13 frozen attempt exited with native Windows code0xc000070a
after164.14 full-owned seconds, without a checkpoint or result. We preserved
the failed artifacts, charged the attempt, verified captured process cleanup
and reconciled only its dead-owner lock. A bounded CUDA smoke passed before
one fresh-output exact-recipe retry. The native failure's cause remains unknown.

ADR0020 proposes one parameter-free band-conditioning change: apply per-channel
GELU to acoustic-value and native-metadata features before observed-channel
averaging. Synthetic checks demonstrate cancellation in the original affine
averaging path, but do not establish the cause of its real performance gap.
The revision preserves architecture dimensions, parameter count, objective,
initialization order, samples and schedules. The owner authorized11 seed7
recipes including controls and at most12 additional full-owned Band GPU-hours
inside the existing96-hour programme limit. The independently admitted real
Band screening queue is active at this snapshot; its results are pending.

## Reusable inference and remaining evidence

The repository contains real pretrained encoder weights, frozen forecast
readouts and context-only inference. The new load-only latent interface uses
safe weights-only decoding, exact tensor/config/scaler guards and inaccessible
ancestral paths as metadata. Its encoder operation returns selected shared or
CF EMA features. CF ordinal-zone prediction uses the saved ONLINE encoder and
three saved predictors. Shared predictions return latent future-block
coordinates for native offsets1/3/6. These outputs do not represent acoustic
dB or depth profiles. FullH96 CF forward-zone inference lies outside its sampled
forward crop-view support; it does not constitute a trained rollout.

The distinct reviewer approved the latent source; root verified40 bindings.
Root89 CPU checks and actual synthetic weights replay from a Unicode disk path
passed for shared, CF and Band. Real64-context immutable-weight replay remains
under separate admission. Synthetic checks establish software correctness
within their cases, not representation quality or competitiveness.

The final report still requires whole-site native0–230m forecasts, fixed missing
secondary-channel controls and common-support held-out comparisons. The prefix
study also requires real1/7/30-day label fits with fixed source-calendar
boundaries, seven-day separation and a common later suffix. Prefix adaptations
become fitted ancestors and must remain separate from zero-shot claims. We must
freeze selected families, weights, scalers, seeds and evaluators before opening
held-out values, then obtain independent access and execution review.

We have not established JEPA value, field SOTA, biological causation, tuna
biomass, species, catch or fuel benefits. These public experiments do not
validate Marine production integration. App and release work remain frozen.

## Evidence files

- `evidence/ssl-research-v1/development-comparison-summary-v2.json`
- `evidence/ssl-research-v1/development-comparison-v2-numeric-review-final.json`
- `evidence/ssl-research-v1/NATIVE_SSL_DEVELOPMENT_REVIEW_ADDENDUM_V2.md`
- `evidence/ssl-research-v1/cf-strong-three-seed-completion-v1.json`
- `evidence/ssl-research-v1/direct-replication-completion-v1.json`
- `evidence/ssl-research-v1/native-latent-software-review-final.json`
- `evidence/ssl-latent-builder-v1/root-durable-fixture-v1.log`
- `orchestration/native_band_budget_owner_resolution_v1.json`
- `orchestration/native_ssl_run_ledger_v1.json` (live executor-owned accounting)
- `evidence/ssl-research-v1/source-pins.json`

Snapshot evidence paths describe local files. This document does not authorize
public publication, paid resources or additional recipes.
