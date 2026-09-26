# Research choices as of 2026-09-26

This is a targeted implementation selection, not a claim of exhaustive SOTA or established
marine superiority. All comparative claims come from the cited original paper/project, not
from our own reproduction. Prioritize a working three-family experiment over implementing
all newly published architectures.

## Core ideas

TS-JEPA [S09] is directly relevant to temporal representations. LeJEPA [S07] supplies the
SIGReg regularization idea; LeWorldModel [S08] combines compact latent dynamics and that
regularizer in an end-to-end visual control setting. Our observation-only acoustic adaptation
is not a reproduction of action-conditioned visual control results. EMA-JEPA is a separate
candidate with distinct target-gradient semantics.

Use an acoustic-aware encoder and a supervised reference trained on the same automatic future
outcomes. A hybrid head can consume raw history plus learned/predicted latents, following the
useful engineering pattern in the user's Kaleido work [S17]. It must beat a comparably budgeted
reference before being promoted. A latent-space loss alone is not the product metric.

## Optional and deferred

Chronos-2 [S10] is a useful frozen numerical forecasting reference if its current package and
weights license fit the environment. Cap execution and adapters; do not let it displace the
mandatory classical and matched-neural baselines.

JEPA-Anything [S18], newly posted in September2026, is a research reference for predictive
factorization. It is not in P0 because an additional new method plus uncertain data engineering
is excessive scope for the meeting. Orthogonal factors must not be casually labelled fish,
environment or sensor causes without a separate identification experiment.

V-JEPA/video models, giant multimodal models, reinforcement learning, diffusion generators,
learned planners, drone perception and MASS feeding are outside this MVP. MASS is interesting
for future action-conditioned work, but the selected public corpus has no verified feeding
actions or shrimp growth outcomes. Do not invent actions from passive telemetry.

## Hypothesis failure handling

If SIGReg collapses, preserve diagnostics and reject that candidate; do not substitute a
random embedding under the same name. If a strong tree wins, serve it and report exactly what
JEPA failed to add. If acoustic calibration cannot be verified, stop physical-unit claims and
keep raw-count demonstrations explicitly separate from the forecast benchmark.
