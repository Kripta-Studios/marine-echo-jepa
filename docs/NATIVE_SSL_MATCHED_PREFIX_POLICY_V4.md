# Prospective matched-prefix policy V4

Status: prepared for genuine review; no final-access or prefit approval.
This extends the existing prefix policy to matched CF controls and conventional
prefix fitting. It does not alter historical fits or the original 47-method study.
The machine-readable registration is `orchestration/native_prefix_matched_policy_v4.json`.

All arms use the same permitted deployment-specific data and observed labels.
Keep H96, four source-day context warmup, 1/7/30 labelled source-day prefixes,
fully prefix-contained targets and the fixed common suffix starting at source
day 41. This is a one-day labelled prefix after warmup, not one calendar day of
total data. Never obtain missing history from future or suffix intervals.

Frozen pretrained, random and supervised encoders use fresh matched heads;
fresh scratch uses the matching backbone. Preserve full-H96 CF extraction and
the original crop mismatch in every arm. Neural fits retain the original 2000
updates, 500 cadence and 64 batch size. Preserve original TRAIN scalers and all
fitted ancestry, including historical archives consumed by descendants.

LightGBM uses the existing native reference feature function and its pinned
quantile recipe, 15 boosters, 300 iterations and four CPU threads. The prospective
DEV selection schedule is 75/150/225/300 iterations with earliest strict
improvement on original AEON4 DEV daily pinball. This adds four prespecified
selection points to prefix fitting; it does not revise the historical fixed 300
reference. No hyperparameter search or prefix/suffix scaler fit is introduced.
Its original raw-dB features and targets retain their original scaling semantics.
Seeds 7/13/23 and the same allowed prefix labels apply. Persistence and seasonal24
remain unchanged no-fit references.

Freeze exact finalists, weight/scaler ancestry, source geometry and per-cell
policies before reserved numerical access. No DEV win over LightGBM is required
for transfer eligibility. Reserved scores cannot select among candidates or
change the prefix policy. This registration does not yet name final artifacts,
approve access, or claim that any adaptation cell has been fitted.
