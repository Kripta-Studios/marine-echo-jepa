# ADR 0010: AEON development candidate selection

Date: 2026-09-27. Status: PROPOSED FOR INDEPENDENT REVIEW.
Study: `aeon3_geb_2024_hourly_sv_v1`. This prospective rule is written before
the forward-EMA JEPA and Chronos-2 validation outcomes are opened. It resolves
the family/seed ambiguity in `docs/06_EXPERIMENT_PROTOCOL.md` for this new
study; it does not change historical v1/v2 decisions or make the post-hoc
extension confirmatory.

## Fixed validation support and score

Use only the already approved ADR 0009 corrected eligible-target-source-date
pinball evaluator. Require the same 1,219 issued validation row IDs, target
truth and masks, and the same eligible source-date IDs for each horizon in
every comparison. Do not rank on the original all-scored-day diagnostic.
For a three-seed family, first take the arithmetic mean of the three saved
five-quantile forecasts at each row/horizon/quantile, then apply the same
monotone quantile rearrangement, and score that ensemble. Do not pick its best
seed. Report all individual seed scores and variability alongside the ensemble.
Single-model families keep their saved forecasts. Controls are diagnostics,
not selection candidates. A tie at the exact stored primary score is resolved
by the following fixed order, not by secondary metrics: persistence, seasonal,
ridge, raw-only histogram gradient boosting, direct-neural ensemble,
EMA-JEPA ensemble, shared-SIGReg ensemble, EMA hybrid ensemble,
shared-SIGReg hybrid ensemble, LightGBM, forward-EMA JEPA ensemble, Chronos-2.

The core conventional reference is the lowest-scoring one among persistence,
seasonal, ridge, raw-only histogram gradient boosting and the direct-neural
three-seed ensemble. The core JEPA comparator is the lowest-scoring one among
EMA-JEPA, shared-SIGReg and their corresponding raw-plus-latent hybrid
three-seed ensembles. These are the protocol-family choices. Record their
scores and selected checkpoint/model hashes before numerical CAL access.

The post-hoc extension compares one LightGBM model, the fixed three-seed
forward-EMA JEPA ensemble, and frozen Chronos-2 against both core choices.
Its lowest validation score may identify an exploratory operational candidate,
but must be labelled `POST_HOC_DEVELOPMENT_SELECTION` and cannot repair the
core confirmatory claim. Report the core and exploratory choices separately.
If a reviewed candidate cannot complete, record its exact failed/not-run
status and do not silently replace it or select from a partially fitted seed
family. Any operational default must be chosen only from completed, reviewed
models that the offline app can actually serve.

## Paired comparison and final boundary

Report primary and per-horizon scores, valid coverage, 90% raw interval
coverage/width and paired daily loss differences for every completed family.
Use the existing pure evaluator's 48-hour source-date blocks and 2,000 draws
with its fixed seed 20260926 for
paired bootstrap 95% intervals; these intervals describe this deployment and
do not establish external generalization. State the number of distinct source
dates and blocks, selection-induced optimism and the unknown source-clock
timezone. The bootstrap is descriptive on validation, not a new tuning target.

After an independent outcome review of every completed model and a separate
review of this rule and the comparison implementation, freeze the selected
models, exact weights, checkpoint hashes, CAL widening rule, TEST metadata
candidate universe, reader/QC/metric code and bootstrap before opening
retrospective TEST acoustic values. CAL may only widen 90% interval endpoints;
it cannot change the choices. Apply the original >=5% incremental loss gate,
paired 95% interval and <=10% per-horizon regression guard on the frozen
retrospective comparison, while clearly marking post-hoc arms exploratory.
The TEST period is retrospective, not sealed. A failed JEPA value gate is a
valid scientific outcome; an unexecuted candidate is not a negative result.
