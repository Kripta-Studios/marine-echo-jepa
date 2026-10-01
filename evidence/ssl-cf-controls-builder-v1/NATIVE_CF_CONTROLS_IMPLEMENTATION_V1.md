CF matched controls engineering delivery
=======================================

This delivery adds `cf_random_frozen` and `cf_direct_supervised`, with explicit
zero SSL identity. It establishes no independent approval, scientific fitting,
quality result or held-out assessment. ADR0025 supersedes ADR0022's older scope
uncertainty: six prospective fits at seeds 7/13/23, a separate 12 full-owned GPU
hours inside 96 aggregate hours, retaining 12 aggregate hours for evaluation.

The model retains only the original factory's corresponding fresh CF encoder
branch and fresh QueryHead. It has no online branch, latent predictors, SSL
loss or EMA update. The factory's forked seed and head seed+100000 are unchanged;
initial encoder/head tensors match the original strong-endpoint reset at all
three seeds. H96/width256/latent128/depth5 and all original numerical helpers are
preserved. The encoder has 412,800 parameters and the head 18,565. Frozen mode
optimizes 18,565; direct mode optimizes 431,365. Frozen encoding uses eval/no_grad
and checks every parameter and BatchNorm buffer after each optimizer step.

Source matching clarification: original TRAIN pinball balances observed
horizons within each batch; eligible source-date -> horizon -> deployment
balancing/floor18 is the original DEV selection metric. These exact source
semantics are reused, rather than changing the training objective. Full H96
feature use retains the comparator's documented difference from sampled SSL
training crops. Source dates carry no verified UTC claim.

Admission requires exact current source/config/cohort/TRAIN/DEV/split/protocol,
executor and budget bindings, distinct APPROVED_CF_CONTROL_PREFIT identities and
an owned execution receipt before numerical/checkpoint decoding or model/RNG/
optimizer setup. Real CLI supports only cuda:0. Real recipe is frozen2000/500
or direct3000/750, batch64, AdamW0.0003/0.0001, clip1, original warmup10%/cosine
floor10%, patience4/floor18. Neither final-test nor an existing fitted weight
ancestor is admitted. The core_config is explicitly a zero-pretrain source
factory/inference descriptor; the control config governs supervised counters.

ROOT must supply the reviewed executor and owner budget receipt. Budget schema
requires status ROOT_RESOLVED, study CF_MATCHED_CONTROLS, fits6, cf_gpu_hours12,
aggregate_gpu_hours96, evaluation_reserve_hours12 and seeds[7,13,23]. The child
receipt has kind native_cf_control_owned_execution_v1/status RUNNING_CUDA,
actual child_pid, review/executor/budget/config SHA256, output/device/method/seed,
remaining_cf_hours, remaining_aggregate_hours and deadline_seconds. Deadline is
at most min(remaining CF, remaining aggregate minus12)*3600. This receipt is
runtime evidence, not a reviewer signature. ROOT's future executor must verify
whole-attempt accounting/idle journals/ownership and supervise the complete
attempt, including failure/startup/save. No such executor is implemented here.
Original Resources retains its existing RSS22GiB/CUDA<10GiB/owner gates.

Artifacts: resumable native_cf_control_resume_v1 checkpoints preserve optimizer,
schedule, all source/review/data/config identities, RNG, sampler, selection and
cumulative resources. Final weights/scalers/reports use exclusive creates.
native_cf_control_weights_only_inference_v1 contains encoder/readout only.
Selected kinds are native_cf_random_control_encoder_v1 and
native_cf_supervised_control_encoder_v1. Old SSL encoder loading rejects both.
The new load-only APIs use weights_only CPU loading, strict finite float32 meta
templates/assign, frozen eval state and opaque source paths; no ancestor reads,
RNG setup, optimizer, compilation or CUDA initialization occurs on CPU.
encode(x,observed,metadata) returns features; forecast additionally takes issued
native query and returns supervised quantiles in original TRAIN inverse scale.
They never accept targets/future inputs. Native230 remains230; a200 relabel is
rejected. Random features are untrained, while their head/scalers are fitted;
direct features are supervised, never SSL pretrained.

Actual builder verification: frozen-BN behavioral red exited1; final focused
pytest exited0 with65 passed/four ROOT-only skipped (11.80s). Ruff check and
format-check exited0. Physical Unicode safe synthetic checkpoint replay and
exclusive-file collisions passed for both controls. The final source proof
records26 unchanged protected dependencies,24 runtime source files and explicit
original helper calls/AST hashes. Earlier collection/import/proof failures are
retained; the final tests pin authoritative MAIN helpers before adding only
new builder modules. The first check harness encountered stdout encoding error;
its preserved pytest exit4 is separate from harness exit1.

The four actual optimizer/resume cases were NOT_RUN in builder, respecting the
retained optimizer/cache restriction without retry. After integrating exact
bytes, ROOT runs from MAIN using its existing environment:

    set NATIVE_CF_CONTROLS_ROOT_RUNNER_CHECKS=1
    .venv\Scripts\python.exe -m pytest --import-mode=importlib -q -p no:cacheprovider tests/unit/test_native_cf_controls.py tests/integration/test_native_cf_controls.py
    .venv\Scripts\python.exe -m ruff check --no-cache src/marine_echo/training/native_cf_controls.py src/marine_echo/inference/native_cf_controls.py tests/unit/test_native_cf_controls.py tests/integration/test_native_cf_controls.py

ROOT-only fixtures exercise both actual AdamW trajectories, exact continuation
and moments/schedule/RNG, frozen versus updated encoder, four original-DEV
selection opportunities, saved forecast/mask replay, and tampered sampler/frozen
resume denial before optimizer. These remain synthetic correctness checks.

Prospective public entrypoint, only through a separately reviewed ROOT guard:

    python -m marine_echo.training.native_cf_controls --train TRAIN --dev DEV --train-cohort TRAIN_COHORT --dev-cohort DEV_COHORT --split SPLIT --adr0016 ADR0016 --protocol PROTOCOL --config CONFIG --review REVIEW --executor ROOT_EXECUTOR --budget OWNER_BUDGET --execution OWNED_RECEIPT --output FRESH_OUTPUT --device cuda:0 [--resume SAME_OUTPUT/latest.pt]

Load-only APIs: marine_echo.inference.native_cf_controls.load_encoder and
load_inference. Existing SSL/assessment/prefix loaders are intentionally not
extended to these new kinds. Independent source/prefit review, resource wrapper,
real fits and held-out evaluation are NOT_RUN. No commit or original-source,
ledger, lock, permission, installation or public numerical access occurred.
Integrate only the four authored files and small top-level evidence; exclude
private SYNTHETIC_CORRECTNESS_ONLY-* fixture trees and __pycache__.
