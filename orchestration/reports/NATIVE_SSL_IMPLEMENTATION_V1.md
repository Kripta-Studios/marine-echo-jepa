# Native SSL implementation V1

Date: 2026-09-30. Implementer session: `01a0ef20-7397-7ba1-a98c-f59bc38ddcc1`.
Owner-assigned GPT-6.1 Sol/high route retained; no agents spawned. Branch:
`research/marine-jepa-vnext-builder`. Numeric acquisition, split reservation,
protocol, independent approval and GPU execution remain root-owned.

## Delivered code

`src/marine_echo/models/native_temporal.py`: shared patch4 channel projection,
observed-count channel pooling, native measurement conditioning, four temporal
attention blocks, unconstrained latent projection and horizon/query prediction.
Default width192/latent64 measures 2,424,448 encoder parameters and 2,457,621
complete model parameters. Feed-forward expansion6 provides temporal computation
capacity; no unused parameter inflation. Masked fills/right padding cannot enter
features. Empty crops return exact zero with an explicit safe attention sentinel.

Shared SSL independently encodes three actual future4 crops with the same
encoder and differentiable targets. Horizon-mean MSE combines with0.03 times
existing source-pinned SlicedEppsPulley on encoded context/future observations.
Predictor outputs are not regularizer inputs. Four latent rows per sample are
correlated, including overlapping future crops; they are not four independent
observations. Regularizer buffers/global step are checkpointed. Permuted SSL
rolls the same batch by one, with identical contexts/target multiset/SIGReg inputs.

Masked reconstruction shares this backbone and trains its latent projection.
It hides25% of observed train values and reconstructs only actual withheld
observations. Direct learns features with quantile supervision using a separate
copy of the common initialized head. Random features remain fully frozen.
Frozen readouts initialize identically when their latent input dimensions match,
including across the CF and shared architectures.

`src/marine_echo/training/native_ssl.py`: all six methods, strict prefit gate,
reviewed TRAIN source/deployment membership, persisted observed-channel/target
scalers, deterministic sampling, warmup/cosine schedules, frozen probes, bounded
development selection, safe full resume and inference artifacts. No final-test
path is discovered or opened. Roles are checked before numeric members; interval
IDs, masks, identities, dimensions and supplied per-observation metadata broadcast
invariants are checked. SSL eligibility never filters development issuance.

## Matched selection policy

The final2026-09-30 root budget amendment supersedes all earlier cadence notes:
SSL6000/cadence1500 and direct3000/cadence750 yield four encoder candidates.
Every candidate receives a fresh common frozen head, the same TRAIN readout
sequence and up to500 updates. The runner validates total planned supervision
across the joint screen, not just each probe. Development is
scored every250, with patience4 inside each readout and over encoder candidates.
Development supplies scores/selection only, never gradients/scalers. Random has
one unchanged encoder and the same inner head opportunities.

The full SSL screen can spend4*500 readout updates plus6000 SSL updates.
The direct screen can spend3000 supervised backbone plus4*500 probe updates,
meeting the total5000-supervised-update ceiling. Default readout budget is500;
root's later2000-update downstream-only frozen endpoints are separate trajectories.
All probe/backbone/total updates and synchronized device-owned elapsed time are
reported. Every probe is a separate supervised trajectory within its ceiling.
Direct's extra supervised feature updates are explicit. Equal downstream frozen
supervision is not equal full-recipe supervision or measured compute. Selection
records the best development encoder/head and actual steps, rather than silently
calling the latest endpoint selected. Resume restores scalers without refitting.

The evaluator averages source dates with>=18 anchors, then horizons, then
deployments. An unsupported horizon leaves the primary score undefined; no
deployment silently disappears. Shared support/coverage are explicit, every
issuance has finite sorted quantiles, and native geometry is retained. Saved NPZ
contains predictions/targets/masks/dates, row/deployment/cutoff identities, encoded
query, bounds in metres, frequency and interval duration. JSON retains daily
scores and coverage. Root's separate evaluator can reconstruct independently.

## Root execution and review interface

Review must have status `APPROVED_PREFIT`, a reviewer session distinct from the
implementer, and an allowed method. Required `bindings` map absolute paths to
exact SHA256 of TRAIN/DEV NPZs, runner, model, existing SIGReg source, config,
protocol and split. Metric code is in the bound runner. Named train/dev/split/
protocol hashes must also match. Missing/empty/rejected/stale records fail before
loading numerical members or fitting. Runtime config equals reviewed config.
CF also requires bindings for the four inspected author sources and LICENSE.

Use the existing sibling environment, with this source first:

```bat
set PYTHONPATH=src
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m marine_echo.training.native_ssl --train TRAIN.npz --dev DEV.npz --output OUTPUT --method shared_ssl --seed 7 --history 96 --pretrain-updates 6000 --pretrain-cadence 1500 --readout-updates 500 --batch-size 64 --device cuda --review REVIEW.json --prepare-config-only
```

Preparation creates only a new config for review; no corpus or fit is opened.
After independent approval, run without the preparation flag. `--config PATH`
loads root's exact config; explicit CLI changes must still match the reviewed
file. Direct uses methoddirect/pretrain-updates3000/pretrain-cadence750. Resume
adds `--resume PATH`. Default split/ADR0016 point to main; explicit paths are
supported. The above is a root command contract, not an executed real fit.

Checkpoints use weights-only deserialization and save encoder/model/predictors/
SIGReg, both optimizers/schedulers, selected states, phase/sampler steps,
initial/best heads, scalers, Python/NumPy/Torch/CUDA RNG and resource totals.
CPU smokes never initialize CUDA. Training continuation requires the original
device kind. Portable inference needs no optional CF checkout. Cross-device
replay tolerances are predeclared; CPU continuation/replay uses exact equality.
Inference requires only x/observed/metadata/query, without assessment arrays.
SSL batches likewise exclude scalar forecast labels. A focused failing regression
on inputs-only inference is retained in `red-inputs-only-inference.txt`, followed
by the final passing suite after removing label preprocessing from inference.

CUDA execution uses an exclusive native-run lock, rejects another GPU compute
process, requires CUDA PyTorch, caps allocator memory and checks allocated/
reserved<10GiB and RSS<22GiB. Time includes probes/evaluation. Stale locks are
not automatically removed. Hardware enforcement remains NOT_RUN by this builder.

## CF-JEPA provenance and adaptations

Inspected root-pinned `encoder.py`, `trainer.py`, `losses.py`, `cf_jepa.py` at
commit `5d3d2fd1273c283fbfa03249c078619245e84033` before implementation.
MIT copyright/permission/disclaimer are retained in the model module. Preserved:
kernels3/9/15, dilated depthwise residual convolutions, pointwise mixing and
BatchNorm; near-identity horizon predictors; normalized L1 prediction; PARAMS128
variance/covariance/invariance coefficients; linearly annealed prediction weight;
cosine EMA; parameter-only EMA and target eval/no gradients. Target BN buffers
remain initialized as in author code. Forecast features come from EMA.

The final CF route uses author128 dimensions256/128/depth5, LR0.00034 and
weight decay0.05, configured separately from shared encoder dimensions.
Author symmetric convolution padding, crop bounds/four crops and three remaining
future zones are retained. TRAIN trajectory appends unique future steps1..9 to
the context: future[0], future[1,2:4], future[2,1:4]. Overlapping native horizon
blocks do not duplicate intervals. Online crops shift across this TRAIN-only
trajectory; full-trajectory EMA targets are detached, with future zones strictly
later than each online crop. Forecasting uses only the observed prefix ending
at issuance, so symmetric encoding never receives held-out future input.

Explicit marine adaptations: values/masks/metadata form48 input fields; observed
masking is explicit; masked mean pooling replaces8-bin flattening; a query-aware
quantile head replaces Ridge; observed-only TRAIN scalers replace zero-filled
all-value statistics; update-based warmup/cosine and reviewed screen selection
replace author epoch scheduling. Crop-mean invariance retains the author's
pool-bin collapse. Author crop RNG uses NumPy and is checkpointed. Readout tensor
equality across CF and shared is meaningful at the same latent width128; the
default shared64 and source CF128 have different input dimensions. This is a
source-grounded marine adaptation, never full paper reproduction. No official
paper-dataset experiment or CF scientific result is claimed.

## Actual verification and limitations

Explicit `shell=cmd.exe` works without permission changes. Harmless `cd` and
Python version exited0; default PowerShell failed before execution. Initial
pytest exited2 on missing specified modules (`red-tests.txt`). First implemented
suite exited1:10 passed; two fixture writes raised PermissionError in the allowed
evidence tree. Failures are preserved in `green-attempt-01.txt`. Those filesystem
operations were not retried or bypassed.

Independent CPU checks use real torch/NumPy serialization with in-memory artifact
transport. They exercise the complete state graph, safe loads, continuation and
inference, but do not prove durable writes or atomic replacement. Builder durable
fixture/checkpoint/relocation tests remain BLOCKED. Root can set
`MARINE_NATIVE_DURABLE_TESTS=1` in its authorized environment. Parent successes
are never represented as builder executions.

Final focused suite:34 passed, exit0,12.94seconds in `final-tests-v3.txt`.
Final Ruff check and format verification both exited0. Checks cover
masks/padding/empty blocks, geometry/query,
context/target gradients, encoder SIGReg/state, masked latent gradients, CF EMA
detachment/buffers, head initialization, batch/pairing equality, train scalers/
membership, every required review binding before fit, test-role rejection,
interval/metadata invariants, direct ceilings/opportunities, independent daily
weighting, frozen encoders, RNG and full serialized CPU continuation/replay for
all six methods at pretraining/readout interruptions. Only labelled synthetic
CPU optimizer steps were performed. Runtime: Python3.12.13, torch2.11.0+cu128.
Pinned author encoder parity uses copied weights and exact equality. An
additional check inspects the actual deduplicated CF target trajectory.

CLI help exited0. Initial Ruff found import/style issues; fix and format exited0.
Original handoff verification exited1 on the pre-existing `.editorconfig` hash
mismatch (`handoff-check.txt`); that file is unchanged and outside assigned scope.
Final suite/lint/format/scope/commit evidence is recorded in the implementation
ledger. Authorship is restricted to assigned new files/evidence. Existing code,
contracts/lockfiles/data/app/release/historical evidence remain unchanged.

NOT_RUN: real fitting, CUDA checks, independent review, final-test access,
fine-tuning, references/multi-seed campaign, uncertainty/adaptation experiments
and scientific package review. Root owns those campaign tasks. No cloud/spend,
credential discovery, scientific outcome or independent approval is supplied.

## Integration and commit status

The scoped `git add` exited128 because it could not create
`../marine-echo-jepa/.git/worktrees/marine-jepa-vnext-builder/index.lock`:
Permission denied. Exact output is preserved in `git-add-attempt.txt`. No retry,
permission change, alternate execution tool or commit attempt bypassed that
restriction. Commit: none; root integration is required. Suggested coherent
commit message: `Implement reviewed native temporal SSL and matched frozen probes`.

Integrate exactly these authored paths; leave the pre-existing untracked mission
file untouched:

- `src/marine_echo/models/native_temporal.py`
- `src/marine_echo/training/native_ssl.py`
- `tests/unit/test_native_temporal.py`
- `tests/integration/test_native_ssl.py`
- `evidence/ssl-builder-v1/`
- `orchestration/reports/NATIVE_SSL_IMPLEMENTATION_V1.md`

Exact four code/test hashes and actual check statuses are in
`evidence/ssl-builder-v1/implementation_ledger.json`. Final source code is
validated by34 tests and both Ruff checks. Durable artifacts, independent review
and real fits remain separate pending gates, regardless of software test success.
