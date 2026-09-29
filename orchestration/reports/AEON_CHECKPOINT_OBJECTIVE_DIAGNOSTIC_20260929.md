# AEON checkpoint and objective diagnostic at 991368d

The immediate recommendation is **no further fitting before the meeting**.
The saved direct 1500-step models already supply a competitive shorter-training
reference. The EMA gradient restriction is reproduced on initialized models and
saved pretraining states, and separated statistics expose a substantial objective
scale mismatch. These findings justify replacing automatic nine-trajectory
replication with a specific research decision. They do not retire the project,
invalidate the preserved historical results, or establish that JEPA is universally
ineffective.

This is a post-hoc diagnostic on exposed TRAIN/VAL at source commit `991368d`.
No scientific training, optimizer steps, CAL/TEST numeric access, 30k/50k
extension, completed-fit rerun or release build occurred. The distinct review
sidecar is `orchestration/reviews/AEON_CHECKPOINT_OBJECTIVE_DIAGNOSTIC_REVIEW_20260929.json`.
The detailed execution report is
`orchestration/reports/AEON_CHECKPOINT_DIAGNOSTIC_TECHNICAL_20260929.md`.
The parent verified the implementation session as GPT-6.1 Sol / high from
its exact runtime record; evidence is `verified-implementation-routing.json`.
Native role attempts that ignored the owner's model constraint were stopped
and supplied no implementation artifacts. Review uses a fresh, separately
verified GPT-6.1 Sol session.

## What the saved direct endpoints establish

All nine requested supervised checkpoints and the three original EMA final SSL
checkpoints were recovered and verified against the original manifest and slot
indexes. There are no missing requested model artifacts. Forecasts were replayed
on the reconstructed 1219-row original VAL cohort and scored with the corrected
at-least-18-anchor daily evaluator: 50 eligible source dates per horizon and
1194/1192/1189 eligible rows at +1/+3/+6 intervals. CPU replays differ from saved
CUDA predictions by at most 0.00001526 dB; they are numerically faithful, not
bit-exact. No historical forecast or score was overwritten.

| Comparison | Direct 1500 supervised | Direct 3000 supervised | EMA 1500 SSL + 1500 supervised |
| --- | ---: | ---: | ---: |
| Seed 7 | 0.645004 | 0.651154 | 0.648805 |
| Seed 13 | 0.640577 | 0.647964 | 0.647989 |
| Seed 23 | 0.643002 | 0.644897 | 0.647978 |
| Equal-three-seed mean-prediction ensemble | 0.639577 | 0.639062 | 0.642170 |

Values are corrected daily mean pinball in dB, with lower values better.
The ensemble row scores averaged predictions; it is not the average of the
three single-seed scores. Direct 1500 beats EMA at each matched seed and as an
equal-three-seed ensemble on this VAL. It uses the same number of supervised
updates and half the total optimizer-update count. Direct 3000 matches EMA's
total update count. Update count is not identical wall time or floating-point
work across objectives.

The three individual direct scores worsen between 1500 and 3000, but the
direct ensemble improves slightly. Both statements must remain visible.
These are small development differences, not a new prospective winner or an
independent significance claim. The 1500 checkpoints are fixed prefixes of
historical 3000-step runs, chosen for this diagnostic by the owner's instruction,
not historical minima. They have not been evaluated on CAL/TEST in this task.
Initial forecast heads differ between direct and SSL even at the same seed, so
these comparisons cannot isolate the causal contribution of pretraining.

## Trajectories and forecast errors

Per-update TRAIN-loss curves were not saved. This missing evidence remains
explicit. The diagnostic instead evaluates each predetermined trajectory
endpoint on the same 512 chronological, evenly spaced TRAIN windows, without
fitting. These TRAIN errors are row-weighted descriptive pinball, whereas VAL
uses the corrected daily metric; their absolute difference is not a like-for-like
generalization-gap estimate.

| Saved trajectory, seed 7 | Fixed-sample TRAIN pinball, early -> final | Corrected VAL pinball, early -> final |
| --- | ---: | ---: |
| Long direct, supervised 2500 -> 30000 | 0.523975 -> 0.179783 | 0.650889 -> 1.178379 |
| Long EMA, supervised 2500 -> 15000 after 15000 SSL | 0.525353 -> 0.354225 | 0.664914 -> 0.913424 |
| Original direct, supervised 1500 -> 3000 | 0.536609 -> 0.516958 | 0.645004 -> 0.651154 |

Original direct seeds 13 and 23 likewise reduce fixed-sample TRAIN loss while
their individual VAL loss rises from 1500 to 3000. The long-run opposing trends
provide stronger evidence consistent with overfitting than VAL deterioration
alone. A simple explanation in which training itself merely diverged is less
consistent with these sampled TRAIN errors. Nonstationary calendar/distribution
shift, fixed-learning-rate dynamics, capacity and objective/view mismatch remain
possible contributors. The sparse TRAIN probes are not a reconstructed training
curve. Intermediate VAL checks are diagnostic; none was selected or promoted.

## Reproduced gradient path and separated objective statistics

The supplied 18 synthetic source probes passed against verified blobs. Another
18 initialized-model TRAIN-batch probes and 12 saved-state TRAIN-batch probes
record first-layer gradients at each of the 24 positions, separately for value
and observation-mask columns. The six saved states are original seed-7 EMA and
shared-SIGReg at SSL500/1500, long EMA at SSL15000 and expanded EMA at SSL1500.
Each uses two fixed TRAIN batches of 64, whose row/source memberships and hashes
are saved. Expanded diagnostic batches differ from the historical source-
homogeneous SSL sampler: one is prior-year only and one mixes sources. Rank
and scale observations are conditional on these diagnostic batches.

EMA masks both values and masks in positions 18--23, and detaches its suffix
teacher target. All 6144 corresponding first-layer weights have exactly zero
data-gradient from prediction loss, weighted regularizer and total SSL loss.
Teacher and forecast-head gradients remain absent. Shared-SIGReg and direct
have nonzero suffix gradients in the tested cases. This is a demonstrated
learning-path restriction. AdamW decay can still alter weights, shared later
layers change, and supervised full-context updates can learn suffix inputs.
It is not evidence that the weights are absolutely frozen or that the restriction
caused a particular forecasting result.

Teacher suffix target, online short context, online full context and predictor
are measured separately. Shared-SIGReg has no EMA teacher: its suffix target
uses the online encoder, despite the compatible JSON field name
`teacher_suffix_target`. Full means, centered standard deviations, covariance
traces, effective ranks, prediction/regularizer losses and component gradients
are in `objective-gradients.json`.

| Saved state | Suffix-target centered RMS | Online full-context centered RMS | Predictor centered RMS | Suffix-target effective rank |
| --- | ---: | ---: | ---: | ---: |
| Original EMA SSL1500 | 0.00805--0.01242 | 1.11121--1.16486 | 0.99730--1.11705 | 3.60--4.97 |
| Long EMA SSL15000 | 0.01564--0.02147 | 0.79251--1.51270 | 0.82489--1.39005 | 3.63--4.05 |
| Expanded EMA SSL1500 | 0.00961--0.01241 | 0.58783--0.69396 | 0.91555--1.04467 | 4.03--4.98 |
| Original shared-SIGReg SSL1500 | 0.73807--1.21649 | 0.43505--0.97119 | 0.44662--0.86018 | 2.75--2.85 |

Ranges cover the two fixed batches, not confidence intervals. Effective rank
is covariance eigenvalue entropy after centering; 64 rows bound rank at 63.
The EMA predictor/teacher variation mismatch persists after removing means.
At these final EMA states the weighted-regularizer gradient norm is about
2.67--4.32 times the prediction gradient norm over trainable parameters.
Gradient cosines vary in sign: there is no uniformly opposing direction.
The regularizer acts on the predictor, whose distributional objective differs
from matching a small-variation teacher. This supports an objective-design
concern, not causal attribution of forecast loss. Shared-SIGReg also remains
low rank despite suffix gradient support, so suffix starvation alone is not
a sufficient explanation. No exactly constant representation was observed.
All backward operations used disposable copies; original checkpoint bytes,
loaded model state and regularizer buffers were checked unchanged.

## Initialization, sampling, expanded TRAIN and evaluation lineage

Tensor equality checks at seeds 7/13/23 establish identical initial encoders
and different forecast heads. SSL constructs its predictor before its head,
consuming additional RNG draws. Same seed therefore does not mean all tensors
match.

The current five-file training-code digest exactly matches the historical
campaign manifest and Git revision `3755cd4`. Both aligned and shuffled SSL
sample 64 distinct contexts per batch, but shuffled SSL cycles through
interval-ID-modulo-24 pools and rolls targets within those batches. Aligned
SSL samples the full TRAIN pool. Reconstructed context sequences differ,
all 4965 contexts are eventually exposed, and shuffled target separation is
at least 24 intervals. Extra shift draws change SSL RNG consumption. The actual
supervised phase resets RNG to `SeedSequence([seed,2])`, yielding identical
supervised index sequences. A hypothetical continued-RNG difference is not an
actual historical supervised-sampling difference. No rejection/duplicate
algorithm appears in the bound historical source. The old control changed
context-batch distribution as well as pairing.

Expanded TRAIN jointly normalizes 8507 prior-year and 4965 original windows
(13472 total), uses source-homogeneous SSL batches and pooled supervision,
and changes serial/deployment/calendar composition. At fixed updates, nominal
presentations fall from 38.67 to 14.25 per window for direct and from 19.34 to
7.13 in each EMA phase. These are overlapping, dependent presentations, not
independent epochs. Input mean/std change from -82.95869/6.48199 to
-83.08755/5.83542; target mean/std from -87.68227/4.30697 to
-87.90529/4.19104. Recorded source counts are in `expanded-exposure.json`.
The modest direct-only expanded result cannot be attributed solely to volume.

The prior-year archive is TRAIN for expanded models and any descendants
carrying their fitted weights or preprocessing. It cannot be their external
test. The original frozen-model prior-year transfer remains unchanged in its
historical descriptive scope. The earlier reviewed retrospective core result,
including its scoped value-gate conclusion, is preserved. No new test result,
confirmatory claim, biological inference or business validation is added here.

## Mechanisms, correlations and unresolved questions

| Evidence class | Established scope |
| --- | --- |
| Mechanisms and implementation facts | Masking/detachment removes the EMA suffix data-gradient path; constructor RNG ordering changes forecast-head initialization; the old shuffled sampler changes context-batch distribution as well as pairing. |
| Correlations | Lower sampled TRAIN error accompanies worsening long-run VAL; low rank, centered scale mismatch and large regularizer gradients coexist; expanded cohort/scaler/sampling/exposure changes accompany its scores. |
| Unresolved | Which mechanisms cause which forecast differences; the contribution of sampling, decoder initialization and objective geometry to historical transfer; whether a controlled objective intervention would help; generalization beyond these exposed development cohorts. |

## Decision and conditional follow-up

The highest-value immediate task is to use this reviewed evidence at the meeting,
with no further fitting. Historical nine-trajectory attribution would multiply
an objective whose restrictions are already demonstrated. Direct replication
has lower immediate value because three saved 1500-step runs are available.
The unchanged long route already failed its continuation gate. Another release
would not answer a scientific question.

One later controlled objective/gradient-routing intervention could be considered;
it is **a proposal only, unapproved and not queued**:

- **Hypothesis:** moving the 0.03 regularizer from predictor outputs to online
  full-context representations restores suffix encoder data-gradients and
  reduces the centered teacher/predictor scale mismatch without sacrificing
  forecast quality.
- **Matched comparators:** current EMA objective versus that single location
  change. Copy every initial tensor and buffer, including forecast heads, from
  one common initialization; use identical original TRAIN rows, scaler,
  batch indices, optimizer settings and supervised schedule. Keep the student
  prefix and detached teacher suffix unchanged. No expanded data or other
  architecture/hyperparameter changes.
- **Fixed budget:** one seed-7 pair, 1500 SSL + 1500 supervised updates per arm,
  6000 total optimizer updates, one local process, 20-minute fitting wall-time
  cap, below 10 GiB GPU and 22 GiB RAM; no cloud or sweeps. This requires a new
  reviewed protocol and explicit owner authorization after the meeting.
- **Exposure labels:** original 4965-row TRAIN; original 1219-row repeatedly
  inspected development VAL; no CAL/TEST or prior-year evaluation. A VAL gain
  would remain exploratory.
- **Continuation rule:** stop if resource caps, tensor/batch matching or finite
  objective checks fail. Complete the fixed endpoints without selecting a
  historical minimum. Consider a separately authorized replication only if
  the changed objective gives nonzero suffix gradients on both fixed TRAIN
  batches, reduces `abs(log(predictor_centered_RMS/teacher_centered_RMS))`
  by at least 25% on both batches, and lowers corrected final VAL pinball by
  at least 1% versus its matched arm. Otherwise stop that intervention.
  Meeting claims and historical conclusions remain unchanged regardless.

## Evidence and preservation

The completed evidence is under
`evidence/aeon-checkpoint-diagnostic-20260929/run-02/`. Attempt 1 completed
endpoint and backward probes but exited 1 on the 30k metadata schema;
its log and recovered tool-source snapshot remain preserved. The snapshot is
labelled reconstructed after failure, not captured at process start.
Attempt 2 reused those hash-checked
phases, finished the trajectory/error diagnostics, and exited zero in
126.4 seconds. It recorded peak RSS of 852996096 bytes, used CPU, allocated
zero GPU bytes and performed zero optimizer steps. Runtime was Python 3.12.13,
PyTorch 2.11.0+cu128. The completed prefix contains 18 supplied synthetic probes,
18 initialized TRAIN probes and 12 saved-state TRAIN probes. Focused tests and
lint logs accompany the technical report; original handoff verification reports
the pre-existing `.editorconfig` mismatch.

The reviewed r3 ZIP retains SHA-256
`5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57`.
The before/after preservation evidence covers 383 historical files, including
models, predictions, protocols, reviews and original ledgers. Software-release,
experiment, historical JEPA-value and business-validation gates retain their
existing scopes. This diagnostic creates no new release or universal value gate.
