# ADR 0019: deterministic CF pooling and owned Windows process execution

Date: 2026-09-30. Status: implementation delta requiring renewed independent
prefit approval. The completed shared SSL trajectory, source, weights and reviews
are historical evidence; they are not edited or rerun by this decision.

The first real CF job failed on its first backward pass, before optimizer updates.
PyTorch2.11 CUDA rejects adaptive-average pooling backward with deterministic
algorithms enabled. Replace only CF's three adaptive-pooling call sites with
ordinary slice means and a stack using exact floor/ceil adaptive-bin boundaries.
This preserves the mathematical objective, overlapping bins and gradient weights;
kernel reduction rounding may differ slightly. Do not disable deterministic
algorithms, enable warn-only, change crops/zones/loss coefficients or alter source
pins. CPU output/gradient parity and root-owned deterministic CUDA backward/replay
must pass before renewed review and a new real output directory.

LightGBM4.6's first fixed-recipe attempt fitted its first quantile booster and
failed in the native filename save API under the Unicode checkout. The error does
not establish an OS permission denial. Preserve that failed path and attempt;
do not retry it, relocate to ASCII, alter permissions or silently reuse weights.
Use the documented complete model-to-string API and Python UTF-8 exclusive file
creation on separately reviewed NEW run paths. Propagate permission/existence
errors without fallback. Reload via Booster(model_str=text), verifying prediction
parity. All fifteen fresh quantile recipes and their TRAIN-only supervision remain
unchanged. Successful synthetic Unicode IO demonstrates only the new text path.

The first comparator receipt measured only the Windows venv Python redirector,
not the interpreter descendant. Its reported RAM peak is incomplete and must
not be represented as a valid full-process measurement. The renewed outer
supervisor tracks the owned Popen process plus observed recursive descendants,
conservatively sums their OS peak working sets (sampled-RSS fallback elsewhere),
and enforces the22GiB limit and full-attempt deadline. If stopping is required,
terminate only captured owned process objects, preserving creation-time identity;
wait and verify tree exit before retry. Unknown ownership or monitor denial is
blocking. Ordinary desktop apps are never termination targets.

The neural and separate downstream wrappers use explicit CUDA device0. They
validate distinct review bindings before execution and supervise one scientific
process tree. CUDA allocated/reserved remains below10GiB through the existing
allocator cap and resource checks. Cleanup of a stopped attempt's GPU lock
requires verified owned-tree exit, a captured owned PID and exact output identity;
unknown/stale locks remain intact. One root coordinator updates the ledger.

All full owned attempts, including failures/resumes, count toward the96 GPU-hour
ceiling. The CF direct attempt lacked a process timer; its actual time is marked
unmeasured and a conservative console-creation-to-journal observation window is
charged explicitly. Chronos retains its cumulative two-hour development slot,
including loading/preparation/inference/saving and failed/resumed attempts.

The original shared model module is retained byte-identically as a bound source
snapshot. Shared class definitions remain unchanged by the CF-only repair; a
distinct downstream review must authorize that compatibility explicitly while
preserving original ancestor hashes. No historical data/protocol/scaler/member
binding may be substituted. No final-test numerical access, scientific claims,
new tuning configurations, app/release work, cloud/paid-resource approval,
credentials, permissions or denied junction operation is authorized by this ADR.
