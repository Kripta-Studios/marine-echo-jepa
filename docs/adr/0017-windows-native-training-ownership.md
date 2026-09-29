# ADR 0017: Windows desktop-aware training ownership

Date: 2026-09-30. Status: proposed before the first scientific optimizer update;
requires distinct prefit review of the exact executor change. This operational
correction changes neither the reserved split nor the objectives, schedules,
selection rule or historical protocols in ADR0015/0016.

The installed WDDM driver lists ordinary desktop graphics programs in
`nvidia-smi --query-compute-apps`, including Explorer and the desktop compositor.
The current runner therefore rejects the otherwise eligible local trajectory.
An actual prefit process-only check reproduced the issue; no training was
started to discover it. NVIDIA documents unavailable per-process memory under
WDDM because Windows manages those allocations.
[NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/),
[NVML process documentation](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlProcessInfo__t.html).

Retain the exclusive local GPU-owner lock and exactly one serialized training
process. Parse the local NVIDIA process snapshot, then inspect executable names
only. On WDDM, admit a fixed explicit list of ordinary desktop graphics programs.
Other Python/ML runtimes and unknown executable names block execution. A denied
process-name read stays unknown and blocks: it never triggers a permission
change or another access mechanism. On TCC or non-Windows reporting, all other
reported compute processes continue to block. A process that exits between
snapshots is omitted. Never terminate the user's desktop programs.

Record the driver, observed PID/name pairs, admitted desktop entries, blockers
and exact policy in each run. This establishes local training ownership, not
proof of exclusive hardware use or complete visibility of all GPU work.
Hardware exclusivity is explicitly `UNVERIFIABLE_WDDM`. Per-process Torch
allocated/reserved peaks remain below10GiB, process RSS below22GiB, aggregate
owned scientific-job runtime at most96GPU-hours, and cloud remains disabled.
The coordinator's serial job ledger conservatively includes data loading and
evaluation time in the aggregate budget.

The fixed desktop list and fail-closed unknown-runtime checks have focused
red/green unit evidence and a successful live process-only check. Those are
operational correctness checks, not scientific results. First fitting still
requires an exact renewed independent approval binding the changed runner,
ownership helper, operational ADR and frozen configuration/data/source files.
