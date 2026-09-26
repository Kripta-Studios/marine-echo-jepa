# Compute, Windows and cloud budget

## Hardware interpretation

Target machine: Windows 11, NVIDIA RTX 5070 Ti Laptop with approximately 12 GB VRAM and
32 GB **system RAM**. The two pools are not interchangeable; this is not a 32 GB GPU.
Design for <=10 GiB allocated GPU memory and <=22 GiB process-tree system RAM. Measure actual
free resources before each run. One trainer owns the GPU; at most two CPU ingestion workers.
Default Windows DataLoader workers=0; raise to 2 only after deterministic spawn tests.
Always protect worker entry points with `if __name__ == "__main__"`.

Choose Python 3.12 initially. Verify a current official PyTorch CUDA build supports the actual
GPU architecture. The owner's earlier working 2.10.0+cu128 setup is a lead, not a universally
correct installation recipe. A successful `torch.cuda.is_available()` is insufficient:
execute device matmul, Conv1d/2d, attention forward/backward, optimizer update and checkpoint
reload. Record GPU name, capability, torch build, driver and kernel outcome. Official source:
[S15]. Never silently benchmark a CPU fallback as CUDA training.

Start BF16 only after a finite forward/backward comparison to FP32. Use native PyTorch
operations; no mandatory FlashAttention, custom CUDA extension, torch.compile, Triton or WSL.
CPU-only inference and preprocessing must remain supported. WSL2/Linux cloud is an optional
separately tested runtime, not a hidden prerequisite of the Windows application.

## Memory and data

Train from compact processed daily shards, not raw files loaded into RAM. Inspect one hourly
raw file first. Calibration/extraction must be resumable, with per-shard hashes. Zarr version
must be pinned to the tested Echopype version; current Echopype releases changed Zarr support
[S05]. A separate `data-env/pyproject.toml` and lockfile is acceptable if acoustic dependencies
conflict with the app/training environment. Document the two commands; do not silently use a
second interpreter.

Planned disk caps: raw downloads 40 GiB, extracted/preprocessed working set 80 GiB. Verify
at least 100 GiB free before a full extraction, adjusting a declared cap to available storage
without deleting unrelated user files. The published D1 archive is ~4.4 GB compressed, not
an assurance of expanded size. Cap an individual response at 8 GiB. Preserve raw files read-only.

Default model <1.5 million parameters, hard cap 3 million. Start batch16, context96, 64 range
bins, latent width96. Profile 100 updates and extrapolate elapsed time before a long run.
Increase to batch32 only inside the measured memory budget. Checkpoint at most every 250
updates and at end/best validation. Avoid hundreds of epochs by accident: steps are the cap.
An OOM retry reduces batch once and records the changed configuration; do not alter temporal
resolution, input channels or model dimensions inside a supposedly identical run.

## Training allocation, not a speed promise

Plan approximately 20–60 local GPU-hours across development, selected three-seed replications,
controls, hybrids and final export, contingent on the 100-step profile. This is a project
resource allocation, not measured runtime. If projected execution exceeds the remaining
schedule, drop P1 work before touching P0 evidence. Reuse cached frozen embeddings for tree
heads and CPU evaluation. Do not run multiple deep-learning trainers on the laptop GPU.

## Optional cloud budget

The user set a ceiling of USD500 for compute. This document does not authorize creating
accounts, purchasing machines, exposing secrets or deploying publicly. Prefer local first;
request owner provisioning of one machine only when measured runtime/memory justifies it.
Use the same split/config hashes and import cloud results as a separately recorded runtime.

Public Runpod pricing checked 2026-09-26 lists RTX4090 24GB around USD0.74/hour and RTX5090
32GB around USD0.99/hour [S16]. These are dated examples, not guaranteed availability or an
all-in quote. At USD0.74/hour, 100 GPU-hours are USD74 before storage, transfer and tax.
More money is not a reason to enlarge the model or run unbounded hyperparameter search.

Budget allocation ceiling:
- GPU time: USD250;
- storage/transfer: USD75;
- contingency: USD75;
- protected reserve: USD100.

Plan to stay well below USD500. A ledger records paid, committed, hourly rate and projected
stop time. Refuse new paid jobs if paid+committed would exceed USD400; keep the USD100 reserve
for finishing/copying evidence. Hard ceiling including applicable fees: USD500. The owner
must inspect provider billing; a local ledger is not a provider-enforced billing cap.
Agent/LLM subscriptions and tokens are outside this GPU allocation unless the owner explicitly
reassigns it. Do not imply three large coding agents are free.

Every cloud launch needs a runtime expiry and manual/provider auto-stop protection. Export
checkpoints and verify hashes, then stop compute and verify that only authorized storage
remains. Data rights must permit upload; no proprietary Marine data is included in this phase.
Do not put cloud credentials in repo config or generated shell transcripts.
