# ADR 0001: immutable local source and bounded runtime

Date: 2026-09-26. Prospective decision, before any model comparisons.

The owner provided the primary corpus locally. Existing verified files must not be
downloaded again. The omitted manual is registered in configs/datasets.json. Preserve
all XML and DPL files. Local SHA-256 and ZIP CRC establish local integrity, not a
publisher signature. The raw manual contains operational details and will not be
redistributed in the meeting package; only safe calibration findings and its hash.

Runtime measured 62.6 GiB free, below the preferred 100 GiB. ZIP metadata declares
10,169,592,780 expanded bytes. Set a 20 GiB extraction cap and retain at least 10 GiB
free. Preserve the original ZIP, extract transactionally, and avoid duplicate copies
across worktrees. No change to dataset identity or scientific eligibility.

Use a local Python 3.12 data/app environment. The owner identified an existing CUDA
environment under EVOCON_JEPA_Codex_Handoff; test it before reusing it for model checks.
Do not modify that environment. A redundant CUDA download was cancelled.
The portable dependency specification retains the explicit official CUDA index.
Node 24 LTS will be pinned locally; no global runtime or security changes.
