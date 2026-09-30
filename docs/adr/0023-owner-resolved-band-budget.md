# ADR 0023: Explicit owner resolution of the band revision budget

Status: OWNER_AUTHORIZED_BUDGET; exact code/configuration prefit review remains required.
Date: 2026-09-30.

The owner answered the pending ADR0020 question: "Allow up to 11 recipes, including controls". The question explicitly limits the single band-conditioning revision to12 additional local GPU-hours inside the existing96-hour aggregate. This is affirmative authorization; elapsed time was never treated as consent.

The eleven recipes are the six original seed7 recipes plus band shared SSL, band masked reconstruction, band permuted pairing, band random frozen features and band directly supervised end-to-end. All controls count. The band direct recipe uses the authored fixed3,000-update downstream direct endpoint because the band core deliberately rejects a separate supervised screening path. It does not add a sixth revision recipe or a repeat of the old direct screen.

The proposed CF backbone matched controls in ADR0022 have not been implemented or fitted. They are deferred and do not consume the authorized eleven slots. Existing CF results retain their cross-architecture attribution limitation. No other histories, latent sizes, objectives, learning rates or architecture variants are authorized by this resolution.

Count all band-family scientific GPU work conservatively against12 hours: screens, controls, later permitted replication and readouts, GPU inference/assessment, failures, resumes, startup, evaluation and saving. Those same operations also count against96 aggregate hours. Before launch, the required supervisor must enforce the lesser remaining allowance, using immutable receipts and the active run ledger. One scientific process, indexedcuda:0, allocated/reserved memory below10GiB and owned process-tree RAM below22GiB remain mandatory.

The owner reply does not substitute for independent exact source/data/configuration prefit review, permit final-site numerical access, expand the maximum two SSL finalists, authorize cloud or paid resources, or change historical protocols. Preserve ADR0020's earlier proposal and the independent request for clarification as historical evidence. The machine-readable resolution is `orchestration/native_band_budget_owner_resolution_v1.json`.
