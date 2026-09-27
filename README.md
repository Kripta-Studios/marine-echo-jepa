# Marine Echo JEPA

**A reproducible public-data acoustic forecasting demonstrator with an offline-first web application.**

Current status: **working offline engineering application; scientific P0 remains incomplete**.
The existing app, read-only API, real raw-count replay and model software are implemented.
The historical engineering diagnostic is preserved at `release/meeting-20260926-r2.zip`.
It contains no completed benchmark or supported JEPA-value result.

Continuation work has recovered the correct factory calibration evidence, tested bounded
environmental assumptions and repaired a pinned upstream parser edge case. The complete
TRAIN eligibility audit is complete: the original target permits at most 67 overall days;
even a generous bound for any fixed 38 kHz band permits at most 73, below the required 90.
Independent review accepted those data findings; all 25 benchmark runs remain blocked.
See the current checkpoint rather than treating the original handoff as the implementation.

The product asks: **What acoustic state should we expect one, three, and six hours from now,
and when is our observation too old to support a useful forecast?** It does not estimate tuna
catch, certify biomass, direct fishing, or claim validation on Marine Instruments equipment.

## Start

[Full package index](INDEX.md)

Read [AGENTS.md](AGENTS.md), [the continuation contract](CODEX_CONTINUE_AFTER_DIAGNOSTIC.md),
[current status](orchestration/STATUS.md), and [release validation](release/RELEASE_VALIDATION.md).
[START_HERE.md](START_HERE.md) and [CODEX_START_PROMPT.md](CODEX_START_PROMPT.md) preserve the
original handoff context. The numbered documents in `docs/` are the
implementation contract. Source facts and access limitations are in `references/SOURCES.md`.

Current results and tested local launch command: [continuation outcome](orchestration/reports/CONTINUATION_OUTCOME_20260927.md).

## Intended deliverable

A local English-language application with an acoustic replay viewer, three-horizon forecasts,
a baseline/model comparison lab, a clearly labelled observation-age experiment, and an
evidence explorer. It must work without internet at the meeting after preparation.

The preferred public corpus is MOSAiC downward-looking AZFP data, PANGAEA 949811. OOI
moored EK60 data provide a documented contingency. Neither corpus is tropical tuna FAD data.

## Hardware and cost

Windows 11; RTX 5070 Ti Laptop, approximately 12 GB dedicated VRAM; 32 GB system RAM.
Cloud is disabled without explicit owner approval. The infrastructure ceiling is USD 500,
with a USD 400 stop and reserve; it is not a spending target. No purchases or metered agent
API use are authorized by that ceiling.

## Important boundaries

- Public research data are not Marine Instruments customer data.
- Acoustic intensity is not automatically biomass or species identity.
- Model novelty and model superiority are different deliverables.
- A negative JEPA result does not invalidate an honest working application.
- No raw data, large weights, secrets, or private Marine exports belong in Git.
- This is an independent proposal, not an endorsed Marine Instruments product.
