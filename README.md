# Marine Echo JEPA

**A reproducible public-data acoustic forecasting demonstrator with an offline-first web application.**

Status of this distribution: **implementation handoff with tested bootstrap utilities, not a trained MVP**.
Prepared on 2026-09-26. Planning target: a meeting approximately two weeks later (around
2026-10-10); the actual meeting date has not been provided.

The implementation agent must build the complete product described here, acquire permitted
public data, execute real experiments, independently review the work, and package a runnable
meeting release. It must not stop after scaffolding, a synthetic demo, or a plan.

The product asks: **What acoustic state should we expect one, three, and six hours from now,
and when is our observation too old to support a useful forecast?** It does not estimate tuna
catch, certify biomass, direct fishing, or claim validation on Marine Instruments equipment.

## Start

[Full package index](INDEX.md)

Read [START_HERE.md](START_HERE.md), [AGENTS.md](AGENTS.md), and
[CODEX_START_PROMPT.md](CODEX_START_PROMPT.md). The numbered documents in `docs/` are the
implementation contract. Source facts and access limitations are in `references/SOURCES.md`.

## Deliverable

A local English-language application with an acoustic replay viewer, three-horizon forecasts,
a baseline/model comparison lab, a clearly labelled observation-age experiment, and an
evidence explorer. It must work without internet at the meeting after preparation.

The preferred public corpus is MOSAiC downward-looking AZFP data, PANGAEA 949811. OOI
moored EK60 data provide a documented contingency. Neither corpus is tropical tuna FAD data.

## Hardware and cost

Windows 11; RTX 5070 Ti Laptop, approximately 12 GB dedicated VRAM; 32 GB system RAM.
Optional cloud infrastructure ceiling: USD 500, not a spending target. Agent subscriptions/API
usage are separate unless the owner explicitly reallocates this budget.

## Important boundaries

- Public research data are not Marine Instruments customer data.
- Acoustic intensity is not automatically biomass or species identity.
- Model novelty and model superiority are different deliverables.
- A negative JEPA result does not invalidate an honest working application.
- No raw data, large weights, secrets, or private Marine exports belong in Git.
- This is an independent proposal, not an endorsed Marine Instruments product.
