# ADR 0020: proposed single frequency-conditioning revision

Date: 2026-09-30. Status: PROPOSED; no fitting authorized by this document.
Study: native_acoustic_ssl_v1. Historical protocols and evidence are preserved.

The fixed-mask shared encoder pools linear acoustic projections and additive
channel metadata before a nonlinearity. Opposing normalized changes in two bands
can therefore cancel. A synthetic correctness proof using the actual selected
encoder reproduced this invariance. It does not establish the cause of the
development score gap. The independent learning-path audit found no defect in
the inspected gradient, frozen-state, sampling or selection paths.

The sole proposed revision inserts parameter-free GELU on each channel's
projected values plus frequency/geometry metadata before observed-count pooling.
Everything else remains fixed: history96, patch4, width192, four temporal blocks,
four heads, latent64, temporal mean pooling, native geometry, encoded SIGReg.03,
shared context/future gradients, objectives, data, schedules and selection.
No history24/latent128 screen or further architectural revision is proposed.
New artifact kinds and architecture identity prevent reinterpretation of old
weights. The CF-JEPA architecture and all completed evidence remain unchanged.

## Explicit finite accounting proposal

ADR0015 permits at most eight seed7 configuration screens across candidates A/B,
one hypothesis revision, two SSL finalists and matched direct seeds7/13/23,
and96 aggregate GPU-hours including controls and failed attempts. This proposal
uses three A/B configurations: A/CF-H96, B/linear-band-pool-H96, and the single
B/nonlinear-band-pool-H96 revision. It freezes the other five possible A/B
configurations unused. It does not treat matched controls as free compute.

| Family | Seed7 encoder screen recipes | Strong endpoints |
| --- | --- | --- |
| Original CF | CF SSL | fresh frozen2000 and full3000 |
| Original shared | shared SSL, masked SSL, permuted SSL, direct, random | shared frozen2000/full3000, masked frozen2000/full3000, permuted frozen2000, direct frozen2000, random frozen2000, fresh direct3000 |
| Proposed revised shared | shared SSL, masked SSL, permuted SSL, direct, random | shared frozen2000/full3000, masked frozen2000/full3000, permuted frozen2000, direct frozen2000, random frozen2000, fresh direct3000 |

This explicitly totals11 seed7 encoder-screen recipes, of which three are A/B
candidate configurations and eight are matched controls. Original six recipe
screens and approved strong endpoints continue independently. The proposal is
NOT within a stricter eight-total-recipes interpretation. A distinct reviewer
must resolve this discrepancy against the owner's finite mission before any
revision fit; rejection cannot be bypassed by renaming controls or dropping
their compute/selection opportunities from the record.

SSL screens use6000 updates and four fresh500-update probes (2000 total), with
four development checkpoint opportunities. Direct screens use3000 supervised
updates and four500-update frozen probes (5000 total). Random screens train
no encoder and use one500-update probe. Strong frozen endpoints use2000 updates
and four500-step checkpoint opportunities; full/fresh-direct endpoints use3000
and four750-step opportunities. Initializations and train sample indices are
matched within each architecture. No dev gradients or final-test access.

Each eventual finalist is a distinct architecture selected on the declared
strong frozen endpoint, with fine-tuning reported separately. At most two SSL
families plus matched direct expand to seeds7/13/23; controls required for those
comparisons must be enumerated and approved before expansion. This document
does not authorize expansion or new seeds. No retrospective checkpoint rescoring
or extra probes for completed original runs.

Reserve at most12 additional owned GPU-hours for all proposed seed7 revision
screens and their strong endpoints. Full-owned failed/retried time counts.
The aggregate96-hour ceiling remains controlling; stop before launching a job
whose approved allocation exceeds the remaining limit. Later seed expansion,
held-out inference and packaging retain the unused reserve. Actual allocation
must be enforced by the reviewed supervisor/ledger, not estimates alone.

## Gates

Budget interpretation approval is separate from source/prefit approval. Require
durable coordinator tests, precise old/new source proof, native data/scaler and
ancestor bindings, distinct prefit and new safe inference review before fits.
The proposed builder implementation is not a training authorization. A rejected
revision leaves the original authorized research programme active. Final AEON2
values remain unopened until finalists, artifacts and evaluation code are frozen
and distinct final numeric-access review passes. No app, release, cloud, paid
resource, credential or junction action is part of this revision.
