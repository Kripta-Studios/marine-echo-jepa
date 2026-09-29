# AEON scaling and expanded TRAIN offline r3 validation

The new immutable r3 candidate built from source commit
`f975d2b0a425a9cbdadbacf9f50d0cad216d93c1`. It presents the independently
reviewed original retrospective study, descriptive external-transfer result,
negative same-data 30k development result and expanded-TRAIN 3k development
result. The scaling studies remain development only.

## Exact package

- Archive: `release/aeon-offline-scale-expanded-20260928-r3.zip`.
- SHA-256: `5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57`.
- Archive size: 26,226,838 bytes; 107 hashed assets plus `SHA256SUMS`.
- Manifest SHA-256: `225f607f89dfd2634d9e5c87c8632ce5b8179d126a2d16a47423a330ab596def`.
- `SOURCE_REVISION.json` SHA-256:
  `0d08e3198b401a3944c5695ed8ba3faaae66ad6aca9df51732797d1a48408c18`.

R1 remains unapproved because its README cited the wrong new review locations.
R2 remains unapproved because unbroken evidence hashes caused mobile horizontal
overflow. Their original archive bytes are preserved. The distinct source review
`orchestration/reviews/AEON_SCALE_EXPANDED_OFFLINE_R3_SOURCE_REVIEW_20260929.json`
passed the scoped `.panel code` wrapping fix. R3 changes no scientific evidence,
model predictions or checkpoint bytes relative to r2.

## Actual execution

The production `npm run build` exited zero. The portable builder exited zero
and verified all 107 packaged assets. A new extraction under
`outputs/aeon-scale-expanded-relocated-r3/` was launched with its own
`Run-AEON-Research.ps1 -Port 8782`. It created a fresh offline CPython 3.12.13
environment, verified 107 assets again, and installed 13 bundled app
distributions with `uv pip install --offline --no-index`. The API served only
`127.0.0.1`; no training or new acoustic materialization was performed.

The reproducible Chromium regression is:

```powershell
cd web
node tests/aeon-release-smoke.mjs http://127.0.0.1:8782 ../outputs/aeon-scale-expanded-r3-browser-smoke.json
```

It exited zero and checked HTTP 200 health/study responses, all four exact
development scores, 13,472 expanded TRAIN windows and a truth-free historical
replay page of 12 rows from 1,216 cutoffs. Both new study sections displayed
their correctly rounded scores. At 1,440 and 390 pixel viewports the document
width was exactly 1,440 and 390 pixels. Page errors and external browser
requests were both empty. Desktop/mobile screenshots are saved next to the
JSON under `outputs/`.

The same script against the preserved relocated r2 exited one on the mobile
fit assertion: 531 pixels of document width in a 390 pixel viewport. This is a
new reproducible red observation; the earlier ad-hoc r2 smoke recorded 489
pixels and is retained separately. The API, metrics and desktop checks passed
before that expected mobile failure. The regression establishes that the
assertion detects the old defect and passes for the new archive.

- R3 browser JSON SHA-256:
  `fb5001270344c2aeb5259905b22a482645af2b99aca78aba564c0c48e7aec363`.
- R2 red regression JSON SHA-256:
  `feef8d66578c96b92b8839fae49de8d39b9d297b67a9ba5a20924446b580ef62`.
- Final regression script SHA-256:
  `8a818389d9a3fe97df2feeebfdef6cd4b2a63583193f877078f86e6a7a19a2d6`.
- Desktop screenshot SHA-256:
  `356cfd82d3e1d8aea00c572f48773cf3c0bef3ca6ce8d9d3127d8767ec7301a5`.
- Mobile screenshot SHA-256:
  `a952bf4ce39db4248d670cd12c211361fb70b67b1258e5c223d5c9598e5d3907`.

The compact red/green JSON evidence and final desktop/mobile PNGs are also
preserved under `evidence/browser/` for repository review. The smoke host used
Node 25.8.1 and Chromium; the packaged app serves built assets and requires no
Node runtime. Test servers were deliberately stopped after successful
validation; their serving sessions are not left running.

## Review status and interpretation

The distinct `/root/release_reviewer` approved this exact archive for offline
research release promotion, with severity NONE. The structured external review
is `orchestration/reviews/AEON_SCALE_EXPANDED_OFFLINE_R3_REVIEW_20260929.json`.
Its SHA-256 is
`42cb8a077edf4c7ffe9a972ab525adbb8c69ac2ebaaba68f957036c620b5ec0b`.
The reviewer independently recomputed the ZIP/manifest/sidecar, all 11 source
revision evidence pins and 12 packaged slot/prediction/final-checkpoint pins;
checked both package verifiers, offline installed dependencies, listener and
API; and inspected the coordinator's browser script, red/green evidence and
screenshots. Browser execution belongs to the coordinator, not the reviewer.
The immutable internal package class correctly records final release review
pending at build time; this subsequent external approval governs those exact
unchanged bytes. R1 and r2 remain unapproved.

The expanded direct result is a modest single-seed, repeatedly inspected
validation observation. The 30k outcome is negative for both trained families.
Neither establishes SOTA, external generalization, useful JEPA representations
or commercial Marine validation. The earlier retrospective positive comparison
belongs to its original frozen families and cannot evaluate the newly derived
expanded-TRAIN models. Raw acoustic archives and intermediate checkpoints are
excluded; eight reviewed final development prediction/checkpoint binaries are
included. Historical v1 eligibility failures, 25 blocked registry entries and
prior research releases remain preserved.
