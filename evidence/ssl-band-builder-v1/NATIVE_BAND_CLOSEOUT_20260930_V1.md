# Native band builder closeout — 2026-09-30

This closes the bounded implementation delivery, not scientific approval. Only this new report and the companion [strict handoff JSON](closeout-handoff-20260930-v1.json) were authored during closeout. Existing sources, evidence, reports, denial records and partial events remain unchanged.

Eight authored source/test SHA256 values match the frozen [AST proof](ast-proof-final-v1.json). The closeout used standard-library file/hash reads only, with exit 0; no project or Torch imports, tests or scientific execution were repeated. Frozen proof SHA256: `3b63c857c7ad5fe31fea806a2646c4c90d7983adeaa53ca100b2a9efeee8ca59`. An initial quoting SyntaxError returned exit 1 before file reads; correcting command quoting through the same cmd.exe launcher succeeded. This was not a filesystem denial.

The sole encoder change is parameter-free GELU on each channel's projected values plus native frequency/geometry metadata before observed-count pooling. Parameter tensors, initialization order and counts remain matched: 2,457,621 total, 2,424,448 encoder and 5,189 readout. Shared context/independent future target gradients and encoded SIGReg 0.03 remain; CF is unchanged and rejected by this family. Frozen AST evidence records unchanged training/selection loop structure and preserved original sources.

Band loaders enforce distinct band artifact kinds, exact architecture/config/scalers/tensors and safe weights-only loading. CPU inference freezes state without RNG reset, determinism or cache initialization. Encoder-only input is context/masks/native metadata; forecast additionally accepts issued query, never targets or future crops. Native 0–230 m stays 0–230 m. Labels distinguish SSL, permuted pairing control, supervised direct and untrained random encoders. Downstream supervised ancestry remains explicit.

Recorded builder evidence: [red mechanism](red-mechanism-v1.txt), exit 1 with the meaningful legacy cancellation failure; [final green](green-final-v1.txt), exit 0, 77 passed and six optimizer-dependent checks NOT_RUN/skipped in 43.69 s; [Ruff](lint-final-v1.txt), [format check](format-check-final-v1.txt) and frozen AST proof, exit 0. Original optimizer/cache denial is preserved; no retry occurred.

Root separately reports 83 synthetic CPU checks passed in 64.18 s with NATIVE_BAND_ROOT_RUNNER_CHECKS=1. Its first run had 77 passes and six failures from NumPy ZipFile io.open bypassing explicit virtual transport: FileNotFoundError, not permission denial. Root repaired only its test-only band_test_support.py to pass a virtual file handle through the real NumPy codec. Its second run exercised synthetic optimizers, exact resume, frozen versus updated encoders and inference replay. These are owner-reported root checks, not builder reruns or independent review. Root evidence is ../marine-echo-jepa/evidence/ssl-research-v1/band-root-durable-tests-01.log, band-root-durable-tests-02.log and band-snapshot-integration.json. All eight authored bytes remained unchanged.

Budget is OWNER_CLARIFICATION_REQUIRED; the owner question remains pending and there is no source-prefit approval. Non-smoke real direct screening is intentionally rejected by native_band_ssl.run in favor of the fixed native_band_downstream direct_end_to_end trajectory. ADR0020's eleven proposed ceiling does not mean all eleven recipes are executable or required. This closeout authorizes no budget expansion, fit, selection, scientific assessment or improvement claim.

Current import closure may now resolve identical newly integrated root band source. Do not rerun the historical proof expecting exclusive builder import paths. The next held-out executor task requires a separate explicit resume.

| Authored path | SHA256 |
| --- | --- |
| src/marine_echo/models/native_band_temporal.py | 1550bb3f130e3981f57b46031e39d0d42ce7dac4738aea9280362210dfaca96b |
| src/marine_echo/training/native_band_ssl.py | ca33bf3141ea17c5adb3d0a4cd2ed0d186be8efb846a5b581003f2088a105e4f |
| src/marine_echo/training/native_band_downstream.py | 95becf2cd01a7ab07c67a4da8d26c8e8bf58297f56513a439b0ab2b2be776619 |
| src/marine_echo/inference/native_band_acoustic.py | 6181baf22b75be73767f4b76370c65bfc37fdeb189da3deabcabe8a7c2dcd1bf |
| tests/unit/test_native_band_temporal.py | 6e98d15bda040fe0122c1495fd112ca7ee43fe90c7cbb7814697ba5b49dc09fc |
| tests/unit/test_native_band_inference.py | 7ae660584a7980e339aa035d1520c1c06d06a257943c739e31e09ecabb70752a |
| tests/integration/test_native_band_ssl.py | aca0374a21ebc28a4af357e2431722a34a6b2dc8b9a4741a7c1140334d834fab |
| tests/integration/test_native_band_downstream.py | 4d20681481e7c78918e46ae6d30b3c28779b845bf24164573f44c21e90d0e2c7 |

