# Native prefix configuration follow-up V3

SYNTHETIC_CORRECTNESS_ONLY. Implemented the bounded follow-up in orchestration/ssl_vnext_prefix_native_configuration_followup.txt (SHA256 d19602f92fe71b694f8f16391181954b4edaddb95540c1392f77d6620472f0d6). No public acoustic arrays, predictions or weights were decoded. No scientific fitting, GPU operation, approval or quality claim occurred.

The inspected native reader preserves nominal positive future indices even when a slot is structurally unavailable. The executor now separates those ordinals from actual raw observation identities. An unavailable target requires a false forecast mask, empty native metadata, UNKNOWN processing, zero ping counts and an independently bound structural-gap explanation. A positive requested ordinal or explicit -1 sentinel does not create an observation ID or timestamp. Available masked labels retain their actual identity/date; missing observed identities fail closed. Metadata-only future-chain records are distinguished from fitted contexts/labels.

The separately versioned native_prefix_raw_intervals_v2 registry uses configuration_mode per_source_configuration_map_v1 and a bound native_prefix_source_configuration_map_v1 document. Each source retains its deployment/site/archive/lifetime and complete contiguous ordinal segments. Each segment identifies actual per-channel geometry/processing/pings or an explicit source gap. Eligible 150- and 180-ping configurations remain in one deployment; 165-ping quarantine is retained. Observed context channels must match one issued 96-interval configuration. Masked secondary structural absence requires a channel-specific receipt. Future configuration/ping/clock boundaries require receipts rather than fabricated raw observations. Known raw timestamps cannot be contradicted by a clock-gap receipt.

Root must supply the separately bound configuration_map, source_metadata paths and canonical native_prefix_source_configuration_metadata_v1 receipts, with complete source identities/configurations/segments matching the derived map exactly. Independent source-gap receipts and their hashes remain required. The legacy single-configuration registry is explicitly synthetic-only; real admission requires V2 metadata. No scientific-access shortcut was added.

The 1/7/30-day prefixes, source-start+4-day label start and common source-start+41-day suffix remain fixed. Source-calendar timestamps are not verified UTC. Native primary 0–230m and original development 0–225m are preserved; actual secondary geometry is not relabelled. No row intersection, mask filling, recipe/seed/optimizer/schedule/scaler/model change or method selection was introduced. Typed supervised parents, SSL/control semantics and the existing band seed-seven guard remain unchanged.

Executed builder checks:
- Meaningful red: new configuration-map support failed against V2 (exit 1). Independent unchanged-source receipt versus forged derived geometry also failed (exit 1), then passed after the semantic provenance guard.
- Final command: ..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q --import-mode=importlib -p no:cacheprovider tests/unit/test_native_prefix_transfer.py tests/integration/test_native_prefix_transfer.py
- Final result: exit 0; 211 passed, eight skipped, 191.61 seconds. These are synthetic CPU correctness checks, including actual reader metadata behavior and private NumPy codecs.
- Ruff check and format-check of the three authored code/test files and three evidence helpers exited 0. Commands/results are recorded in lint-final-v3.log.
- Source proof exited 0: 27 tracked main files unchanged, all 28 closed V2 evidence files unchanged, other closed builder sources unchanged, and 29 recipe/model/ancestry/checkpoint AST entries unchanged.

Eight optimizer/resume/durable cases remain NOT_RUN in builder because their operations are root-only. No denied optimizer/cache/Git operation was retried. After checked integration, root runs the same focused pytest command from main with its authorized NATIVE_PREFIX_ROOT_RUNNER_CHECKS=1 environment, and independently validates durable receipts, real metadata compatibility and exact source/protocol review before any scientific fit. Builder did not execute these root checks or approve any fit.

Historical evidence is preserved, including the initial alias-fixture attempt that passed (not a red test), its corrected independent fixture, an earlier four-failure synthetic NaN fixture run, and a Windows import-launch diagnostic of unknown cause. All final checks use finite observed fixtures. Two ancillary read commands during closeout failed on command quoting (rg exit 2; python -c syntax exit 1); neither was a filesystem denial or scientific operation, and no permission/cache/process changes followed. An earlier oversized stdout snapshot transport was split into bounded read modes without a denied write retry.

Delivery: three changed allowed paths, before/final byte snapshots, bounded-final-v3.patch, source-proof-final-v3.json, immutable red/green logs, executed-checks-v3.json and closeout-handoff-v3.json. Root integrates only checked differences. Main remains read-only; no commit or Git index operation was attempted.

Final authored SHA256:
- src/marine_echo/training/native_prefix_transfer.py: 448e6a0e3044d457ed9f8d04fb6d07948eceb2240d6e0ed6f96ab5bfc8466181
- tests/unit/test_native_prefix_transfer.py: 9751a823476d6a6d170b0d9705ada3b8c03c3d804f374f7cdddb10e26c56a485
- tests/integration/test_native_prefix_transfer.py: b6166fc71a8e75de2cece68e1269a521ff7ad49e9de6520df7d71508bfcd0b80

