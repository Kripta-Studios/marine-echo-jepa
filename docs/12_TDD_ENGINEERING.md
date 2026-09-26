# Engineering workflow

Keep changes small and vertically usable: contract -> failing test -> minimal implementation
-> passing test -> refactor -> independent review -> integrate. Avoid an initial giant
implementation dump followed by superficial tests. All code, comments, UI, commit summaries,
issues, run reports and documentation are English.

One integrator owns dependency locks, shared contracts, experiment configs and main. Assign
explicit path ownership and a task branch/worktree to each concurrent builder. Do not allow
multiple agents to write one checkout's index, lockfiles or GPU state. Agents can research in
parallel; overlapping implementations are serialized or placed in separate worktrees and
reviewed before merging. Native subagent configuration is not filesystem isolation by itself.

Use conventional commits (`feat`, `fix`, `test`, `docs`, `refactor`, `chore`). Include task ID
in message or body. Never reset another worktree, force-push, rewrite history or remove
historical negative results. Default local commits; push only to the private repository remote
explicitly authorized by the owner. No automatic public visibility change.

Pin dependencies after resolution on the target runtime; commit `uv.lock` and the selected npm
lockfile. No `latest` tags in a reproducible release. Runtime manifests include Python, Node,
package versions, OS, hardware, git SHA, dirty status, seeds, time/cost and source hashes.
Dirty development runs are diagnostic. Final benchmark uses a clean committed code/config
state, and run output is written outside tracked source.

Use Pydantic for external boundaries, dataclasses/typed arrays for internal objects, explicit
units and UTC. Keep acoustic calibration in an adapter, not a web component. Avoid global
state, silent exception swallowing, unknown-column coercion, hardcoded user paths and broad
retry loops. Retry transient network faults with bounded exponential delay; do not retry
permission errors forever. All long loops report progress and resumable transactional state.

Runtime artifact names are unique, content-linked and never overwritten in place. Invalid
outputs move to a retained diagnostic namespace with a reason, not into the published catalog.
A lock identifies an actual PID/host and run ID; reclaim a stale lock only after proving its
owner is gone. Do not kill all Python or Node processes on the user's machine.

Keep architecture decision records for scope changes, calibration assumptions, model objective,
version conflicts, data fallback and test invalidation. Two-week constraints authorize dropping
P1 work; they do not authorize weakening numerical gates after outcomes or inventing a demo
result. Prefer a smaller complete vertical slice with honest baselines over an unfinished
research platform.
