# Native Codex orchestration and accountability

Official model identifiers and Codex project subagent configuration were checked on
2026-09-26 [S11–S14]. Account access, client version and supported effort levels must still
be verified locally before delegation. Requested identities:

| Role | Model | Effort | Responsibility |
|---|---|---|---|
| Leader | `gpt-6-astra` | `high` | design, task DAG, integration, scientific decisions |
| Builder | `gpt-6-sol` | `high` | data, models, API and substantive implementation |
| Reviewer | `gpt-5.6-sol` | `high` | independent adversarial/scientific/security review |
| Small-task builder | `gpt-6-luna` | `max` | narrowly scoped presentational/config/script tasks |

The main session uses `.codex/config.toml`; named subagents use `.codex/agents/*.toml`.
Do not implement an LLM API wrapper, fake orchestration dashboard or homegrown chat proxy.
Do not merely put four role names in one prompt and claim four independent models ran.
Record actual session/agent identifiers, model binding and effort in the task report when
exposed by the client; otherwise record exactly what was configured and what cannot be verified.
An unavailable binding is a real orchestration blocker. Never substitute/relabel without owner
approval. Continue unaffected work where honest, clearly marking lack of independent review.

## Resource and ownership rules

At most three concurrent subagent threads per leader session. Start one Sol implementation
thread, one Luna small-task thread and an independent reviewer when their inputs are ready.
Do not spawn a new team per function or create recursively expanding agents. Subagents cannot
spawn additional agents without leader approval. The leader remains responsible for the full
pipeline and does not disappear after writing tasks.

Separate Git worktrees for concurrent writers, or strict serialized writes when the runtime
cannot change an agent's working directory safely. Include absolute assigned working directory
and allowed paths in each task. Shared source contracts, locks and experiment configs have one
owner. Changes outside the task scope require escalation, not an optimistic merge.

Luna may implement layout skeletons, ordinary form components, copy, straightforward PowerShell
wrappers or JSON configuration formatting with exact acceptance tests. Sol reviews all such
changes. Units, timestamp logic, calibration, data splits, loss functions, checkpoint loading,
scientific plots, security and budget enforcement are never delegated solely to Luna.

The reviewer does not author the feature it approves. It receives source/spec and redacted
outcome-free protocol for pre-test review, runs or proposes adversarial tests in an isolated
review worktree, and returns a structured verdict. Its default sandbox is read-only; executable
tests needing writes require a separately approved temporary workspace or a reviewed execution
by an implementer. Do not claim independent execution when only logs were inspected.
The leader persists reviewer findings and resolves issues through a different builder.

## Task state machine

`TODO -> READY -> RUNNING -> REVIEW -> DONE`, with `BLOCKED`, `FAILED` and `DROPPED_P1` explicit.
`DONE` requires dependencies, tests, artifacts and review evidence. Updating a checkbox is not
a deliverable. Each report includes task ID, parent dependencies, actual model/session, branch,
commit, owned paths, commands, red/green logs, artifacts, issues, time/cost and next action.
`orchestration/tasks.json` is the initial DAG; update status without deleting its requirements.

Checkpoint `orchestration/STATUS.md` after each integrated task. On session limits, leave all
running jobs in a documented safe state with PID, log and resume command. Do not assume another
agent can attach to a training process without checking it. Use `CODEX_RESUME_PROMPT.md`.

## Independent review checkpoints

R0: data/units/license and loader/calibration feasibility.
R1: past-only windows, split/protocol/metrics, before model selection.
R2: encoder objectives and matched comparisons, before final test.
R3: frozen release, security/offline behaviour, final metrics reconstruction.

A reviewer veto of a material flaw blocks promotion until fixed. The leader may disagree in a
written ADR, but cannot relabel an unresolved veto as an independent approval. Preserve a
negative research result, and serve a conventional baseline when it is better supported.
