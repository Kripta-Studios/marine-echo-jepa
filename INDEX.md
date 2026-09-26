# Package index

Start with [START_HERE.md](START_HERE.md) and [CODEX_START_PROMPT.md](CODEX_START_PROMPT.md).

## English specification and execution documents

- [Repository instructions](AGENTS.md)
- [Resume `marine-echo-jepa` as GPT-6 Astra high. Read AGENTS.md, CODEX_START_PROMPT.md,](CODEX_RESUME_PROMPT.md)
- [You are the end-to-end leader for `marine-echo-jepa`, running GPT-6 Astra with high reasoning](CODEX_START_PROMPT.md)
- [Marine Echo JEPA](README.md)
- [Start here](START_HERE.md)
- [Independent review task](agent_prompts/INDEPENDENT_REVIEWER.md)
- [Leader task contract](agent_prompts/LEADER.md)
- [Small, bounded implementation task](agent_prompts/LUNA_SMALL_TASKS.md)
- [Substantive implementation task](agent_prompts/SOL_BUILDER.md)
- [Product brief and decision](docs/00_PRODUCT_BRIEF.md)
- [Scope and acceptance](docs/01_SCOPE_ACCEPTANCE.md)
- [Datasets and access plan](docs/02_DATASETS_ACCESS.md)
- [Acoustic data contract](docs/03_ACOUSTIC_DATA_CONTRACT.md)
- [Preprocessing and acoustic integrity](docs/04_PREPROCESSING.md)
- [Models and training implementation](docs/05_MODELS.md)
- [Experiment protocol — freeze before accessing test outcomes](docs/06_EXPERIMENT_PROTOCOL.md)
- [Compute, Windows and cloud budget](docs/07_COMPUTE_WINDOWS_CLOUD.md)
- [System architecture](docs/08_SYSTEM_ARCHITECTURE.md)
- [API contract v1](docs/09_API_CONTRACT.md)
- [Web app: Acoustic Forecast & Freshness Lab](docs/10_WEB_APP_UX.md)
- [Test strategy and proof obligations](docs/11_TEST_STRATEGY.md)
- [Engineering workflow](docs/12_TDD_ENGINEERING.md)
- [Native Codex orchestration and accountability](docs/13_AGENT_ORCHESTRATION.md)
- [Two-week execution schedule](docs/14_TWO_WEEK_EXECUTION.md)
- [Security, permissions and licensing](docs/15_SECURITY_LICENSES.md)
- [Meeting playbook](docs/16_MEETING_PLAYBOOK.md)
- [Proposed Marine Instruments transfer — not implemented integration](docs/17_MARINE_TRANSFER_DATA_REQUEST.md)
- [Reuse and source audit](docs/18_REUSE_SOURCE_AUDIT.md)
- [Research choices as of 2026-09-26](docs/19_RESEARCH_CHOICES.md)
- [Definition of done and release matrix](docs/20_DEFINITION_OF_DONE.md)
- [Risks and predetermined fallback decisions](docs/21_RISKS_FALLBACKS.md)
- [Command contract for the implementation agent](docs/22_COMMAND_REFERENCE.md)
- [Handoff preparation QA — 2026-09-26](evidence/HANDOFF_QA.md)
- [Execution status](orchestration/STATUS.md)
- [Source ledger](references/SOURCES.md)

## Machine-readable and executable assets

`configs/`: dataset links/access limits, model/compute defaults, finite experiment budget and agent bindings.
`schemas/`: forecast interchange schema; semantic oracles are separate.
`.codex/`: project leader settings and three standalone native subagent configurations.
`orchestration/tasks.json`:22 dependency-linked tasks, all initially TODO.
`tools/`: access/runtime probes, original-handoff verification, numerical/forecast oracles and ZIP metadata guard.
`scripts/`: PowerShell launch/verification wrappers, not yet Windows-executed.
`tests_handoff/`:51 helper tests and a strictly synthetic contract fixture.
`evidence/`: preparation logs and limitations; not trained-model or application evidence.
`HANDOFF_SHA256SUMS`:original-distribution integrity manifest.
