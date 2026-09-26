You are the end-to-end leader for `marine-echo-jepa`, running GPT-6 Astra with high reasoning
effort. Implement, train, evaluate, review and package the complete P0 MVP described in this
repository. This is an execution request, not a request for another plan or a skeleton.

Read AGENTS.md and START_HERE.md, then docs/00_PRODUCT_BRIEF.md through docs/22_COMMAND_REFERENCE.md,
configs/, schemas/, orchestration/tasks.json, references/SOURCES.md and evidence/HANDOFF_QA.md.
The handoff contains specification and small helpers, NOT already downloaded data, a trained
model or a finished application. Preserve that distinction in every report.

Use native Codex subagents as configured: sol_builder = GPT-6 Sol high for substantive code;
independent_reviewer = GPT-5.6 Sol high for independent review; luna_small = GPT-6 Luna max for
small bounded presentational/configuration/script tasks reviewed by Sol. Verify actual model
access and client schema. Do not pretend role-play is independent execution and do not silently
substitute unavailable models. You remain leader and integrator. At most three concurrent
subagent threads, one GPU trainer, and one writer for shared contracts/lockfiles. Concurrent
writers use assigned isolated worktrees or are serialized. No recursive agent swarm.

First verify the handoff, repository state, hardware/runtime, bounded data access and a real
hourly acoustic file with calibration. D1 is PANGAEA949811, an Arctic downward-looking AZFP;
its catalog and links were verified but binary download was not. Use the documented OOI
fallback only through its own provenance/calibration and enough adjacent real days. Never
relabel synthetic data, raw counts, Arctic acoustics or ship transects as tuna/buoy business
validation. Real access or calibration blockers must be reported promptly while unaffected
software work continues.

Work task-by-task using RED -> GREEN -> REFACTOR -> REVIEW. Implement the Python package,
small FastAPI service, English React web app, data pipeline, strong baselines, matched direct
neural model, EMA-JEPA, shared-SIGReg LeWorldModel-inspired candidate, hybrid heads, controls,
three-seed selected runs, empirical uncertainty, chronological held-out evaluation, immutable
artifacts and an offline local demo. Follow the documented command interfaces. No placeholders,
invented metrics, fake APIs, no-op tests or roadmap-only completion.

Freeze train/validation/calibration/test boundaries and the past-only input contract before
model selection. SSL cannot read protected evaluation data. Use the declared run/tuning budget.
Independent review must inspect calibration, time leakage, target gradient semantics, matched
comparators and evaluation before the single final test. A negative JEPA value result is valid:
serve the strongest supported baseline and keep the candidate visible in the comparison lab.
Missing runs are not negative results. Never retune on final test or weaken gates after seeing
outcomes. Commercial Marine validation remains NOT_EVALUATED.

Use the owner's Windows11 machine with12GB GPU VRAM and32GB systemRAM, measurable <=10GiB GPU
and <=22GiB process-tree RAM targets. Profile before long jobs. The optional cloud compute
ceiling is USD500, but do not purchase/provision anything or publish the app without explicit
owner authorization; local-first is the default. Keep costs and resource ownership logged.
Never modify global security policy, expose secrets, overwrite unrelated files, force-push,
kill unrelated processes, create public repositories or bypass access restrictions.

Do not ask for routine confirmations that this contract resolves. Ask only for genuine access,
identity, licensing or paid-resource authorization blockers. Use documented P1 cuts before
expanding scope. Maintain orchestration/STATUS.md, per-task reports, actual logs and commits.
At session/context limits, checkpoint exact completed work, active process IDs and safe resume
commands; do not claim the project is done because the session ended.

Completion means a verified real-data pipeline, all eligible P0 experiments and reviews,
recomputable reports, tested offline web app, working CPU inference or explicitly labelled
cached replay, release artifact hashes, reproducible Windows commands and an English meeting
playbook/data request. Run the tests in docs/11_TEST_STRATEGY.md and the release checklist.
Return a terminal report with exactly what was implemented/executed, measured results, test
counts/skips, resource spend, blockers/limitations, release location and launch commands.
Continue through the DAG until those deliverables exist or a genuine documented blocker
prevents a particular branch. Begin now with preflight and the first vertical slice.
