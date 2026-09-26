# Start here

## What you received

This package is an English implementation specification, native Codex role configuration,
initial task graph, data-access tools, runtime preflight, and handoff checks. It does **not**
contain downloaded acoustic archives, trained checkpoints, or the finished web application.
`evidence/HANDOFF_QA.md` records only checks actually run while creating this handoff.

## Create a private repository

Name: `marine-echo-jepa`

Description:
`Public-data acoustic forecasting MVP with compact JEPA/LeWorldModel-inspired models, uncertainty-aware replay, and an offline-first web app.`

Use a new repository rather than forking the complete industrial or logistics applications.
Keep reusable upstream components small, attributed, and pinned.

## Windows setup

Use a short local path, for example `C:\dev\marine-echo-jepa`. Extract this ZIP directly into
that folder. The ZIP has files at its root; do not extract it into an existing project.

In PowerShell:

```powershell
$Repo = 'C:\dev\marine-echo-jepa'
New-Item -ItemType Directory -Force $Repo | Out-Null
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\marine-echo-jepa-codex-handoff.zip" -DestinationPath $Repo
Set-Location $Repo
git init -b main
git add .
git commit -m "docs: add Marine Echo JEPA implementation contract"
# Optional: requires installed/authenticated GitHub CLI.
gh repo create Kripta-Studios/marine-echo-jepa --private --source . --remote origin --push
```

The commit may require your existing Git name/email configuration. Do not invent an identity.
If you created the repository through GitHub's website, add its remote instead of running
`gh repo create` again. No global dependency installation is required to open Codex.

## Start the leader

```powershell
Set-Location 'C:\dev\marine-echo-jepa'
codex --version
codex --cd . --model gpt-6-astra
```

The project configuration sets the leader effort to `high`; verify it in the session.
Paste the complete contents of `CODEX_START_PROMPT.md`. Alternatively, after reviewing the
scripts, run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Start-Codex.ps1`;
it sends a short instruction to read that file, not thousands of characters through a shell.
The execution-policy flag is process-local; no permanent policy change is needed.

Codex must verify the installed client supports the provided project-scoped agent files and
that your account can access each specified model/effort. Official names were checked at
preparation, but availability is account-specific. Never silently substitute a model.

## Optional handoff checks before asking the agent to work

With `uv` already installed:

```powershell
uv run --no-project --python 3.12 python tools/verify_handoff.py
uv run --no-project --python 3.12 python -m unittest discover -s tests_handoff -v
uv run --no-project --python 3.12 python tools/probe_data.py
```

These commands test the handoff and probe data hosts, not the future application. The data
probe does not download the multi-gigabyte archive. Read its recorded outcome.

## Expected agent behaviour

The leader works through the task DAG. It delegates bounded implementation to Sol High,
small low-risk changes to Luna Max, and independent review to GPT-5.6 Sol High. One process
owns the GPU and one integrator owns shared lockfiles. Agent sessions and GPU processes are
separate resources: pausing an agent does not magically stop or resume a training process.

The owner may need to complete login, permissions, cloud provisioning, or a service signup.
These are real access boundaries, not reasons to skip unaffected work. No prompt guarantees
uninterrupted completion through quotas, authentication, or session limits; durable checkpoints
and `CODEX_RESUME_PROMPT.md` provide continuation.
