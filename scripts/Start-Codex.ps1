[CmdletBinding()]
param([string]$Repo = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path -LiteralPath $Repo).Path
if (-not (Test-Path -LiteralPath (Join-Path $Root 'CODEX_START_PROMPT.md'))) {
    throw 'CODEX_START_PROMPT.md is missing. Use the extracted handoff repository.'
}
$Codex = Get-Command codex -ErrorAction Stop
Set-Location -LiteralPath $Root
$Prompt = 'Read CODEX_START_PROMPT.md and execute the complete P0 implementation contract. Begin with preflight, native role binding verification and real data access.'
& $Codex.Source --cd $Root --model 'gpt-6-astra' $Prompt
if ($LASTEXITCODE -ne 0) { throw "Codex exited with code $LASTEXITCODE. Inspect the actual error; do not bypass permissions." }
