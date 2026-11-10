# Is the bot still behaving like the known-good version (18 Oct 2026, gentle-tone update)?
#   1) offline tests: tools, booking flow, safety guards, Exotel bridge (free, ~1 min)
#   2) 19 scripted test conversations with the real Claude, graded by a second Claude
#      (~6-9 min, uses your Anthropic credits), each also checked for polite, gentle wording.
#      Known-good result: 100% pass, 100% safety.
# Run it after ANY change: clinic data, prompts, code, .env.local, or a package update.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force logs | Out-Null

Write-Host "`n== 1/2  Offline tests" -ForegroundColor Cyan
uv run --frozen python tools/logtee.py logs\check-bot-offline.log -- python -m pytest -q
$offline = $LASTEXITCODE

Write-Host "`n== 2/2  Live test conversations (19 scenarios x 2, Claude Sonnet)" -ForegroundColor Cyan
uv run --frozen python tools/logtee.py logs\check-bot-live.log -- python bench/evaluate.py --runs 2 --concurrency 4
$live = $LASTEXITCODE

Write-Host ""
if ($offline -eq 0 -and $live -eq 0) {
    Write-Host "PASS: behaves like the known-good version (>= 95% of conversations pass, 100% of safety ones)." -ForegroundColor Green
} else {
    Write-Host "FAIL: something changed (offline tests exit $offline, live conversations exit $live)." -ForegroundColor Red
    Write-Host "Details: logs\check-bot-offline.log and logs\check-bot-live.log" -ForegroundColor Red
    Write-Host "To go back: unzip ..\known-good\voicebot-known-good-2026-10-18-tone.zip over this folder." -ForegroundColor Red
}
Read-Host "Press Enter to close"
