# CI Preflight — Run before pushing to catch CI failures
# Matches upstream (yichuan-w/LEANN) CI checks

$ErrorActionPreference = "Stop"
$repoRoot = (git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) { Write-Error "Not in a git repo"; exit 1 }
Push-Location $repoRoot

Write-Host "`n=== CI Preflight (lint + type-check) ===" -ForegroundColor Cyan
Write-Host ""

# 1. Lint
Write-Host "[1/2] Lint and Format Check..." -ForegroundColor Yellow
uv run --only-group lint pre-commit run --all-files --show-diff-on-failure
if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAILED. Install uv: winget install astral-sh.uv" -ForegroundColor Red
    Pop-Location
    exit 1
}
Write-Host "  OK" -ForegroundColor Green

# 2. Type check
Write-Host "[2/2] Type Check (ty)..." -ForegroundColor Yellow
uv tool install ty 2>$null | Out-Null
ty check packages/leann-core/src apps tests
if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAILED" -ForegroundColor Red
    Pop-Location
    exit 1
}
Write-Host "  OK" -ForegroundColor Green

Pop-Location
Write-Host "`nPreflight passed. Safe to push." -ForegroundColor Green
exit 0
