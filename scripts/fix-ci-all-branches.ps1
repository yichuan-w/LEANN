# Fix CI (lint/ruff) on all 8 PR branches
# Run from repo root - matches CI: pre-commit run --all-files

$ErrorActionPreference = "Continue"
$branches = @(
    "abinav/issue-177-cold-start-LONFb",
    "abinav/issue-233-hybrid-search-LONFb",
    "abinav/issue-141-reindex-LONFb",
    "abinav/issue-158-ocr-LONFb",
    "abinav/issue-217-llamaindex-LONFb",
    "abinav/issue-166-warmup-LONFb",
    "abinav/issue-96-obsidian-LONFb",
    "abinav/issue-47-local-cursor-LONFb"
)

$repoRoot = (git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) { Write-Error "Not in git repo"; exit 1 }
Set-Location $repoRoot

foreach ($branch in $branches) {
    Write-Host "`n========== $branch ==========" -ForegroundColor Cyan
    git checkout $branch 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Skip (branch not found)" -ForegroundColor Yellow
        continue
    }

    # Run ruff (same as pre-commit ruff hooks)
    $ruffCheck = python -m ruff check packages/leann-core apps tests --fix 2>&1
    $ruffFormat = python -m ruff format packages/leann-core apps tests 2>&1

    $changed = git status --porcelain
    if ($changed) {
        git add -A
        git commit -m "fix(ci): apply ruff check and format"
        git push origin $branch
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Pushed" -ForegroundColor Green
        } else {
            Write-Host "  Push failed" -ForegroundColor Red
        }
    } else {
        Write-Host "  No changes" -ForegroundColor Gray
    }
}

Write-Host "`nDone." -ForegroundColor Cyan
