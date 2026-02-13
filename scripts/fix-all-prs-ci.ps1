# Fix CI (ruff lint/format) on all 8 PR branches and push
# Run from repo root

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
Push-Location $repoRoot

foreach ($branch in $branches) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Branch: $branch" -ForegroundColor Cyan
    
    git checkout $branch 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Skip (branch not found)" -ForegroundColor Yellow
        continue
    }
    
    # Run ruff (matches pre-commit)
    $ruffCheck = python -m ruff check packages/leann-core apps tests --fix 2>&1
    $ruffFormat = python -m ruff format packages/leann-core apps tests 2>&1
    
    $changed = git status --porcelain
    if ($changed) {
        git add -A
        git commit -m "fix: apply ruff check and format for CI"
        git push origin $branch
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Pushed" -ForegroundColor Green
        } else {
            Write-Host "  Push failed" -ForegroundColor Red
        }
    } else {
        Write-Host "  No changes needed" -ForegroundColor Gray
    }
}

Pop-Location
Write-Host "`nDone." -ForegroundColor Cyan
