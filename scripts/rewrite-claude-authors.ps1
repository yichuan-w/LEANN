# Rewrite Claude-authored commits to Abinav (abinav_rao@berkeley.edu)
# Uses filter-branch to only rewrite branch-specific commits (fast, no full history rewrite)

# Ensure we're in repo root
$repoRoot = (git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) { Write-Error "Not in a git repo"; exit 1 }
Push-Location $repoRoot

$branches = @(
    "claude/feat-ocr-support-LONFb",
    "claude/feat-reindex-cli-LONFb",
    "claude/fix-cold-start-zmq-LONFb",
    "claude/issue-141-reindex-LONFb",
    "claude/issue-158-ocr-LONFb",
    "claude/issue-166-warmup-LONFb",
    "claude/issue-177-cold-start-LONFb",
    "claude/issue-217-llamaindex-LONFb",
    "claude/issue-233-hybrid-search-LONFb",
    "claude/issue-47-local-cursor-LONFb",
    "claude/issue-96-obsidian-LONFb",
    "claude/refactor-uv-workspace-LONFb",
    "claude/review-codebase-structure-LONFb"
)

foreach ($branch in $branches) {
    $remoteBranch = "origin/$branch"
    $newBranchName = $branch -replace "^claude/", "abinav/"
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Processing: $branch -> $newBranchName" -ForegroundColor Cyan
    
    $claudeCount = (git log $remoteBranch --author="Claude" --oneline 2>$null | Measure-Object -Line).Lines
    if ($claudeCount -eq 0) {
        Write-Host "  No Claude commits, skipping" -ForegroundColor Yellow
        continue
    }
    
    $base = (git merge-base origin/main $remoteBranch).Trim()
    $commitCount = (git rev-list --count $base..$remoteBranch)
    Write-Host "  $commitCount commit(s) to rewrite" -ForegroundColor Green
    
    # Remove refs/original from previous run (filter-branch backup)
    git for-each-ref --format="%(refname)" refs/original/ 2>$null | ForEach-Object { git update-ref -d $_ 2>$null }
    
    # Checkout and create new branch
    git checkout -B $newBranchName $remoteBranch 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Failed to checkout" -ForegroundColor Red
        git checkout main 2>$null
        continue
    }
    
    
    # filter-branch: use base..branch (explicit ref - HEAD can be ambiguous in some contexts)
    # Source the script so exports apply to filter-branch's shell; use absolute path
    $env:FILTER_BRANCH_SQUELCH_WARNING = "1"
    $scriptPath = "$($repoRoot -replace '\\', '/')/scripts/rewrite-author.sh"
    $range = "$base..$newBranchName"
    $filterResult = git filter-branch -f --env-filter ". $scriptPath" -- $range 2>&1
    
    if ($filterResult -match "was rewritten") {
        Write-Host "  Success! Top commits:" -ForegroundColor Green
        git log $newBranchName --format="  %h %an <%ae> %s" -5
        # Unset upstream so new branch doesn't track old claude remote
        git branch --unset-upstream $newBranchName 2>$null
    } else {
        Write-Host "  Failed: $($filterResult -join ' ')" -ForegroundColor Red
    }
    
    git checkout main 2>$null
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Done. New branches: abinav/*" -ForegroundColor Green
Write-Host "Push with: git push origin abinav/feat-ocr-support-LONFb abinav/feat-reindex-cli-LONFb ..." -ForegroundColor Yellow
Write-Host "Or: git push origin 'refs/heads/abinav/*:refs/heads/abinav/*'" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Pop-Location
