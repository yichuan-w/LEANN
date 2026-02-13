# CI Preflight — Run Before Pushing PRs

**Target:** PRs to [yichuan-w/LEANN](https://github.com/yichuan-w/LEANN) (upstream)

The upstream CI runs these checks. Run them locally before pushing to catch failures early.

---

## Prerequisites

```powershell
# Install uv (recommended - matches CI exactly)
winget install astral-sh.uv
# OR: pip install uv
```

---

## Commands (match upstream CI exactly)

### 1. Lint and Format Check
```powershell
uv run --only-group lint pre-commit run --all-files --show-diff-on-failure
```
- Runs: ruff (check + fix), ruff-format, pre-commit-hooks
- Fails if any file is modified (fix and re-run)

### 2. Type Check with ty
```powershell
uv tool install ty
ty check packages/leann-core/src apps tests
```
- Catches type errors before CI

### 3. Quick script
```powershell
.\scripts\ci-preflight.ps1
```

---

## Upstream CI Jobs (yichuan-w/LEANN)

| Job | Command | Runs on |
|-----|---------|---------|
| Lint and Format Check | `uv run --only-group lint pre-commit run --all-files --show-diff-on-failure` | ubuntu-latest, Python 3.11 |
| Type Check with ty | `ty check packages/leann-core/src apps tests` | ubuntu-latest |
| Build | Multi-platform (Linux, macOS, ARM64) | After lint + type-check pass |
| Tests | `pytest tests/ -v --tb=short` | After build |
| Arch smoke test | Import + minimal runtime check | After build |

---

## Note: Fork vs Upstream

Your fork (`raoabinav/LEANN`) may have an older workflow without the **Type Check with ty** job. PRs to upstream use upstream's CI, which includes type-check. Always run both lint and type-check before pushing.
