# Release Guide

## 📋 Prerequisites

Before releasing, ensure:
1. ✅ All code changes are committed and pushed
2. ✅ CI has passed on the latest commit (check [Actions](https://github.com/yichuan-w/LEANN/actions/workflows/ci.yml))
3. ✅ You have determined the new version number

### Optional: TestPyPI Configuration

To enable TestPyPI testing (recommended but not required):
1. Get a TestPyPI API token from https://test.pypi.org/manage/account/token/
2. Add it to repository secrets: Settings → Secrets → Actions → New repository secret
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: Your TestPyPI token (starts with `pypi-`)

**Note**: TestPyPI testing is optional. If not configured, the release will skip TestPyPI and proceed.

## 🚀 Recommended: Manual Release Workflow

### Via GitHub UI (Most Reliable)

1. **Verify CI Status**: Check that the latest commit has a green checkmark ✅
2. Go to [Actions → Manual Release](https://github.com/yichuan-w/LEANN/actions/workflows/release-manual.yml)
3. Click "Run workflow"
4. Enter version (e.g., `0.1.1`)
5. Toggle "Test on TestPyPI first" if desired
6. Click "Run workflow"

**What happens:**
- ✅ Validates version format
- ✅ Downloads pre-built packages from CI (no rebuild needed!)
- ✅ Updates all package versions
- ✅ Optionally tests on TestPyPI
- ✅ Creates tag and GitHub release
- ✅ Automatically triggers PyPI publish

### Via Command Line

```bash
gh workflow run release-manual.yml -f version=0.1.1 -f test_pypi=true
```

## ⚡ Quick Release (One-Line)

For experienced users who want the fastest path:

```bash
./scripts/release.sh 0.1.1
```

This script will:
1. Update all package versions
2. Commit and push changes
3. Create GitHub release
4. CI automatically builds and publishes to PyPI

⚠️ **Note**: If CI fails, you'll need to manually fix and re-tag

## Manual Testing Before Release

For testing specific packages locally (especially DiskANN on macOS):

```bash
# Build specific package locally
./scripts/build_and_test.sh diskann  # or hnsw, core, meta, all

# Test installation in a clean environment
python -m venv test_env
source test_env/bin/activate
pip install packages/*/dist/*.whl

# Upload to Test PyPI (optional)
./scripts/upload_to_pypi.sh test

# Upload to Production PyPI (use with caution)
./scripts/upload_to_pypi.sh prod
```

## First-time setup

1. Install GitHub CLI:
   ```bash
   brew install gh
   gh auth login
   ```

2. Set PyPI token in GitHub:
   ```bash
   gh secret set PYPI_API_TOKEN
   # Paste your PyPI token when prompted
   ``` 