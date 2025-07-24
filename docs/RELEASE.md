# Release Guide

## One-line Release 🚀

```bash
./scripts/release.sh 0.1.1
```

That's it! This script will:
1. Update all package versions
2. Commit and push changes
3. Create GitHub release
4. CI automatically builds and publishes to PyPI

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

### Why Manual Build for DiskANN?

DiskANN's complex dependencies (protobuf, abseil, etc.) sometimes require local testing before release. The build script will:
- Compile the C++ extension
- Use `delocate` (macOS) or `auditwheel` (Linux) to bundle system libraries
- Create a self-contained wheel with no external dependencies

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