# LEANN Tests

This directory contains automated tests for the LEANN project, primarily used in CI/CD pipelines.

## Test Files

### `test_ci_basic.py`
Basic functionality tests that verify:
- All packages can be imported correctly
- C++ extensions (FAISS, DiskANN) load properly
- Basic index building and searching works for both HNSW and DiskANN backends

### `test_main_cli.py`
Tests the main CLI example functionality:
- Tests with facebook/contriever embeddings
- Tests with OpenAI embeddings (if API key is available)
- Verifies that normalized embeddings are detected and cosine distance is used

## Running Tests Locally

### Basic tests:
```bash
python tests/test_ci_basic.py
```

### Main CLI tests:
```bash
# Without OpenAI API key
python tests/test_main_cli.py

# With OpenAI API key
OPENAI_API_KEY=your-key-here python tests/test_main_cli.py
```

## CI/CD Integration

These tests are automatically run in the GitHub Actions workflow:
1. After building wheel packages
2. On multiple Python versions (3.9 - 3.13)
3. On both Ubuntu and macOS

### Known Issues

- On macOS, there might be C++ standard library compatibility issues that cause tests to fail
- The CI is configured to continue on macOS failures to avoid blocking releases
- OpenAI tests are skipped if no API key is provided in GitHub secrets

## Test Data

Tests use the example data in `examples/data/`:
- `PrideandPrejudice.txt` - Text file for testing
- PDF files for document processing tests 