# LEANN Tests

This directory contains automated tests for the LEANN project using pytest.

## Test Files

### `test_readme_examples.py`
Tests the examples shown in README.md:
- The basic example code that users see first (parametrized for both HNSW and DiskANN backends)
- Import statements work correctly
- Different backend options (HNSW, DiskANN)
- Different LLM configuration options (parametrized for both backends)
- **All main README examples are tested with both HNSW and DiskANN backends using pytest parametrization**

### `test_basic.py`
Basic functionality tests that verify:
- All packages can be imported correctly
- C++ extensions (FAISS, DiskANN) load properly
- Basic index building and searching works for both HNSW and DiskANN backends
- Uses parametrized tests to test both backends

### `test_document_rag.py`
Tests the document RAG example functionality:
- Tests with facebook/contriever embeddings
- Tests with OpenAI embeddings (if API key is available)
- Tests error handling with invalid parameters
- Verifies that normalized embeddings are detected and cosine distance is used

### `test_diskann_partition.py`
Tests DiskANN graph partitioning functionality:
- Tests DiskANN index building without partitioning (baseline)
- Tests automatic graph partitioning with `is_recompute=True`
- Verifies that partition files are created and large files are cleaned up for storage saving
- Tests search functionality with partitioned indices
- Validates medoid and max_base_norm file generation and usage
- Includes performance comparison between DiskANN (with partition) and HNSW
- **Note**: These tests are skipped in CI due to hardware requirements and computation time

## Running Tests

### Install test dependencies:
```bash
# Using extras
uv pip install -e ".[test]"
```

### Run all tests:
```bash
pytest tests/

# Or with coverage
pytest tests/ --cov=leann --cov-report=html

# Run in parallel (faster)
pytest tests/ -n auto
```

### Run specific tests:
```bash
# Only basic tests
pytest tests/test_basic.py

# Only tests that don't require OpenAI
pytest tests/ -m "not openai"

# Skip slow tests
pytest tests/ -m "not slow"

# Run DiskANN partition tests (requires local machine, not CI)
pytest tests/test_diskann_partition.py
```

### Run with specific backend:
```bash
# Test only HNSW backend
pytest tests/test_basic.py::test_backend_basic[hnsw]
pytest tests/test_readme_examples.py::test_readme_basic_example[hnsw]

# Test only DiskANN backend
pytest tests/test_basic.py::test_backend_basic[diskann]
pytest tests/test_readme_examples.py::test_readme_basic_example[diskann]

# All DiskANN tests (parametrized + specialized partition tests)
pytest tests/ -k diskann
```

## CI/CD Integration

Tests are automatically run in GitHub Actions:
1. After building wheel packages
2. On multiple Python versions (3.9 - 3.13)
3. On both Ubuntu and macOS
4. Using pytest with appropriate markers and flags

### pytest.ini Configuration

The `pytest.ini` file configures:
- Test discovery paths
- Default timeout (600 seconds)
- Environment variables (HF_HUB_DISABLE_SYMLINKS, TOKENIZERS_PARALLELISM)
- Custom markers for slow and OpenAI tests
- Verbose output with short tracebacks

### Known Issues

- OpenAI tests are automatically skipped if no API key is provided
