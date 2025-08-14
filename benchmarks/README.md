# 🧪 LEANN Benchmarks & Testing

This directory contains performance benchmarks and comprehensive tests for the LEANN system, including backend comparisons and sanity checks across different configurations.

## 📁 Test Files

### `diskann_vs_hnsw_speed_comparison.py`
Performance comparison between DiskANN and HNSW backends:
- ✅ **Search latency** comparison with both backends using recompute
- ✅ **Index size** and **build time** measurements
- ✅ **Score validity** testing (ensures no -inf scores)
- ✅ **Configurable dataset sizes** for different scales

```bash
# Quick comparison with 500 docs, 10 queries
python benchmarks/diskann_vs_hnsw_speed_comparison.py

# Large-scale comparison with 2000 docs, 20 queries
python benchmarks/diskann_vs_hnsw_speed_comparison.py 2000 20
```

### `test_distance_functions.py`
Tests all supported distance functions across DiskANN backend:
- ✅ **MIPS** (Maximum Inner Product Search)
- ✅ **L2** (Euclidean Distance)
- ✅ **Cosine** (Cosine Similarity)

```bash
uv run python tests/sanity_checks/test_distance_functions.py
```

### `test_l2_verification.py`
Specifically verifies that L2 distance is correctly implemented by:
- Building indices with L2 vs Cosine metrics
- Comparing search results and score ranges
- Validating that different metrics produce expected score patterns

```bash
uv run python tests/sanity_checks/test_l2_verification.py
```

### `test_sanity_check.py`
Comprehensive end-to-end verification including:
- Distance function testing
- Embedding model compatibility
- Search result correctness validation
- Backend integration testing

```bash
uv run python tests/sanity_checks/test_sanity_check.py
```

## 🎯 What These Tests Verify

### ✅ Distance Function Support
- All three distance metrics (MIPS, L2, Cosine) work correctly
- Score ranges are appropriate for each metric type
- Different metrics can produce different rankings (as expected)

### ✅ Backend Integration
- DiskANN backend properly initializes and builds indices
- Graph construction completes without errors
- Search operations return valid results

### ✅ Embedding Pipeline
- Real-time embedding computation works
- Multiple embedding models are supported
- ZMQ server communication functions correctly

### ✅ End-to-End Functionality
- Index building → searching → result retrieval pipeline
- Metadata preservation through the entire flow
- Error handling and graceful degradation

## 🔍 Expected Output

When all tests pass, you should see:

```
📊 测试结果总结:
  mips      : ✅ 通过
  l2        : ✅ 通过
  cosine    : ✅ 通过

🎉 测试完成!
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**: Ensure you're running from the project root:
```bash
cd /path/to/leann
uv run python tests/sanity_checks/test_distance_functions.py
```

**Memory Issues**: Reduce graph complexity for resource-constrained systems:
```python
builder = LeannBuilder(
    backend_name="diskann",
    graph_degree=8,  # Reduced from 16
    complexity=16    # Reduced from 32
)
```

**ZMQ Port Conflicts**: The tests use different ports to avoid conflicts, but you may need to kill existing processes:
```bash
pkill -f "embedding_server"
```

## 📊 Performance Expectations

### Typical Timing (3 documents, consumer hardware):
- **Index Building**: 2-5 seconds per distance function
- **Search Query**: 50-200ms
- **Recompute Mode**: 5-15 seconds (higher accuracy)

### Memory Usage:
- **Index Storage**: ~1-2 MB per distance function
- **Runtime Memory**: ~500MB (including model loading)

## 🔗 Integration with CI/CD

These tests are designed to be run in automated environments:

```yaml
# GitHub Actions example
- name: Run Sanity Checks
  run: |
    uv run python tests/sanity_checks/test_distance_functions.py
    uv run python tests/sanity_checks/test_l2_verification.py
```

The tests are deterministic and should produce consistent results across different platforms.
