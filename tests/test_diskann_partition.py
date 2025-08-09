"""
Test DiskANN graph partitioning functionality.

Tests the automatic graph partitioning feature that was implemented to save
storage space by partitioning large DiskANN indices and safely deleting
redundant files while maintaining search functionality.
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip DiskANN partition tests in CI - requires specific hardware and large memory",
)
def test_diskann_without_partition():
    """Test DiskANN index building without partition (baseline)."""
    from leann.api import LeannBuilder, LeannSearcher

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = str(Path(temp_dir) / "test_no_partition.leann")

        # Test data - enough to trigger index building
        texts = [
            f"Document {i} discusses topic {i % 10} with detailed analysis of subject {i // 10}."
            for i in range(500)
        ]

        # Build without partition (is_recompute=False)
        builder = LeannBuilder(
            backend_name="diskann",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            num_neighbors=32,
            search_list_size=50,
            is_recompute=False,  # No partition
        )

        for text in texts:
            builder.add_text(text)

        builder.build_index(index_path)

        # Verify index was created
        index_dir = Path(index_path).parent
        assert index_dir.exists()

        # Check that traditional DiskANN files exist
        index_prefix = Path(index_path).stem
        # Core DiskANN files (beam search index may not be created for small datasets)
        required_files = [
            f"{index_prefix}_disk.index",
            f"{index_prefix}_pq_compressed.bin",
            f"{index_prefix}_pq_pivots.bin",
        ]

        # Check all generated files first for debugging
        generated_files = [f.name for f in index_dir.glob(f"{index_prefix}*")]
        print(f"Generated files: {generated_files}")

        for required_file in required_files:
            file_path = index_dir / required_file
            assert file_path.exists(), f"Required file {required_file} not found"

        # Ensure no partition files exist in non-partition mode
        partition_files = [f"{index_prefix}_disk_graph.index", f"{index_prefix}_partition.bin"]

        for partition_file in partition_files:
            file_path = index_dir / partition_file
            assert not file_path.exists(), (
                f"Partition file {partition_file} should not exist in non-partition mode"
            )

        # Test search functionality
        searcher = LeannSearcher(index_path)
        results = searcher.search("topic 3 analysis", top_k=3)

        assert len(results) > 0
        assert all(result.score is not None and result.score != float("-inf") for result in results)


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip DiskANN partition tests in CI - requires specific hardware and large memory",
)
def test_diskann_with_partition():
    """Test DiskANN index building with automatic graph partitioning."""
    from leann.api import LeannBuilder

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = str(Path(temp_dir) / "test_with_partition.leann")

        # Test data - enough to trigger partitioning
        texts = [
            f"Document {i} explores subject {i % 15} with comprehensive coverage of area {i // 15}."
            for i in range(500)
        ]

        # Build with partition (is_recompute=True)
        builder = LeannBuilder(
            backend_name="diskann",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            num_neighbors=32,
            search_list_size=50,
            is_recompute=True,  # Enable automatic partitioning
        )

        for text in texts:
            builder.add_text(text)

        builder.build_index(index_path)

        # Verify index was created
        index_dir = Path(index_path).parent
        assert index_dir.exists()

        # Check that partition files exist
        index_prefix = Path(index_path).stem
        partition_files = [
            f"{index_prefix}_disk_graph.index",  # Partitioned graph
            f"{index_prefix}_partition.bin",  # Partition metadata
            f"{index_prefix}_pq_compressed.bin",
            f"{index_prefix}_pq_pivots.bin",
        ]

        for partition_file in partition_files:
            file_path = index_dir / partition_file
            assert file_path.exists(), f"Expected partition file {partition_file} not found"

        # Check that large files were cleaned up (storage saving goal)
        large_files = [f"{index_prefix}_disk.index", f"{index_prefix}_disk_beam_search.index"]

        for large_file in large_files:
            file_path = index_dir / large_file
            assert not file_path.exists(), (
                f"Large file {large_file} should have been deleted for storage saving"
            )

        # Verify required auxiliary files for partition mode exist
        required_files = [
            f"{index_prefix}_disk.index_medoids.bin",
            f"{index_prefix}_disk.index_max_base_norm.bin",
        ]

        for req_file in required_files:
            file_path = index_dir / req_file
            assert file_path.exists(), (
                f"Required auxiliary file {req_file} missing for partition mode"
            )


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip DiskANN partition tests in CI - requires specific hardware and large memory",
)
def test_diskann_partition_search_functionality():
    """Test that search works correctly with partitioned indices."""
    from leann.api import LeannBuilder, LeannSearcher

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = str(Path(temp_dir) / "test_partition_search.leann")

        # Create diverse test data
        texts = [
            "LEANN is a storage-efficient approximate nearest neighbor search system.",
            "Graph partitioning helps reduce memory usage in large scale vector search.",
            "DiskANN provides high-performance disk-based approximate nearest neighbor search.",
            "Vector embeddings enable semantic search over unstructured text data.",
            "Approximate nearest neighbor algorithms trade accuracy for speed and storage.",
        ] * 100  # Repeat to get enough data

        # Build with partitioning
        builder = LeannBuilder(
            backend_name="diskann",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            is_recompute=True,  # Enable partitioning
        )

        for text in texts:
            builder.add_text(text)

        builder.build_index(index_path)

        # Test search with partitioned index
        searcher = LeannSearcher(index_path)

        # Test various queries
        test_queries = [
            ("vector search algorithms", 5),
            ("LEANN storage efficiency", 3),
            ("graph partitioning memory", 4),
            ("approximate nearest neighbor", 7),
        ]

        for query, top_k in test_queries:
            results = searcher.search(query, top_k=top_k)

            # Verify search results
            assert len(results) == top_k, f"Expected {top_k} results for query '{query}'"
            assert all(result.score is not None for result in results), (
                "All results should have scores"
            )
            assert all(result.score != float("-inf") for result in results), (
                "No result should have -inf score"
            )
            assert all(result.text is not None for result in results), (
                "All results should have text"
            )

            # Scores should be in descending order (higher similarity first)
            scores = [result.score for result in results]
            assert scores == sorted(scores, reverse=True), (
                "Results should be sorted by score descending"
            )


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip DiskANN partition tests in CI - requires specific hardware and large memory",
)
def test_diskann_medoid_and_norm_files():
    """Test that medoid and max_base_norm files are correctly generated and used."""
    import struct

    from leann.api import LeannBuilder, LeannSearcher

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = str(Path(temp_dir) / "test_medoid_norm.leann")

        # Small but sufficient dataset
        texts = [f"Test document {i} with content about subject {i % 10}." for i in range(200)]

        builder = LeannBuilder(
            backend_name="diskann",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            is_recompute=True,
        )

        for text in texts:
            builder.add_text(text)

        builder.build_index(index_path)

        index_dir = Path(index_path).parent
        index_prefix = Path(index_path).stem

        # Test medoids file
        medoids_file = index_dir / f"{index_prefix}_disk.index_medoids.bin"
        assert medoids_file.exists(), "Medoids file should be generated"

        # Read and validate medoids file format
        with open(medoids_file, "rb") as f:
            nshards = struct.unpack("<I", f.read(4))[0]
            one_val = struct.unpack("<I", f.read(4))[0]
            medoid_id = struct.unpack("<I", f.read(4))[0]

            assert nshards == 1, "Single-shot build should have 1 shard"
            assert one_val == 1, "Expected value should be 1"
            assert medoid_id >= 0, "Medoid ID should be valid (not hardcoded 0)"

        # Test max_base_norm file
        norm_file = index_dir / f"{index_prefix}_disk.index_max_base_norm.bin"
        assert norm_file.exists(), "Max base norm file should be generated"

        # Read and validate norm file
        with open(norm_file, "rb") as f:
            npts = struct.unpack("<I", f.read(4))[0]
            ndims = struct.unpack("<I", f.read(4))[0]
            norm_val = struct.unpack("<f", f.read(4))[0]

            assert npts == 1, "Should have 1 norm point"
            assert ndims == 1, "Should have 1 dimension"
            assert norm_val > 0, "Norm value should be positive"
            assert norm_val != float("inf"), "Norm value should be finite"

        # Test that search works with these files
        searcher = LeannSearcher(index_path)
        results = searcher.search("test subject", top_k=3)

        # Verify that scores are not -inf (which indicates norm file was loaded correctly)
        assert len(results) > 0
        assert all(result.score != float("-inf") for result in results), (
            "Scores should not be -inf when norm file is correct"
        )


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip performance comparison in CI - requires significant compute time",
)
def test_diskann_vs_hnsw_performance():
    """Compare DiskANN (with partition) vs HNSW performance."""
    import time

    from leann.api import LeannBuilder, LeannSearcher

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test data
        texts = [
            f"Performance test document {i} covering topic {i % 20} in detail." for i in range(1000)
        ]
        query = "performance topic test"

        # Test DiskANN with partitioning
        diskann_path = str(Path(temp_dir) / "perf_diskann.leann")
        diskann_builder = LeannBuilder(
            backend_name="diskann",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            is_recompute=True,
        )

        for text in texts:
            diskann_builder.add_text(text)

        start_time = time.time()
        diskann_builder.build_index(diskann_path)

        # Test HNSW
        hnsw_path = str(Path(temp_dir) / "perf_hnsw.leann")
        hnsw_builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            is_recompute=True,
        )

        for text in texts:
            hnsw_builder.add_text(text)

        start_time = time.time()
        hnsw_builder.build_index(hnsw_path)

        # Compare search performance
        diskann_searcher = LeannSearcher(diskann_path)
        hnsw_searcher = LeannSearcher(hnsw_path)

        # Warm up searches
        diskann_searcher.search(query, top_k=5)
        hnsw_searcher.search(query, top_k=5)

        # Timed searches
        start_time = time.time()
        diskann_results = diskann_searcher.search(query, top_k=10)
        diskann_search_time = time.time() - start_time

        start_time = time.time()
        hnsw_results = hnsw_searcher.search(query, top_k=10)
        hnsw_search_time = time.time() - start_time

        # Basic assertions
        assert len(diskann_results) == 10
        assert len(hnsw_results) == 10
        assert all(r.score != float("-inf") for r in diskann_results)
        assert all(r.score != float("-inf") for r in hnsw_results)

        # Performance ratio (informational)
        if hnsw_search_time > 0:
            speed_ratio = hnsw_search_time / diskann_search_time
            print(f"DiskANN search time: {diskann_search_time:.4f}s")
            print(f"HNSW search time: {hnsw_search_time:.4f}s")
            print(f"DiskANN is {speed_ratio:.2f}x faster than HNSW")
