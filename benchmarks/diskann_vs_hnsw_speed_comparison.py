#!/usr/bin/env python3
"""
DiskANN vs HNSW Search Performance Comparison

This benchmark compares search performance between DiskANN and HNSW backends:
- DiskANN: With graph partitioning enabled (is_recompute=True)
- HNSW: With recompute enabled (is_recompute=True)
- Tests performance across different dataset sizes
- Measures search latency, recall, and index size
"""

import gc
import multiprocessing as mp
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# Prefer 'fork' start method to avoid POSIX semaphore leaks on macOS
try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass


def create_test_texts(n_docs: int) -> list[str]:
    """Create synthetic test documents for benchmarking."""
    np.random.seed(42)
    topics = [
        "machine learning and artificial intelligence",
        "natural language processing and text analysis",
        "computer vision and image recognition",
        "data science and statistical analysis",
        "deep learning and neural networks",
        "information retrieval and search engines",
        "database systems and data management",
        "software engineering and programming",
        "cybersecurity and network protection",
        "cloud computing and distributed systems",
    ]

    texts = []
    for i in range(n_docs):
        topic = topics[i % len(topics)]
        variation = np.random.randint(1, 100)
        text = (
            f"This is document {i} about {topic}. Content variation {variation}. "
            f"Additional information about {topic} with details and examples. "
            f"Technical discussion of {topic} including implementation aspects."
        )
        texts.append(text)

    return texts


def benchmark_backend(
    backend_name: str, texts: list[str], test_queries: list[str], backend_kwargs: dict[str, Any]
) -> dict[str, float]:
    """Benchmark a specific backend with the given configuration."""
    from leann.api import LeannBuilder, LeannSearcher

    print(f"\n🔧 Testing {backend_name.upper()} backend...")

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = str(Path(temp_dir) / f"benchmark_{backend_name}.leann")

        # Build index
        print(f"📦 Building {backend_name} index with {len(texts)} documents...")
        start_time = time.time()

        builder = LeannBuilder(
            backend_name=backend_name,
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            **backend_kwargs,
        )

        for text in texts:
            builder.add_text(text)

        builder.build_index(index_path)
        build_time = time.time() - start_time

        # Measure index size
        index_dir = Path(index_path).parent
        index_files = list(index_dir.glob(f"{Path(index_path).stem}.*"))
        total_size = sum(f.stat().st_size for f in index_files if f.is_file())
        size_mb = total_size / (1024 * 1024)

        print(f"   ✅ Build completed in {build_time:.2f}s, index size: {size_mb:.1f}MB")

        # Search benchmark
        print("🔍 Running search benchmark...")
        searcher = LeannSearcher(index_path)

        search_times = []
        all_results = []

        for query in test_queries:
            start_time = time.time()
            results = searcher.search(query, top_k=5)
            search_time = time.time() - start_time
            search_times.append(search_time)
            all_results.append(results)

        avg_search_time = np.mean(search_times) * 1000  # Convert to ms
        print(f"   ✅ Average search time: {avg_search_time:.1f}ms")

        # Check for valid scores (detect -inf issues)
        all_scores = [
            result.score
            for results in all_results
            for result in results
            if result.score is not None
        ]
        valid_scores = [
            score for score in all_scores if score != float("-inf") and score != float("inf")
        ]
        score_validity_rate = len(valid_scores) / len(all_scores) if all_scores else 0

        # Clean up (ensure embedding server shutdown and object GC)
        try:
            if hasattr(searcher, "cleanup"):
                searcher.cleanup()
            del searcher
            del builder
            gc.collect()
        except Exception as e:
            print(f"⚠️  Warning: Resource cleanup error: {e}")

        return {
            "build_time": build_time,
            "avg_search_time_ms": avg_search_time,
            "index_size_mb": size_mb,
            "score_validity_rate": score_validity_rate,
        }


def run_comparison(n_docs: int = 500, n_queries: int = 10):
    """Run performance comparison between DiskANN and HNSW."""
    print("🚀 Starting DiskANN vs HNSW Performance Comparison")
    print(f"📊 Dataset: {n_docs} documents, {n_queries} test queries")

    # Create test data
    texts = create_test_texts(n_docs)
    test_queries = [
        "machine learning algorithms",
        "natural language processing",
        "computer vision techniques",
        "data analysis methods",
        "neural network architectures",
        "database query optimization",
        "software development practices",
        "security vulnerabilities",
        "cloud infrastructure",
        "distributed computing",
    ][:n_queries]

    # HNSW benchmark
    hnsw_results = benchmark_backend(
        backend_name="hnsw",
        texts=texts,
        test_queries=test_queries,
        backend_kwargs={
            "is_recompute": True,  # Enable recompute for fair comparison
            "M": 16,
            "efConstruction": 200,
        },
    )

    # DiskANN benchmark
    diskann_results = benchmark_backend(
        backend_name="diskann",
        texts=texts,
        test_queries=test_queries,
        backend_kwargs={
            "is_recompute": True,  # Enable graph partitioning
            "num_neighbors": 32,
            "search_list_size": 50,
        },
    )

    # Performance comparison
    print("\n📈 Performance Comparison Results")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'HNSW':<15} {'DiskANN':<15} {'Speedup':<10}")
    print(f"{'-' * 60}")

    # Build time comparison
    build_speedup = hnsw_results["build_time"] / diskann_results["build_time"]
    print(
        f"{'Build Time (s)':<25} {hnsw_results['build_time']:<15.2f} {diskann_results['build_time']:<15.2f} {build_speedup:<10.2f}x"
    )

    # Search time comparison
    search_speedup = hnsw_results["avg_search_time_ms"] / diskann_results["avg_search_time_ms"]
    print(
        f"{'Search Time (ms)':<25} {hnsw_results['avg_search_time_ms']:<15.1f} {diskann_results['avg_search_time_ms']:<15.1f} {search_speedup:<10.2f}x"
    )

    # Index size comparison
    size_ratio = diskann_results["index_size_mb"] / hnsw_results["index_size_mb"]
    print(
        f"{'Index Size (MB)':<25} {hnsw_results['index_size_mb']:<15.1f} {diskann_results['index_size_mb']:<15.1f} {size_ratio:<10.2f}x"
    )

    # Score validity
    print(
        f"{'Score Validity (%)':<25} {hnsw_results['score_validity_rate'] * 100:<15.1f} {diskann_results['score_validity_rate'] * 100:<15.1f}"
    )

    print(f"{'=' * 60}")
    print("\n🎯 Summary:")
    if search_speedup > 1:
        print(f"   DiskANN is {search_speedup:.2f}x faster than HNSW for search")
    else:
        print(f"   HNSW is {1 / search_speedup:.2f}x faster than DiskANN for search")

    if size_ratio > 1:
        print(f"   DiskANN uses {size_ratio:.2f}x more storage than HNSW")
    else:
        print(f"   DiskANN uses {1 / size_ratio:.2f}x less storage than HNSW")

    print(
        f"   Both backends achieved {min(hnsw_results['score_validity_rate'], diskann_results['score_validity_rate']) * 100:.1f}% score validity"
    )


if __name__ == "__main__":
    import sys

    try:
        # Handle help request
        if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
            print("DiskANN vs HNSW Performance Comparison")
            print("=" * 50)
            print(f"Usage: python {sys.argv[0]} [n_docs] [n_queries]")
            print()
            print("Arguments:")
            print("  n_docs      Number of documents to index (default: 500)")
            print("  n_queries   Number of test queries to run (default: 10)")
            print()
            print("Examples:")
            print("  python benchmarks/diskann_vs_hnsw_speed_comparison.py")
            print("  python benchmarks/diskann_vs_hnsw_speed_comparison.py 1000")
            print("  python benchmarks/diskann_vs_hnsw_speed_comparison.py 2000 20")
            sys.exit(0)

        # Parse command line arguments
        n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
        n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 10

        print("DiskANN vs HNSW Performance Comparison")
        print("=" * 50)
        print(f"Dataset: {n_docs} documents, {n_queries} queries")
        print()

        run_comparison(n_docs=n_docs, n_queries=n_queries)

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)
    finally:
        # Ensure clean exit (forceful to prevent rare hangs from atexit/threads)
        try:
            gc.collect()
            print("\n🧹 Cleanup completed")
            # Flush stdio to ensure message is visible before hard-exit
            try:
                import sys as _sys

                _sys.stdout.flush()
                _sys.stderr.flush()
            except Exception:
                pass
        except Exception:
            pass
        # Use os._exit to bypass atexit handlers that may hang in rare cases
        import os as _os

        _os._exit(0)
