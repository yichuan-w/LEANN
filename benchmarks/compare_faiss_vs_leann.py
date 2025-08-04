#!/usr/bin/env python3
"""
Memory comparison between Faiss HNSW and LEANN HNSW backend
"""

import gc
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
from llama_index.core.node_parser import SentenceSplitter

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def print_memory_stats(stage: str, start_mem: float):
    """Print memory statistics"""
    current_mem = get_memory_usage()
    diff = current_mem - start_mem
    print(f"[{stage}] Memory: {current_mem:.1f} MB (+{diff:.1f} MB)")
    return current_mem


class MemoryTracker:
    def __init__(self, name: str):
        self.name = name
        self.start_mem = get_memory_usage()
        self.stages = []

    def checkpoint(self, stage: str):
        current_mem = print_memory_stats(f"{self.name} - {stage}", self.start_mem)
        self.stages.append((stage, current_mem))
        return current_mem

    def summary(self):
        print(f"\n=== {self.name} Memory Summary ===")
        for stage, mem in self.stages:
            print(f"{stage}: {mem:.1f} MB")
        peak_mem = max(mem for _, mem in self.stages)
        print(f"Peak Memory: {peak_mem:.1f} MB")
        print(f"Total Memory Increase: {peak_mem - self.start_mem:.1f} MB")
        return peak_mem


def test_faiss_hnsw():
    """Test Faiss HNSW Vector Store in subprocess"""
    print("\n" + "=" * 50)
    print("TESTING FAISS HNSW VECTOR STORE")
    print("=" * 50)

    try:
        result = subprocess.run(
            [sys.executable, "benchmarks/faiss_only.py"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)

        if result.returncode != 0:
            return {
                "peak_memory": float("inf"),
                "error": f"Process failed with code {result.returncode}",
            }

        # Parse peak memory from output
        lines = result.stdout.split("\n")
        peak_memory = 0.0

        for line in lines:
            if "Peak Memory:" in line:
                peak_memory = float(line.split("Peak Memory:")[1].split("MB")[0].strip())

        return {"peak_memory": peak_memory}

    except Exception as e:
        return {
            "peak_memory": float("inf"),
            "error": str(e),
        }


def test_leann_hnsw():
    """Test LEANN HNSW Search Memory (load existing index)"""
    print("\n" + "=" * 50)
    print("TESTING LEANN HNSW SEARCH MEMORY")
    print("=" * 50)

    tracker = MemoryTracker("LEANN HNSW Search")

    # Import and setup
    tracker.checkpoint("Initial")

    from leann.api import LeannSearcher

    tracker.checkpoint("After imports")

    from leann.api import LeannBuilder
    from llama_index.core import SimpleDirectoryReader

    # Load and parse documents
    documents = SimpleDirectoryReader(
        "data",
        recursive=True,
        encoding="utf-8",
        required_exts=[".pdf", ".txt", ".md"],
    ).load_data()

    tracker.checkpoint("After document loading")

    # Parse into chunks
    node_parser = SentenceSplitter(
        chunk_size=256, chunk_overlap=20, separator=" ", paragraph_separator="\n\n"
    )

    all_texts = []
    for doc in documents:
        nodes = node_parser.get_nodes_from_documents([doc])
        for node in nodes:
            all_texts.append(node.get_content())
    print(f"Total number of chunks: {len(all_texts)}")

    tracker.checkpoint("After text chunking")

    # Build LEANN index
    INDEX_DIR = Path("./test_leann_comparison")
    INDEX_PATH = str(INDEX_DIR / "comparison.leann")

    # Check if index already exists
    if os.path.exists(INDEX_PATH + ".meta.json"):
        print("Loading existing LEANN HNSW index...")
        tracker.checkpoint("After loading existing index")
    else:
        print("Building new LEANN HNSW index...")
        # Clean up previous index
        import shutil

        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            graph_degree=32,
            complexity=64,
            is_compact=True,
            is_recompute=True,
            num_threads=1,
        )

        tracker.checkpoint("After builder setup")

        print("Building LEANN HNSW index...")

        for chunk_text in all_texts:
            builder.add_text(chunk_text)

        builder.build_index(INDEX_PATH)
        del builder
        gc.collect()

        tracker.checkpoint("After index building")

    # Find existing LEANN index
    index_paths = [
        "./test_leann_comparison/comparison.leann",
    ]
    index_path = None
    for path in index_paths:
        if os.path.exists(path + ".meta.json"):
            index_path = path
            break

    if not index_path:
        print("❌ LEANN index not found. Please build it first")
        return {"peak_memory": float("inf"), "error": "Index not found"}

    # Measure runtime memory overhead
    print("\nMeasuring runtime memory overhead...")
    runtime_start_mem = get_memory_usage()
    print(f"Before load memory: {runtime_start_mem:.1f} MB")
    tracker.checkpoint("Before load memory")

    # Load searcher
    searcher = LeannSearcher(index_path)
    tracker.checkpoint("After searcher loading")

    print("Running search queries...")
    queries = [
        "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面,任务令一般在什么城市颁发",
        "What is LEANN and how does it work?",
        "华为诺亚方舟实验室的主要研究内容",
    ]

    for i, query in enumerate(queries):
        start_time = time.time()
        # Use same parameters as Faiss: top_k=20, ef=120 (complexity parameter)
        _ = searcher.search(query, top_k=20, ef=120)
        query_time = time.time() - start_time
        print(f"Query {i + 1} time: {query_time:.3f}s")
        tracker.checkpoint(f"After query {i + 1}")

    runtime_end_mem = get_memory_usage()
    runtime_overhead = runtime_end_mem - runtime_start_mem

    peak_memory = tracker.summary()
    print(f"Runtime Memory Overhead: {runtime_overhead:.1f} MB")

    # Get storage size before cleanup
    storage_size = 0
    INDEX_DIR = Path(index_path).parent
    if INDEX_DIR.exists():
        total_size = 0
        for dirpath, _, filenames in os.walk(str(INDEX_DIR)):
            for filename in filenames:
                # Only count actual index files, skip text data and backups
                if filename.endswith((".old", ".tmp", ".bak", ".jsonl", ".json")):
                    continue
                # Count .index, .idx, .map files (actual index structures)
                if filename.endswith((".index", ".idx", ".map")):
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        storage_size = total_size / (1024 * 1024)  # Convert to MB

    # Clean up
    del searcher
    gc.collect()

    return {
        "peak_memory": peak_memory,
        "storage_size": storage_size,
    }


def main():
    """Run comparison tests"""
    print("Storage + Search Memory Comparison: Faiss HNSW vs LEANN HNSW")
    print("=" * 60)

    # Test Faiss HNSW
    faiss_results = test_faiss_hnsw()

    # Force garbage collection
    gc.collect()
    time.sleep(2)

    # Test LEANN HNSW
    leann_results = test_leann_hnsw()

    # Final comparison
    print("\n" + "=" * 60)
    print("STORAGE + SEARCH MEMORY COMPARISON")
    print("=" * 60)

    # Get storage sizes
    faiss_storage_size = 0
    leann_storage_size = leann_results.get("storage_size", 0)

    # Get Faiss storage size using Python
    if os.path.exists("./storage_faiss"):
        total_size = 0
        for dirpath, _, filenames in os.walk("./storage_faiss"):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        faiss_storage_size = total_size / (1024 * 1024)  # Convert to MB

    print("Faiss HNSW:")
    if "error" in faiss_results:
        print(f"  ❌ Failed: {faiss_results['error']}")
    else:
        print(f"  Search Memory: {faiss_results['peak_memory']:.1f} MB")
        print(f"  Storage Size: {faiss_storage_size:.1f} MB")

    print("\nLEANN HNSW:")
    if "error" in leann_results:
        print(f"  ❌ Failed: {leann_results['error']}")
    else:
        print(f"  Search Memory: {leann_results['peak_memory']:.1f} MB")
        print(f"  Storage Size: {leann_storage_size:.1f} MB")

    # Calculate improvements only if both tests succeeded
    if "error" not in faiss_results and "error" not in leann_results:
        memory_ratio = faiss_results["peak_memory"] / leann_results["peak_memory"]

        print("\nLEANN vs Faiss Performance:")
        memory_saving = faiss_results["peak_memory"] - leann_results["peak_memory"]
        print(f"  Search Memory: {memory_ratio:.1f}x less ({memory_saving:.1f} MB saved)")

        # Storage comparison
        if leann_storage_size > faiss_storage_size:
            storage_ratio = leann_storage_size / faiss_storage_size
            print(f"  Storage Size: {storage_ratio:.1f}x larger (LEANN uses more storage)")
        elif faiss_storage_size > leann_storage_size:
            storage_ratio = faiss_storage_size / leann_storage_size
            print(f"  Storage Size: {storage_ratio:.1f}x smaller (LEANN uses less storage)")
        else:
            print("  Storage Size: similar")
    else:
        if "error" not in leann_results:
            print("\n✅ LEANN HNSW completed successfully!")
            print(f"📊 Search Memory: {leann_results['peak_memory']:.1f} MB")
            print(f"📊 Storage Size: {leann_storage_size:.1f} MB")
        if "error" not in faiss_results:
            print("\n✅ Faiss HNSW completed successfully!")
            print(f"📊 Search Memory: {faiss_results['peak_memory']:.1f} MB")
            print(f"📊 Storage Size: {faiss_storage_size:.1f} MB")


if __name__ == "__main__":
    main()
