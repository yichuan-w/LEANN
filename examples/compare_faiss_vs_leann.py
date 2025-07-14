#!/usr/bin/env python3
"""
Memory comparison between Faiss HNSW and LEANN HNSW backend
"""

import logging
import sys
import time
import psutil
import gc

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
    """Test Faiss HNSW Vector Store"""
    print("\n" + "=" * 50)
    print("TESTING FAISS HNSW VECTOR STORE")
    print("=" * 50)

    try:
        import faiss
        from llama_index.core import (
            SimpleDirectoryReader,
            VectorStoreIndex,
            StorageContext,
            Settings,
        )
        from llama_index.vector_stores.faiss import FaissVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        print(f"❌ Missing dependencies for Faiss test: {e}")
        print("Please install:")
        print("  pip install faiss-cpu")
        print("  pip install llama-index-vector-stores-faiss")
        print("  pip install llama-index-embeddings-huggingface")
        return {
            "build_time": float("inf"),
            "peak_memory": float("inf"),
            "error": str(e),
        }

    tracker = MemoryTracker("Faiss HNSW")

    # Import and setup
    tracker.checkpoint("Initial")

    tracker.checkpoint("After imports")

    # Setup embedding model (same as LEANN)
    embed_model = HuggingFaceEmbedding(model_name="facebook/contriever")
    Settings.embed_model = embed_model

    tracker.checkpoint("After embedding model setup")

    # Create Faiss index
    d = 768  # facebook/contriever embedding dimension
    faiss_index = faiss.IndexHNSWFlat(d, 32)  # M=32 same as LEANN
    faiss_index.hnsw.efConstruction = 64  # same as LEANN complexity

    tracker.checkpoint("After Faiss index creation")

    # Load documents
    documents = SimpleDirectoryReader(
        "examples/data",
        recursive=True,
        encoding="utf-8",
        required_exts=[".pdf", ".txt", ".md"],
    ).load_data()

    tracker.checkpoint("After document loading")

    # Create vector store and index
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index
    print("Building Faiss HNSW index...")
    start_time = time.time()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    build_time = time.time() - start_time

    tracker.checkpoint("After index building")

    # Save index
    index.storage_context.persist("./storage_faiss")
    tracker.checkpoint("After index saving")

    # Test queries
    query_engine = index.as_query_engine(similarity_top_k=20)

    print("Running queries...")
    queries = [
        "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面，任务令一般在什么城市颁发",
        "What is LEANN and how does it work?",
        "华为诺亚方舟实验室的主要研究内容",
    ]

    for i, query in enumerate(queries):
        start_time = time.time()
        response = query_engine.query(query)
        query_time = time.time() - start_time
        print(f"Query {i + 1} time: {query_time:.3f}s")
        tracker.checkpoint(f"After query {i + 1}")

    peak_memory = tracker.summary()

    # Clean up
    del index, vector_store, storage_context, faiss_index
    gc.collect()

    return {"build_time": build_time, "peak_memory": peak_memory, "tracker": tracker}


def test_leann_hnsw():
    """Test LEANN HNSW Backend"""
    print("\n" + "=" * 50)
    print("TESTING LEANN HNSW BACKEND")
    print("=" * 50)

    tracker = MemoryTracker("LEANN HNSW")

    # Import and setup
    tracker.checkpoint("Initial")

    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    from leann.api import LeannBuilder, LeannChat
    from pathlib import Path

    tracker.checkpoint("After imports")

    # Load and parse documents
    documents = SimpleDirectoryReader(
        "examples/data",
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

    tracker.checkpoint("After text chunking")

    # Build LEANN index
    INDEX_DIR = Path("./test_leann_comparison")
    INDEX_PATH = str(INDEX_DIR / "comparison.leann")

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
    start_time = time.time()

    for chunk_text in all_texts:
        builder.add_text(chunk_text)

    builder.build_index(INDEX_PATH)
    build_time = time.time() - start_time

    tracker.checkpoint("After index building")

    # Ensure any existing embedding server is updated with new data
    print("Ensuring embedding server data synchronization...")
    meta_file = INDEX_DIR / "comparison.leann.meta.json"
    if meta_file.exists():
        from leann.embedding_server_manager import (
            _check_port,
            _update_server_meta_path,
            _check_server_meta_path,
        )

        port = 5557  # Default port for HNSW backend
        if _check_port(port):
            print(f"Updating server meta path to: {meta_file}")
            success = _update_server_meta_path(port, str(meta_file))
            if success:
                print("✅ Server meta path updated successfully")
            else:
                print("❌ Failed to update server meta path")

            # Verify the update and ensure server is ready
            time.sleep(2)  # Give server time to reload data
            max_retries = 5
            for retry in range(max_retries):
                if _check_server_meta_path(port, str(meta_file)):
                    print("✅ Server meta path verification successful")
                    break
                else:
                    print(
                        f"⏳ Server meta path verification failed (attempt {retry + 1}/{max_retries})"
                    )
                    if retry < max_retries - 1:
                        time.sleep(1)
                    else:
                        print(
                            "❌ Server meta path verification failed after all retries - may cause query issues"
                        )
        else:
            print("No existing server found on port 5557")

    # Test queries
    chat = LeannChat(
        index_path=INDEX_PATH, llm_config={"type": "simulated", "model": "test"}
    )

    tracker.checkpoint("After chat setup")

    print("Running queries...")
    queries = [
        "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面，任务令一般在什么城市颁发",
        "What is LEANN and how does it work?",
        "华为诺亚方舟实验室的主要研究内容",
    ]

    for i, query in enumerate(queries):
        start_time = time.time()
        response = chat.ask(
            query, top_k=20, recompute_beighbor_embeddings=True, complexity=32
        )
        query_time = time.time() - start_time
        print(f"Query {i + 1} time: {query_time:.3f}s")
        tracker.checkpoint(f"After query {i + 1}")

    peak_memory = tracker.summary()

    # Clean up
    del chat, builder
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    gc.collect()

    return {"build_time": build_time, "peak_memory": peak_memory, "tracker": tracker}


def main():
    """Run comparison tests"""
    print("Memory Usage Comparison: Faiss HNSW vs LEANN HNSW")
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
    print("FINAL COMPARISON")
    print("=" * 60)

    print(f"Faiss HNSW:")
    if "error" in faiss_results:
        print(f"  ❌ Failed: {faiss_results['error']}")
    else:
        print(f"  Build Time: {faiss_results['build_time']:.3f}s")
        print(f"  Peak Memory: {faiss_results['peak_memory']:.1f} MB")

    print(f"\nLEANN HNSW:")
    print(f"  Build Time: {leann_results['build_time']:.3f}s")
    print(f"  Peak Memory: {leann_results['peak_memory']:.1f} MB")

    # Calculate improvements only if Faiss test succeeded
    if "error" not in faiss_results:
        time_ratio = faiss_results["build_time"] / leann_results["build_time"]
        memory_ratio = faiss_results["peak_memory"] / leann_results["peak_memory"]

        print(f"\nLEANN vs Faiss:")
        print(
            f"  Build Time: {time_ratio:.2f}x {'faster' if time_ratio > 1 else 'slower'}"
        )
        print(
            f"  Memory Usage: {memory_ratio:.2f}x {'less' if memory_ratio > 1 else 'more'}"
        )

        print(
            f"\nMemory Savings: {faiss_results['peak_memory'] - leann_results['peak_memory']:.1f} MB"
        )
    else:
        print(f"\n✅ LEANN HNSW ran successfully!")
        print(f"📊 LEANN Memory Usage: {leann_results['peak_memory']:.1f} MB")


if __name__ == "__main__":
    main()
