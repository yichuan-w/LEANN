#!/usr/bin/env python3
"""Test only Faiss HNSW"""

import os
import sys
import time

import psutil


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class MemoryTracker:
    def __init__(self, name: str):
        self.name = name
        self.start_mem = get_memory_usage()
        self.stages = []

    def checkpoint(self, stage: str):
        current_mem = get_memory_usage()
        diff = current_mem - self.start_mem
        print(f"[{self.name} - {stage}] Memory: {current_mem:.1f} MB (+{diff:.1f} MB)")
        self.stages.append((stage, current_mem))
        return current_mem

    def summary(self):
        peak_mem = max(mem for _, mem in self.stages)
        print(f"Peak Memory: {peak_mem:.1f} MB")
        return peak_mem


def main():
    try:
        import faiss
    except ImportError:
        print("Faiss is not installed.")
        print(
            "Please install it with `uv pip install faiss-cpu` and you can  then run this script again"
        )
        sys.exit(1)

    from llama_index.core import (
        Settings,
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore

    tracker = MemoryTracker("Faiss HNSW")
    tracker.checkpoint("Initial")

    embed_model = HuggingFaceEmbedding(model_name="facebook/contriever")
    Settings.embed_model = embed_model
    tracker.checkpoint("After embedding model setup")

    d = 768
    faiss_index = faiss.IndexHNSWFlat(d, 32)
    faiss_index.hnsw.efConstruction = 64
    tracker.checkpoint("After Faiss index creation")

    documents = SimpleDirectoryReader(
        "data",
        recursive=True,
        encoding="utf-8",
        required_exts=[".pdf", ".txt", ".md"],
    ).load_data()
    tracker.checkpoint("After document loading")

    # Parse into chunks using the same splitter as LEANN
    node_parser = SentenceSplitter(
        chunk_size=256, chunk_overlap=20, separator=" ", paragraph_separator="\n\n"
    )

    tracker.checkpoint("After text splitter setup")

    # Check if index already exists and try to load it
    index_loaded = False
    if os.path.exists("./storage_faiss"):
        print("Loading existing Faiss HNSW index...")
        try:
            # Use the correct Faiss loading pattern from the example
            vector_store = FaissVectorStore.from_persist_dir("./storage_faiss")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir="./storage_faiss"
            )
            from llama_index.core import load_index_from_storage

            index = load_index_from_storage(storage_context=storage_context)
            print("Index loaded from ./storage_faiss")
            tracker.checkpoint("After loading existing index")
            index_loaded = True
        except Exception as e:
            print(f"Failed to load existing index: {e}")
            print("Cleaning up corrupted index and building new one...")
            # Clean up corrupted index
            import shutil

            if os.path.exists("./storage_faiss"):
                shutil.rmtree("./storage_faiss")

    if not index_loaded:
        print("Building new Faiss HNSW index...")

        # Use the correct Faiss building pattern from the example
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, transformations=[node_parser]
        )
        tracker.checkpoint("After index building")

        # Save index to disk using the correct pattern
        index.storage_context.persist(persist_dir="./storage_faiss")
        tracker.checkpoint("After index saving")

    # Measure runtime memory overhead
    print("\nMeasuring runtime memory overhead...")
    runtime_start_mem = get_memory_usage()
    print(f"Before load memory: {runtime_start_mem:.1f} MB")
    tracker.checkpoint("Before load memory")

    query_engine = index.as_query_engine(similarity_top_k=20)
    queries = [
        "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面,任务令一般在什么城市颁发",
        "What is LEANN and how does it work?",
        "华为诺亚方舟实验室的主要研究内容",
    ]

    for i, query in enumerate(queries):
        start_time = time.time()
        _ = query_engine.query(query)
        query_time = time.time() - start_time
        print(f"Query {i + 1} time: {query_time:.3f}s")
        tracker.checkpoint(f"After query {i + 1}")

    runtime_end_mem = get_memory_usage()
    runtime_overhead = runtime_end_mem - runtime_start_mem

    peak_memory = tracker.summary()
    print(f"Peak Memory: {peak_memory:.1f} MB")
    print(f"Runtime Memory Overhead: {runtime_overhead:.1f} MB")


if __name__ == "__main__":
    main()
