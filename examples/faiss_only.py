#!/usr/bin/env python3
"""Test only Faiss HNSW"""

import sys
import time
import psutil
import gc


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
        print("Please install it with `uv pip install faiss-cpu`")
        sys.exit(1)

    from llama_index.core import (
        SimpleDirectoryReader,
        VectorStoreIndex,
        StorageContext,
        Settings,
    )
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
        "examples/data",
        recursive=True,
        encoding="utf-8",
        required_exts=[".pdf", ".txt", ".md"],
    ).load_data()
    tracker.checkpoint("After document loading")

    print("Building Faiss HNSW index...")
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    tracker.checkpoint("After index building")

    index.storage_context.persist("./storage_faiss")
    tracker.checkpoint("After index saving")

    query_engine = index.as_query_engine(similarity_top_k=20)
    queries = [
        "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面，任务令一般在什么城市颁发",
        "What is LEANN and how does it work?",
        "华为诺亚方舟实验室的主要研究内容",
    ]

    for i, query in enumerate(queries):
        start_time = time.time()
        _ = query_engine.query(query)
        query_time = time.time() - start_time
        print(f"Query {i + 1} time: {query_time:.3f}s")
        tracker.checkpoint(f"After query {i + 1}")

    peak_memory = tracker.summary()
    print(f"Peak Memory: {peak_memory:.1f} MB")


if __name__ == "__main__":
    main()
