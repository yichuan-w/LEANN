#!/usr/bin/env python3
"""
Document search demo with recompute mode
"""

import os
from pathlib import Path
import shutil
import time

# Import backend packages to trigger plugin registration
try:
    import leann_backend_diskann
    import leann_backend_hnsw
    print("INFO: Backend packages imported successfully.")
except ImportError as e:
    print(f"WARNING: Could not import backend packages. Error: {e}")

# Import upper-level API from leann-core
from leann.api import LeannBuilder, LeannSearcher, LeannChat


def load_sample_documents():
    """Create sample documents for demonstration"""
    docs = [
        {"title": "Intro to Python", "content": "Python is a high-level, interpreted language known for simplicity."},
        {"title": "ML Basics", "content": "Machine learning builds systems that learn from data."},
        {"title": "Data Structures", "content": "Data structures like arrays, lists, and graphs organize data."},
    ]
    return docs

def main():
    print("==========================================================")
    print("=== Leann Document Search Demo (DiskANN + Recompute) ===")
    print("==========================================================")
    
    INDEX_DIR = Path("./test_indices")
    INDEX_PATH = str(INDEX_DIR / "documents.diskann")
    BACKEND_TO_TEST = "diskann"

    if INDEX_DIR.exists():
        print(f"--- Cleaning up old index directory: {INDEX_DIR} ---")
        shutil.rmtree(INDEX_DIR)

    # --- 1. Build index ---
    print(f"\n[PHASE 1] Building index using '{BACKEND_TO_TEST}' backend...")
    
    builder = LeannBuilder(
        backend_name=BACKEND_TO_TEST, 
        graph_degree=32, 
        complexity=64
    )
    
    documents = load_sample_documents()
    print(f"Loaded {len(documents)} sample documents.")
    for doc in documents:
        builder.add_text(doc["content"], metadata={"title": doc["title"]})
        
    builder.build_index(INDEX_PATH)
    print(f"\nIndex built!")

    # --- 2. Basic search demo ---
    print(f"\n[PHASE 2] Basic search using '{BACKEND_TO_TEST}' backend...")
    searcher = LeannSearcher(index_path=INDEX_PATH)
    
    query = "What is machine learning?"
    print(f"\nQuery: '{query}'")
    
    print("\n--- Basic search mode (PQ computation) ---")
    start_time = time.time()
    results = searcher.search(query, top_k=2)
    basic_time = time.time() - start_time
    
    print(f"⏱️  Basic search time: {basic_time:.3f} seconds")
    print(">>> Basic search results <<<")
    for i, res in enumerate(results, 1):
        print(f"  {i}. ID: {res.id}, Score: {res.score:.4f}, Text: '{res.text}', Metadata: {res.metadata}")

    # --- 3. Recompute search demo ---
    print(f"\n[PHASE 3] Recompute search using embedding server...")
    
    print("\n--- Recompute search mode (get real embeddings via network) ---")
    
    # Configure recompute parameters
    recompute_params = {
        "recompute_beighbor_embeddings": True,  # Enable network recomputation
        "USE_DEFERRED_FETCH": False,           # Don't use deferred fetch
        "skip_search_reorder": True,           # Skip search reordering
        "dedup_node_dis": True,               # Enable node distance deduplication
        "prune_ratio": 0.1,                   # Pruning ratio 10%
        "batch_recompute": False,             # Don't use batch recomputation
        "global_pruning": False,              # Don't use global pruning
        "zmq_port": 5555,                     # ZMQ port
        "embedding_model": "sentence-transformers/all-mpnet-base-v2"
    }
    
    print("Recompute parameter configuration:")
    for key, value in recompute_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔄 Executing Recompute search...")
    try:
        start_time = time.time()
        recompute_results = searcher.search(query, top_k=2, **recompute_params)
        recompute_time = time.time() - start_time
        
        print(f"⏱️  Recompute search time: {recompute_time:.3f} seconds")
        print(">>> Recompute search results <<<")
        for i, res in enumerate(recompute_results, 1):
            print(f"  {i}. ID: {res.id}, Score: {res.score:.4f}, Text: '{res.text}', Metadata: {res.metadata}")
        
        # Compare results
        print(f"\n--- Result comparison ---")
        print(f"Basic search time: {basic_time:.3f} seconds")
        print(f"Recompute time: {recompute_time:.3f} seconds")
        
        print("\nBasic search vs Recompute results:")
        for i in range(min(len(results), len(recompute_results))):
            basic_score = results[i].score
            recompute_score = recompute_results[i].score
            score_diff = abs(basic_score - recompute_score)
            print(f"  Position {i+1}: PQ={basic_score:.4f}, Recompute={recompute_score:.4f}, Difference={score_diff:.4f}")
        
        if recompute_time > basic_time:
            print(f"✅ Recompute mode working correctly (more accurate but slower)")
        else:
            print(f"ℹ️  Recompute time is unusually fast, network recomputation may not be enabled")
            
    except Exception as e:
        print(f"❌ Recompute search failed: {e}")
        print("This usually indicates an embedding server connection issue")

    # --- 4. Chat demo ---
    print(f"\n[PHASE 4] Starting chat session...")
    chat = LeannChat(index_path=INDEX_PATH)
    chat_response = chat.ask(query)
    print(f"You: {query}")
    print(f"Leann: {chat_response}")

    print("\n==========================================================")
    print("✅ Demo finished successfully!")
    print("==========================================================")


if __name__ == "__main__":
    main()