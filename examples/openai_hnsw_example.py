#!/usr/bin/env python3
"""
OpenAI Embedding Example

Complete example showing how to build and search with OpenAI embeddings using HNSW backend.
"""

import os
import dotenv
from pathlib import Path
from leann.api import LeannBuilder, LeannSearcher

# Load environment variables
dotenv.load_dotenv()

def main():
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return False
    
    print(f"✅ OpenAI API key found: {api_key[:10]}...")
    
    # Sample texts
    sample_texts = [
        "Machine learning is a powerful technology that enables computers to learn from data.",
        "Natural language processing helps computers understand and generate human language.",
        "Deep learning uses neural networks with multiple layers to solve complex problems.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
        "Data science combines statistics, math, and programming to extract insights from data.",
        "Artificial intelligence aims to create machines that can perform human-like tasks.",
        "Python is a popular programming language used extensively in data science and AI.",
        "Neural networks are inspired by the structure and function of the human brain.",
        "Big data refers to extremely large datasets that require special tools to process."
    ]
    
    INDEX_DIR = Path("./simple_openai_test_index")
    INDEX_PATH = str(INDEX_DIR / "simple_test.leann")
    
    print(f"\n=== Building Index with OpenAI Embeddings ===")
    print(f"Index path: {INDEX_PATH}")
    
    try:
        # Use proper configuration for OpenAI embeddings
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="text-embedding-3-small",
            embedding_mode="openai",
            # HNSW settings for OpenAI embeddings
            M=16,                    # Smaller graph degree
            efConstruction=64,       # Smaller construction complexity  
            is_compact=True,         # Enable compact storage for recompute
            is_recompute=True,       # MUST enable for OpenAI embeddings
            num_threads=1,
        )
        
        print(f"Adding {len(sample_texts)} texts to the index...")
        for i, text in enumerate(sample_texts):
            metadata = {"id": f"doc_{i}", "topic": "AI"}
            builder.add_text(text, metadata)
        
        print("Building index...")
        builder.build_index(INDEX_PATH)
        print(f"✅ Index built successfully!")
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n=== Testing Search ===")
    
    try:
        searcher = LeannSearcher(INDEX_PATH)
        
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Programming languages for data science"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = searcher.search(query, top_k=3)
            
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"   {i+1}. Score: {result.score:.4f}")
                print(f"      Text: {result.text[:80]}...")
        
        print(f"\n✅ Search test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Simple OpenAI index test completed successfully!")
    else:
        print(f"\n💥 Simple OpenAI index test failed!")