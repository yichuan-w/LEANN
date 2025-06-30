"""
Simple demo showing basic leann usage
Run: uv run python examples/simple_demo.py
"""

from leann import LeannBuilder, LeannSearcher, LeannChat


def main():
    print("=== Leann Simple Demo ===")
    print()
    
    # Sample knowledge base
    chunks = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Deep learning uses neural networks with multiple layers to process data and make decisions.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information from images and videos.",
        "Reinforcement learning teaches agents to make decisions by receiving rewards or penalties for their actions.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data.",
        "Big data refers to extremely large datasets that require special tools and techniques to process.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
    ]
    
    print("1. Building index (no embeddings stored)...")
    builder = LeannBuilder(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        prune_ratio=0.7,  # Keep 30% of connections
    )
    builder.add_chunks(chunks)
    builder.build_index("demo_knowledge.leann")
    print()
    
    print("2. Searching with real-time embeddings...")
    searcher = LeannSearcher("demo_knowledge.leann")
    
    queries = [
        "What is machine learning?",
        "How does neural network work?", 
        "Tell me about data processing",
    ]
    
    for query in queries:
        print(f"Query: {query}")
        results = searcher.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.3f}")
            print(f"     Text: {result.text[:100]}...")
        print()
    
    print("3. Memory stats:")
    stats = searcher.get_memory_stats()
    print(f"   Cache size: {stats.embedding_cache_size}")
    print(f"   Cache memory: {stats.embedding_cache_memory_mb:.1f} MB") 
    print(f"   Total chunks: {stats.total_chunks}")
    print()
    
    print("4. Interactive chat demo:")
    print("   (Note: Requires OpenAI API key for real responses)")
    
    chat = LeannChat("demo_knowledge.leann")
    
    # Demo questions
    demo_questions: list[str] = [
        "What is the difference between machine learning and deep learning?",
        "How is data science related to big data?",
    ]
    
    for question in demo_questions:
        print(f"   Q: {question}")
        response = chat.ask(question)
        print(f"   A: {response}")
        print()
    
    print("Demo completed! Try running:")
    print("   uv run python examples/document_search.py")


if __name__ == "__main__":
    main()