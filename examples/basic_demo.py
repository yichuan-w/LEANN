"""
Simple demo showing basic leann usage
Run: uv run python examples/basic_demo.py
"""

import argparse

from leann import LeannBuilder, LeannChat, LeannSearcher


def main():
    parser = argparse.ArgumentParser(
        description="Simple demo of Leann with selectable embedding models."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="The embedding model to use, e.g., 'sentence-transformers/all-mpnet-base-v2' or 'text-embedding-ada-002'.",
    )
    args = parser.parse_args()

    print(f"=== Leann Simple Demo with {args.embedding_model} ===")
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
        embedding_model=args.embedding_model,
        backend_name="hnsw",
    )
    for chunk in chunks:
        builder.add_text(chunk)
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

    print("3. Interactive chat demo:")
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
    print("   uv run python apps/document_rag.py")


if __name__ == "__main__":
    main()
