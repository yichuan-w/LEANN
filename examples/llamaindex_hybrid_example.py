import os

from leann.api import LeannBuilder
from leann.integrations.llamaindex import LeannHybridRetriever, LeannRetriever

# Setup LlamaIndex Settings
# Ensure OPENAI_API_KEY is set in environment since LlamaIndex uses it by default for response generation
if not os.environ.get("OPENAI_API_KEY"):
    print(
        "Warning: OPENAI_API_KEY is not set. The Retrieval part will work, but QueryEngine text generation will fail unless using a mock LLM."
    )
    # For demo purposes, we can try to use a mock LLM or let it fail gracefully on the query engine step


def main():
    index_path = "example_docs.leann"

    # 1. Build the dummy database
    print("Building LEANN index...")
    builder = LeannBuilder(backend_name="hnsw", embedding_model="BAAI/bge-small-en-v1.5")
    builder.add_text("LEANN achieves 97% storage reduction.", metadata={"source": "doc1"})
    builder.add_text("Vector databases store embeddings.", metadata={"source": "doc2"})
    builder.add_text(
        "Hybrid search combines vector and keyword search.", metadata={"source": "doc3"}
    )
    builder.build_index(index_path)

    # 2. Example: Pure Vector Search
    print("\n=== Pure Vector Search ===")
    retriever = LeannRetriever(index_path=index_path, top_k=2)

    # Retrieve directly (without LLM generation) to show it works even without API keys
    nodes = retriever.retrieve("How does LEANN reduce storage?")
    for node in nodes:
        print(f"ID: {node.node.id_} | Score: {node.score:.4f} | Text: {node.node.text}")

    # 3. Example: Hybrid Search (Recommended)
    print("\n=== Hybrid Search (70% vector, 30% keyword) ===")
    hybrid_retriever = LeannHybridRetriever(
        index_path=index_path,
        top_k=2,
        bm25_weight=0.3,  # 30% keyword weight mapping to LEANN's `gemma = 0.7` internally
    )

    nodes = hybrid_retriever.retrieve("hybrid search combination")
    for node in nodes:
        print(f"ID: {node.node.id_} | Score: {node.score:.4f} | Text: {node.node.text}")

    print("\nRetrieval successful! The LlamaIndex integration is fully functional.")


if __name__ == "__main__":
    main()
