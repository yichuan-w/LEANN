from leann import LeannBuilder, LeannSearcher, LeannChat
from pathlib import Path
INDEX_PATH = str(Path("./").resolve() / "demo.leann")

# Build an index
builder = LeannBuilder(backend_name="hnsw")
builder.add_text("LEANN saves 97% storage compared to traditional vector databases.")
builder.add_text("Tung Tung Tung Sahur called—they need their banana‑crocodile hybrid back")
builder.build_index(INDEX_PATH)

# Search
print("Searching the index...")
searcher = LeannSearcher(INDEX_PATH)
results = searcher.search("fantastical AI-generated creatures", top_k=1)
print(f"Search results: {len(results)} found")
for i, result in enumerate(results, 1):
    print(f"  {i}. Score: {result.score:.3f}")
    print(f"     Text: {result.text}")
print()

# Chat with your data using Ollama
print("Setting up chat with Ollama...")
try:
    # Using Ollama with available model (gemma3:4b, llama3:latest, or gpt-oss:20b)
    chat = LeannChat(INDEX_PATH, llm_config={"type": "ollama", "model": "gemma3:4b"})
    response = chat.ask("How much storage does LEANN save?", top_k=1)
    print(f"Chat response: {response}")
except Exception as e:
    print(f"Chat failed: {e}")
    print("Available Ollama models: gemma3:4b, llama3:latest, gpt-oss:20b")
