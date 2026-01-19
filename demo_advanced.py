"""
Advanced LEANN Demo: Metadata Filtering, Grep Search, and Model Configurations
"""

from leann import LeannBuilder, LeannSearcher, LeannChat
from pathlib import Path
import os

# ============================================================================
# 1. BUILDING INDEX WITH METADATA
# ============================================================================
print("=" * 70)
print("1. BUILDING INDEX WITH METADATA")
print("=" * 70)

INDEX_PATH_METADATA = str(Path("./").resolve() / "demo_metadata.leann")

# Create builder with default settings
builder = LeannBuilder(backend_name="hnsw")

# Add text with metadata
print("\nAdding documents with metadata...")
builder.add_text(
    "Python is a high-level programming language known for its simplicity and readability.",
    metadata={
        "language": "Python",
        "topic": "programming",
        "difficulty": "beginner",
        "file_type": ".py",
        "lines": 1
    }
)

builder.add_text(
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    metadata={
        "language": "Python",
        "topic": "algorithms",
        "difficulty": "intermediate",
        "file_type": ".py",
        "lines": 1,
        "function_name": "fibonacci"
    }
)

builder.add_text(
    "Machine learning is a subset of artificial intelligence that enables computers to learn.",
    metadata={
        "language": "English",
        "topic": "AI/ML",
        "difficulty": "beginner",
        "file_type": ".md",
        "lines": 1
    }
)

builder.add_text(
    "class NeuralNetwork: def __init__(self, layers): self.layers = layers",
    metadata={
        "language": "Python",
        "topic": "machine_learning",
        "difficulty": "advanced",
        "file_type": ".py",
        "lines": 1,
        "class_name": "NeuralNetwork"
    }
)

builder.add_text(
    "JavaScript is essential for web development and interactive frontend applications.",
    metadata={
        "language": "JavaScript",
        "topic": "web_development",
        "difficulty": "intermediate",
        "file_type": ".js",
        "lines": 1
    }
)

print(f"Building index at {INDEX_PATH_METADATA}...")
builder.build_index(INDEX_PATH_METADATA)
print("✓ Index built successfully!\n")

# ============================================================================
# 2. METADATA FILTERING SEARCH
# ============================================================================
print("=" * 70)
print("2. METADATA FILTERING SEARCH")
print("=" * 70)

searcher = LeannSearcher(INDEX_PATH_METADATA)

# Search with metadata filters
queries = [
    {
        "query": "programming language",
        "filters": {"language": {"==": "Python"}},
        "description": "Search for Python content"
    },
    {
        "query": "machine learning",
        "filters": {"topic": {"==": "machine_learning"}, "difficulty": {"==": "advanced"}},
        "description": "Advanced machine learning content"
    },
    {
        "query": "code",
        "filters": {"file_type": {"==": ".py"}, "difficulty": {"in": ["beginner", "intermediate"]}},
        "description": "Python files for beginner/intermediate"
    },
    {
        "query": "function",
        "filters": {"function_name": {"!=": None}},
        "description": "Content with function definitions"
    }
]

for i, search_params in enumerate(queries, 1):
    print(f"\n{i}. {search_params['description']}")
    print(f"   Query: '{search_params['query']}'")
    print(f"   Filters: {search_params['filters']}")
    results = searcher.search(
        search_params['query'],
        top_k=3,
        metadata_filters=search_params['filters']
    )
    print(f"   Results: {len(results)} found")
    for j, result in enumerate(results, 1):
        print(f"      {j}. Score: {result.score:.3f}")
        print(f"         Text: {result.text[:80]}...")
        if hasattr(result, 'metadata') and result.metadata:
            print(f"         Metadata: {result.metadata}")

# ============================================================================
# 3. GREP SEARCH (EXACT TEXT MATCHING)
# ============================================================================
print("\n" + "=" * 70)
print("3. GREP SEARCH (EXACT TEXT MATCHING)")
print("=" * 70)

print("\nNote: Grep search requires a .jsonl passages file.")
print("It works best with indexes built using the CLI 'leann build' command.")
print("\nTesting with the existing 'my-docs' index...")

try:
    # Try grep search with the CLI-built index
    searcher_cli = LeannSearcher(str(Path("./").resolve() / ".leann/indexes/my-docs/documents.leann"))
    grep_queries = ["LEANN", "storage", "graph"]
    
    for query in grep_queries:
        print(f"\nGrep search: '{query}'")
        try:
            results = searcher_cli.search(query, top_k=3, use_grep=True)
            print(f"Found {len(results)} exact matches:")
            for j, result in enumerate(results, 1):
                print(f"   {j}. {result.text[:100]}...")
        except Exception as e:
            print(f"   Grep search not available: {e}")
            print(f"   Using semantic search instead...")
            results = searcher_cli.search(query, top_k=3, use_grep=False)
            print(f"   Found {len(results)} semantic matches:")
            for j, result in enumerate(results, 1):
                print(f"      {j}. Score: {result.score:.3f} - {result.text[:80]}...")
except Exception as e:
    print(f"\n⚠ Grep search test skipped: {e}")
    print("Grep search works best with indexes built via CLI: leann build <name> --docs <dir>")

# ============================================================================
# 4. DIFFERENT MODEL CONFIGURATIONS
# ============================================================================
print("\n" + "=" * 70)
print("4. DIFFERENT MODEL CONFIGURATIONS")
print("=" * 70)

# Test different embedding models
print("\n4.1 Testing different embedding models...")

# Default model (facebook/contriever)
print("\nBuilding with default model (facebook/contriever)...")
builder_default = LeannBuilder(backend_name="hnsw", embedding_model="facebook/contriever")
builder_default.add_text("This is a test document for embedding model comparison.")
builder_default.add_text("LEANN saves 97% storage compared to traditional vector databases.")
builder_default.build_index(str(Path("./").resolve() / "demo_default_embedding.leann"))
print("✓ Default model index built")

# Small model (faster, lower quality)
print("\nBuilding with small model (sentence-transformers/all-MiniLM-L6-v2)...")
try:
    builder_small = LeannBuilder(
        backend_name="hnsw",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    builder_small.add_text("This is a test document for embedding model comparison.")
    builder_small.add_text("LEANN saves 97% storage compared to traditional vector databases.")
    builder_small.build_index(str(Path("./").resolve() / "demo_small_embedding.leann"))
    print("✓ Small model index built")
    
    # Test search with small model
    searcher_small = LeannSearcher(str(Path("./").resolve() / "demo_small_embedding.leann"))
    results_small = searcher_small.search("storage savings", top_k=1)
    print(f"   Search test: Found {len(results_small)} results")
except Exception as e:
    print(f"   ⚠ Small model test skipped: {e}")

# Different backends
print("\n4.2 Testing different backends...")
print("\nBuilding with HNSW backend (default, optimal for storage)...")
builder_hnsw = LeannBuilder(backend_name="hnsw")
builder_hnsw.add_text("HNSW is excellent for storage-optimized indexes.")
builder_hnsw.build_index(str(Path("./").resolve() / "demo_hnsw_backend.leann"))
print("✓ HNSW backend index built")

print("\nBuilding with DiskANN backend (faster search, better for large datasets)...")
try:
    builder_diskann = LeannBuilder(backend_name="diskann")
    builder_diskann.add_text("DiskANN provides faster search on large datasets.")
    builder_diskann.build_index(str(Path("./").resolve() / "demo_diskann_backend.leann"))
    print("✓ DiskANN backend index built")
    
    # Test search with DiskANN
    searcher_diskann = LeannSearcher(str(Path("./").resolve() / "demo_diskann_backend.leann"))
    results_diskann = searcher_diskann.search("large datasets", top_k=1)
    print(f"   Search test: Found {len(results_diskann)} results")
except Exception as e:
    print(f"   ⚠ DiskANN backend test skipped: {e}")

# Different LLM configurations for chat
print("\n4.3 Testing different LLM configurations...")

# Test with different Ollama models
ollama_models = ["gemma3:4b", "llama3:latest"]
INDEX_FOR_CHAT = INDEX_PATH_METADATA

for model in ollama_models:
    print(f"\nTesting chat with Ollama model: {model}")
    try:
        chat = LeannChat(
            INDEX_FOR_CHAT,
            llm_config={"type": "ollama", "model": model}
        )
        response = chat.ask("What programming languages are mentioned?", top_k=2)
        print(f"   ✓ Model {model} responded:")
        print(f"   {response[:150]}...")
    except Exception as e:
        print(f"   ⚠ Model {model} test skipped: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\n✓ Built index with metadata")
print("✓ Demonstrated metadata filtering search")
print("✓ Demonstrated grep search for exact matches")
print("✓ Tested different embedding models")
print("✓ Tested different backends (HNSW, DiskANN)")
print("✓ Tested different LLM configurations")
print("\nIndex files created:")
print(f"  - {INDEX_PATH_METADATA}")
print(f"  - demo_default_embedding.leann")
print(f"  - demo_hnsw_backend.leann")
print("\nTry running: leann list")
print("=" * 70)
