"""
Comprehensive LEANN Guide: Building Indexes, Advanced Features, and Model Configurations

This script demonstrates:
1. Building indexes from different document types and sources
2. Advanced features (metadata filtering, grep search)
3. Different model configurations and their trade-offs
"""

from leann import LeannBuilder, LeannSearcher, LeannChat
from pathlib import Path
import os

print("=" * 80)
print("COMPREHENSIVE LEANN GUIDE")
print("=" * 80)

# ============================================================================
# PART 1: BUILDING INDEXES FROM DIFFERENT DOCUMENTS
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: BUILDING INDEXES FROM DIFFERENT DOCUMENTS")
print("=" * 80)

# 1.1 Building from Python API (manual text addition)
print("\n1.1 Building Index from Python API (Manual Text Addition)")
print("-" * 80)

builder1 = LeannBuilder(backend_name="hnsw")
builder1.add_text("Machine learning is transforming industries across the globe.")
builder1.add_text("Deep neural networks enable computers to recognize patterns.")
builder1.add_text("Natural language processing helps machines understand human language.")
builder1.add_text("Computer vision enables AI systems to interpret visual data.")

index1_path = str(Path("./").resolve() / "index_manual_text.leann")
builder1.build_index(index1_path)
print(f"✓ Built index: {index1_path}")
print(f"  Documents: 4 text passages")

# 1.2 Building with metadata (for filtering)
print("\n1.2 Building Index with Metadata (For Filtering)")
print("-" * 80)

builder2 = LeannBuilder(backend_name="hnsw")

# Add documents with rich metadata
documents_with_metadata = [
    {
        "text": "Python is excellent for data science and machine learning.",
        "metadata": {
            "language": "Python",
            "domain": "data_science",
            "difficulty": "beginner",
            "file_type": ".py",
            "category": "tutorial"
        }
    },
    {
        "text": "class MLModel(nn.Module): def forward(self, x): return self.layer(x)",
        "metadata": {
            "language": "Python",
            "domain": "deep_learning",
            "difficulty": "advanced",
            "file_type": ".py",
            "category": "code",
            "class_name": "MLModel"
        }
    },
    {
        "text": "JavaScript enables interactive web applications and dynamic content.",
        "metadata": {
            "language": "JavaScript",
            "domain": "web_development",
            "difficulty": "intermediate",
            "file_type": ".js",
            "category": "tutorial"
        }
    },
    {
        "text": "function calculateTotal(items) { return items.reduce((a, b) => a + b.price, 0); }",
        "metadata": {
            "language": "JavaScript",
            "domain": "web_development",
            "difficulty": "intermediate",
            "file_type": ".js",
            "category": "code",
            "function_name": "calculateTotal"
        }
    },
    {
        "text": "SQL queries help extract insights from relational databases efficiently.",
        "metadata": {
            "language": "SQL",
            "domain": "database",
            "difficulty": "intermediate",
            "file_type": ".sql",
            "category": "tutorial"
        }
    }
]

for doc in documents_with_metadata:
    builder2.add_text(doc["text"], metadata=doc["metadata"])

index2_path = str(Path("./").resolve() / "index_with_metadata.leann")
builder2.build_index(index2_path)
print(f"✓ Built index: {index2_path}")
print(f"  Documents: {len(documents_with_metadata)} with metadata")

# 1.3 Different backend options
print("\n1.3 Building with Different Backends")
print("-" * 80)

# HNSW backend (default, storage-optimized)
builder3_hnsw = LeannBuilder(backend_name="hnsw")
builder3_hnsw.add_text("HNSW provides excellent storage efficiency through recomputation.")
index3_hnsw = str(Path("./").resolve() / "index_hnsw.leann")
builder3_hnsw.build_index(index3_hnsw)
print(f"✓ HNSW backend: {index3_hnsw}")
print("  Best for: Maximum storage savings, full recomputation")

# DiskANN backend (faster search on large datasets)
try:
    builder3_diskann = LeannBuilder(backend_name="diskann")
    builder3_diskann.add_text("DiskANN provides faster search on large datasets.")
    index3_diskann = str(Path("./").resolve() / "index_diskann.leann")
    builder3_diskann.build_index(index3_diskann)
    print(f"✓ DiskANN backend: {index3_diskann}")
    print("  Best for: Large datasets, faster search, better scaling")
except Exception as e:
    print(f"⚠ DiskANN backend: {e}")

# 1.4 Different embedding models
print("\n1.4 Building with Different Embedding Models")
print("-" * 80)

# Default model (facebook/contriever - 110M params)
builder4_default = LeannBuilder(
    backend_name="hnsw",
    embedding_model="facebook/contriever"  # Default
)
builder4_default.add_text("Default embedding model provides balanced performance.")
index4_default = str(Path("./").resolve() / "index_default_embedding.leann")
builder4_default.build_index(index4_default)
print(f"✓ Default model (facebook/contriever): {index4_default}")
print("  Size: 110M params, Dimension: 768")
print("  Best for: Balanced performance, general use")

# Small model (all-MiniLM-L6-v2 - 22M params)
try:
    builder4_small = LeannBuilder(
        backend_name="hnsw",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    builder4_small.add_text("Small embedding model provides fast indexing and search.")
    index4_small = str(Path("./").resolve() / "index_small_embedding.leann")
    builder4_small.build_index(index4_small)
    print(f"✓ Small model (all-MiniLM-L6-v2): {index4_small}")
    print("  Size: 22M params, Dimension: 384")
    print("  Best for: Speed, prototyping, interactive use")
except Exception as e:
    print(f"⚠ Small model: {e}")

# ============================================================================
# PART 2: ADVANCED FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: ADVANCED FEATURES")
print("=" * 80)

# 2.1 Metadata Filtering Search
print("\n2.1 Metadata Filtering Search")
print("-" * 80)

searcher_metadata = LeannSearcher(index2_path)

print("\nExample 1: Filter by language (Python only)")
print("Query: 'programming' | Filter: language == 'Python'")
results = searcher_metadata.search(
    "programming",
    top_k=5,
    metadata_filters={"language": {"==": "Python"}}
)
for i, r in enumerate(results, 1):
    print(f"  {i}. Score: {r.score:.3f} - {r.text[:60]}...")
    if hasattr(r, 'metadata') and r.metadata:
        print(f"     Metadata: {r.metadata}")

print("\nExample 2: Filter by domain and difficulty")
print("Query: 'learning' | Filter: domain == 'deep_learning' AND difficulty == 'advanced'")
results = searcher_metadata.search(
    "learning",
    top_k=5,
    metadata_filters={
        "domain": {"==": "deep_learning"},
        "difficulty": {"==": "advanced"}
    }
)
for i, r in enumerate(results, 1):
    print(f"  {i}. Score: {r.score:.3f} - {r.text[:60]}...")

print("\nExample 3: Filter with 'in' operator")
print("Query: 'code' | Filter: difficulty in ['beginner', 'intermediate']")
results = searcher_metadata.search(
    "code",
    top_k=5,
    metadata_filters={
        "difficulty": {"in": ["beginner", "intermediate"]}
    }
)
for i, r in enumerate(results, 1):
    print(f"  {i}. Score: {r.score:.3f} - {r.text[:60]}...")

print("\nExample 4: Filter by existence (has function_name)")
print("Query: 'function' | Filter: function_name != None")
results = searcher_metadata.search(
    "function",
    top_k=5,
    metadata_filters={
        "function_name": {"!=": None}
    }
)
for i, r in enumerate(results, 1):
    print(f"  {i}. Score: {r.score:.3f} - {r.text[:60]}...")
    if hasattr(r, 'metadata') and r.metadata:
        print(f"     Function: {r.metadata.get('function_name', 'N/A')}")

# 2.2 Grep Search (Exact Text Matching)
print("\n2.2 Grep Search (Exact Text Matching)")
print("-" * 80)

# Grep search works best with CLI-built indexes
print("Note: Grep search requires CLI-built indexes (.jsonl passages file)")
print("Testing with 'my-docs' index built from CLI...")

try:
    cli_index_path = str(Path("./").resolve() / ".leann/indexes/my-docs/documents.leann")
    if os.path.exists(cli_index_path):
        searcher_grep = LeannSearcher(cli_index_path)
        
        print("\nExample 1: Exact text search for 'LEANN'")
        results = searcher_grep.search("LEANN", top_k=3, use_grep=True)
        print(f"Found {len(results)} exact matches:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.text[:80]}...")
        
        print("\nExample 2: Exact text search for 'storage'")
        results = searcher_grep.search("storage", top_k=3, use_grep=True)
        print(f"Found {len(results)} exact matches:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.text[:80]}...")
        
        print("\nExample 3: Compare semantic vs grep search")
        query = "graph"
        print(f"\nQuery: '{query}'")
        
        semantic_results = searcher_grep.search(query, top_k=3, use_grep=False)
        print(f"Semantic search (meaning-based): {len(semantic_results)} results")
        for i, r in enumerate(semantic_results[:2], 1):
            print(f"  {i}. Score: {r.score:.3f} - {r.text[:60]}...")
        
        grep_results = searcher_grep.search(query, top_k=3, use_grep=True)
        print(f"\nGrep search (exact match): {len(grep_results)} results")
        for i, r in enumerate(grep_results[:2], 1):
            print(f"  {i}. {r.text[:60]}...")
    else:
        print("⚠ CLI-built index not found. Create one with: leann build my-docs --docs ./data")
except Exception as e:
    print(f"⚠ Grep search test: {e}")
    print("  Create a CLI-built index first: leann build my-docs --docs ./data")

# ============================================================================
# PART 3: DIFFERENT MODEL CONFIGURATIONS
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: DIFFERENT MODEL CONFIGURATIONS")
print("=" * 80)

# 3.1 Embedding Model Comparison
print("\n3.1 Embedding Model Comparison")
print("-" * 80)

embedding_models = [
    {
        "name": "facebook/contriever",
        "size": "110M params",
        "dim": 768,
        "description": "Default, balanced performance",
        "use_case": "General purpose RAG"
    },
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "22M params",
        "dim": 384,
        "description": "Small, fast, lightweight",
        "use_case": "Prototyping, speed-critical, interactive"
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "size": "110M params",
        "dim": 768,
        "description": "High quality, English-focused",
        "use_case": "Production English RAG"
    }
]

print("\nAvailable embedding models:")
for i, model in enumerate(embedding_models, 1):
    print(f"\n{i}. {model['name']}")
    print(f"   Size: {model['size']}, Dimension: {model['dim']}")
    print(f"   Description: {model['description']}")
    print(f"   Best for: {model['use_case']}")

# 3.2 Backend Comparison
print("\n3.2 Backend Comparison")
print("-" * 80)

backends = [
    {
        "name": "HNSW",
        "storage": "Maximum savings (97%+)",
        "search_speed": "Fast",
        "best_for": "Most datasets, storage-optimized",
        "recompute": "Full recomputation support"
    },
    {
        "name": "DiskANN",
        "storage": "Good savings (90%+)",
        "search_speed": "3x+ faster on large datasets",
        "best_for": "Large datasets (100k+ docs), faster search",
        "recompute": "PQ-based traversal + reranking"
    }
]

print("\nAvailable backends:")
for i, backend in enumerate(backends, 1):
    print(f"\n{i}. {backend['name']}")
    print(f"   Storage: {backend['storage']}")
    print(f"   Search Speed: {backend['search_speed']}")
    print(f"   Best for: {backend['best_for']}")
    print(f"   Recompute: {backend['recompute']}")

# 3.3 LLM Configuration Examples
print("\n3.3 LLM Configuration Examples")
print("-" * 80)

print("\nAvailable LLM providers and models:")

# Ollama models
ollama_configs = [
    {"model": "gemma3:4b", "size": "3.3 GB", "speed": "Ultra-fast", "quality": "Good"},
    {"model": "llama3:latest", "size": "4.7 GB", "speed": "Fast", "quality": "Very good"},
    {"model": "gpt-oss:20b", "size": "13 GB", "speed": "Medium", "quality": "Excellent (reasoning)"}
]

print("\nOllama (Local, Free):")
for config in ollama_configs:
    print(f"  • {config['model']}")
    print(f"    Size: {config['size']}, Speed: {config['speed']}, Quality: {config['quality']}")
    print(f"    Example: LeannChat(index, llm_config={{'type': 'ollama', 'model': '{config['model']}'}})")

print("\nOpenAI (Cloud, Paid):")
print("  • gpt-4o-mini - Fast, cheap, good quality")
print("  • gpt-4o - Best quality, higher cost")
print("  • o3/o3-mini - Reasoning models (requires thinking_budget)")

print("\nHuggingFace (Local, Free):")
print("  • Qwen/Qwen3-1.7B-FP8 - Good quality, local inference")

# 3.4 Practical Configuration Examples
print("\n3.4 Practical Configuration Examples")
print("-" * 80)

print("\nExample 1: Fast Prototyping Setup")
print("-" * 40)
print("""
# Fast, lightweight setup
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
chat = LeannChat(
    index_path,
    llm_config={"type": "ollama", "model": "gemma3:4b"}
)
""")

print("\nExample 2: Production Setup")
print("-" * 40)
print("""
# High quality, production-ready
builder = LeannBuilder(
    backend_name="diskann",
    embedding_model="BAAI/bge-base-en-v1.5",
    build_complexity=128
)
chat = LeannChat(
    index_path,
    llm_config={"type": "ollama", "model": "llama3:latest"}
)
""")

print("\nExample 3: Maximum Storage Savings")
print("-" * 40)
print("""
# Maximum storage efficiency
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="facebook/contriever",
    graph_degree=32,
    build_complexity=64
)
# Uses full recomputation for 97%+ storage savings
""")

print("\nExample 4: Large Dataset Setup")
print("-" * 40)
print("""
# Best for large datasets (100k+ documents)
builder = LeannBuilder(
    backend_name="diskann",
    embedding_model="facebook/contriever",
    graph_degree=64,
    build_complexity=128
)
# Faster search on large datasets with good storage savings
""")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n✓ Built indexes from different sources:")
print("  - Manual text addition (Python API)")
print("  - Documents with metadata")
print("  - Different backends (HNSW, DiskANN)")
print("  - Different embedding models")

print("\n✓ Demonstrated advanced features:")
print("  - Metadata filtering with various operators")
print("  - Grep search for exact text matching")
print("  - Comparison of semantic vs grep search")

print("\n✓ Showed different model configurations:")
print("  - Embedding models (small, medium, large)")
print("  - Backends (HNSW, DiskANN)")
print("  - LLM providers (Ollama, OpenAI, HuggingFace)")
print("  - Practical setup examples")

print("\n" + "=" * 80)
print("Next Steps:")
print("=" * 80)
print("1. Try building your own index: leann build my-index --docs ./your-docs")
print("2. Experiment with metadata filtering on your documents")
print("3. Test different embedding models for your use case")
print("4. Use grep search for exact code/function matching")
print("5. Configure for your specific needs (speed vs quality vs storage)")

print("\n" + "=" * 80)
