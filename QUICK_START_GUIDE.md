# LEANN Quick Start Guide

A comprehensive guide covering building indexes, advanced features, and model configurations.

## Part 1: Building Indexes from Different Documents

### 1.1 Python API - Manual Text Addition

```python
from leann import LeannBuilder

builder = LeannBuilder(backend_name="hnsw")
builder.add_text("Your first document text here.")
builder.add_text("Your second document text here.")
builder.build_index("my_index.leann")
```

### 1.2 Python API - With Metadata

```python
from leann import LeannBuilder

builder = LeannBuilder(backend_name="hnsw")

# Add text with metadata for filtering
builder.add_text(
    "Python is great for data science.",
    metadata={
        "language": "Python",
        "domain": "data_science",
        "difficulty": "beginner",
        "file_type": ".py"
    }
)

builder.add_text(
    "class MLModel(nn.Module): ...",
    metadata={
        "language": "Python",
        "domain": "deep_learning",
        "difficulty": "advanced",
        "file_type": ".py",
        "class_name": "MLModel"
    }
)

builder.build_index("index_with_metadata.leann")
```

### 1.3 CLI - From Document Directory

```bash
# Build index from directory
leann build my-docs --docs ./documents

# Build from multiple directories
leann build my-code --docs ./src ./tests ./config

# Build from specific file types
leann build my-ppts --docs ./ --file-types .pptx,.pdf

# Build with specific backend
leann build my-index --docs ./docs --backend diskann
```

### 1.4 Different Embedding Models

```python
# Default model (balanced performance)
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="facebook/contriever"  # 110M params, 768 dims
)

# Small model (fast, lightweight)
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # 22M params, 384 dims
)

# Large model (best quality)
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="Qwen/Qwen3-Embedding-0.6B"  # 600M params
)
```

### 1.5 Different Backends

```python
# HNSW (default, maximum storage savings)
builder = LeannBuilder(backend_name="hnsw")

# DiskANN (faster search on large datasets)
builder = LeannBuilder(backend_name="diskann")
```

## Part 2: Advanced Features

### 2.1 Metadata Filtering Search

#### Basic Filters

```python
from leann import LeannSearcher

searcher = LeannSearcher("index_with_metadata.leann")

# Filter by exact match
results = searcher.search(
    "programming",
    metadata_filters={"language": {"==": "Python"}}
)

# Filter by multiple conditions (AND)
results = searcher.search(
    "learning",
    metadata_filters={
        "domain": {"==": "deep_learning"},
        "difficulty": {"==": "advanced"}
    }
)

# Filter with 'in' operator
results = searcher.search(
    "code",
    metadata_filters={
        "difficulty": {"in": ["beginner", "intermediate"]}
    }
)

# Filter by existence (not None)
results = searcher.search(
    "function",
    metadata_filters={
        "function_name": {"!=": None}
    }
)
```

#### Supported Filter Operators

- `==`: Equal
- `!=`: Not equal
- `<`, `<=`, `>`, `>=`: Comparison
- `in`: In list
- `not_in`: Not in list
- `contains`: String contains
- `starts_with`: String starts with
- `ends_with`: String ends with
- `is_true`, `is_false`: Boolean checks

### 2.2 Grep Search (Exact Text Matching)

**Important:** Grep search works best with CLI-built indexes (creates `.jsonl` passages file).

```python
from leann import LeannSearcher

# Use CLI-built index
searcher = LeannSearcher(".leann/indexes/my-docs/documents.leann")

# Exact text search
results = searcher.search("LEANN", use_grep=True)

# Find function definitions
results = searcher.search("def train_model", use_grep=True)

# Find class definitions
results = searcher.search("class SearchResult", use_grep=True)

# Find error messages
results = searcher.search("FileNotFoundError", use_grep=True)
```

#### When to Use Grep vs Semantic Search

- **Grep Search**: Exact text matches, function names, error messages, code patterns
- **Semantic Search**: Meaning-based, natural language queries, similar concepts

```python
# Compare both approaches
query = "graph"

# Semantic search (meaning-based)
semantic_results = searcher.search(query, use_grep=False)

# Grep search (exact match)
grep_results = searcher.search(query, use_grep=True)
```

## Part 3: Different Model Configurations

### 3.1 Embedding Models Comparison

| Model | Size | Dimension | Speed | Quality | Use Case |
|-------|------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 22M | 384 | ⚡⚡⚡ | ⭐⭐ | Prototyping, speed-critical |
| `facebook/contriever` | 110M | 768 | ⚡⚡ | ⭐⭐⭐ | Default, balanced (recommended) |
| `BAAI/bge-base-en-v1.5` | 110M | 768 | ⚡⚡ | ⭐⭐⭐⭐ | Production English RAG |
| `Qwen/Qwen3-Embedding-0.6B` | 600M | Variable | ⚡ | ⭐⭐⭐⭐⭐ | Maximum quality |

### 3.2 Backend Comparison

| Backend | Storage Savings | Search Speed | Best For |
|---------|----------------|--------------|----------|
| **HNSW** (default) | 97%+ | Fast | Most datasets, maximum storage savings |
| **DiskANN** | 90%+ | 3x+ faster | Large datasets (100k+ docs), faster search |

### 3.3 LLM Configuration Examples

#### Ollama (Local, Free)

```python
from leann import LeannChat

# Fast, lightweight
chat = LeannChat(
    index_path,
    llm_config={"type": "ollama", "model": "gemma3:4b"}
)

# Balanced
chat = LeannChat(
    index_path,
    llm_config={"type": "ollama", "model": "llama3:latest"}
)

# Reasoning model
chat = LeannChat(
    index_path,
    llm_config={"type": "ollama", "model": "gpt-oss:20b"}
)
```

#### OpenAI (Cloud, Paid)

```python
# Fast, cheap
chat = LeannChat(
    index_path,
    llm_config={"type": "openai", "model": "gpt-4o-mini"}
)

# Best quality
chat = LeannChat(
    index_path,
    llm_config={"type": "openai", "model": "gpt-4o"}
)

# Reasoning model (with thinking budget)
chat = LeannChat(
    index_path,
    llm_config={
        "type": "openai",
        "model": "o3",
        "thinking_budget": "medium"  # low, medium, high
    }
)
```

### 3.4 Complete Configuration Examples

#### Example 1: Fast Prototyping Setup

```python
# Fast, lightweight
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

chat = LeannChat(
    index_path,
    llm_config={"type": "ollama", "model": "gemma3:4b"}
)
```

#### Example 2: Production Setup

```python
# High quality, production-ready
builder = LeannBuilder(
    backend_name="diskann",
    embedding_model="BAAI/bge-base-en-v1.5",
    build_complexity=128,
    graph_degree=64
)

chat = LeannChat(
    index_path,
    llm_config={"type": "ollama", "model": "llama3:latest"}
)
```

#### Example 3: Maximum Storage Savings

```python
# Maximum storage efficiency (97%+ savings)
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="facebook/contriever",
    graph_degree=32,
    build_complexity=64
)
# Uses full recomputation for maximum storage savings
```

#### Example 4: Large Dataset Setup

```python
# Best for large datasets (100k+ documents)
builder = LeannBuilder(
    backend_name="diskann",
    embedding_model="facebook/contriever",
    graph_degree=64,
    build_complexity=128
)
# Faster search on large datasets with good storage savings
```

## Quick Reference Commands

### CLI Commands

```bash
# Build index
leann build my-index --docs ./documents

# Search
leann search my-index "your query"

# Chat (single question)
leann ask my-index "your question"

# Chat (interactive)
leann ask my-index --interactive

# List all indexes
leann list

# Remove index
leann remove my-index
```

### Python API

```python
from leann import LeannBuilder, LeannSearcher, LeannChat

# Build
builder = LeannBuilder(backend_name="hnsw")
builder.add_text("text", metadata={...})
builder.build_index("index.leann")

# Search
searcher = LeannSearcher("index.leann")
results = searcher.search("query", top_k=10, metadata_filters={...})

# Chat
chat = LeannChat("index.leann", llm_config={...})
response = chat.ask("question", top_k=5)
```

## Next Steps

1. **Build your first index**: Start with a small document set
2. **Experiment with metadata**: Add metadata to enable filtering
3. **Try grep search**: Use CLI to build indexes for exact matching
4. **Compare models**: Test different embedding models for your use case
5. **Optimize configuration**: Tune for speed vs quality vs storage

## Resources

- **GitHub**: https://github.com/yichuan-w/LEANN
- **Examples**: See `./examples/` directory
- **Advanced Guide**: See `ADVANCED_FEATURES_DEMO.md`
- **Configuration**: See `COMPREHENSIVE_GUIDE.py`

---

**Happy searching with LEANN! 🎉**
