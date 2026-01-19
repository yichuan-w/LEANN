# LEANN Advanced Features Demo

This document demonstrates advanced LEANN features including metadata filtering, grep search, and different model configurations.

## 1. Building Index with Metadata

Metadata allows you to filter search results by custom criteria.

### Example: Python API

```python
from leann import LeannBuilder, LeannSearcher

# Create builder
builder = LeannBuilder(backend_name="hnsw")

# Add text with metadata
builder.add_text(
    "Python is a high-level programming language.",
    metadata={
        "language": "Python",
        "topic": "programming",
        "difficulty": "beginner",
        "file_type": ".py"
    }
)

builder.add_text(
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    metadata={
        "language": "Python",
        "topic": "algorithms",
        "difficulty": "intermediate",
        "file_type": ".py",
        "function_name": "fibonacci"
    }
)

# Build index
builder.build_index("demo_metadata.leann")
```

## 2. Metadata Filtering Search

Search with metadata filters to find specific content.

### Supported Operators

- `==`: Equal
- `!=`: Not equal
- `<`, `<=`, `>`, `>=`: Comparison
- `in`: In list
- `not_in`: Not in list
- `contains`: String contains
- `starts_with`: String starts with
- `ends_with`: String ends with
- `is_true`, `is_false`: Boolean checks

### Example: Filtered Search

```python
from leann import LeannSearcher

searcher = LeannSearcher("demo_metadata.leann")

# Search for Python content only
results = searcher.search(
    "programming language",
    top_k=5,
    metadata_filters={
        "language": {"==": "Python"}
    }
)

# Search for advanced machine learning content
results = searcher.search(
    "machine learning",
    top_k=3,
    metadata_filters={
        "topic": {"==": "machine_learning"},
        "difficulty": {"==": "advanced"}
    }
)

# Search for beginner/intermediate Python files
results = searcher.search(
    "code",
    top_k=5,
    metadata_filters={
        "file_type": {"==": ".py"},
        "difficulty": {"in": ["beginner", "intermediate"]}
    }
)

# Search for content with function definitions
results = searcher.search(
    "function",
    top_k=3,
    metadata_filters={
        "function_name": {"!=": None}
    }
)
```

### Example Output

```
Search for Python content
Query: 'programming language'
Filters: {'language': {'==': 'Python'}}
Results: 2 found
  1. Score: 0.970
     Text: Python is a high-level programming language...
     Metadata: {'language': 'Python', 'topic': 'programming', 'difficulty': 'beginner', 'file_type': '.py'}
```

## 3. Grep Search (Exact Text Matching)

Grep search finds exact text matches, useful for:
- Finding specific function/class names
- Searching for error messages
- Finding exact code patterns
- Keyword-based search

### Important Note

Grep search works best with indexes built using the **CLI** command:
```bash
leann build my-docs --docs ./documents
```

This creates a `.jsonl` passages file that grep search requires.

### Example: Grep Search

```python
from leann import LeannSearcher

# Load CLI-built index
searcher = LeannSearcher(".leann/indexes/my-docs/documents.leann")

# Exact text search
results = searcher.search("LEANN", top_k=5, use_grep=True)

# Find function definitions
results = searcher.search("def train_model", top_k=3, use_grep=True)

# Find error messages
results = searcher.search("FileNotFoundError", use_grep=True)

# Find class definitions
results = searcher.search("class SearchResult", top_k=5, use_grep=True)
```

### CLI Example

```bash
# Grep search via CLI (future feature)
leann search my-docs "LEANN" --grep
```

## 4. Different Model Configurations

### 4.1 Embedding Models

#### Small Models (< 100M params)
**Best for**: Fast prototyping, speed-critical applications

```python
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # 22M params
)
```

**Pros**: Fast, lightweight  
**Cons**: Lower semantic understanding

#### Medium Models (100M-500M params)
**Best for**: Balanced performance (default)

```python
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="facebook/contriever"  # 110M params (default)
)

# Alternative
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="BAAI/bge-base-en-v1.5"  # 110M params
)
```

**Pros**: Good balance of speed and quality  
**Cons**: Moderate compute requirements

#### Large Models (500M+ params)
**Best for**: Production use, maximum quality

```python
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="Qwen/Qwen3-Embedding-0.6B"  # 600M params
)
```

**Pros**: Best semantic understanding, near OpenAI performance  
**Cons**: Slower, more compute required

### 4.2 Backend Selection

#### HNSW (Default)
**Best for**: Most datasets, maximum storage savings

```python
builder = LeannBuilder(backend_name="hnsw")
```

**Pros**: 
- Excellent storage efficiency
- Full recomputation support
- 95%+ storage savings

**Cons**: 
- Higher memory during build
- Full recomputation required

#### DiskANN
**Best for**: Large datasets, faster search

```python
builder = LeannBuilder(backend_name="diskann")
```

**Pros**:
- 3x+ faster search on large datasets
- Smart storage with graph partitioning
- Better scaling for 100k+ documents

**Cons**:
- Slightly larger index size
- More complex configuration

### 4.3 LLM Configurations

#### Ollama (Local, Free)
```python
chat = LeannChat(
    index_path,
    llm_config={
        "type": "ollama",
        "model": "gemma3:4b"  # or llama3:latest, gpt-oss:20b
    }
)
```

**Available Models**:
- `gemma3:4b` - Fast, lightweight
- `llama3:latest` - Balanced
- `gpt-oss:20b` - Reasoning model (supports thinking_budget)

#### OpenAI (Cloud, Paid)
```python
chat = LeannChat(
    index_path,
    llm_config={
        "type": "openai",
        "model": "gpt-4o-mini"  # or gpt-4o, o3
    }
)
```

#### HuggingFace (Local, Free)
```python
chat = LeannChat(
    index_path,
    llm_config={
        "type": "hf",
        "model": "Qwen/Qwen3-1.7B-FP8"
    }
)
```

### 4.4 Advanced Configuration Options

#### Build Complexity
Controls thoroughness during index construction:

```python
builder = LeannBuilder(
    backend_name="hnsw",
    build_complexity=64  # 32: quick, 64: balanced (default), 128: production
)
```

#### Search Complexity
Controls search thoroughness at query time:

```python
results = searcher.search(
    query,
    top_k=20,
    search_complexity=64  # 16: fast, 32: balanced, 64+: maximum quality
)
```

#### Graph Degree
Number of connections per node:

```python
builder = LeannBuilder(
    backend_name="hnsw",
    graph_degree=32  # HNSW: 16-32 (default: 32), DiskANN: 32-128 (default: 64)
)
```

## 5. Complete Example

See `demo_advanced.py` for a complete working example demonstrating all these features.

## 6. CLI Commands

### Build with Metadata (Future)
```bash
leann build my-code --docs ./src --metadata-config metadata.yaml
```

### Search with Filters (Future)
```bash
leann search my-code "function" --filter language=Python --filter difficulty=intermediate
```

### Grep Search
```bash
# Use Python API or CLI with --grep flag (when available)
```

## Resources

- [LEANN GitHub](https://github.com/yichuan-w/LEANN)
- [Configuration Guide](https://github.com/yichuan-w/LEANN/blob/main/docs/configuration-guide.md)
- [Examples Directory](./examples/)
