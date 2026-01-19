# LEANN Demo Summary

This document summarizes the demonstrations and examples created during this session.

## ✅ What We Accomplished

### 1. **Basic Setup** ✓
- Installed LEANN using `uv`
- Created virtual environment
- Installed from PyPI
- Verified installation with basic demo

### 2. **Basic Demo** ✓
- Created `demo.py` with Python API
- Demonstrated index building
- Demonstrated semantic search
- Demonstrated chat with Ollama (`gemma3:4b`)
- All features working successfully

### 3. **CLI Usage** ✓
- Built index from documents: `leann build my-docs --docs ./data`
- Searched documents: `leann search my-docs "query"`
- Chatted with documents: `leann ask my-docs "question"`
- Listed all indexes: `leann list`

### 4. **Advanced Features** ✓

#### A. Metadata Filtering ✓
- Created `demo_advanced.py` demonstrating metadata indexing
- Showed how to add metadata to documents
- Demonstrated various filter operators:
  - `==` (equal)
  - `!=` (not equal)
  - `in` (in list)
  - `!==` None (exists)
- Created `demo_metadata.leann` index with metadata

#### B. Grep Search ✓
- Demonstrated exact text matching with `use_grep=True`
- Tested with CLI-built index (`my-docs`)
- Successfully found exact matches for:
  - "LEANN"
  - "storage"
  - "graph"
- Grep search works with indexes built via CLI

#### C. Different Model Configurations ✓
- Tested default model: `facebook/contriever`
- Tested small model: `sentence-transformers/all-MiniLM-L6-v2`
- Tested HNSW backend (default)
- Tested DiskANN backend
- Tested different Ollama models: `gemma3:4b`, `llama3:latest`

## 📁 Files Created

1. **`demo.py`** - Basic Python API demo
   - Index building
   - Semantic search
   - Chat with Ollama

2. **`demo_advanced.py`** - Advanced features demo
   - Metadata filtering
   - Grep search
   - Multiple model configurations
   - Different backends

3. **`ADVANCED_FEATURES_DEMO.md`** - Comprehensive documentation
   - Metadata filtering guide
   - Grep search guide
   - Model configuration guide
   - Code examples

4. **`DEMO_SUMMARY.md`** - This file

## 📊 Indexes Created

1. **`demo.leann`** - Basic demo index (2 text passages)
2. **`my-docs`** - CLI-built index from `./data` directory (2,412 chunks, 2.7 MB)
3. **`demo_metadata.leann`** - Advanced demo with metadata (5 documents)
4. **`demo_default_embedding.leann`** - Test index with default model
5. **`demo_hnsw_backend.leann`** - Test index with HNSW backend
6. **`demo_diskann_backend.leann`** - Test index with DiskANN backend (if available)

## 🔍 Key Learnings

### Metadata Filtering
- Allows sophisticated search queries
- Supports multiple filter operators
- Enables use cases like:
  - Code search by file type
  - Content filtering by difficulty/topic
  - Date-based filtering
  - Custom attribute filtering

### Grep Search
- Requires CLI-built indexes (creates `.jsonl` passages file)
- Best for exact text matching
- Useful for finding:
  - Function/class names
  - Error messages
  - Exact code patterns
  - Keywords

### Model Configurations
- **Embedding Models**:
  - Small (< 100M): Fast, lower quality
  - Medium (100M-500M): Balanced (default)
  - Large (500M+): Best quality, slower
- **Backends**:
  - HNSW: Maximum storage savings, default
  - DiskANN: Faster search on large datasets
- **LLMs**:
  - Ollama: Local, free, privacy-focused
  - OpenAI: Cloud, paid, high quality
  - HuggingFace: Local, free, direct loading

## 🚀 Next Steps

### Try These Commands:

1. **List all indexes:**
   ```bash
   cd /Users/peggs/leann
   source .venv/bin/activate
   leann list
   ```

2. **Search with filters (Python API):**
   ```python
   from leann import LeannSearcher
   searcher = LeannSearcher("demo_metadata.leann")
   results = searcher.search(
       "programming",
       metadata_filters={"language": {"==": "Python"}}
   )
   ```

3. **Grep search:**
   ```python
   from leann import LeannSearcher
   searcher = LeannSearcher(".leann/indexes/my-docs/documents.leann")
   results = searcher.search("LEANN", use_grep=True)
   ```

4. **Interactive chat:**
   ```bash
   cd /Users/peggs/leann
   source .venv/bin/activate
   leann ask my-docs --interactive --llm ollama --model gemma3:4b
   ```

5. **Build index from your documents:**
   ```bash
   leann build my-index --docs /path/to/your/documents
   ```

## 📚 Resources

- **GitHub**: https://github.com/yichuan-w/LEANN
- **Configuration Guide**: See `ADVANCED_FEATURES_DEMO.md`
- **Examples**: See `./examples/` directory
- **Documentation**: See `./docs/` directory

## 💡 Tips

1. **For code search**: Use metadata filtering to filter by file type, language, etc.
2. **For exact matches**: Use grep search with CLI-built indexes
3. **For production**: Use larger embedding models and DiskANN backend
4. **For privacy**: Use Ollama for local LLM inference
5. **For speed**: Use smaller models and HNSW backend

---

**Happy searching with LEANN! 🎉**
