# LEANN - The smallest vector index in the world

LEANN is a revolutionary vector database that democratizes personal AI. Transform your laptop into a powerful RAG system that can index and search through millions of documents while using **97% less storage** than traditional solutions **without accuracy loss**.

## Installation

```bash
# Default installation (HNSW backend, recommended)
uv pip install leann

# With DiskANN backend (for large-scale deployments)
uv pip install leann[diskann]
```

## Quick Start

```python
from leann import LeannBuilder, LeannSearcher, LeannChat

# Build an index
builder = LeannBuilder(backend_name="hnsw")
builder.add_text("LEANN saves 97% storage compared to traditional vector databases.")
builder.build_index("my_index.leann")

# Search
searcher = LeannSearcher("my_index.leann")
results = searcher.search("storage savings", top_k=3)

# Chat with your data
chat = LeannChat("my_index.leann", llm_config={"type": "ollama", "model": "llama3.2:1b"})
response = chat.ask("How much storage does LEANN save?")
```

## Documentation

For full documentation, visit [https://leann.readthedocs.io](https://leann.readthedocs.io)

## License

MIT License 