# Normalized Embeddings Support in LEANN

LEANN now automatically detects normalized embedding models and sets the appropriate distance metric for optimal performance.

## What are Normalized Embeddings?

Normalized embeddings are vectors with L2 norm = 1 (unit vectors). These embeddings are optimized for cosine similarity rather than Maximum Inner Product Search (MIPS).

## Automatic Detection

When you create a `LeannBuilder` instance with a normalized embedding model, LEANN will:

1. **Automatically set `distance_metric="cosine"`** if not specified
2. **Show a warning** if you manually specify a different distance metric
3. **Provide optimal search performance** with the correct metric

## Supported Normalized Embedding Models

### OpenAI
All OpenAI text embedding models are normalized:
- `text-embedding-ada-002`
- `text-embedding-3-small`
- `text-embedding-3-large`

### Voyage AI
All Voyage AI embedding models are normalized:
- `voyage-2`
- `voyage-3`
- `voyage-large-2`
- `voyage-multilingual-2`
- `voyage-code-2`

### Cohere
All Cohere embedding models are normalized:
- `embed-english-v3.0`
- `embed-multilingual-v3.0`
- `embed-english-light-v3.0`
- `embed-multilingual-light-v3.0`

## Example Usage

```python
from leann.api import LeannBuilder

# Automatic detection - will use cosine distance
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="text-embedding-3-small",
    embedding_mode="openai"
)
# Warning: Detected normalized embeddings model 'text-embedding-3-small'...
# Automatically setting distance_metric='cosine'

# Manual override (not recommended)
builder = LeannBuilder(
    backend_name="hnsw",
    embedding_model="text-embedding-3-small",
    embedding_mode="openai",
    distance_metric="mips"  # Will show warning
)
# Warning: Using 'mips' distance metric with normalized embeddings...
```

## Non-Normalized Embeddings

Models like `facebook/contriever` and other sentence-transformers models that are not normalized will continue to use MIPS by default, which is optimal for them.

## Why This Matters

Using the wrong distance metric with normalized embeddings can lead to:
- **Poor search quality** due to HNSW's early termination with narrow score ranges
- **Incorrect ranking** of search results
- **Suboptimal performance** compared to using the correct metric

For more details on why this happens, see our analysis in the [embedding detection code](../packages/leann-core/src/leann/api.py) which automatically handles normalized embeddings and MIPS distance metric issues.
