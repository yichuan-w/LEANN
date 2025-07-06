# HNSW Index Storage Optimization

This document explains the storage optimization features available in the HNSW backend.

## Storage Modes

The HNSW backend supports two orthogonal optimization techniques:

### 1. CSR Compression (`is_compact=True`)
- Converts the graph structure from standard format to Compressed Sparse Row (CSR) format
- Reduces memory overhead from graph adjacency storage
- Maintains all embedding data for direct access

### 2. Embedding Pruning (`is_recompute=True`) 
- Removes embedding vectors from the index file
- Replaces them with a NULL storage marker
- Requires recomputation via embedding server during search
- Must be used with `is_compact=True` for efficiency

## Performance Impact

**Storage Reduction (100 vectors, 384 dimensions):**
```
Standard format:     168 KB (embeddings + graph)
CSR only:           160 KB (embeddings + compressed graph)  
CSR + Pruned:         6 KB (compressed graph only)
```

**Key Benefits:**
- **CSR compression**: ~5% size reduction from graph optimization
- **Embedding pruning**: ~95% size reduction by removing embeddings
- **Combined**: Up to 96% total storage reduction

## Usage

```python
# Standard format (largest)
builder = LeannBuilder(
    backend_name="hnsw",
    is_compact=False,
    is_recompute=False
)

# CSR compressed (medium)
builder = LeannBuilder(
    backend_name="hnsw", 
    is_compact=True,
    is_recompute=False
)

# CSR + Pruned (smallest, requires embedding server)
builder = LeannBuilder(
    backend_name="hnsw",
    is_compact=True,      # Required for pruning
    is_recompute=True     # Default: enabled
)
```

## Trade-offs

| Mode | Storage | Search Speed | Memory Usage | Setup Complexity |
|------|---------|--------------|--------------|------------------|
| Standard | Largest | Fastest | Highest | Simple |
| CSR | Medium | Fast | Medium | Simple |
| CSR + Pruned | Smallest | Slower* | Lowest | Complex** |

*Requires network round-trip to embedding server for recomputation  
**Needs embedding server and passages file for search