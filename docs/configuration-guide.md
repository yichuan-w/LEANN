# LEANN Configuration Guide

This guide helps you optimize LEANN for different use cases and understand the trade-offs between various configuration options.

## Getting Started: Simple is Better

When first trying LEANN, start with a small dataset to quickly validate your approach:

**For document RAG**: The default `data/` directory works perfectly - just a few PDFs let you test in minutes
```bash
python -m apps.document_rag --query "What techniques does LEANN use?"
```

**For other data sources**: Limit the dataset size for quick testing
```bash
# WeChat: Test with recent messages only
python -m apps.wechat_rag --max-items 100 --query "昨天聊了什么"

# Browser history: Last few days
python -m apps.browser_rag --max-items 500 --query "AI papers I read"

# Email: Recent inbox
python -m apps.email_rag --max-items 200 --query "meeting schedules"
```

Once validated, scale up gradually:
- 100 documents → 1,000 → 10,000 → full dataset
- This helps identify issues early before committing to long processing times

## Embedding Model Selection: Understanding the Trade-offs

Based on our experience developing LEANN, embedding models fall into three categories:

### Small Models (< 100M parameters)
**Example**: `sentence-transformers/all-MiniLM-L6-v2` (22M params)
- **Pros**: Lightweight, fast for both indexing and inference
- **Cons**: Lower semantic understanding, may miss nuanced relationships
- **Use when**: Speed is critical, handling simple queries, on interactive mode or just experimenting with LEANN

### Medium Models (100M-500M parameters)
**Example**: `facebook/contriever` (110M params), `BAAI/bge-base-en-v1.5` (110M params)
- **Pros**: Balanced performance, good multilingual support, reasonable speed
- **Cons**: Requires more compute than small models
- **Use when**: Need quality results without extreme compute requirements, general-purpose RAG applications

### Large Models (500M+ parameters)
**Example**: `Qwen/Qwen3-Embedding-0.6B` (600M params), `intfloat/multilingual-e5-large` (560M params)
- **Pros**: Best semantic understanding, captures complex relationships, excellent multilingual support
- **Cons**: Slower inference, longer index build times
- **Use when**: Quality is paramount and you have sufficient compute resources

### Cloud vs Local Trade-offs

**OpenAI Embeddings** (`text-embedding-3-small/large`)
- **Pros**: No local compute needed, consistently fast, high quality
- **Cons**: Requires API key, costs money, data leaves your system, [known limitations with certain languages](https://yichuan-w.github.io/blog/lessons_learned_in_dev_leann/)
- **When to use**: Prototyping, non-sensitive data, need immediate results

**Local Embeddings**
- **Pros**: Complete privacy, no ongoing costs, full control, can sometimes outperform OpenAI embeddings
- **Cons**: Slower than cloud APIs, requires local compute resources
- **When to use**: Production systems, sensitive data, cost-sensitive applications

## Index Selection: Matching Your Scale

### HNSW (Hierarchical Navigable Small World)
**Best for**: Small to medium datasets (< 10M vectors)
- Full recomputation required
- High memory usage during build phase
- Excellent recall (95%+)

```bash
# Optimal for most use cases
--backend-name hnsw --graph-degree 32 --build-complexity 64
```

### DiskANN
**Best for**: Large datasets (> 10M vectors, 10GB+ index size)
- Uses Product Quantization (PQ) for coarse filtering during graph traversal
- Recomputes only top candidates for exact distance calculation

```bash
# For billion-scale deployments
--backend-name diskann --graph-degree 64 --build-complexity 128
```

## LLM Selection: Engine and Model Comparison

### LLM Engines

**OpenAI** (`--llm openai`)
- **Pros**: Best quality, consistent performance, no local resources needed
- **Cons**: Costs money ($0.15-2.5 per million tokens), requires internet, data privacy concerns
- **Models**: `gpt-4o-mini` (fast, cheap), `gpt-4o` (best quality), `o3-mini` (reasoning, not so expensive)

**Ollama** (`--llm ollama`)
- **Pros**: Fully local, free, privacy-preserving, good model variety
- **Cons**: Requires local GPU/CPU resources, slower than cloud APIs, need to pre-download models by `ollama pull`
- **Models**: `qwen3:1.7b` (best general quality), `deepseek-r1:1.5b` (reasoning)

**HuggingFace** (`--llm hf`)
- **Pros**: Free tier available, huge model selection, direct model loading (vs Ollama's server-based approach)
- **Cons**: More complex initial setup
- **Models**: `Qwen/Qwen3-1.7B-FP8`

## Parameter Tuning Guide

### Search Complexity Parameters

**`--build-complexity`** (index building)
- Controls thoroughness during index construction
- Higher = better recall but slower build
- Recommendations:
  - 32: Quick prototyping
  - 64: Balanced (default)
  - 128: Production systems
  - 256: Maximum quality

**`--search-complexity`** (query time)
- Controls search thoroughness
- Higher = better results but slower
- Recommendations:
  - 16: Fast/Interactive search (500-1000ms on consumer hardware)
  - 32: High quality with diversity (1000-2000ms)
  - 64+: Maximum accuracy (2000ms+)

### Top-K Selection

**`--top-k`** (number of retrieved chunks)
- More chunks = better context but slower LLM processing
- Should be always smaller than `--search-complexity`
- Guidelines:
  - 3-5: Simple factual queries
  - 5-10: General questions (default)
  - 10+: Complex multi-hop reasoning

**Trade-off formula**:
- Retrieval time ∝ log(n) × search_complexity
- LLM processing time ∝ top_k × chunk_size
- Total context = top_k × chunk_size tokens

### Graph Degree (HNSW/DiskANN)

**`--graph-degree`**
- Number of connections per node in the graph
- Higher = better recall but more memory
- HNSW: 16-32 (default: 32)
- DiskANN: 32-128 (default: 64)


## Performance Optimization Checklist

### If Embedding is Too Slow

1. **Switch to smaller model**:
   ```bash
   # From large model
   --embedding-model Qwen/Qwen3-Embedding
   # To small model
   --embedding-model sentence-transformers/all-MiniLM-L6-v2
   ```

2. **Use MLX on Apple Silicon**:
   ```bash
   --embedding-mode mlx --embedding-model mlx-community/multilingual-e5-base-mlx
   ```

3. **Limit dataset size for testing**:
   ```bash
   --max-items 1000  # Process first 1k items only
   ```

### If Search Quality is Poor

1. **Increase retrieval count**:
   ```bash
   --top-k 30  # Retrieve more candidates
   ```

2. **Tune chunk size for your content**:
   - Technical docs: `--chunk-size 512`
   - Chat messages: `--chunk-size 128`
   - Mixed content: `--chunk-size 256`

3. **Upgrade embedding model**:
   ```bash
   # For English
   --embedding-model BAAI/bge-base-en-v1.5
   # For multilingual
   --embedding-model intfloat/multilingual-e5-large
   ```

## Understanding the Trade-offs

Every configuration choice involves trade-offs:

| Factor | Small/Fast | Large/Quality |
|--------|------------|---------------|
| Embedding Model | `all-MiniLM-L6-v2` | `Qwen/Qwen3-Embedding-0.6B` |
| Chunk Size | 512 tokens | 128 tokens |
| Index Type | HNSW | DiskANN |
| LLM | `qwen3:1.7b` | `gpt-4o` |

The key is finding the right balance for your specific use case. Start small and simple, measure performance, then scale up only where needed.

## Deep Dive: Critical Configuration Decisions

### When to Disable Recomputation

LEANN's recomputation feature provides exact distance calculations but can be disabled for extreme QPS requirements:

```bash
--no-recompute  # Disable selective recomputation
```

**Trade-offs**:
- **With recomputation** (default): Exact distances, best quality, higher latency, minimal storage (only stores metadata, recomputes embeddings on-demand)
- **Without recomputation**: Must store full embeddings, significantly higher memory and storage usage (10-100x more), but faster search

**Disable when**:
- You have abundant storage and memory
- Need extremely low latency (< 100ms)
- Running a read-heavy workload where storage cost is acceptable

## Further Reading

- [Lessons Learned Developing LEANN](https://yichuan-w.github.io/blog/lessons_learned_in_dev_leann/)
- [LEANN Technical Paper](https://arxiv.org/abs/2506.08276)
- [DiskANN Original Paper](https://papers.nips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
