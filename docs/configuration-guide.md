# LEANN Configuration Guide

This guide helps you optimize LEANN for different use cases and understand the trade-offs between various configuration options.

## Getting Started: Simple is Better

When first trying LEANN, start with a small dataset to quickly validate your approach. Use the default `data/` directory which contains just a few files - this lets you test the full pipeline in minutes rather than hours.

```bash
# Quick test with minimal data
python -m apps.document_rag --max-items 100 --query "What techniques does LEANN use?"
```

Once validated, scale up gradually:
- 100 documents → 1,000 → 10,000 → full dataset
- This helps identify issues early before committing to long processing times

## Embedding Model Selection: Understanding the Trade-offs

Based on our experience developing LEANN, embedding models fall into three categories:

### Small Models (384-768 dims)
**Example**: `sentence-transformers/all-MiniLM-L6-v2`
- **Pros**: Fast inference (10-50ms, 384 dims), good for real-time applications
- **Cons**: Lower semantic understanding, may miss nuanced relationships
- **Use when**: Speed is critical, handling simple queries

### Medium Models (768-1024 dims)
**Example**: `facebook/contriever`
- **Pros**: Balanced performance, good multilingual support, reasonable speed
- **Cons**: Requires more compute than small models
- **Use when**: Need quality results without extreme compute requirements

### Large Models (1024+ dims)
**Example**: `Qwen/Qwen3-Embedding`
- **Pros**: Best semantic understanding, captures complex relationships, excellent multilingual support
- **Cons**: Slow inference, high memory usage, may overfit on small datasets
- **Use when**: Quality is paramount and you have sufficient compute

### Cloud vs Local Trade-offs

**OpenAI Embeddings** (`text-embedding-3-small/large`)
- **Pros**: No local compute needed, consistently fast, high quality
- **Cons**: Requires API key, costs money, data leaves your system, [known limitations with certain languages](https://yichuan-w.github.io/blog/lessons_learned_in_dev_leann/)
- **When to use**: Prototyping, non-sensitive data, need immediate results

**Local Embeddings**
- **Pros**: Complete privacy, no ongoing costs, full control
- **Cons**: Requires GPU for good performance, setup complexity
- **When to use**: Production systems, sensitive data, cost-sensitive applications

## Index Selection: Matching Your Scale

### HNSW (Hierarchical Navigable Small World)
**Best for**: Small to medium datasets (< 10M vectors)
- Fast search (1-10ms latency)
- Full recomputation required (no double queue optimization)
- High memory usage during build phase
- Excellent recall (95%+)

```bash
# Optimal for most use cases
--backend-name hnsw --graph-degree 32 --build-complexity 64
```

### DiskANN
**Best for**: Large datasets (> 10M vectors, 10GB+ index size)
- Uses Product Quantization (PQ) for coarse filtering in double queue architecture
- Extremely fast search through selective recomputation

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
- **Cons**: Requires local GPU/CPU resources, slower than cloud
- **Models**: `qwen3:1.7b` (best general quality), `deepseek-r1:1.5b` (reasoning)

**HuggingFace** (`--llm hf`)
- **Pros**: Free tier available, huge model selection, direct model loading (vs Ollama's server-based approach)
- **Cons**: API rate limits, local mode needs significant resources, more complex setup
- **Models**: `Qwen/Qwen3-1.7B-FP8`


### Model Size Trade-offs

| Model Size | Speed | Quality | Memory | Use Case |
|------------|-------|---------|---------|----------|
| 1B params | 50-100 tok/s | Basic | 2-4GB | Quick answers, simple queries |
| 3B params | 20-50 tok/s | Good | 4-8GB | General purpose RAG |
| 7B params | 10-20 tok/s | Excellent | 8-16GB | Complex reasoning |
| 13B+ params | 5-10 tok/s | Best | 16-32GB+ | Research, detailed analysis |

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

## Common Configurations by Use Case

### 1. Quick Experimentation
```bash
python -m apps.document_rag \
  --max-items 1000 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --backend-name hnsw \
  --llm ollama --llm-model llama3.2:1b
```

### 2. Personal Knowledge Base
```bash
python -m apps.document_rag \
  --embedding-model facebook/contriever \
  --chunk-size 512 --chunk-overlap 128 \
  --backend-name hnsw \
  --llm ollama --llm-model llama3.2:3b
```

### 3. Production RAG System
```bash
python -m apps.document_rag \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --chunk-size 256 --chunk-overlap 64 \
  --backend-name diskann \
  --llm openai --llm-model gpt-4o-mini \
  --top-k 20 --search-complexity 64
```

### 4. Multi-lingual Support (e.g., WeChat)
```bash
python -m apps.wechat_rag \
  --embedding-model intfloat/multilingual-e5-base \
  --chunk-size 192 --chunk-overlap 48 \
  --backend-name hnsw \
  --llm ollama --llm-model qwen3:8b
```

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

3. **Process in batches**:
   ```bash
   --max-items 10000  # Process incrementally
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
| Embedding Model | all-MiniLM-L6-v2 | BAAI/bge-large |
| Chunk Size | 128 tokens | 512 tokens |
| Index Type | HNSW | DiskANN |
| LLM | llama3.2:1b | gpt-4o |

The key is finding the right balance for your specific use case. Start small and simple, measure performance, then scale up only where needed.

## Deep Dive: Critical Configuration Decisions

### When to Disable Recomputation

LEANN's recomputation feature provides exact distance calculations but can be disabled for extreme QPS requirements:

```bash
--no-recompute  # Disable selective recomputation
```

**Trade-offs**:
- **With recomputation** (default): Exact distances, best quality, higher latency
- **Without recomputation**: Approximate distances via PQ, 2-5x faster, significantly lower memory and storage usage

**Disable when**:
- QPS requirements > 1000/sec
- Slight accuracy loss is acceptable
- Running on resource-constrained hardware

## Performance Monitoring

Key metrics to watch:
- Index build time
- Query latency (p50, p95, p99)
- Memory usage during build and search
- Disk I/O patterns (for DiskANN)
- Recomputation ratio (% of candidates recomputed)

## Further Reading

- [Lessons Learned Developing LEANN](https://yichuan-w.github.io/blog/lessons_learned_in_dev_leann/)
- [LEANN Technical Paper](https://arxiv.org/abs/2506.08276)
- [DiskANN Original Paper](https://papers.nips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
