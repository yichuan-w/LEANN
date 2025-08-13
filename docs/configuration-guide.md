# LEANN Configuration Guide

This guide helps you optimize LEANN for different use cases and understand the trade-offs between various configuration options.

## Getting Started: Simple is Better

When first trying LEANN, start with a small dataset to quickly validate your approach:

**For document RAG**: The default `data/` directory works perfectly - includes 2 AI research papers, Pride and Prejudice literature, and a technical report
```bash
python -m apps.document_rag --query "What techniques does LEANN use?"
```

**For other data sources**: Limit the dataset size for quick testing
```bash
# WeChat: Test with recent messages only
python -m apps.wechat_rag --max-items 100 --query "What did we discuss about the project timeline?"

# Browser history: Last few days
python -m apps.browser_rag --max-items 500 --query "Find documentation about vector databases"

# Email: Recent inbox
python -m apps.email_rag --max-items 200 --query "Who sent updates about the deployment status?"
```

Once validated, scale up gradually:
- 100 documents → 1,000 → 10,000 → full dataset (`--max-items -1`)
- This helps identify issues early before committing to long processing times

## Embedding Model Selection: Understanding the Trade-offs

Based on our experience developing LEANN, embedding models fall into three categories:

### Small Models (< 100M parameters)
**Example**: `sentence-transformers/all-MiniLM-L6-v2` (22M params)
- **Pros**: Lightweight, fast for both indexing and inference
- **Cons**: Lower semantic understanding, may miss nuanced relationships
- **Use when**: Speed is critical, handling simple queries, interactive mode, or just experimenting with LEANN. If time is not a constraint, consider using a larger/better embedding model

### Medium Models (100M-500M parameters)
**Example**: `facebook/contriever` (110M params), `BAAI/bge-base-en-v1.5` (110M params)
- **Pros**: Balanced performance, good multilingual support, reasonable speed
- **Cons**: Requires more compute than small models
- **Use when**: Need quality results without extreme compute requirements, general-purpose RAG applications

### Large Models (500M+ parameters)
**Example**: `Qwen/Qwen3-Embedding-0.6B` (600M params), `intfloat/multilingual-e5-large` (560M params)
- **Pros**: Best semantic understanding, captures complex relationships, excellent multilingual support. **Qwen3-Embedding-0.6B achieves nearly OpenAI API performance!**
- **Cons**: Slower inference, longer index build times
- **Use when**: Quality is paramount and you have sufficient compute resources. **Highly recommended** for production use

### Quick Start: Cloud and Local Embedding Options

**OpenAI Embeddings (Fastest Setup)**
For immediate testing without local model downloads(also if you [do not have GPU](https://github.com/yichuan-w/LEANN/issues/43) and do not care that much about your document leak, you should use this, we compute the embedding and recompute using openai API):
```bash
# Set OpenAI embeddings (requires OPENAI_API_KEY)
--embedding-mode openai --embedding-model text-embedding-3-small
```

**Ollama Embeddings (Privacy-Focused)**
For local embeddings with complete privacy:
```bash
# First, pull an embedding model
ollama pull nomic-embed-text

# Use Ollama embeddings
--embedding-mode ollama --embedding-model nomic-embed-text
```

<details>
<summary><strong>Cloud vs Local Trade-offs</strong></summary>

**OpenAI Embeddings** (`text-embedding-3-small/large`)
- **Pros**: No local compute needed, consistently fast, high quality
- **Cons**: Requires API key, costs money, data leaves your system, [known limitations with certain languages](https://yichuan-w.github.io/blog/lessons_learned_in_dev_leann/)
- **When to use**: Prototyping, non-sensitive data, need immediate results

**Local Embeddings**
- **Pros**: Complete privacy, no ongoing costs, full control, can sometimes outperform OpenAI embeddings
- **Cons**: Slower than cloud APIs, requires local compute resources
- **When to use**: Production systems, sensitive data, cost-sensitive applications

</details>

## Index Selection: Matching Your Scale

### HNSW (Hierarchical Navigable Small World)
**Best for**: Small to medium datasets (< 10M vectors) - **Default and recommended for extreme low storage**
- Full recomputation required
- High memory usage during build phase
- Excellent recall (95%+)

```bash
# Optimal for most use cases
--backend-name hnsw --graph-degree 32 --build-complexity 64
```

### DiskANN
**Best for**: Large datasets (> 10M vectors, 10GB+ index size) - **⚠️ Beta version, still in active development**
- Uses Product Quantization (PQ) for coarse filtering during graph traversal
- Novel approach: stores only PQ codes, performs rerank with exact computation in final step
- Implements a corner case of double-queue: prunes all neighbors and recomputes at the end

```bash
# For billion-scale deployments
--backend-name diskann --graph-degree 64 --build-complexity 128
```

## LLM Selection: Engine and Model Comparison

### LLM Engines

**OpenAI** (`--llm openai`)
- **Pros**: Best quality, consistent performance, no local resources needed
- **Cons**: Costs money ($0.15-2.5 per million tokens), requires internet, data privacy concerns
- **Models**: `gpt-4o-mini` (fast, cheap), `gpt-4o` (best quality), `o3` (reasoning), `o3-mini` (reasoning, cheaper)
- **Thinking Budget**: Use `--thinking-budget low/medium/high` for o-series reasoning models (o3, o3-mini, o4-mini)
- **Note**: Our current default, but we recommend switching to Ollama for most use cases

**Ollama** (`--llm ollama`)
- **Pros**: Fully local, free, privacy-preserving, good model variety
- **Cons**: Requires local GPU/CPU resources, slower than cloud APIs, need to install extra [ollama app](https://github.com/ollama/ollama?tab=readme-ov-file#ollama) and pre-download models by `ollama pull`
- **Models**: `qwen3:0.6b` (ultra-fast), `qwen3:1.7b` (balanced), `qwen3:4b` (good quality), `qwen3:7b` (high quality), `deepseek-r1:1.5b` (reasoning)
- **Thinking Budget**: Use `--thinking-budget low/medium/high` for reasoning models like GPT-Oss:20b

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
  - 16: Fast/Interactive search
  - 32: High quality with diversity
  - 64+: Maximum accuracy

### Top-K Selection

**`--top-k`** (number of retrieved chunks)
- More chunks = better context but slower LLM processing
- Should be always smaller than `--search-complexity`
- Guidelines:
  - 10-20: General questions (default: 20)
  - 30+: Complex multi-hop reasoning requiring comprehensive context

**Trade-off formula**:
- Retrieval time ∝ log(n) × search_complexity
- LLM processing time ∝ top_k × chunk_size
- Total context = top_k × chunk_size tokens

### Thinking Budget for Reasoning Models

**`--thinking-budget`** (reasoning effort level)
- Controls the computational effort for reasoning models
- Options: `low`, `medium`, `high`
- Guidelines:
  - `low`: Fast responses, basic reasoning (default for simple queries)
  - `medium`: Balanced speed and reasoning depth
  - `high`: Maximum reasoning effort, best for complex analytical questions
- **Supported Models**:
  - **Ollama**: `gpt-oss:20b`, `gpt-oss:120b`
  - **OpenAI**: `o3`, `o3-mini`, `o4-mini`, `o1` (o-series reasoning models)
- **Note**: Models without reasoning support will show a warning and proceed without reasoning parameters
- **Example**: `--thinking-budget high` for complex analytical questions

**📖 For detailed usage examples and implementation details, check out [Thinking Budget Documentation](THINKING_BUDGET_FEATURE.md)**

**💡 Quick Examples:**
```bash
# OpenAI o-series reasoning model
python apps/document_rag.py --query "What are the main techniques LEANN explores?" \
  --index-dir hnswbuild --backend hnsw \
  --llm openai --llm-model o3 --thinking-budget medium

# Ollama reasoning model
python apps/document_rag.py --query "What are the main techniques LEANN explores?" \
  --index-dir hnswbuild --backend hnsw \
  --llm ollama --llm-model gpt-oss:20b --thinking-budget high
```

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
   --embedding-model Qwen/Qwen3-Embedding-0.6B
   # To small model
   --embedding-model sentence-transformers/all-MiniLM-L6-v2
   ```

2. **Limit dataset size for testing**:
   ```bash
   --max-items 1000  # Process first 1k items only
   ```

3. **Use MLX on Apple Silicon** (optional optimization):
   ```bash
   --embedding-mode mlx --embedding-model mlx-community/Qwen3-Embedding-0.6B-8bit
   ```
    MLX might not be the best choice, as we tested and found that it only offers 1.3x acceleration compared to HF, so maybe using ollama is a better choice for embedding generation

4. **Use Ollama**
   ```bash
   --embedding-mode ollama --embedding-model nomic-embed-text
   ```
   To discover additional embedding models in ollama, check out https://ollama.com/search?c=embedding or read more about embedding models at https://ollama.com/blog/embedding-models, please do check the model size that works best for you
### If Search Quality is Poor

1. **Increase retrieval count**:
   ```bash
   --top-k 30  # Retrieve more candidates
   ```

2. **Upgrade embedding model**:
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
