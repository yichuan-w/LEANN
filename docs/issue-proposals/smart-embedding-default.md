# Smart default embedding model based on platform and corpus size

## Summary

Propose platform- and corpus-aware default embedding model selection for `leann build` when `--embedding-model` is not explicitly specified. This would improve out-of-the-box experience for different deployment scenarios (macOS CPU, NVIDIA GPU, etc.) without changing behavior when users pass an explicit model.

## Motivation

- **Current default**: `facebook/contriever` (~420MB, 768 dim) — heavy for CPU-only builds on large corpora
- **macOS users** often hit slow builds on 20K+ chunks; lighter models like `all-MiniLM-L6-v2` (~90MB) are much faster
- **NVIDIA GPU users** can leverage stronger models; smaller corpora benefit from quality (e.g. Qwen3-Embedding-0.6B), larger ones from balanced models (e.g. bge-base-en-v1.5)

## Proposed logic

| Platform | Chunk count | Default model |
|----------|-------------|---------------|
| **macOS** | ≥ 20,000 | `sentence-transformers/all-MiniLM-L6-v2` |
| **macOS** | < 20,000 | `intfloat/e5-small-v2` |
| **NVIDIA GPU** | < 5,000 | `Qwen/Qwen3-Embedding-0.6B` |
| **NVIDIA GPU** | ≥ 5,000 | `BAAI/bge-base-en-v1.5` |
| **Other** | any | `facebook/contriever` (unchanged) |

## Implementation notes

1. **Platform detection**: `torch.cuda.is_available()` for NVIDIA; `sys.platform == "darwin"` for macOS
2. **Chunk count**: Known only after loading/chunking; may need to either:
   - Do a lightweight pre-scan (e.g. file count × rough chunks per file), or
   - Defer default choice until after first chunking pass (and cache for incremental)
3. **Explicit override**: If user passes `--embedding-model`, always use it; this logic applies only when the flag is omitted

## Model references

- `sentence-transformers/all-MiniLM-L6-v2`: ~90MB, 384 dim, fast on CPU
- `intfloat/e5-small-v2`: ~90MB, 384 dim
- `Qwen/Qwen3-Embedding-0.6B`: 0.6B params, 1024 dim, strong retrieval
- `BAAI/bge-base-en-v1.5`: ~110M params, 768 dim, good MTEB scores

## Open questions

- Should we add a `--embedding-model auto` to explicitly opt into this logic?
- Pre-scan vs post-chunk decision: trade-off between accuracy and implementation complexity
