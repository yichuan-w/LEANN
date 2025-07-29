# Parameter Consistency Guide

This document ensures that the new unified interface maintains exact parameter compatibility with the original examples.

## Parameter Mapping

### Common Parameters (All Examples)

| Parameter | Default Value | Notes |
|-----------|--------------|-------|
| `backend_name` | `"hnsw"` | All examples use HNSW backend |
| `graph_degree` | `32` | Consistent across all |
| `complexity` | `64` | Consistent across all |
| `is_compact` | `True` | NOT `compact_index` |
| `is_recompute` | `True` | NOT `use_recomputed_embeddings` |
| `num_threads` | `1` | Force single-threaded mode |
| `chunk_size` | `256` | Consistent across all |

### Example-Specific Defaults

#### document_rag.py (replaces main_cli_example.py)
- `index_dir`: `"./test_doc_files"` (matches original)
- `chunk_overlap`: `128` (matches original)
- `embedding_model`: `"facebook/contriever"`
- `embedding_mode`: `"sentence-transformers"`
- No max limit by default

#### email_rag.py (replaces mail_reader_leann.py)
- `index_dir`: `"./mail_index"` (matches original)
- `max_items`: `1000` (was `max_emails`)
- `chunk_overlap`: `25` (matches original)
- `embedding_model`: `"facebook/contriever"`
- NO `embedding_mode` parameter in LeannBuilder (original doesn't have it)

#### browser_rag.py (replaces google_history_reader_leann.py)
- `index_dir`: `"./google_history_index"` (matches original)
- `max_items`: `1000` (was `max_entries`)
- `chunk_overlap`: `25` (primary value in original)
- `embedding_model`: `"facebook/contriever"`
- `embedding_mode`: `"sentence-transformers"`

#### wechat_rag.py (replaces wechat_history_reader_leann.py)
- `index_dir`: `"./wechat_history_magic_test_11Debug_new"` (matches original)
- `max_items`: `50` (was `max_entries`, much lower default)
- `chunk_overlap`: `25` (matches original)
- `embedding_model`: `"Qwen/Qwen3-Embedding-0.6B"` (special model for Chinese)
- NO `embedding_mode` parameter in LeannBuilder (original doesn't have it)

## Implementation Notes

1. **Parameter Names**: The original files use `is_compact` and `is_recompute`, not the newer names.

2. **Chunk Overlap**: Most examples use `25` except for documents which uses `128`.

3. **Embedding Mode**: Only `google_history_reader_leann.py` and `main_cli_example.py` have this parameter.

4. **Max Items**: Each example has different defaults:
   - Email/Browser: 1000
   - WeChat: 50
   - Documents: unlimited

5. **Special Cases**:
   - WeChat uses a specific Chinese embedding model
   - Email reader includes HTML processing option 