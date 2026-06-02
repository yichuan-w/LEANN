# LlamaIndex Integration

LEANN exposes LlamaIndex retrievers in `leann.integrations.llamaindex`:

- `LeannRetriever`: pure dense-vector retrieval from an existing LEANN index.
- `LeannHybridRetriever`: dense-vector plus BM25 keyword retrieval using LEANN's hybrid search.

Both retrievers wrap `LeannSearcher.search()` and convert each LEANN `SearchResult` into a
LlamaIndex `TextNode` wrapped in `NodeWithScore`, preserving passage ID, text, score, and dictionary
metadata.

## Usage

```python
from leann.integrations.llamaindex import LeannHybridRetriever, LeannRetriever

vector_retriever = LeannRetriever(
    index_path=".leann/indexes/my-code/documents.leann",
    top_k=8,
)

hybrid_retriever = LeannHybridRetriever(
    index_path=".leann/indexes/my-code/documents.leann",
    top_k=8,
    bm25_weight=0.3,
)

nodes = hybrid_retriever.retrieve("where is the incremental update logic?")
for node in nodes:
    print(node.node.id_, node.score, node.node.metadata)
```

`bm25_weight` is the keyword-side weight in the closed interval `[0.0, 1.0]`:

- `0.0`: pure dense-vector retrieval.
- `0.3`: 70 percent dense-vector weight, 30 percent BM25 keyword weight.
- `1.0`: pure BM25 keyword retrieval.

Internally, `LeannHybridRetriever` calls LEANN with `vector_weight = 1.0 - bm25_weight`. Invalid
weights raise `ValueError` instead of being silently clamped.

Additional LEANN query-time options can be passed with `search_kwargs`:

```python
filtered = LeannHybridRetriever(
    index_path=".leann/indexes/my-code/documents.leann",
    bm25_weight=0.4,
    search_kwargs={"metadata_filters": {"file_path": {"contains": "tests/"}}},
)
```

## Notes

- Build or refresh the LEANN index before constructing the retriever.
- Dictionary metadata from LEANN passages is preserved on the LlamaIndex node.
- Non-dictionary metadata is normalized to `{}` because LlamaIndex node metadata expects a mapping.
- If a pure-vector retriever reads an index whose distance metric is lower-is-better L2 distance,
  the adapter converts that raw distance to `1 / (1 + distance)` for the LlamaIndex score and stores
  the original value as `node.metadata["leann_raw_score"]`.
- LlamaIndex `QueryBundle` objects must contain plain text queries. Custom embedding strings and
  precomputed query embeddings are rejected because LEANN owns query embedding computation.
