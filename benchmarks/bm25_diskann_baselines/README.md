BM25 vs DiskANN Baselines

```bash
aws s3 sync s3://powerrag-diskann-rpj-wiki-20250824-224037-194d640c/bm25_rpj_wiki/index_en_only/ benchmarks/data/indices/bm25_index/
aws s3 sync s3://powerrag-diskann-rpj-wiki-20250824-224037-194d640c/diskann_rpj_wiki/ benchmarks/data/indices/diskann_rpj_wiki/
```

- Dataset: `benchmarks/data/queries/nq_open.jsonl` (Natural Questions)
- Machine-specific; results measured locally with the current repo.

DiskANN (NQ queries, search-only)
- Command: `uv run benchmarks/bm25_diskann_baselines/run_diskann.py`
- Settings: `recompute_embeddings=False`, embeddings precomputed (excluded from timing), batching off, caching off (`cache_mechanism=2`, `num_nodes_to_cache=0`)
- Result: avg 0.019339 s/query, QPS 51.71 (p50 ~0.018936 s, p95 ~0.023573 s)

BM25
- Command: `uv run --script ./benchmarks/run_bm25.py`
- Settings: `k=10`, `k1=0.9`, `b=0.4`, queries=100
- Result: avg 0.026976 s/query, QPS 37.07 (p50 0.024729 s, p90 0.042158 s, p95 0.047099 s, p99 0.053520 s)

Notes
- DiskANN measures search-only latency on real NQ queries (embeddings computed beforehand and excluded from timing).
- Use `benchmarks/bm25_diskann_baselines/run_diskann.py` for DiskANN; `benchmarks/run_bm25.py` for BM25.
