BM25 vs DiskANN Baselines

```bash
aws s3 sync s3://powerrag-diskann-rpj-wiki-20250824-224037-194d640c/bm25_rpj_wiki/index_en_only/ benchmarks/data/indices/bm25_index/
aws s3 sync s3://powerrag-diskann-rpj-wiki-20250824-224037-194d640c/diskann_rpj_wiki/ benchmarks/data/indices/diskann_rpj_wiki/
```

- Dataset: `benchmarks/data/queries/nq_open.jsonl` (Natural Questions)
- Machine-specific; the sample results below are historical local output and should be regenerated
  for current hardware, dependencies, and code.
- Current scripts compute p50 as the median sample summary and p90/p95/p99 as lower nearest-rank
  observed samples, matching the shared LEANN benchmark timing helpers.

DiskANN (NQ queries, search-only)
- Command from the repository root: `uv run --script benchmarks/bm25_diskann_baselines/run_diskann.py --report benchmark-results/diskann-baseline.json`
- Settings: `recompute_embeddings=False`, embeddings precomputed (excluded from timing), batching off, caching off (`cache_mechanism=2`, `num_nodes_to_cache=0`)
- Result: avg 0.011093 s/query, QPS 90.15 (p50 0.010731 s, p95 0.015000 s)

BM25
- Command from the repository root: `uv run --script benchmarks/bm25_diskann_baselines/run_bm25.py --report benchmark-results/bm25-baseline.json`
- Settings: `k=10`, `k1=0.9`, `b=0.4`, queries=100
- Result: avg 0.028589 s/query, QPS 34.97 (p50 0.026060 s, p90 0.043695 s, p95 0.053260 s, p99 0.055257 s)

Notes
- DiskANN measures search-only latency on real NQ queries (embeddings computed beforehand and excluded from timing).
- Use `benchmarks/bm25_diskann_baselines/run_diskann.py` for DiskANN; `benchmarks/bm25_diskann_baselines/run_bm25.py` for BM25.
- Pass `--data-source` and `--data-revision` when generating reports for PR review. JSON reports
  include schema version, benchmark name, query-file SHA256, shell-quoted script command arguments,
  baseline settings, timing scope, recursive index-directory storage bytes/file count, latency
  summaries, and local environment metadata. BM25 reports use `timing_scope=search_with_doc_fetch`
  when `--fetch-docs` is enabled; otherwise both BM25 and DiskANN report
  `timing_scope=search_only`.
