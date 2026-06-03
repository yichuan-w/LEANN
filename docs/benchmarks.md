# Benchmarks

This page inventories the benchmark entry points currently available in this repository. It is a
documentation scaffold for the P0 benchmark roadmap item; it does not claim that every suite is
fully automated in CI or that historical numbers have been freshly reproduced on the current
branch.

## Quick Local Checks

These scripts are the best first stop when checking backend behavior on a developer machine.

| Area | Entry point | What it measures | Notes |
| --- | --- | --- | --- |
| Backend latency and size | `benchmarks/diskann_vs_hnsw_speed_comparison.py` | DiskANN vs HNSW search latency, build time, and index size | Accepts optional document and query counts. |
| Storage comparison | `benchmarks/compare_faiss_vs_leann.py` | LEANN storage savings against a FAISS baseline | Good for a quick sanity check of the storage story. |
| Embedding throughput | `benchmarks/benchmark_embeddings.py` | Embedding provider throughput | Depends on the configured local/provider model. |
| No-recompute baseline | `benchmarks/benchmark_no_recompute.py` | Search behavior when stored embeddings are available | Useful when isolating recompute overhead. |
| Retrieval backend comparison | `benchmarks/compare_retrieval_backends.py` | Recall, hit rate, latency, and storage for multiple prebuilt indexes under one manifest | Evaluation-only; does not build indexes. |

Example:

```bash
uv run python benchmarks/diskann_vs_hnsw_speed_comparison.py
uv run python benchmarks/compare_faiss_vs_leann.py
```

## Curated Reports

Curated benchmark reports should be committed only when they include the full command, data source,
revision, environment, and result context needed for review. Generated artifacts under
`benchmark-results/` should remain local unless they are intentionally promoted into a documented
report.

## P0 Benchmark Acceptance Checklist

The P0 benchmark roadmap item is not complete until at least one reviewable benchmark pack includes
real retrieval results, not just synthetic helper metrics. A benchmark pack should contain:

- A retrieval evaluation artifact from `benchmarks/run_evaluation.py` with JSON and Markdown
  outputs.
- A query-log summary from `benchmarks/summarize_query_log.py` when replay logs are used for
  comparison.
- Dataset name, corpus size, query count, ground-truth source, benchmark data source, and exact
  commands. For repository-provided retrieval data, record the Hugging Face dataset
  `LEANN-RAG/leann-rag-evaluation-data` and the downloaded revision or download date.
- Backend and index settings, including backend name, compact/recompute mode, `top_k`,
  `complexity`, BM25/vector weighting, and passage ID scheme.
- Embedding model, embedding mode, provider options that affect embeddings, and whether embeddings
  were recomputed during search.
- Hardware, operating system, Python version, LEANN commit SHA, and whether GPU acceleration was
  used.
- Recall@k or hit rate@k where ground truth exists, latency distribution, and local index storage
  bytes.

Generated JSON/Markdown under `benchmark-results/` stays local or is attached to CI/PR artifacts.
Commit only curated reports that include the full context above and call out limitations clearly.

A minimal first P0 pack should compare the main supported retrieval path against at least one
baseline on a real dataset with ground truth. A broader pack should cover HNSW and IVF, include
BM25/hybrid settings when relevant, and separate retrieval latency from any optional LLM generation.
Use the single-index retrieval and query-log artifact tools for the first reports, but do not treat
manual one-off runs as a complete backend comparison; a reviewable cross-backend pack should keep
dataset, query slice, ground truth, `top_k`, complexity, embedding model, and output schema
identical across rows.

## Retrieval Evaluation

`benchmarks/run_evaluation.py` is the main retrieval evaluation driver. It can download evaluation
data when the local `benchmarks/data/` tree is missing required queries, ground truth, or at least
one real `.index` artifact under `indices/`, and it can reuse a previously downloaded index path
for larger runs. The repository-provided prebuilt indexes are large, so check local disk capacity
before relying on the automatic download path; use `--queries-file`, `--ground-truth`, and a
prebuilt or locally built index path when running a smaller custom pack. Evaluation is
retrieval-only by default; use `--run-llm` only when generation should be measured separately from
retrieval latency.

```bash
uv run benchmarks/run_evaluation.py
uv run benchmarks/run_evaluation.py benchmarks/data/indices/rpj_wiki/rpj_wiki \
  --dataset rpj_wiki \
  --data-source LEANN-RAG/leann-rag-evaluation-data \
  --data-revision 2026-06-02-download \
  --num-queries 2000 \
  --top-k 3 \
  --complexity 120 \
  --format markdown \
  --json-output benchmark-results/retrieval-rpj-wiki.json \
  --markdown-output benchmark-results/retrieval-rpj-wiki.md
```

Use this path for recall-oriented comparisons. The JSON artifact records the dataset, data source
and revision, index path, backend, embedding model/mode, query count, `top_k`, complexity,
hardware/platform, local LEANN commit, branch, dirty-worktree flag, query-file SHA256 and
ground-truth-file SHA256 when file paths are available, shell-quoted script command arguments,
storage bytes, whether embeddings were recomputed, per-query result IDs, per-query golden IDs,
missing result-ID counts, and duplicate result/golden text counters. Recall is text-overlap based
for compatibility with indexes that use different passage ID schemes; the duplicate-text counters
are included so reviewers can identify queries where text-set matching may collapse distinct
passages. For custom datasets, pass
`--queries-file`, `--ground-truth`, `--dataset`, `--data-source`, and `--data-revision` explicitly
so the artifact does not depend on path-name inference.

## Retrieval Backend Comparisons

`benchmarks/compare_retrieval_backends.py` runs the single-index retrieval evaluator across
multiple prebuilt index paths from one JSON manifest. It is evaluation-only: build or download the
indexes first, then use the manifest to prove that every row uses the same dataset, query slice,
ground truth, `top_k`, complexity, and batch size.

Example manifest:

```json
{
  "dataset": "rpj_wiki",
  "data_source": "LEANN-RAG/leann-rag-evaluation-data",
  "data_revision": "2026-06-02-download",
  "queries_file": "benchmarks/data/queries/nq_open.jsonl",
  "ground_truth_file": "benchmarks/data/ground_truth/rpj_wiki/flat_results_nq_k3.json",
  "num_queries": 2000,
  "top_k": 3,
  "complexity": 120,
  "batch_size": 0,
  "runs": [
    {
      "name": "hnsw",
      "backend": "hnsw",
      "index_path": "benchmarks/data/indices/rpj_wiki/rpj_wiki_hnsw"
    },
    {
      "name": "ivf",
      "backend": "ivf",
      "index_path": "benchmarks/data/indices/rpj_wiki/rpj_wiki_ivf"
    }
  ]
}
```

Run it with:

```bash
uv run python benchmarks/compare_retrieval_backends.py benchmark-results/retrieval-comparison.json \
  --format markdown \
  --json-output benchmark-results/retrieval-comparison-summary.json \
  --markdown-output benchmark-results/retrieval-comparison-summary.md
```

The combined artifact includes one normalized row per run with backend, passage ID scheme,
embedding model/mode, recall@k, hit rate@k, evaluated query count, missing ground-truth and golden
passage counts, missing result-ID counts, duplicate result/golden text counters, latency
mean/median/p95, storage bytes, local LEANN commit, dirty-worktree flag, and SHA256 hashes for the
query and ground-truth files. CLI-generated comparison artifacts also record the shell-quoted script
command arguments. Keep the full per-index evaluation summaries in the JSON artifact for reviewer
drill-down.

Reviewable benchmark artifacts use one shared timing-statistics helper for mean, median, p95, min,
and max. p95 is the lower nearest-rank observed sample from the sorted timings, not an interpolated
value, so retrieval and query-log reports can be compared without reinterpreting the percentile
rule. The standalone BM25 and DiskANN baseline scripts use the same observed-sample percentile
helpers for their latency report fields.

## Query Log Summaries

`leann search --query-log <path>` and `leann ask --query-log <path>` write replay-oriented JSONL
records. New records include `duration_ms`, result IDs, backend settings, and search mode. Use
`benchmarks/summarize_query_log.py` to turn those logs into reviewable benchmark summaries:

```bash
uv run python benchmarks/summarize_query_log.py queries.jsonl \
  --ground-truth ground_truth.json \
  --index-path .leann/indexes/my-index/documents.leann \
  --data-source my-dataset \
  --data-revision 2026-06-02-download \
  --k 10 \
  --format markdown \
  --json-output benchmark-results/my-index-summary.json \
  --markdown-output benchmark-results/my-index-summary.md
```

Ground truth can be either a JSON object mapping query text to relevant passage IDs, or JSONL rows
with `query` plus `relevant_ids`, `gold_ids`, `expected_ids`, or `ids`. The summarizer reports
recall@k, hit rate@k, result counts, latency mean/median/p95/min/max when present,
backend/search-mode counts, records with missing result IDs, total missing result IDs,
missing-latency/backend/search-mode counters, local index storage bytes when index paths are
available, data source/revision, shell-quoted script command arguments, and local environment
metadata.
`--format` controls stdout; `--json-output` and `--markdown-output` optionally write reviewable
artifacts in the same run. Ground-truth query text must exactly match the `query` value in the
query log.

Generated benchmark artifacts record the Python script path and arguments with POSIX-style shell
quoting. Curated reports should still include the full outer shell command, such as `uv run ...`,
when the wrapper or dependency mode affects reproducibility.

Query logs may contain sensitive query text and, when enabled, query embeddings. Summary artifacts
omit embeddings and result text, but they still include query counts, backend/search-mode details,
latency summaries, recall statistics, missing-ID/missing-metadata counters, and storage file paths.

`benchmark-results/` is the default local scratch directory for generated benchmark summaries and
is gitignored. Commit only curated benchmark reports with enough context to satisfy the reporting
template below, or attach generated artifacts to a pull request or CI job.

## Benchmark Suites

| Suite | Location | Status | Purpose |
| --- | --- | --- | --- |
| BM25 and DiskANN baselines | `benchmarks/bm25_diskann_baselines/` | Data-dependent | Measures Natural Questions search-only latency for BM25 and DiskANN using externally synced artifacts; scripts can write JSON reports with query hashes, timing scope, recursive storage bytes, settings, command arguments, and environment provenance. |
| ContextBench | `benchmarks/contextbench/` | Manual/agent workflow | Runs repository-preparation and trace-driven code-assistant evaluations. |
| Enron emails | `benchmarks/enron_emails/` | Data-dependent | Evaluates retrieval and generation on the Enron email corpus. |
| FinanceBench | `benchmarks/financebench/` | Data-dependent | Evaluates retrieval-augmented generation on financial question answering. |
| LAION multimodal | `benchmarks/laion/` | Data-dependent and multimodal | Evaluates image retrieval and multimodal generation with CLIP/Qwen-style models. |
| Update benchmarks | `benchmarks/update/` | Reproducible with local data, hardware-sensitive | Measures update latency and sequential-vs-offline update strategies. |

Historical or sample result files in benchmark subdirectories should be treated as reference runs
with their documented hardware and settings, not as automatically refreshed results for the current
checkout.

## Reporting Template

When adding benchmark results, include:

- LEANN commit SHA and backend.
- Dataset name, corpus size, and query count.
- Embedding model and embedding mode.
- Search settings such as `top_k`, `complexity`, `beam_width`, `prune_ratio`, and BM25/vector
  weighting when applicable.
- Hardware, operating system, and whether GPU acceleration was used.
- Build time, index size, search latency distribution, and recall@k where ground truth exists.

This keeps benchmark updates self-contained and reviewable.
