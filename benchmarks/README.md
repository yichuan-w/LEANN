# LEANN Benchmarks

This directory contains benchmark scripts and data-dependent evaluation suites for LEANN. The
canonical benchmark inventory now lives in [docs/benchmarks.md](../docs/benchmarks.md).

## Quick Starts

Run a small backend comparison:

```bash
uv run python benchmarks/diskann_vs_hnsw_speed_comparison.py
```

Compare LEANN storage usage against a FAISS baseline:

```bash
uv run python benchmarks/compare_faiss_vs_leann.py
```

Run the main retrieval evaluation driver:

```bash
uv run benchmarks/run_evaluation.py
```

This command treats an incomplete `benchmarks/data/` tree, including a README-only checkout or an
`indices/` tree without any `.index` artifact, as missing benchmark data and downloads the public
data pack. The repository-provided prebuilt indexes are large; check local disk capacity before
relying on the automatic download path.

For larger retrieval runs after data has been downloaded:

```bash
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

Retrieval evaluation is search-only by default. Use `--run-llm` only when generation should be
included as opt-in metadata outside retrieval latency. Retrieval artifacts include per-query result
IDs, golden IDs, and duplicate-text counters so reviewers can audit text-overlap recall when passage
ID schemes differ across indexes. CLI-generated artifacts also record SHA256 hashes for the query
and ground-truth files plus shell-quoted script command arguments.

Retrieval and query-log artifacts share the same timing-statistics helper. p95 is the lower
nearest-rank observed sample from sorted timings, so it is deterministic and comparable across
reviewable benchmark reports. The standalone BM25 and DiskANN baseline scripts use the same
observed-sample percentile helpers for their latency report fields.

Run one retrieval manifest across multiple prebuilt backend indexes:

```bash
uv run python benchmarks/compare_retrieval_backends.py benchmark-results/retrieval-comparison.json \
  --format markdown \
  --json-output benchmark-results/retrieval-comparison-summary.json \
  --markdown-output benchmark-results/retrieval-comparison-summary.md
```

The comparison manifest keeps dataset, query file, ground truth, `top_k`, complexity, and batch
size identical across runs. The combined JSON/Markdown artifacts report per-run backend,
passage-ID scheme, embedding model/mode, recall@k, hit rate@k, latency, storage bytes, query file
hash, ground-truth file hash, shell-quoted script command arguments, missing result-ID counts, and
duplicate result/golden text counters.
See [docs/benchmarks.md](../docs/benchmarks.md) for the manifest fields and a complete example.

Summarize query logs with optional ground truth and storage accounting:

```bash
uv run python benchmarks/summarize_query_log.py queries.jsonl \
  --ground-truth ground_truth.json \
  --index-path .leann/indexes/my-index/documents.leann \
  --data-source my-dataset \
  --data-revision 2026-06-02-download \
  --format markdown \
  --json-output benchmark-results/my-index-summary.json \
  --markdown-output benchmark-results/my-index-summary.md
```

`--format` controls stdout. Use `--json-output` and `--markdown-output` to write durable review
artifacts in the same run. Query-log summaries include latency mean/median/p95/min/max when
available plus counters for records with missing result IDs, total missing result IDs, missing
latency, missing search mode, missing backend metadata, and shell-quoted script command arguments.

The `benchmark-results/` directory is ignored by git and is the default scratch location for local
summary artifacts. Commit only curated reports with full dataset, hardware, and command context.

## Suites

- `bm25_diskann_baselines/`: Natural Questions BM25 and DiskANN search-only baselines that require
  externally synced artifacts, report latency with the shared observed-sample percentile helpers,
  and can write JSON reports with query hashes, timing scope, recursive storage bytes, settings,
  command arguments, and environment provenance.
- `contextbench/`: trace-driven code-assistant benchmark tooling.
- `enron_emails/`: retrieval and generation evaluation on the Enron email corpus.
- `financebench/`: financial document question-answering benchmark.
- `laion/`: multimodal image/text retrieval benchmark.
- `update/`: update-latency and sequential-vs-offline update strategy benchmarks.

Benchmark result files committed under subdirectories are reference outputs from their documented
settings and hardware. Do not treat them as automatically refreshed results for the current branch
unless the relevant command has been rerun.
