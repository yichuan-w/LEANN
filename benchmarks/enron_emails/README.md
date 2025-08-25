# Enron Emails Benchmark

A retrieval-only benchmark for evaluating LEANN search on the Enron email corpus. It mirrors the structure and CLI of the existing FinanceBench and LAION benches, using stage-based evaluation focused on Recall@3.

- Dataset: Enron email CSV (e.g., Kaggle wcukierski/enron-email-dataset) for passages
- Queries: corbt/enron_emails_sample_questions (filtered for realistic questions)
- Metric: Recall@3 vs FAISS Flat baseline

## Layout

benchmarks/enron_emails/
- setup_enron_emails.py: Prepare passages, build LEANN index, build FAISS baseline
- evaluate_enron_emails.py: Evaluate retrieval recall (Stage 2)
- data/: Generated passages, queries, embeddings-related files
- baseline/: FAISS Flat baseline files

## Quickstart

1) Prepare the data and index

cd benchmarks/enron_emails
python setup_enron_emails.py --data-dir data

Notes:
- If `--emails-csv` is omitted, the script attempts to download from Kaggle dataset `wcukierski/enron-email-dataset` using Kaggle API (requires `KAGGLE_USERNAME` and `KAGGLE_KEY`).
  Alternatively, pass a local path to `--emails-csv`.

Notes:
- The script parses emails, chunks header/body into passages, builds a compact LEANN index, and then builds a FAISS Flat baseline from the same passages and embedding model.
- Optionally, it will also create evaluation queries from HuggingFace dataset `corbt/enron_emails_sample_questions`.

2) Run recall evaluation (Stage 2)

python evaluate_enron_emails.py --index data/enron_index_hnsw.leann --stage 2

3) Complexity sweep (Stage 3)

python evaluate_enron_emails.py --index data/enron_index_hnsw.leann --stage 3 --target-recall 0.90 --max-queries 200

Stage 3 uses binary search over complexity to find the minimal value achieving the target Recall@3 (assumes recall is non-decreasing with complexity). The search expands the upper bound as needed and snaps complexity to multiples of 8.

4) Index comparison (Stage 4)

python evaluate_enron_emails.py --index data/enron_index_hnsw.leann --stage 4 --max-queries 100 --output results.json

Notes:
- Minimal CLI: you can run from repo root with only `--index`, defaults match financebench/laion patterns:
  - `--stage` defaults to `all` (runs 2, 3, 4)
  - `--baseline-dir` defaults to `baseline`
  - `--queries` defaults to `data/evaluation_queries.jsonl` (or falls back to the index directory)
- Fail-fast behavior: no silent fallbacks. If compact index cannot run with recompute, it errors out.

4) Index comparison (Stage 4)

python evaluate_enron_emails.py --index data/enron_index_hnsw.leann --stage 4 --max-queries 100 --output results.json

Optional flags:
- --queries data/evaluation_queries.jsonl (custom queries file)
- --baseline-dir baseline (where FAISS baseline lives)
- --complexity 64 (LEANN complexity parameter)

## Files Produced
- data/enron_passages_preview.jsonl: Small preview of passages used (for inspection)
- data/enron_index_hnsw.leann.*: LEANN index files
- baseline/faiss_flat.index + baseline/metadata.pkl: FAISS baseline with passage IDs
- data/evaluation_queries.jsonl: Query file (id + query; includes GT IDs for reference)

## Notes
- We only evaluate retrieval Recall@3 (no generation). This matches the other benches’ style and stage flow.
- The emails CSV must contain a column named "message" (raw RFC822 email) and a column named "file" for source identifier. Message-ID headers are parsed as canonical message IDs when present.

## Stages Summary

- Stage 2 (Recall@3):
  - Compares LEANN vs FAISS Flat baseline on Recall@3.
  - Compact index runs with `recompute_embeddings=True`.

- Stage 3 (Binary Search for Complexity):
  - Builds a non-compact index (`<index>_noncompact.leann`) and runs binary search with `recompute_embeddings=False` to find the minimal complexity achieving target Recall@3 (default 90%).

- Stage 4 (Index Comparison):
  - Reports .index-only sizes for compact vs non-compact.
  - Measures timings on 100 queries by default: non-compact (no recompute) vs compact (with recompute).
  - Fails fast if compact recompute cannot run.
  - If `--complexity` is not provided, the script tries to use the best complexity from Stage 3:
    - First from the current run (when running `--stage all`), otherwise
    - From `enron_stage3_results.json` saved next to the index during the last Stage 3 run.
    - If neither exists, Stage 4 will error and ask you to run Stage 3 or pass `--complexity`.

## Example Results

These are sample results obtained on a subset of Enron data using all-mpnet-base-v2.

- Stage 3 (Binary Search):
  - Minimal complexity achieving 90% Recall@3: 88
  - Sampled points:
    - C=8 → 59.9% Recall@3
    - C=72 → 89.4% Recall@3
    - C=88 → 90.2% Recall@3
    - C=96 → 90.7% Recall@3
    - C=112 → 91.1% Recall@3
    - C=136 → 91.3% Recall@3
    - C=256 → 92.0% Recall@3

- Stage 4 (Index Sizes, .index only):
  - Compact: ~2.17 MB
  - Non-compact: ~82.03 MB
  - Storage saving by compact: ~97.35%

- Stage 4 (Timing, 100 queries, complexity=88):
  - Non-compact (no recompute): ~0.0074 s avg per query
  - Compact (with recompute): ~1.947 s avg per query
  - Speed ratio (non-compact/compact): ~0.0038x

Full JSON output for Stage 4 is saved by the script (see `--output`), e.g.:
`benchmarks/enron_emails/results_enron_stage4.json`.
