# Update Benchmarks

This directory hosts two benchmark suites that exercise LEANN’s HNSW “update +
search” pipeline under different assumptions:

1. **RNG recompute latency** – measure how random-neighbour pruning and cache
   settings influence incremental `add()` latency when embeddings are fetched
   over the ZMQ embedding server.
2. **Update strategy comparison** – compare a fully sequential update pipeline
   against an offline approach that keeps the graph static and fuses results.

Both suites build a non-compact, `is_recompute=True` index so that new
embeddings are pulled from the embedding server. Benchmark outputs are written
under `.leann/bench/` by default and appended to CSV files for later plotting.

## Benchmarks

### 1. HNSW RNG Recompute Benchmark

`bench_hnsw_rng_recompute.py` evaluates incremental update latency under four
random-neighbour (RNG) configurations. Each scenario uses the same dataset but
changes the forward / reverse RNG pruning flags and whether the embedding cache
is enabled:

| Scenario name                      | Forward RNG | Reverse RNG | ZMQ embedding cache |
| ---------------------------------- | ----------- | ----------- | ------------------- |
| `baseline`                         | Enabled     | Enabled     | Enabled             |
| `no_cache_baseline`                | Enabled     | Enabled     | **Disabled**        |
| `disable_forward_rng`              | **Disabled**| Enabled     | Enabled             |
| `disable_forward_and_reverse_rng`  | **Disabled**| **Disabled**| Enabled             |

For each scenario the script:
1. (Re)builds a `is_recompute=True` index and writes it to `.leann/bench/`.
2. Starts `leann_backend_hnsw.hnsw_embedding_server` for remote embeddings.
3. Appends the requested updates using the scenario’s RNG flags.
4. Records total time, latency per passage, ZMQ fetch counts, and stage-level
   timings before appending a row to the CSV output.

**Run:**
```bash
LEANN_HNSW_LOG_PATH=.leann/bench/hnsw_server.log \
LEANN_LOG_LEVEL=INFO \
uv run -m benchmarks.update.bench_hnsw_rng_recompute \
  --runs 1 \
  --index-path .leann/bench/test.leann \
  --initial-files data/PrideandPrejudice.txt \
  --update-files data/huawei_pangu.md \
  --max-initial 300 \
  --max-updates 1 \
  --add-timeout 120
```

**Output:**
- `benchmarks/update/bench_results.csv` – per-scenario timing statistics
  (including ms/passage) for each run.
- `.leann/bench/hnsw_server.log` – detailed ZMQ/server logs (path controlled by
  `LEANN_HNSW_LOG_PATH`).
  _The reference CSVs checked into this branch were generated on a workstation with an NVIDIA RTX 4090 GPU; throughput numbers will differ on other hardware._

### 2. Sequential vs. Offline Update Benchmark

`bench_update_vs_offline_search.py` compares two end-to-end strategies on the
same dataset:

- **Scenario A – Sequential Update**
  - Start an embedding server.
  - Sequentially call `index.add()`; each call fetches embeddings via ZMQ and
    mutates the HNSW graph.
  - After all inserts, run a search on the updated graph.
  - Metrics recorded: update time (`add_total_s`), post-update search time
    (`search_time_s`), combined total (`total_time_s`), and per-passage
    latency.

- **Scenario B – Offline Embedding + Concurrent Search**
  - Stop Scenario A’s server and start a fresh embedding server.
  - Spawn two threads: one generates embeddings for the new passages offline
    (graph unchanged); the other computes the query embedding and searches the
    existing graph.
  - Merge offline similarities with the graph search results to emulate late
    fusion, then report the merged top‑k preview.
  - Metrics recorded: embedding time (`emb_time_s`), search time
    (`search_time_s`), concurrent makespan (`makespan_s`), and scenario total.

**Run (both scenarios):**
```bash
uv run -m benchmarks.update.bench_update_vs_offline_search \
  --index-path .leann/bench/offline_vs_update.leann \
  --max-initial 300 \
  --num-updates 1
```

You can pass `--only A` or `--only B` to run a single scenario. The script will
print timing summaries to stdout and append the results to CSV.

**Output:**
- `benchmarks/update/offline_vs_update.csv` – per-scenario timing statistics for
  Scenario A and B.
- Console output includes Scenario B’s merged top‑k preview for quick sanity
  checks.
  _The sample results committed here come from runs on an RTX 4090-equipped machine; expect variations if you benchmark on different GPUs._

### 3. Visualisation

`plot_bench_results.py` combines the RNG benchmark and the update strategy
benchmark into a single two-panel plot.

**Run:**
```bash
uv run -m benchmarks.update.plot_bench_results \
  --csv benchmarks/update/bench_results.csv \
  --csv-right benchmarks/update/offline_vs_update.csv \
  --out benchmarks/update/bench_latency_from_csv.png
```

**Options:**
- `--broken-y` – Enable a broken Y-axis (default: true when appropriate).
- `--csv` – RNG benchmark results CSV (left panel).
- `--csv-right` – Update strategy results CSV (right panel).
- `--out` – Output image path (PNG/PDF supported).

**Output:**
- `benchmarks/update/bench_latency_from_csv.png` – visual comparison of the two
  suites.
- `benchmarks/update/bench_latency_from_csv.pdf` – PDF version, suitable for
  slides/papers.

## Parameters & Environment

### Common CLI Flags
- `--max-initial` – Number of initial passages used to seed the index.
- `--max-updates` / `--num-updates` – Number of passages to treat as updates.
- `--index-path` – Base path (without extension) where the LEANN index is stored.
- `--runs` – Number of repetitions (RNG benchmark only).

### Environment Variables
- `LEANN_HNSW_LOG_PATH` – File to receive embedding-server logs (optional).
- `LEANN_LOG_LEVEL` – Logging verbosity (DEBUG/INFO/WARNING/ERROR).
- `CUDA_VISIBLE_DEVICES` – Set to empty string if you want to force CPU
  execution of the embedding model.

With these scripts you can easily replicate LEANN’s update benchmarks, compare
multiple RNG strategies, and evaluate whether sequential updates or offline
fusion better match your latency/accuracy trade-offs.
