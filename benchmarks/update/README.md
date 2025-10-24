# Update Benchmarks

This directory contains benchmarks for evaluating LEANN's update performance and RNG (Random Neighbor Graph) recomputation strategies.

## Overview

The benchmarks measure:
- **RNG Configuration Impact**: Latency comparison across different RNG caching and recomputation strategies
- **Update Strategy Comparison**: Sequential updates vs. offline batch rebuilding

## Benchmarks

### 1. HNSW RNG Recompute Benchmark

Evaluates latency under different RNG configurations (baseline, cache, forward RNG, etc.).

**Run:**
```bash
LEANN_HNSW_LOG_PATH=.leann/bench/hnsw_server.log \
LEANN_LOG_LEVEL=INFO \
uv run -m examples.bench_hnsw_rng_recompute \
  --runs 1 \
  --index-path .leann/bench/test.leann \
  --initial-files data/PrideandPrejudice.txt \
  --update-files data/huawei_pangu.md \
  --max-initial 300 \
  --max-updates 1 \
  --add-timeout 120
```

**Output:** `bench_results.csv` - Contains latency measurements (ms/passage) for each RNG scenario.

### 2. Sequential vs. Offline Update Benchmark

Compares sequential incremental updates against offline batch index rebuilding.

**Run:**
```bash
rm -f .leann/bench/offline_vs_update.*
uv run -m examples.bench_update_vs_offline_search \
  --index-path .leann/bench/offline_vs_update.leann \
  --max-initial 300 \
  --num-updates 1 > ./a.log
```

**Output:** `offline_vs_update.csv` - Contains total time (seconds) for both strategies.

### 3. Visualization

Generate plots from the benchmark results.

**Run:**
```bash
uv run scripts/plot_bench_results.py \
  --csv bench_results.csv \
  --csv-right offline_vs_update.csv \
  --out bench_latency_from_csv.png
```

**Options:**
- `--broken-y`: Use broken Y-axis for better visualization (default: true)
- `--csv`: Path to RNG benchmark results
- `--csv-right`: Path to update strategy comparison results
- `--out`: Output image path

**Output:**
- `bench_latency_from_csv.png` - Side-by-side comparison plots
- `bench_latency_from_csv.pdf` - PDF version for papers

## Results

The plots show:
- **Left subplot**: Latency (s/passage) across 4 RNG configurations with broken axis
- **Right subplot**: Total time (s) for Sequential vs. Offline update strategies

## Parameters

### Common Parameters
- `--max-initial`: Number of initial passages to index
- `--max-updates`: Number of passages to add as updates
- `--index-path`: Path to store the LEANN index

### Environment Variables
- `LEANN_HNSW_LOG_PATH`: Path for HNSW server logs
- `LEANN_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)