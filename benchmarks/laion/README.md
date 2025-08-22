# LAION Multimodal Benchmark

A multimodal benchmark for evaluating image retrieval performance using LEANN with CLIP embeddings on LAION dataset subset.

## Overview

This benchmark evaluates:
- **Image retrieval timing** using caption-based queries
- **Recall@K performance** for image search
- **Complexity analysis** across different search parameters
- **Index size and storage efficiency**

## Dataset Configuration

- **Dataset**: LAION-400M subset (10,000 images)
- **Embeddings**: Pre-computed CLIP ViT-B/32 (512 dimensions)
- **Queries**: 200 random captions from the dataset
- **Ground Truth**: Self-recall (query caption → original image)

## Quick Start

### 1. Setup the benchmark

```bash
cd benchmarks/laion
python setup_laion.py --num-samples 10000 --num-queries 200
```

This will:
- Create dummy LAION data (10K samples)
- Generate CLIP embeddings (512-dim)
- Build LEANN index with HNSW backend
- Create 200 evaluation queries

### 2. Run evaluation

```bash
# Run all evaluation stages
python evaluate_laion.py --index data/laion_index.leann

# Run specific stages
python evaluate_laion.py --index data/laion_index.leann --stage timing
python evaluate_laion.py --index data/laion_index.leann --stage recall
python evaluate_laion.py --index data/laion_index.leann --stage complexity
```

### 3. Save results

```bash
python evaluate_laion.py --index data/laion_index.leann --output results.json
```

## Configuration Options

### Setup Options
```bash
python setup_laion.py \
  --num-samples 10000 \
  --num-queries 200 \
  --index-path data/laion_index.leann \
  --backend hnsw
```

### Evaluation Options
```bash
python evaluate_laion.py \
  --index data/laion_index.leann \
  --queries data/evaluation_queries.jsonl \
  --complexity 64 \
  --top-k 3 \
  --num-samples 100 \
  --stage all
```

## Evaluation Stages

### Stage 1: Index Analysis
- Analyzes index file sizes and metadata
- Reports storage efficiency

### Stage 2: Search Timing
- Measures average search latency
- Tests with configurable complexity and top-k
- Reports searches per second

### Stage 3: Recall Evaluation
- Evaluates Recall@K using ground truth
- Self-recall: query caption should retrieve original image

### Stage 4: Complexity Analysis
- Tests performance across different complexity levels [16, 32, 64, 128]
- Analyzes speed vs. accuracy tradeoffs

## Output Metrics

### Timing Metrics
- Average/median/min/max search time
- Standard deviation
- Searches per second
- Latency in milliseconds

### Recall Metrics
- Recall@K percentage
- Number of queries with ground truth

### Index Metrics
- Total index size (MB)
- Component breakdown (index, passages, metadata)
- Backend and embedding model info

## Example Results

```
🎯 LAION MULTIMODAL BENCHMARK RESULTS
============================================================

📏 Index Information:
  Total size: 145.2 MB
  Backend: hnsw
  Embedding model: clip-vit-b-32
  Total passages: 10000

⚡ Search Performance:
  Total queries: 200
  Average search time: 0.023s
  Median search time: 0.021s
  Min/Max search time: 0.012s / 0.089s
  Std dev: 0.008s
  Complexity: 64
  Top-K: 3

📊 Recall Performance:
  Recall@3: 85.5%
  Queries with ground truth: 200

⚙️ Complexity Analysis:
  Complexity  16: 0.015s avg
  Complexity  32: 0.019s avg
  Complexity  64: 0.023s avg
  Complexity 128: 0.031s avg

🚀 Performance Summary:
  Searches per second: 43.5
  Latency (ms): 23.0ms
```

## Directory Structure

```
benchmarks/laion/
├── setup_laion.py           # Setup script
├── evaluate_laion.py        # Evaluation script
├── README.md               # This file
└── data/                   # Generated data
    ├── laion_images/       # Image files (placeholder)
    ├── laion_metadata.jsonl # Image metadata
    ├── laion_passages.jsonl # LEANN passages
    ├── laion_embeddings.npy # CLIP embeddings
    ├── evaluation_queries.jsonl # Evaluation queries
    └── laion_index.leann/  # LEANN index files
```

## Notes

- Current implementation uses dummy data for demonstration
- For real LAION data, implement actual download logic in `setup_laion.py`
- CLIP embeddings are randomly generated - replace with real CLIP model for production
- Adjust `num_samples` and `num_queries` based on available resources
- Consider using `--num-samples` during evaluation for faster testing