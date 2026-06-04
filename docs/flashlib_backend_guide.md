# FlashLib Backend Guide

LEANN now ships an optional, GPU-accelerated search backend powered by
[**FlashLib**](https://github.com/FlashML-org/flashlib) — a library of classical
machine-learning operators built on Triton and CuteDSL. This guide covers what it
is, when to use it, how to install it, and how it plugs into LEANN.

## What is FlashLib?

FlashLib (`pip install flashlib`) is a GPU library of classical ML primitives
(k-means, DBSCAN, PCA, SVD, UMAP, t-SNE, IVF-Flat ANN, and more). LEANN uses its
`IVFFlat` index — an inverted-file flat approximate-nearest-neighbor (ANN) index
that runs entirely on CUDA tensors. At a fixed `(nlist, nprobe)` it probes the same
candidate set as a reference IVF-Flat (FAISS / cuVS), so recall is predictable,
while the search itself is accelerated on the GPU.

```python
import torch
from flashlib import IVFFlat

db = torch.randn(1_000_000, 128, device="cuda")
index = IVFFlat(nlist=1024, nprobe=16).fit(db)
distances, indices = index.kneighbors(torch.randn(10_000, 128, device="cuda"), n_neighbors=10)
```

## When to use it

| Backend | Best for | Storage | Hardware |
|---------|----------|---------|----------|
| `hnsw` (default) | Laptop / CPU, max storage savings via recomputation | ~3% of raw (pruned graph) | CPU |
| `diskann` | Larger-than-memory datasets | On-disk graph | CPU |
| `ivf` | Incremental add/remove without rebuild | Full vectors (FAISS) | CPU |
| **`flashlib`** | **High-throughput search on a CUDA GPU** | Full vectors (`.npy`) | **CUDA GPU** |
| **`flashlib_ivf`** | **GPU IVF-Flat (approximate) — the GPU counterpart of `ivf`** | Full vectors (`.pt`) | **CUDA GPU** |

Use FlashLib when you already have a GPU and want fast IVF-Flat search. It stores
the full float32 vectors rather than a pruned graph, so it trades LEANN's storage
savings for raw GPU search speed.

## Requirements

- A **CUDA GPU** (required at *search* time; building the index only needs numpy).
- `flashlib` and `torch` (installed automatically with the extra below).

## Installation

The backend is **optional** — it is not pulled in by a default LEANN install.

```bash
# From a LEANN source checkout
uv sync --extra flashlib

# Or as a standalone package
pip install leann-backend-flashlib
```

LEANN auto-discovers any installed `leann-backend-*` package, so once it is
installed the `flashlib` backend name is available with no further configuration.

## Usage

### Python API

```python
from leann import LeannBuilder, LeannSearcher

builder = LeannBuilder(backend_name="flashlib")   # nlist=1024, distance_metric="mips"
builder.add_text("LEANN recomputes embeddings on the fly to cut storage by ~97%.")
builder.add_text("FlashLib runs IVF-Flat search on the GPU.")
builder.build_index("demo.leann")

searcher = LeannSearcher("demo.leann")
results = searcher.search("How does LEANN save storage?", top_k=3)
for r in results:
    print(r.score, r.text)
```

### Example apps / CLI

```bash
source .venv/bin/activate
python -m apps.document_rag \
    --query "What are the main techniques LEANN explores?" \
    --backend-name flashlib
```

## How it works

FlashLib's `IVFFlat` builds its index in GPU memory and has **no on-disk format**.
The LEANN backend bridges that gap:

1. **Build** (`FlashlibBuilder`): persists the raw float32 vectors as
   `<index>.flashlib.npy` and an id map as `<index>.flashlib_id_map.json`.
2. **Search** (`FlashlibSearcher`): loads the vectors into a CUDA tensor and
   reconstructs the index once via `IVFFlat(nlist, nprobe).fit(db)` at start-up,
   then answers every query with `index.kneighbors(...)`.

FlashLib's only distance metric is **squared L2**. For `mips` / `cosine` the
backend L2-normalizes both the database and the query vectors; on unit vectors,
squared-L2 ranking is equivalent to inner-product / cosine ranking, so results
match the other backends. `nlist` is clamped to the corpus size (the k-means
constraint), and `nprobe` is derived from the search `complexity` knob.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `nlist` (build) | `1024` | Number of IVF partitions; clamped to the number of vectors. |
| `distance_metric` (build) | `"mips"` | `mips`, `cosine`, or `l2`. |
| `nprobe` (search) | derived from `complexity` | Partitions probed per query — the recall knob (higher = better recall, slower). |

## FlashLib IVF backend (`flashlib_ivf`)

The **`flashlib_ivf`** backend (`leann-backend-flashlib-ivf`) is the GPU counterpart
of LEANN's FAISS `ivf` backend: an IVF-Flat (inverted file) index that
coarse-quantizes the corpus into `nlist` cells with k-means and, at search time,
scans only the `nprobe` nearest cells — entirely on CUDA tensors via FlashLib's
`flash_ivf_flat`. At a fixed `(nlist, nprobe)` the GPU and CPU IVF probe nearly the
same candidate set, so recall is comparable; the only difference is GPU vs CPU
kernels. `nprobe` is the recall knob (defaults to `min(complexity, nlist)`).

```bash
uv sync --extra flashlib-ivf          # or: pip install leann-backend-flashlib-ivf
```

```python
from leann import LeannBuilder, LeannSearcher

builder = LeannBuilder(backend_name="flashlib_ivf", nlist=4096, distance_metric="cosine")
builder.add_text("LEANN recomputes embeddings on the fly to cut storage by ~97%.")
builder.build_index("demo.leann")

searcher = LeannSearcher("demo.leann")
results = searcher.search("How does LEANN save storage?", top_k=10, complexity=32)  # nprobe=32
```

**How it works:** the builder trains the coarse quantizer on the GPU and persists the
built index tensors with `torch.save` (`<index>.flashlib_ivf.pt`) plus an id map
(`<index>.flashlib_ivf_id_map.json`); the searcher reloads them onto the GPU once (no
k-means re-train). A CUDA GPU is required at **both** build (k-means) and search time.
FlashLib IVF ranks by squared L2, so `mips`/`cosine` L2-normalize the vectors (squared-L2
ranking then matches inner-product/cosine).

### Speed — IVF (GPU) vs IVF (CPU)

```bash
python benchmarks/flashlib_ivf_vs_faiss_ivf.py \
    --sizes 100000 1000000 --nprobe-sweep 1 8 32 128 --cpu-threads 8
```

`flashlib_ivf` (GPU) vs the FAISS `ivf` backend (CPU) at the **same `nlist`**, sweeping
`nprobe` (NVIDIA H200, `faiss-cpu` at 8 threads, 768-dim, top-k=10). GPU latency stays
~flat while CPU latency grows linearly with `nprobe`, so the GPU lead widens exactly as
you raise recall:

| Corpus | nprobe | GPU lat | CPU lat | GPU q/s | CPU q/s | Recall (GPU/CPU) | Speedup (lat / tpt) |
|--------|--------|---------|---------|---------|---------|------------------|---------------------|
| 1M | 8 | 0.45 ms | 1.14 ms | 107k | 5.9k | 0.340 / 0.321 | 2.6× / 18× |
| 1M | 32 | 0.46 ms | 3.00 ms | 141k | 1.9k | 0.400 / 0.350 | 6.5× / 75× |
| 1M | 128 | 0.55 ms | 9.91 ms | 95k | 0.6k | 0.539 / 0.423 | 18× / 159× |

Build (1M, `nlist=4096`): GPU **10.6 s** vs FAISS CPU **140.7 s** — a **13× faster
build** (GPU k-means vs CPU k-means training).

Honest caveats: at very low `nprobe` (e.g. 1) single-query GPU latency (~0.44 ms) is
*higher* than CPU, because the per-query work is tiny and GPU kernel-launch overhead
dominates; the GPU advantage grows with `nprobe` (higher recall), batch size, and corpus
size. The absolute recall above is low because the synthetic mixture-of-Gaussians corpus
has more clusters than `nlist`; on real embeddings recall is far higher — this benchmark
isolates the GPU-vs-CPU *relative* comparison at matched `(nlist, nprobe)`.

## Notes & limitations

- GPU-only at search time; building an index works on a CPU-only machine.
- Stores full vectors, so it does not benefit from LEANN's graph-pruning storage
  savings — pick `hnsw` if minimizing disk footprint is the priority.
- Query embeddings are computed through the standard LEANN embedding path (the
  HNSW ZMQ embedding server when available, otherwise direct model loading), so
  any LEANN-supported embedding model works.
