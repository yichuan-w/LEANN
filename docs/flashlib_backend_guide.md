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

## Notes & limitations

- GPU-only at search time; building an index works on a CPU-only machine.
- Stores full vectors, so it does not benefit from LEANN's graph-pruning storage
  savings — pick `hnsw` if minimizing disk footprint is the priority.
- Query embeddings are computed through the standard LEANN embedding path (the
  HNSW ZMQ embedding server when available, otherwise direct model loading), so
  any LEANN-supported embedding model works.
