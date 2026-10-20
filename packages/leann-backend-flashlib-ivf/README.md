# leann-backend-flashlib-ivf

GPU-accelerated [FlashLib](https://github.com/FlashML-org/flashlib) IVF-Flat
(inverted file) backend for LEANN - the GPU counterpart of the FAISS
[`leann-backend-ivf`](../leann-backend-ivf) backend.

FlashLib is a GPU library of classical ML operators built on Triton / CuteDSL.
Its IVF-Flat index coarse-quantizes the corpus into `nlist` cells and, at search
time, scans only the `nprobe` nearest cells - entirely on CUDA tensors. At a fixed
`(nlist, nprobe)` it probes the same candidate set as a reference IVF-Flat
(FAISS / cuVS), so recall is comparable; the difference is GPU vs CPU kernels.

This is registered as the `flashlib_ivf` backend, distinct from the exact GPU
k-NN `flashlib` backend (which does brute-force `NearestNeighbors`, not IVF).

## Requirements

- A CUDA GPU (required at **build** time for k-means training and at **search** time).
- `pip install flashlib` and `torch`.

## Install

```bash
# from a LEANN checkout
uv sync --extra flashlib-ivf
# or
pip install leann-backend-flashlib-ivf
```

## Usage

```python
from leann import LeannBuilder, LeannSearcher

builder = LeannBuilder(backend_name="flashlib_ivf", nlist=1024, distance_metric="cosine")
builder.add_text("LEANN recomputes embeddings to save storage.")
builder.build_index("demo.leann")

searcher = LeannSearcher("demo.leann")
print(searcher.search("How does LEANN save storage?", top_k=3, complexity=32))  # nprobe=min(32, nlist)
```

Or from the example apps:

```bash
python -m apps.document_rag --query "What are the main techniques LEANN explores?" \
    --backend-name flashlib_ivf
```

## How it works

The built IVF-Flat index is a small set of torch tensors (centroids,
cell-contiguous data, row ids, CSR offsets), so this backend persists it with
`torch.save` (`<index>.flashlib_ivf.pt`) plus an id map
(`<index>.flashlib_ivf_id_map.json`) and reloads it onto the GPU at searcher
start-up - no k-means re-train.

FlashLib's only distance metric is squared L2. For `mips` / `cosine` the vectors
are L2-normalized at build and query time, on which squared-L2 ranking is
equivalent to inner-product / cosine ranking.

## Parameters

| kwarg | default | meaning |
|-------|---------|---------|
| `nlist` | `1024` | number of IVF partitions / coarse centroids (clamped to corpus size) |
| `nprobe` (build) | `16` | default partitions probed per query (recall knob) |
| `niter` | `20` | Lloyd k-means iterations for the coarse quantizer |
| `seed` | `0` | RNG seed (deterministic build) |
| `distance_metric` | `"mips"` | `mips`, `cosine`, or `l2` |
| `complexity` (search) | `64` | sets `nprobe = min(complexity, nlist)` when `nprobe` is not given |
