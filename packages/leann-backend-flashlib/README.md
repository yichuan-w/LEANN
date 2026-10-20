# leann-backend-flashlib

GPU-accelerated [FlashLib](https://github.com/FlashML-org/flashlib) `IVFFlat`
backend for LEANN.

FlashLib is a GPU library of classical ML operators built on Triton / CuteDSL.
Its `IVFFlat` index runs approximate nearest-neighbor search entirely on CUDA
tensors and, at a fixed `(nlist, nprobe)`, probes the same candidate set as a
reference IVF-Flat (FAISS / cuVS).

## Requirements

- A CUDA GPU (required at **search** time; index building only needs numpy).
- `pip install flashlib` and `torch`.

## Install

```bash
# from a LEANN checkout
uv sync --extra flashlib
# or
pip install leann-backend-flashlib
```

## Usage

```python
from leann import LeannBuilder, LeannSearcher

builder = LeannBuilder(backend_name="flashlib")   # nlist=1024, distance_metric="mips"
builder.add_text("LEANN recomputes embeddings to save storage.")
builder.build_index("demo.leann")

searcher = LeannSearcher("demo.leann")
print(searcher.search("How does LEANN save storage?", top_k=3))
```

Or from the CLI / example apps:

```bash
python -m apps.document_rag --query "What are the main techniques LEANN explores?" \
    --backend-name flashlib
```

## How it works

FlashLib's `IVFFlat` has no on-disk format, so this backend persists the raw
float32 vectors (`<index>.flashlib.npy`) plus an id map (`<index>.flashlib_id_map.json`)
and rebuilds the GPU index at searcher start-up via `IVFFlat(...).fit(db)`.

FlashLib's only distance metric is squared L2. For `mips` / `cosine` the vectors
are L2-normalized at build and query time, on which squared-L2 ranking is
equivalent to inner-product / cosine ranking.

## Parameters

| kwarg | default | meaning |
|-------|---------|---------|
| `nlist` | `1024` | number of IVF partitions (clamped to corpus size) |
| `distance_metric` | `"mips"` | `mips`, `cosine`, or `l2` |
| `nprobe` (search) | derived from `complexity` | partitions probed per query (recall knob) |
