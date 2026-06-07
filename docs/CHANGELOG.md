# Changelog

Append-only log of major changes to LEANN (new features, breaking changes, important
fixes). Newest entries at the bottom.

## 2026-06-02: GPU FlashLib IVF backend (`flashlib_ivf`)

- Add `leann-backend-flashlib-ivf`, a GPU IVF-Flat (inverted file) approximate-NN
  backend built on FlashLib (`flash_ivf_flat`, Triton/CuteDSL) — the GPU counterpart
  of the FAISS `ivf` backend. Registered as backend name `flashlib_ivf`; install via
  `uv sync --extra flashlib-ivf` or `pip install leann-backend-flashlib-ivf`. Shares
  the `nlist`/`nprobe` recall knobs with the `ivf` backend, so the two are drop-in
  comparable. Requires a CUDA GPU at build (k-means) and search.
- Add `benchmarks/flashlib_ivf_vs_faiss_ivf.py`: head-to-head `flashlib_ivf` (GPU) vs
  `ivf` (FAISS, CPU) at matched `nlist` across an `nprobe` sweep (build time,
  single-query latency, batched throughput, recall@k vs exact ground truth). On an
  NVIDIA H200 at 1M x 768 vectors (nlist=4096, 8 CPU threads): ~13x faster build and,
  at nprobe=32, ~6.5x lower single-query latency / ~75x higher batched throughput at
  comparable recall (GPU latency stays ~flat while CPU grows linearly with nprobe).
- Docs: `docs/flashlib_backend_guide.md` gains a `flashlib_ivf` section.

## 2026-06-07: Content-hash passage IDs as the default

- Make `content-hash` the default generated passage ID scheme so new indexes use
  `sha256(text)[:16]` IDs that remain stable across file moves and chunk reorderings.
- Preserve existing index ID schemes during incremental updates: legacy sequential
  indexes continue appending sequential IDs, while content-hash indexes append
  content-derived IDs.
- Mark precomputed embedding array builds as `external` ID indexes and align passage
  JSONL, offset maps, backend labels, and metadata to the caller-provided IDs.
- Serialize full-rebuild publish and ID-migration rewrites with an index write lock.
