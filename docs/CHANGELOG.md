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

## 2026-06-03: ReAct source policy and safer web tools

- Add `leann react --source-policy local|web|both` and the matching Python API option so
  users can explicitly restrict the agent to local LEANN retrieval, public web research, or both.
- Refuse out-of-policy tool calls at runtime before local search or web requests are executed,
  keeping model-selected actions inside the configured retrieval boundary.
- Add bounded web request timeouts and reject non-HTTP page-fetch URLs for the Serper/Jina web tools.
