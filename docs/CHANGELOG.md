# Changelog

Append-only log of major changes to LEANN (new features, breaking changes, important
fixes). Newest entries at the bottom.

## 2026-03-05: IVF backend incremental update support

- Added `leann-backend-ivf` with FAISS IndexIVFFlat + DirectMap.Hashtable.
- IVF supports in-place `add_vectors` and `remove_ids` without full rebuild.
- `leann build` is now idempotent: re-running on an existing index does incremental update (add new, remove deleted, re-index modified files).
- Fixed incremental build chunking inconsistency and shared metadata dict bug.
- Fixed IVF incremental update duplicate chunks from stale `passages.jsonl`.

## 2026-03-05: MCP server v2 — build, status, and structured search

- Added `leann_build` MCP tool: build or incrementally update indexes directly from Claude Code.
- Added `leann_status` MCP tool: inspect index details (backend, embedding model, chunk/file count, size).
- `leann_search` now uses `--json` output with file paths always included, formatted as markdown code blocks.
- Fixed `float32` JSON serialization bug in `leann search --json`.
- Cleaned up MCP tool descriptions (concise, no emoji).

## 2026-03-05: Documentation — roadmap, vision, and dev guidelines

- Rewrote `docs/roadmap.md` with current P0/P1 priorities from GitHub issue #237.
- Added `docs/ultimate_goal.md` — long-term vision (personal data platform, best code retrieval MCP, multimodal, local-first).
- Added self-contained documentation principle and dev doc maintenance rules to `CLAUDE.md`.

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

## 2026-07-27: Bounded `leann list` index discovery

- Bound App-format metadata discovery to a configurable directory depth; `leann list` defaults to depth 5 and accepts `--max-depth` for deeper projects.
- Prune common dependency, virtual-environment, cache, and system directories during discovery.
- Preserve discovery of registered CLI-format and App-format projects without recursively scanning an entire home directory.
