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

## 2026-06-03: OCR PDF indexing

- Add opt-in OCR for scanned and image-heavy PDFs via `leann build --enable-ocr`.
  LEANN still extracts embedded PDF text first with PyMuPDF; OCR only runs on pages
  that have no embedded text, which keeps normal PDF indexing unchanged.
- Add the `leann-core[ocr]` and `leann[ocr]` extras for `pytesseract` and Pillow,
  while keeping OCR dependencies out of the default install. Users must also have
  the local Tesseract binary available on `PATH`.
- Add `apps/ocr_rag.py` plus `docs/ocr.md` for an end-to-end scanned-PDF RAG example,
  including source metadata fields and deterministic failure behavior when OCR
  dependencies or the Tesseract runtime are missing.
