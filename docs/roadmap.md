# LEANN Roadmap

LEANN aims to be a **personal knowledge layer** — not just a storage-efficient vector database, but a unified, always-up-to-date knowledge base that runs entirely on your own machine. It connects your code, images, and personal data (documents, emails, browser history, chats) into a single multimodal search interface.

Contributions and feedback are welcome. Join our [Slack](https://join.slack.com/t/leann-e2u9779/shared_invite/zt-3ol2ww9ic-Eg_kB8omwe6xmYVd0epr4Q) to discuss.

---

## Completed

- [x] HNSW backend integration
- [x] DiskANN backend with MIPS/L2/Cosine support
- [x] Real-time embedding pipeline
- [x] Memory-efficient graph pruning
- [x] IVF backend with incremental add/remove (#231, #89, #141)
- [x] Merkle tree file-change detection — `leann watch` (#41)

---

## P0 — Core (Q1 2026)

### LEANN MCP — The Best Code Retrieval MCP

The primary near-term goal: make LEANN the go-to MCP server for code-aware AI assistants. This means **dynamic updates** (your index stays current as you edit code), **rich code context** (AST-aware chunking that understands functions, classes, and modules — not just raw text), and a **dead-simple interface** (one command to build, automatic incremental updates, zero configuration for common setups).

- [x] IVF backend — incremental add/remove without full rebuild (#231, #89, #141)
- [x] Merkle tree file-change detection — `leann watch` for automatic re-indexing on file changes (#41)
- [ ] Cold start optimization — faster first-build experience for new users (#166, #177)
- [ ] Live index updates — push index changes as files are saved, not just on rebuild
- [ ] Smarter code context — cross-file symbol resolution, call graph awareness, import tracking

### Search Quality

- [ ] Hybrid search — combine dense vector retrieval with sparse keyword matching (BM25) for better recall on exact identifiers and variable names (#233, #90)

### Documentation

- [ ] ReadTheDocs — hosted documentation site (#234)
- [ ] Benchmarks — recall@k, latency, and storage comparisons across backends

---

## P1 (Q2 2026)

### Multimodal

- [ ] Video retrieval (#160)
- [ ] CLIP support — image-text cross-modal search (#94)
- [ ] OCR — extract text from images/scanned documents (#158)

### Platform & Distribution

- [ ] Windows support (#14)
- [ ] Web UI (#229)

### Applications & Integrations

- [ ] Agent + Deep research (#104)
- [ ] Local Cursor — local model + local retrieval for code assistance (#47)
- [ ] LlamaIndex integration (#217)
- [ ] Obsidian support (#96)

---

## Contributing

If you're interested in working on any of the items above, please reach out to [@yichuan-w](https://github.com/yichuan-w) or [@andylizf](https://github.com/andylizf). See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor workflow.
