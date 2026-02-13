# Create all 8 PRs - run after: gh auth login
# Base branch: master

$prs = @(
    @{
        head = "abinav/issue-177-cold-start-LONFb"
        title = "Fix cold-start latency + ZMQ reliability"
        body = @'
Closes #177, #182

**What changed and why:**
- **Removed redundant `_ensure_server_running` call inside `compute_query_embedding`.** Previously, every query embedding triggered a second server health check even though the caller (`api.py`) already called `_ensure_server_running`. This was the main source of cold-start latency — a double server startup on first query. Now `compute_query_embedding` trusts the port it receives from the caller, eliminating the redundant check.
- **Added ZMQ retry with exponential backoff (0.5s, 1s, 2s) in `_compute_embedding_via_server`.** The original code did a single ZMQ send/recv with no retry — any transient connection failure (server still loading, socket not yet bound) was a hard crash. Now retries up to 3 times with proper socket teardown between attempts.
- **Set `ZMQ.SNDTIMEO=10s` and `ZMQ.LINGER=0`.** Without SNDTIMEO, a dead server caused an indefinite hang on `socket.send()`. Without LINGER=0, `socket.close()` would block waiting for unsent messages, stalling the retry loop.
- **`enable_warmup` is now a kwargs pop, not a forward.** It was being passed through to `start_server()` which didn't understand it. Now it's consumed by `_ensure_server_running` to fire a dummy embedding request after server start, pre-loading the model into GPU memory before the first real query hits.
- **Replaced `print()` with `logging.getLogger(__name__)` in searcher_base.** The original used bare `print("⚠️ ...")` for error reporting, which is invisible in production and breaks structured logging pipelines.
- **Added timing instrumentation across the entire server lifecycle** — `start_server`, `_start_new_server`, `_ensure_server_running`, `compute_query_embedding` all log elapsed time so you can actually profile where cold-start time goes.
'@
    },
    @{
        head = "abinav/issue-233-hybrid-search-LONFb"
        title = "feat: add hybrid search via BM25 (SQLite FTS5) + score fusion"
        body = @'
Closes #233

**What changed and why:**
- **New `bm25.py` — SQLite FTS5-backed BM25 index.** Chose FTS5 over a pure-Python BM25 implementation (like `rank_bm25`) because: (1) FTS5 handles tokenization/stemming natively, (2) it scales to millions of chunks without loading everything into memory, (3) it's zero-dependency (sqlite3 is in the stdlib), and (4) the BM25 ranking is built into the FTS5 `rank` function, so no custom scoring math needed.
- **`build_fts5_index()` is called at index build time** — added to both the `build_index()` and `build_from_embeddings()` paths in `api.py`. The FTS5 DB is stored alongside the LEANN index as `<name>.fts5.db`. For pre-computed embedding imports (where chunk text may be placeholder), it skips FTS5 construction via `skip_if_placeholder`.
- **New `hybrid.py` — weighted score fusion.** Implements min-max normalization on both vector distances and BM25 scores, then linearly combines them: `final = (1 - ratio) * vector_norm + ratio * bm25_norm`. Chose linear fusion over reciprocal rank fusion (RRF) because RRF ignores score magnitude and LEANN's HNSW distances are meaningful. The `sparse_score_ratio` parameter (0.0 = pure vector, 1.0 = pure BM25) is exposed through the search API.
- **BM25 index is optional** — if the `.fts5.db` file doesn't exist (e.g., old indexes built before this PR), search falls back to pure vector. No migration required.
'@
    },
    @{
        head = "abinav/issue-141-reindex-LONFb"
        title = "Add leann reindex CLI command with file manifest tracking"
        body = @'
Closes #141

**What changed and why:**
- **New `leann reindex <name> --docs <dir>` command.** The core problem: after adding/editing files in your docs directory, you had to `leann remove` + `leann build` to update the index. Now `reindex` handles it as a single operation.
- **SHA-256 file manifest for delta detection.** On every `build` or `reindex`, a `manifest.json` (mapping `{absolute_path: sha256}`) is saved alongside the index. On subsequent reindex, the manifest is diffed to report exactly which files are new/changed/deleted/unchanged. This is a content-hash approach (not mtime) so it's resilient to filesystem clock skew and `touch` operations.
- **`LeannBuilder.from_meta()` classmethod** reconstructs a builder from an existing index's `meta.json`. This avoids the user having to re-specify `--backend`, `--embedding-model`, `--graph-degree`, etc. on reindex — it reads the original build config and reuses it exactly.
- **Currently does a full rebuild, not incremental embedding.** Even though delta detection identifies unchanged files, the rebuild still re-embeds everything. True incremental embedding (appending to an existing HNSW graph) would require the backend to support `add_items()` which HNSW doesn't cleanly expose. The manifest delta is still valuable: it gives users visibility into what changed and skips rebuild entirely when nothing changed.
- **`--sync` flag for delete behavior.** By default, deleted files keep their chunks in the index (safe for rollbacks). `--sync` removes them. This mirrors how Elasticsearch reindex handles deletes.
'@
    },
    @{
        head = "abinav/issue-158-ocr-LONFb"
        title = "Add --enable-ocr flag for scanned PDF support"
        body = @'
Closes #158

**What changed and why:**
- **`--enable-ocr` flag on `leann build`.** When set, PDFs with pages containing < 50 chars of embedded text are sent through OCR. The 50-char threshold is intentional — it avoids OCR overhead on text-native PDFs while catching scanned documents and image-heavy slides.
- **Chose MinerU (`magic-pdf`) over pytesseract.** MinerU works at the document level (not page-by-page), understands PDF layout structure, and outputs clean Markdown with heading hierarchy preserved. Pytesseract gives flat text with no structure, requires Tesseract system binary, and is significantly slower on multi-page docs. MinerU is optional — `pip install leann-core[ocr]` pulls it in.
- **Fallback chain:** MinerU → pymupdf built-in OCR → raw text extraction. If MinerU fails or isn't installed, it falls through gracefully. The function signature of `extract_pdf_text_with_pymupdf` is backward compatible (new `use_ocr=False` kwarg).
- **`apps/ocr_rag.py` example app** showing the full pipeline: directory of scanned PDFs → OCR extraction → chunking → index → query. This extends the existing `BaseRAGExample` pattern used by other example apps.
'@
    },
    @{
        head = "abinav/issue-217-llamaindex-LONFb"
        title = "feat: add LlamaIndex integration with hybrid retriever support"
        body = @'
Closes #217

**What changed and why:**
- **`LeannRetriever(BaseRetriever)` — drops LEANN into any LlamaIndex RAG pipeline.** Implements `_retrieve()` and `_aretrieve()` (async just delegates to sync since LEANN's search is CPU-bound, not I/O-bound). Takes `index_path`, `top_k`, `complexity` as constructor args, instantiates `LeannSearcher` internally.
- **`LeannHybridRetriever(BaseRetriever)` — vector + BM25 through LlamaIndex.** Delegates to `LeannSearcher.search(sparse_score_ratio=...)`, reusing the hybrid search infrastructure from PR 2. If PR 2 isn't merged yet, this still works — `sparse_score_ratio` is ignored when no FTS5 index exists.
- **Conversion layer `_results_to_nodes()`** maps LEANN's `SearchResult` objects (which use `.text`, `.id`, `.score`, `.metadata`) to LlamaIndex's `NodeWithScore(TextNode(...))`. Metadata is passed through as-is so LlamaIndex's postprocessors can filter on it.
- **Placed under `leann/integrations/` subpackage** to establish a pattern for future framework integrations (LangChain, Haystack, etc.) without polluting the top-level namespace.
- **BM25 weight is clamped to [0, 1]** in the constructor rather than validated at query time — fail-fast principle.
'@
    },
    @{
        head = "abinav/issue-166-warmup-LONFb"
        title = "feat: add persistent embedding daemon to eliminate cold-start latency"
        body = @'
Closes #166

**What changed and why:**
- **New `leann serve` command** that starts a long-running background process keeping the embedding model loaded in memory. The first `leann search` after boot normally takes 10-30s (model load + tokenizer init). With the daemon pre-warmed, it's < 100ms.
- **Daemon lifecycle: `--foreground`, `--stop`, `--status`.** Writes a PID file to `~/.leann/daemon.pid` and state to `~/.leann/daemon.json`. `--stop` sends SIGTERM and cleans up the PID file. `--status` reads the state file and checks if the process is still alive.
- **`EmbeddingServerManager.connect_to_daemon()` added** — before starting a new server subprocess, the manager now checks if a daemon is already running on the expected port and reuses it. This is the key integration point: existing `leann search` commands transparently benefit from the daemon without any code changes in the search path.
- **Chose process-level daemon over thread-level** because the embedding model (especially sentence-transformers with PyTorch) holds a GIL-heavy computation path. A separate process avoids GIL contention with the ZMQ server's event loop. The daemon uses the same ZMQ protocol as the subprocess-based server — zero changes to the client side.
- **Relationship with PR 1 (cold-start fix):** PR 1 fixes the cold-start for one-off queries (retry + no double-check). This PR eliminates cold-start entirely for repeated usage (daemon stays warm). They're complementary — PR 1 is the safety net, this is the performance optimization.
'@
    },
    @{
        head = "abinav/issue-96-obsidian-LONFb"
        title = "feat: add Obsidian vault support with wikilink/backlink awareness"
        body = @'
Closes #96

**What changed and why:**
- **New `--obsidian` flag on `leann build`.** When set, `.md` files are processed through `ObsidianVaultReader` instead of the standard `SimpleDirectoryReader` → `SentenceSplitter` pipeline. Non-markdown files in the same directory still go through the standard pipeline — the two are merged.
- **`ObsidianVaultReader` does a two-pass scan:** Pass 1 scans all `.md` files to extract wikilinks and build a vault-wide backlink map. Pass 2 re-reads and chunks each file, attaching the computed backlinks as metadata. The two-pass approach is necessary because backlinks are a global property — you can't know who links to Note C until you've scanned every note.
- **Wikilink parser handles `[[Page]]`, `[[Page|alias]]`, `[[Page#heading]]`, `[[Page#heading|alias]]`** via a single regex. Obsidian's link syntax is irregular (case-insensitive targets, optional aliases, heading anchors), so the regex is permissive and normalization happens downstream.
- **Backlink map is keyed on lowercase note name** to match Obsidian's case-insensitive linking behavior. Duplicate note names (e.g., `notes/A.md` and `subfolder/A.md`) are detected and logged as warnings — the full relative path is used as the internal key to avoid collisions.
- **YAML frontmatter parsing uses `yaml.safe_load` with a fallback** to basic `key: value` line parsing if PyYAML isn't installed. Tags from frontmatter are merged with inline `#tags` extracted from the body (code blocks excluded).
- **Chunk metadata includes:** `note_name`, `file_path`, `tags`, `wikilinks` (outgoing), `backlinks` (incoming), `frontmatter`, `chunk_index`, `source: "obsidian"`. This makes graph-aware retrieval possible downstream — you could boost chunks that have many backlinks, or filter by tag.
'@
    },
    @{
        head = "abinav/issue-47-local-cursor-LONFb"
        title = "feat: add local Cursor proxy - local LLM + LEANN code retrieval"
        body = @'
Closes #47

**What changed and why:**
- **New `leann cursor` command** starts an OpenAI-compatible HTTP proxy (`/v1/chat/completions`, `/v1/models`) that intercepts chat requests, runs LEANN retrieval against a local code index, injects the retrieved context into the system prompt, and forwards to a local LLM (Ollama/LM Studio).
- **Architecture decision: HTTP proxy, not a LlamaIndex agent.** A proxy is editor-agnostic — Cursor, Continue, Claude Code, any OpenAI-compatible client works by just pointing `OPENAI_BASE_URL` at it. An agent-based approach would lock you into one framework.
- **Context injection strategy: prepend to system message.** If the conversation already has a system message, context is appended to it. If not, a new system message is inserted at position 0. This preserves the original system prompt's instructions while adding retrieved code. Alternative considered (injecting as a separate user message) was rejected because it pollutes the conversation history and confuses multi-turn context.
- **Uses `ThreadingHTTPServer` from stdlib** — no FastAPI/Flask dependency. For a local proxy that handles one user's requests, stdlib threading is sufficient and keeps the dependency footprint at zero.
- **10 MB request body limit** prevents OOM from malformed requests. CORS headers are set for browser-based editors. `_forward_to_llm` preserves upstream error bodies so debugging LLM connection issues is transparent.
- **`_build_context_block` truncates at `--max-context` chars** (default 8000) to stay within the local model's context window budget. It includes file path headers per snippet so the model can reference sources.
- **Works without an index** (`--index` is optional) — in that case it's a pure model proxy, useful for just routing Cursor to a local LLM without retrieval.
'@
    }
)

$base = "main"
$repo = (git rev-parse --show-toplevel)
Push-Location $repo

foreach ($pr in $prs) {
    Write-Host "`nCreating PR: $($pr.title)" -ForegroundColor Cyan
    $bodyFile = [System.IO.Path]::GetTempFileName()
    $pr.body | Out-File -FilePath $bodyFile -Encoding utf8
    gh pr create --base $base --head $pr.head --title $pr.title --body-file $bodyFile
    Remove-Item $bodyFile -Force -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Failed - run 'gh auth login' if needed" -ForegroundColor Red
        break
    }
}

Pop-Location
Write-Host "`nDone." -ForegroundColor Green
