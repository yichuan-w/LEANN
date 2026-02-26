# LEANN × OpenClaw Integration Plan

## Executive Summary

OpenClaw (229K+ GitHub Stars) is the hottest open-source personal AI assistant.
Its memory system—while feature-rich—has documented pain points around storage
growth, API costs, and search quality. LEANN's graph-pruned recomputation
architecture (97% storage compression + free local embeddings) fills a gap that
no existing solution addresses. The async "sleep time compute" nature of
OpenClaw makes LEANN's recomputation latency invisible to end users.

---

## 1  Competitive Landscape (as of Feb 2026)

### 1.1  Existing Memory / Search Solutions in OpenClaw Ecosystem

| Solution | Backend | Embeddings | Storage | Local? | Notes |
|---|---|---|---|---|---|
| **memory-core** (built-in default) | SQLite + sqlite-vec | Remote: Mistral/Voyage/Gemini/OpenAI; Local: embeddinggemma-300m GGUF (~0.6 GB) | Full vectors stored | Partial | BM25+vector hybrid, MMR, temporal decay |
| **memory-lancedb** (bundled plugin) | LanceDB | OpenAI or Ollama | Full vectors stored | Partial | Auto-capture/auto-recall |
| **QMD** (official alt backend, by Shopify founder Tobi, 10.5K stars) | SQLite (Rust CLI) | Jina v3 GGUF 1024d (~1 GB) | Full vectors stored | Yes | BM25+vector+LLM reranking, <100ms query |
| **memsearch** (Zilliz, 612 stars) | Milvus / Milvus Lite | OpenAI/Gemini/Voyage/Ollama/sentence-transformers | Full vectors stored | Partial | Standalone Python lib extracted from OpenClaw |
| **openclaw-mem** (community) | SQLite + FTS5 | Optional, w/ RRF fusion | Full vectors stored | Yes | Progressive recall, dual-language |
| **Gno** (ClawHub skill) | Custom | BM25+vector+hybrid | Full vectors stored | Yes | Knowledge graph viz, web UI |

**Key observation**: Every single solution stores full embedding vectors.
None offers storage compression.

### 1.2  LEANN's Unique Differentiators

| Dimension | LEANN | All existing solutions |
|---|---|---|
| Embedding storage | 97% compressed (graph pruning + on-demand recomputation) | 100% stored |
| Local embedding quality | sentence-transformers all-MiniLM-L6-v2 (MTEB 56.3, 384d) | embeddinggemma-300m (unranked on MTEB), or remote APIs |
| Embedding API cost | $0 (fully local) | $0-$400/month depending on provider |
| Scale | Tested to 60M+ passages, DiskANN for larger-than-memory | SQLite-scale (practical limit ~100K chunks) |

---

## 2  User Pain Points (from Reddit, HN, GitHub Issues)

### 2.1  API Cost Shock (r/openclaw, multiple threads)

- Users report $60-400/month for cloud OpenClaw usage
- One user: "$25 loaded, drained to $5 in a day while idle"
- Another: "$500 gone overnight" from automated loops
- Memory search embedding calls add to the bill (remote providers by default)
- PSA thread: "Turn on memory search with embeddings — it'll save you money"
  got significant engagement; users are actively looking for cost reduction

**LEANN angle**: Zero embedding API cost. Local sentence-transformers, no API key needed.

### 2.2  Memory Search Quality Issues (GitHub #19913, #24624)

- Bug: session transcripts dominate `memory_search` results, pushing out actual
  memory files (rated "high severity")
- BM25 boilerplate inflation from plugin-injected text
- Feature request: configurable MEMORY.md injection modes
- Users report memory search "requires significant tuning" to work reliably

**LEANN angle**: HNSW-based vector search with no BM25 boilerplate pollution.
Pure semantic retrieval.

### 2.3  Storage Growth on Mac Mini (MacRumors forums, OpenClaw guides)

- Mac Mini M4 base: 256GB (non-upgradeable SSD)
- OpenClaw itself: ~20GB (binaries + history + workspace)
- Ollama models: 8-18GB
- Embedding indexes grow linearly and never shrink
- Users actively discussing external storage solutions for Mac Mini

**LEANN angle**: 97% storage compression. Same data, 1/30th the disk space.

### 2.4  Sleep Time Compute Model

- OpenClaw processes messages asynchronously (yieldMs default 10s, timeout 1800s)
- Users don't expect instant responses from WhatsApp/Telegram bots
- Agent runs are serialized per-session with queue management
- LLM response generation (several seconds) dominates total latency, not search

**LEANN angle**: Recomputation latency (1-2s with warm daemon) is invisible
within the async message processing pipeline.

---

## 3  Integration Strategy

### 3.1  Phase 1: OpenClaw Skill (Low Effort, Quick Win)

**What**: A SKILL.md that wraps `leann` CLI for indexing and searching OpenClaw
memory files.

**Why Skill over Plugin**:
- Skills are just a `SKILL.md` file — no TypeScript needed
- Can be published to ClawHub immediately
- LEANN stays in Python, uses existing CLI infrastructure
- Users install `leann` via `pip install leann-core` and drop in the skill

**Skill directory structure**:
```
~/.openclaw/workspace/skills/leann-memory/
├── SKILL.md
├── setup.sh          # Auto-install leann-core via pip/uv
└── README.md
```

**SKILL.md** (draft):
```yaml
---
name: leann_memory_search
description: >
  Semantic memory search with 97% storage compression. Use when the user
  asks to search memories, notes, documents, or any markdown knowledge base.
  Replaces default memory_search with higher quality local embeddings and
  dramatically lower storage footprint. Triggered by: "search my memories",
  "find in my notes", "what did I say about X", "recall", "remember".
version: 1.0.0
author: LEANN Team
user-invocable: true
---

# LEANN Memory Search

## When to Use
- User asks to search memories, notes, or knowledge bases
- User wants to recall past decisions, conversations, or facts
- User asks "what did we decide about X" or "find my notes on Y"

## Setup (one-time)
1. Check if `leann` CLI is available: `which leann`
2. If not installed: `pip install leann-core` (or `uv tool install leann-core --with leann`)
3. Build index on memory files:
   ```bash
   leann build openclaw-memory \
     --docs ~/.openclaw/workspace/MEMORY.md ~/.openclaw/workspace/memory/ \
     --embedding-mode sentence-transformers \
     --embedding-model all-MiniLM-L6-v2 \
     --backend hnsw
   ```

## Search Workflow
1. Run `leann search openclaw-memory "<user query>" --top-k 5`
2. Parse the JSON output for relevant passages
3. Present results with source file attribution

## Rebuild (periodic)
When memory files change significantly, rebuild the index:
```bash
leann build openclaw-memory \
  --docs ~/.openclaw/workspace/MEMORY.md ~/.openclaw/workspace/memory/ \
  --embedding-mode sentence-transformers \
  --embedding-model all-MiniLM-L6-v2 \
  --backend hnsw
```
```

**Limitations**:
- No live file watching (user must rebuild manually or via cron)
- Does not replace `memory_search` / `memory_get` tools directly
- Skill-level integration, not deep plugin integration

**Timeline**: 1-2 days

### 3.2  Phase 2: Auto-Sync Skill with File Watcher

**What**: Enhanced skill that uses `leann sync` (or a lightweight watcher
script) to automatically re-index when memory files change.

**Key additions**:
- Background daemon that watches `~/.openclaw/workspace/memory/` for changes
- Debounced re-indexing (similar to OpenClaw's 1.5s debounce)
- Cron-based or inotify/fswatch-based triggering

**Implementation**: Add a `--watch` mode to `leann build` or leverage the
existing `FileSynchronizer` class in `leann.sync`.

**Timeline**: 1 week

### 3.3  Phase 3: Full OpenClaw Memory Plugin (High Effort, Maximum Impact)

**What**: A TypeScript OpenClaw plugin (`memory-leann`) that registers as a
`kind: "memory"` plugin and replaces `memory-core` in the plugin slot system.

**Architecture**:
```
┌──────────────────────────────────────────────────┐
│  OpenClaw Gateway (TypeScript, Node.js)          │
│                                                  │
│  ┌──────────────┐     ┌───────────────────────┐  │
│  │ memory-leann │     │ Agent                 │  │
│  │ plugin       │◄────│ memory_search tool    │  │
│  │              │     │ memory_get tool        │  │
│  └──────┬───────┘     └───────────────────────┘  │
│         │                                        │
│         │ subprocess / HTTP / MCP                 │
│         ▼                                        │
│  ┌──────────────────┐                            │
│  │ leann CLI / MCP  │  (Python process)          │
│  │ server (daemon)  │                            │
│  └──────────────────┘                            │
└──────────────────────────────────────────────────┘
```

**Plugin manifest** (`openclaw.plugin.json`):
```json
{
  "id": "memory-leann",
  "kind": "memory",
  "name": "LEANN Memory",
  "description": "97% storage-compressed semantic memory search with local embeddings",
  "version": "1.0.0",
  "configSchema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "embeddingModel": {
        "type": "string",
        "default": "all-MiniLM-L6-v2"
      },
      "backend": {
        "type": "string",
        "enum": ["hnsw", "diskann"],
        "default": "hnsw"
      },
      "indexPath": {
        "type": "string",
        "default": "~/.openclaw/memory/leann-index"
      },
      "syncIntervalMs": {
        "type": "number",
        "default": 5000
      }
    }
  },
  "uiHints": {
    "embeddingModel": { "label": "Embedding Model", "placeholder": "all-MiniLM-L6-v2" },
    "backend": { "label": "Vector Backend" },
    "indexPath": { "label": "Index Path" },
    "syncIntervalMs": { "label": "Sync Interval (ms)" }
  }
}
```

**TypeScript plugin** (index.ts):
```typescript
export default function register(api) {
  // Register memory_search tool
  api.registerTool({
    name: "memory_search",
    description: "Semantic search over memory files using LEANN (97% compressed)",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        top_k: { type: "number", default: 5 }
      },
      required: ["query"]
    },
    async execute(_id, params) {
      // Shell out to `leann search` CLI
      const result = await exec(
        `leann search openclaw-memory "${params.query}" --top-k ${params.top_k} --json`
      );
      return { content: [{ type: "text", text: result.stdout }] };
    }
  });

  // Register memory_get tool (pass-through to file read)
  api.registerTool({
    name: "memory_get",
    description: "Read a specific memory file",
    parameters: { /* ... */ },
    async execute(_id, params) { /* read file directly */ }
  });

  // Register background service for auto-sync
  api.registerService({
    id: "leann-sync",
    start: () => { /* start file watcher + periodic leann build */ },
    stop: () => { /* cleanup */ }
  });
}
```

**Timeline**: 2-4 weeks

**Prerequisite**: User must have Python + `leann-core` installed. The plugin
shells out to the `leann` CLI process.

---

## 4  Marketing & Positioning

### 4.1  Key Messaging

**One-liner**: "97% smaller memory indexes for OpenClaw. Local embeddings, zero API cost."

**Longer pitch**:
> Your OpenClaw memories grow every day. On a 256GB Mac Mini, every megabyte
> counts. LEANN compresses your memory index by 97% through graph-based
> selective recomputation — the same search quality, 1/30th the disk space.
> Plus: high-quality local embeddings (sentence-transformers) with zero API
> costs. No Mistral key, no Voyage key, no OpenAI bill.

### 4.2  Content Strategy

**Blog post** (publish on dev.to, Medium, personal blog):
- Title: "How I Cut My OpenClaw Memory Storage by 97% on a Mac Mini"
- Content: benchmark comparison (storage size, search quality, latency)
  with real OpenClaw memory data
- Include step-by-step setup guide

**Reddit r/openclaw post**:
- Title: "PSA: LEANN gives you 97% storage compression for memory search
  with zero API cost (local embeddings)"
- Reference the popular "Turn on memory search with embeddings" PSA thread
- Focus on the cost savings angle (users care about this most)

**Hacker News**:
- Show HN post when Phase 1 skill is ready
- Title: "Show HN: 97% storage compression for AI agent memory (OpenClaw skill)"
- Emphasize the technical innovation (graph pruning + recomputation)

**GitHub**:
- Add "Works with OpenClaw" badge to LEANN README
- Open a discussion in OpenClaw repo offering LEANN as a memory backend option
- Respond to existing memory-related issues (#19913, #24624) with LEANN as
  a potential solution

### 4.3  Comparison Table for Marketing

| Feature | memory-core | QMD | memsearch | **LEANN** |
|---|---|---|---|---|
| Storage compression | None | None | None | **97%** |
| Embedding cost | Remote API ($) | Free (local GGUF) | Configurable | **Free (local)** |
| Embedding quality (MTEB) | Depends on provider | Jina v3 (good) | Depends | **all-MiniLM-L6-v2 (56.3)** |
| Hybrid search | BM25 + vector | BM25 + vector + rerank | Dense + BM25 (RRF) | Vector (HNSW) |
| Session transcript pollution | Yes (bug #19913) | Yes | N/A | **No** (pure vector) |
| Scale limit | ~100K chunks | ~100K chunks | Milvus-scale | **60M+ passages** |
| Disk footprint (50K chunks, 384d) | ~75 MB | ~200 MB (1024d) | ~75 MB | **~2 MB** |
| Setup complexity | Built-in | Moderate | pip install | pip install |
| Live file watching | Yes | Yes (5min interval) | Yes | Phase 2 |

### 4.4  Target User Segments

1. **Cost-conscious Mac Mini users** — Running OpenClaw 100% local, every GB matters
2. **Heavy memory users** — Months of daily logs + extra indexed directories
3. **Privacy-focused users** — Want zero remote API calls, even for embeddings
4. **Users hitting memory search bugs** — Looking for alternatives to memory-core/QMD

---

## 5  Technical Considerations

### 5.1  Why Storage Matters: OpenClaw Memory Growth Analysis

OpenClaw memory is append-only by design. Data accumulates across four layers,
each contributing to embedding index growth. Below we trace how storage adds up
from raw markdown to vector indexes.

#### 5.1.1  Layer 1 — Core Memory Files

OpenClaw stores memories as plain Markdown:

- `MEMORY.md`: curated long-term facts (recommended < 2,000 words ≈ 10-15 KB)
- `memory/YYYY-MM-DD.md`: one file per day, append-only

A typical daily log captures timestamped entries of conversations, decisions,
tasks, and observations. Estimating 5-20 KB per day:

| Usage level | Daily file size | 1 year total | Chunks (400 tok, 80 overlap) |
|---|---|---|---|
| Light (casual user) | ~5 KB | ~2 MB | ~5,000 |
| Moderate (daily driver) | ~12 KB | ~4.5 MB | ~12,000 |
| Heavy (power user) | ~20 KB | ~7.5 MB | ~20,000 |

These chunks are the units that get embedded. Each chunk ≈ 400 tokens with
80-token overlap between adjacent chunks.

#### 5.1.2  Layer 2 — Session Transcripts (Experimental, Growing Fast)

OpenClaw can index session transcripts (`~/.openclaw/agents/<id>/sessions/*.jsonl`)
via the experimental `sessionMemory` feature. Session data grows much faster
than curated memory:

- One user reported **25 MB of session JSONL in just 3 days** (from 15-second
  cron cycles creating ~4,500 entries)
- Regular chat sessions produce 2-3 MB within hours (~35 messages)
- Sessions with large tool outputs can hit 500 KB+ per session before truncation

When session indexing is enabled, the chunk count balloons:

| Scenario | Session data/month | Added chunks/month |
|---|---|---|
| Light chat use | ~50 MB | ~25,000 |
| Moderate (daily active) | ~200 MB | ~100,000 |
| Heavy (cron + automation) | ~500 MB+ | ~250,000+ |

This is the main growth vector. Session indexing is opt-in today but is the
direction OpenClaw is heading (it enables "recall recent conversations" which
users want).

#### 5.1.3  Layer 3 — Extra Indexed Paths (Unbounded)

OpenClaw supports indexing additional directories beyond the default memory
layout via `memorySearch.extraPaths` or QMD `paths[]`:

```json5
memorySearch: {
  extraPaths: ["~/notes", "~/Documents/meetings", "/srv/shared-docs"]
}
```

```bash
# QMD collections — users routinely add entire directory trees
qmd collection add ~/notes --name notes
qmd collection add ~/Documents/meetings --name meetings
qmd collection add ~/Projects/my-app --name code --mask "*.py,*.ts,*.md"
```

This layer is **unbounded and unpredictable**. Real-world personal data sizes
from knowledge-management communities:

| Data source | Typical size | Example |
|---|---|---|
| Obsidian vault (power user) | 500 MB - 10 GB | 18,000 files over 5 years; 8,000 notes / 2.8M words |
| DEVONthink database | 3 - 9 GB | One user: 9 GB, 82 million words (50% PDFs) |
| Meeting transcripts (Whisper) | 15-20 KB/hour of audio | 200 meetings/year ≈ 3-4 MB text, ~10K chunks |
| Code repository | 1 - 50 MB source | Medium project ≈ 20K-100K chunks |
| Email archive (exported) | 1 - 20 GB | 10 years of email history |
| Slack/Discord export | 500 MB - 5 GB | Active workspace, multi-year |
| Research papers (PDF text) | 1 - 10 GB | Academic user with 500+ papers |

A user who points QMD or memorySearch at `~/Documents` (a 5 GB directory of
mixed markdown, PDFs, and notes) could easily generate **2-5 million chunks**.
At that scale, the embedding index alone would be **6-30 GB** with conventional
full-vector storage — possibly exceeding the free space on a 256 GB Mac Mini.

With LEANN's 97% compression: **180 MB - 900 MB**. The difference between
"need an external drive" and "fits in pocket change".

#### 5.1.4  Layer 4 — The Embedding Index (Where Storage Hurts)

Every chunk gets embedded as a float32 vector. The per-chunk storage cost
depends on the embedding model's dimension:

| Embedding model | Dimensions | Bytes per chunk | Who uses it |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 1,536 B (1.5 KB) | LEANN default |
| embeddinggemma-300m | 768 | 3,072 B (3.0 KB) | OpenClaw local default |
| text-embedding-3-small | 1,536 | 6,144 B (6.0 KB) | OpenClaw remote default |
| Jina v3 (QMD) | 1,024 | 4,096 B (4.0 KB) | QMD default |

Note: these are raw vector bytes only. SQLite/Milvus indexes add ~50%
overhead for metadata, FTS5 tables, and index structures.

#### 5.1.5  Putting It All Together

**Scenario A — Casual user (6 months, memory only)**

| Component | Chunks | Embedding (768d) | With overhead |
|---|---|---|---|
| MEMORY.md | ~200 | 0.6 MB | 0.9 MB |
| Daily logs (180 days) | ~6,000 | 18 MB | 27 MB |
| **Total** | **~6,200** | **18.6 MB** | **~28 MB** |
| LEANN (97% compression) | — | — | **~0.8 MB** |
| **Savings** | | | **~27 MB** |

Not dramatic. User probably doesn't care.

**Scenario B — Daily driver (1 year, memory + sessions)**

| Component | Chunks | Embedding (768d) | With overhead |
|---|---|---|---|
| MEMORY.md | ~500 | 1.5 MB | 2.3 MB |
| Daily logs (365 days) | ~15,000 | 45 MB | 68 MB |
| Session transcripts | ~100,000 | 300 MB | 450 MB |
| **Total** | **~115,500** | **346 MB** | **~520 MB** |
| LEANN (97% compression) | — | — | **~16 MB** |
| **Savings** | | | **~504 MB** |

Half a GB on a 256 GB Mac Mini. This matters.

**Scenario C — Power user (1 year, everything indexed, 1536d embeddings)**

| Component | Chunks | Embedding (1536d) | With overhead |
|---|---|---|---|
| MEMORY.md | ~500 | 3 MB | 4.5 MB |
| Daily logs (365 days) | ~20,000 | 120 MB | 180 MB |
| Session transcripts | ~250,000 | 1.5 GB | 2.25 GB |
| Extra paths (notes/docs) | ~50,000 | 300 MB | 450 MB |
| Code repos | ~100,000 | 600 MB | 900 MB |
| **Total** | **~420,500** | **2.5 GB** | **~3.8 GB** |
| LEANN (97% compression) | — | — | **~114 MB** |
| **Savings** | | | **~3.7 GB** |

3.7 GB freed on a 256 GB Mac Mini = 1.4% of total disk. Combined with
OpenClaw's 20 GB base + Ollama models (8-18 GB), every GB counts.

#### 5.1.6  Growth Trajectory

The critical insight: **embedding storage grows linearly and never shrinks**.
OpenClaw has no garbage collection for embeddings — old daily logs stay
indexed forever (temporal decay affects ranking, not storage). And extra
indexed paths can add an unpredictable one-time spike at any moment
(user decides to index their 5 GB Obsidian vault or email archive).

```
Embedding index size over time (without LEANN):

              Memory only          + Sessions           + Extra paths (5 GB docs)
              ───────────          ───────────          ────────────────────────
Month 1:      ██  28 MB            ███  78 MB           ████████████████  6.1 GB
Month 6:      ████  55 MB          ████████  330 MB     ████████████████  6.4 GB
Year 1:       ████████  110 MB     ████████████  520 MB █████████████████ 6.6 GB
Year 2:       ████████████  220 MB ████████████████ 1 GB████████████████████ 7.1 GB
Year 3:       ████████████████ 330 MB                    ██████████████████████ 7.6 GB+

Same scenarios with LEANN (97% compression):

Month 1:      █  0.8 MB            █  2.3 MB            ██  183 MB
Year 1:       █  3.3 MB            █  16 MB             ███  198 MB
Year 3:       █  10 MB             ██  48 MB            ████  228 MB
```

The extra-paths scenario is the killer use case: a single `qmd collection add
~/Documents` can spike embedding storage by **6+ GB overnight**. With LEANN,
that same operation costs **~180 MB**. This is the difference between
"impossible on a 256 GB Mac Mini" and "barely noticeable".

### 5.2  Storage Comparison Summary

| | memory-core (768d) | QMD (1024d) | memsearch (1536d) | **LEANN (384d, compressed)** |
|---|---|---|---|---|
| 10K chunks | 46 MB | 61 MB | 92 MB | **~1.4 MB** |
| 100K chunks | 460 MB | 614 MB | 922 MB | **~14 MB** |
| 500K chunks | 2.3 GB | 3.1 GB | 4.6 GB | **~69 MB** |
| 1M chunks | 4.6 GB | 6.1 GB | 9.2 GB | **~138 MB** |

LEANN's advantage compounds with scale. At 1M chunks, the gap is **67x**
smaller than the nearest competitor.

### 5.2  Latency Budget

OpenClaw memory_search pipeline:
```
User message → Agent turn → memory_search tool call → [LEANN search] → Agent continues → LLM response
                                                       ^^^^^^^^^^^^
                                                       Budget: configurable
                                                       Default: 4000ms
                                                       LEANN target: <2000ms
```

LEANN search latency breakdown (warm daemon):
- ZMQ query embedding: ~50ms (single 384d vector)
- HNSW graph traversal + neighbor recomputation: ~500-1500ms (depends on efSearch)
- Total: **~600-1600ms** (well within 4s default)

Cold start (first search after boot):
- Model loading: ~3-5s (sentence-transformers all-MiniLM-L6-v2)
- Solution: daemon mode keeps model in memory; auto-start on OpenClaw gateway boot

### 5.3  Python ↔ TypeScript Bridge

For Phase 3 plugin, the TypeScript plugin shells out to Python:

Option A: **CLI subprocess** (`child_process.execFile("leann", ["search", ...])`)
- Simplest, works with existing `leann` CLI
- ~100ms overhead per subprocess spawn

Option B: **Long-running daemon + HTTP/ZMQ**
- LEANN daemon stays running, TypeScript talks via HTTP API
- Lower per-query overhead, but more complex setup

Option C: **MCP bridge**
- LEANN already has `leann_mcp` MCP server
- OpenClaw supports MCP client connections
- Most architecturally clean, but requires MCP plumbing

**Recommendation**: Start with Option A (CLI subprocess) for simplicity.
Move to Option B if latency matters.

---

## 6  Execution Roadmap

### Week 1: Phase 1 Skill + Benchmarks
- [ ] Create `leann-memory` skill directory with SKILL.md
- [ ] Test indexing real OpenClaw memory files (~1 year of daily logs)
- [ ] Benchmark: storage size, search quality (recall@5), latency
- [ ] Compare against memory-core and QMD on same dataset
- [ ] Write blog post with benchmark results

### Week 2: Marketing Launch
- [ ] Publish blog post (dev.to / Medium)
- [ ] Reddit r/openclaw post
- [ ] Submit to ClawHub marketplace
- [ ] Add OpenClaw integration section to LEANN README
- [ ] Engage in OpenClaw GitHub issues (#19913, #24624)

### Week 3-4: Phase 2 Auto-Sync
- [ ] Implement file watcher for memory directory
- [ ] Add debounced incremental re-indexing
- [ ] Test with live OpenClaw instance
- [ ] Update ClawHub listing

### Month 2: Phase 3 Plugin (if Phase 1/2 gain traction)
- [ ] Develop TypeScript plugin shell
- [ ] Implement `memory_search` / `memory_get` tool registration
- [ ] Background service for auto-sync
- [ ] Publish as npm package `@leann/openclaw-memory`
- [ ] Submit PR to OpenClaw for official integration consideration

---

## 7  Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OpenClaw improves memory-core to use compression | Low | High | Move fast, establish presence before they do |
| Users don't care about storage compression at small scale | Medium | Medium | Target heavy users; market the "grows over time" angle |
| LEANN recomputation too slow for some queries | Low | Medium | Daemon mode; 384d model is fast; timeoutMs is configurable |
| Python dependency friction for TypeScript/Node.js users | Medium | Medium | Phase 1 skill requires just `pip install`; clear setup docs |
| QMD or memsearch adds compression | Low | High | Our compression is fundamental architecture, not easy to bolt on |

---

## 8  Success Metrics

- **Phase 1**: 100+ ClawHub installs in first month
- **Phase 2**: 500+ ClawHub installs; mentioned in 3+ community threads
- **Phase 3**: Included in OpenClaw docs as alternative memory backend
- **Long-term**: LEANN GitHub stars increase by 1000+ from OpenClaw traffic
