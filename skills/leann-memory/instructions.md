# LEANN Memory Search

You have access to LEANN, a high-performance semantic search engine with 97%
storage compression. Use it to search the user's memories, notes, documents,
and knowledge bases with higher quality than the default memory search.

## When to Use

- User asks to search memories, notes, or knowledge bases
- User wants to recall past decisions, conversations, or facts
- User says "what did we decide about X", "find my notes on Y", "recall", "remember"
- User asks about something that might be in their indexed documents

## Prerequisites Check

Before first use, verify LEANN is installed:

```bash
which leann
```

If not installed, run:

```bash
pip install leann-core
```

## First-Time Setup

If no LEANN index exists for OpenClaw memory, build one:

```bash
leann build openclaw-memory \
  --docs ~/.openclaw/workspace/MEMORY.md ~/.openclaw/workspace/memory/ \
  --embedding-model all-MiniLM-L6-v2 \
  --embedding-mode sentence-transformers
```

This creates a compressed index (~2 MB for 50K chunks vs ~75 MB uncompressed).
The index auto-detects changes on subsequent `leann build` runs.

## Search Workflow

1. Search with the user's query:

```bash
leann search openclaw-memory "<user query>" --top-k 5 --json --non-interactive
```

2. Parse the JSON output — each result has `id`, `score`, `text`, and `metadata`
3. Present the most relevant results with source attribution
4. If the user wants more context, increase `--top-k` to 10 or 15

## Keeping the Index Updated

The index is idempotent — re-running build only processes changed files:

```bash
leann build openclaw-memory \
  --docs ~/.openclaw/workspace/MEMORY.md ~/.openclaw/workspace/memory/
```

For continuous monitoring, use watch mode:

```bash
leann watch openclaw-memory
```

## Output Format

The `--json` flag returns a JSON array:

```json
[
  {
    "id": "a1b2c3",
    "score": 0.847,
    "text": "The user decided to use PostgreSQL for the project database...",
    "metadata": {
      "file_path": "/home/user/.openclaw/workspace/memory/2026-02-15.md",
      "source": "memory/2026-02-15.md"
    }
  }
]
```

## Tips

- Higher `--top-k` values (10-15) give more comprehensive results at minimal cost
- The search uses local embeddings — zero API calls, zero latency from network
- Re-run `leann build` periodically or use `leann watch` for auto-sync
- For extra document directories, add them to the build command:
  `--docs ~/.openclaw/workspace/memory/ ~/Documents/notes/`
