# OpenClaw + LEANN Setup Guide

Two ways to connect LEANN to your OpenClaw agent: **MCP server** (recommended)
or **ClawHub skill**.

---

## Option A: MCP Server (Recommended)

OpenClaw natively supports MCP tools. LEANN ships an MCP server that exposes
`leann_search` and `leann_list` as tools your agent can call directly.

### 1. Install LEANN

```bash
pip install leann-core
# or
uv tool install leann-core --with leann
```

### 2. Build an index on your memory files

Using Ollama embeddings (recommended if you already run Ollama):

```bash
leann build openclaw-memory \
  --docs ~/.openclaw/workspace/MEMORY.md ~/.openclaw/workspace/memory/ \
  --embedding-mode ollama \
  --embedding-model nomic-embed-text
```

Or using local sentence-transformers (no Ollama required):

```bash
leann build openclaw-memory \
  --docs ~/.openclaw/workspace/MEMORY.md ~/.openclaw/workspace/memory/ \
  --embedding-mode sentence-transformers \
  --embedding-model all-MiniLM-L6-v2
```

Add extra directories if you have them:

```bash
leann build openclaw-memory \
  --docs ~/.openclaw/workspace/MEMORY.md \
        ~/.openclaw/workspace/memory/ \
        ~/Documents/notes/ \
  --embedding-mode ollama \
  --embedding-model nomic-embed-text
```

### 3. Register the MCP server with OpenClaw

Add to `~/.openclaw/openclaw.json`:

```json5
{
  // ... your existing config ...
  "mcpServers": {
    "leann": {
      "command": "leann_mcp",
      "args": [],
      "env": {}
    }
  }
}
```

### 4. Use it

Ask your agent:
- "Search my memories for database decisions"
- "What did we decide about the API design?"
- "Find my notes on deployment"

The agent will call `leann_search` via MCP and return structured results.

### 5. Keep the index fresh

```bash
# Re-run build (idempotent — only processes changed files)
leann build openclaw-memory \
  --docs ~/.openclaw/workspace/MEMORY.md ~/.openclaw/workspace/memory/

# Or use watch mode for continuous auto-sync
leann watch openclaw-memory --interval 30
```

---

## Option B: ClawHub Skill

If you prefer the skill-based approach:

```bash
clawhub install leann-team/leann-memory
```

Or copy `skills/leann-memory/` from this repo to
`~/.openclaw/workspace/skills/leann-memory/`.

The skill tells your agent how to call `leann search` via shell commands.
Setup steps (install + build index) are the same as above.

---

## Important: Ollama Configuration

If you use Ollama as your OpenClaw model provider, make sure your
`~/.openclaw/openclaw.json` uses the **native Ollama API** — not the
OpenAI-compatible endpoint:

```json5
{
  "models": {
    "providers": {
      "ollama": {
        "baseUrl": "http://127.0.0.1:11434",  // no /v1 suffix
        "apiKey": "ollama-local",
        "api": "ollama"  // NOT "openai-completions" or "openai-responses"
      }
    }
  }
}
```

Using `"openai-completions"` or `"openai-responses"` silently breaks tool
calling — the model outputs tool calls as plain text instead of structured
`tool_calls`. See [astral-sh/ty#21243](https://github.com/openclaw/openclaw/issues/21243).

---

## Storage Comparison

| Scenario | Default memory-core | LEANN |
|---|---|---|
| 1 year daily logs (~12K chunks) | ~23 MB | **~0.7 MB** |
| + session transcripts (~100K chunks) | ~190 MB | **~6 MB** |
| + 10 GB indexed documents (~500K chunks) | ~950 MB | **~30 MB** |

All numbers assume 384-dimensional embeddings (all-MiniLM-L6-v2 or
nomic-embed-text).

---

## Troubleshooting

**"leann: command not found"**
Ensure LEANN is on your PATH. If installed via `uv tool install`, run
`uv tool update-shell` and restart your terminal.

**"Index not found"**
Run `leann list` to see available indexes. Build one first with `leann build`.

**Slow first search**
The first query loads the embedding model (~90 MB). Subsequent queries reuse the
warm daemon and are fast (~0.5s). Use `leann warmup openclaw-memory` to
pre-warm.

**Memory files changed but search results are stale**
Re-run `leann build openclaw-memory --docs ...` — it detects changes
automatically and only re-indexes what changed.

**Agent doesn't use LEANN tools**
Make sure your Ollama model supports tool calling (e.g. `qwen3:8b` or larger).
Smaller models like `qwen3:4b` may not reliably invoke tools.
