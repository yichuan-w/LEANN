# Data Source Indexing Commands

LEANN provides `index-*` CLI commands for indexing common personal data sources. Each command reads from a specific data source and builds a searchable LEANN index.

## Available Commands

| Command | Description | Platform |
|---------|-------------|----------|
| `leann index-browser [chrome\|brave]` | Browser history | macOS |
| `leann index-email` | Apple Mail | macOS |
| `leann index-calendar` | Apple Calendar | macOS |
| `leann index-imessage` | iMessage conversations | macOS |
| `leann index-wechat --export-dir <path>` | WeChat chat history | Any |
| `leann index-chatgpt --export-path <path>` | ChatGPT export | Any |
| `leann index-claude --export-path <path>` | Claude export | Any |

## Common Options

All `index-*` commands accept these options:

```bash
--index-name NAME          # Custom index name (each command has a sensible default)
--embedding-model MODEL    # Embedding model (default: facebook/contriever)
--embedding-mode MODE      # Backend: sentence-transformers, openai, mlx, ollama
--max-count N              # Max items to index (default: 1000)
--no-recompute             # Store full embeddings instead of using recomputation
```

## Examples

### Index Chrome browser history

```bash
leann index-browser chrome
leann index-browser brave --index-name brave_history
```

### Index Apple Mail

```bash
leann index-email
```

### Index iMessage

```bash
leann index-imessage
```

### Index Apple Calendar

```bash
leann index-calendar
```

### Index ChatGPT or Claude exports

```bash
# ChatGPT: export from https://chat.openai.com → Settings → Export data
leann index-chatgpt --export-path ~/Downloads/chatgpt-export.zip

# Claude: export from https://claude.ai → Settings → Export data
leann index-claude --export-path ~/Downloads/claude-export.json
```

### Index WeChat

```bash
# Requires exported JSON files from wechat-exporter
leann index-wechat --export-dir ~/wechat-export/
```

## Daily Automation

You can schedule indexing with cron for automatic daily updates:

```bash
# Edit crontab
crontab -e

# Add entries (runs at 2 AM daily):
0 2 * * * cd /path/to/LEANN && leann index-browser chrome
5 2 * * * cd /path/to/LEANN && leann index-email
10 2 * * * cd /path/to/LEANN && leann index-imessage
```

## Searching Indexed Data

After indexing, search with the standard `leann search` command:

```bash
leann search browser_history "github pull request review"
leann search email "meeting notes from last week"
leann search imessage "dinner plans"
```
