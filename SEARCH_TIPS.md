# LEANN Search Tips & Troubleshooting

## Your Search is Loading

When you see:
```
[read_HNSW - CSR NL v4] Reading metadata & CSR indices...
ZmqDistanceComputer initialized: d=768, metric=0
```

This means:
- ✅ Index is loading successfully
- ✅ Embedding server is initializing
- ⏳ It may take 10-30 seconds to start the embedding server on first use

**Wait for it to complete** - you should see search results after the server starts.

## If Search Hangs or Takes Too Long

### Option 1: Wait it out (first time)
The first search after starting LEANN takes longer because it needs to:
1. Load the index
2. Start the embedding server
3. Compute embeddings for your query

Subsequent searches will be much faster.

### Option 2: Use Python API (faster, more control)

Create a file `quick_search.py`:

```python
#!/usr/bin/env python3
from leann import LeannSearcher
import sys

index_path = ".leann/indexes/my-docs/documents.leann"
query = sys.argv[1] if len(sys.argv) > 1 else "project X"

print(f"Searching for: {query}\n")
searcher = LeannSearcher(index_path)
results = searcher.search(query, top_k=10)

print(f"Found {len(results)} results:\n")
for i, r in enumerate(results, 1):
    print(f"{i}. Score: {r.score:.3f}")
    print(f"   {r.text[:200]}...")
    print()
```

Run it:
```bash
cd ~/LEANN
source .venv/bin/activate
python quick_search.py "project X"
```

### Option 3: Check if embedding server is running

```bash
lsof -i :5557
```

If you see a process, the server is running. If not, it's still starting.

## Fix Permission Error for Interactive Mode

Run this in your terminal:

```bash
cd ~/LEANN
./fix_history_permissions.sh
```

Or manually:
```bash
rm -f ~/.leann_history
touch ~/.leann_history
chmod 644 ~/.leann_history
```

Then try interactive mode again:
```bash
leann ask my-docs --interactive --model gemma3:4b
```

## Quick Commands That Work

**1. Simple search (wait for results):**
```bash
leann search my-docs "project X" --top-k 5
```

**2. Non-interactive ask (no permission issues):**
```bash
leann ask my-docs "What did I write about project X?" --model gemma3:4b
```

**3. Python search (fastest, most reliable):**
```python
from leann import LeannSearcher
searcher = LeannSearcher(".leann/indexes/my-docs/documents.leann")
results = searcher.search("project X", top_k=10)
for r in results:
    print(f"{r.score:.3f}: {r.text[:150]}")
```

## Expected Behavior

- **First search**: 10-30 seconds (server startup)
- **Subsequent searches**: 1-3 seconds
- **Index loading**: 1-2 seconds
- **Embedding computation**: 1-2 seconds per query

## If Nothing Works

Check the index is valid:
```bash
leann list
```

You should see `my-docs ✅` in the list.

If the index shows `❌`, rebuild it:
```bash
leann build my-docs --docs ~/Documents --file-types .pdf,.txt,.md --force
```
