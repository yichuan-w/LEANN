# How to Use LEANN - Step by Step Guide

## Prerequisites
Make sure you're in the LEANN directory and have the virtual environment activated:

```bash
cd ~/LEANN
source .venv/bin/activate
```

---

## 1. Search an Existing Index

**Basic search:**
```bash
leann search my-docs "your search query here"
```

**Search with more results:**
```bash
leann search my-docs "vector database" --top-k 10
```

**Example:**
```bash
leann search my-docs "machine learning"
```

---

## 2. Ask Questions (Chat with Your Data)

**Single question:**
```bash
leann ask my-docs "What is LEANN?" --model gemma3:4b
```

**Interactive chat (recommended):**
```bash
leann ask my-docs --interactive --model gemma3:4b
```
This opens an interactive session where you can ask multiple questions. Type `exit` or `quit` to end.

---

## 3. Build a New Index from Your Documents

### Step 1: Prepare your documents
Create a folder with your documents (PDF, TXT, MD, etc.):
```bash
mkdir -p ~/my-documents
# Copy your files there
cp ~/Documents/*.pdf ~/my-documents/
cp ~/Documents/*.txt ~/my-documents/
```

### Step 2: Build the index
```bash
leann build my-new-index --docs ~/my-documents
```

**Build from multiple directories:**
```bash
leann build my-code-index --docs ./src ./docs ./README.md
```

**Build with specific file types:**
```bash
leann build my-pdfs --docs ./ --file-types .pdf,.docx
```

**Build with custom settings:**
```bash
leann build my-index --docs ./documents \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --backend hnsw
```

### Step 3: Verify your index was created
```bash
leann list
```
You should see your new index in the list.

### Step 4: Use your new index
```bash
leann search my-new-index "your query"
leann ask my-new-index "question" --model gemma3:4b
```

---

## 4. List All Your Indexes

```bash
leann list
```

Shows all indexes with their sizes and status.

---

## 5. Remove an Index

```bash
leann remove my-old-index
```

---

## 6. Use Python API (Advanced)

Create a Python script:

```python
from leann import LeannBuilder, LeannSearcher, LeannChat
from pathlib import Path

# Build an index
builder = LeannBuilder(backend_name="hnsw")
builder.add_text("Your first document text here.")
builder.add_text("Your second document text here.")
builder.build_index("my_index.leann")

# Search
searcher = LeannSearcher("my_index.leann")
results = searcher.search("your query", top_k=5)
for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.text}")

# Chat
chat = LeannChat("my_index.leann", llm_config={"type": "ollama", "model": "gemma3:4b"})
response = chat.ask("Your question here")
print(response)
```

Run it:
```bash
python your_script.py
```

---

## Common Use Cases

### Index Your Codebase
```bash
leann build my-codebase --docs ./src ./tests ./docs
leann search my-codebase "authentication function"
```

### Index Your Documents
```bash
leann build my-docs --docs ~/Documents
leann ask my-docs "What did I write about project X?" --model gemma3:4b
```

### Index Specific File Types
```bash
leann build my-pdfs --docs ~/Documents --file-types .pdf
leann search my-pdfs "research findings"
```

---

## Troubleshooting

**If you get "Model not found" error:**
- Use `--model gemma3:4b` flag (you have this installed)
- Or install the model: `ollama pull qwen3:8b`

**If search is slow:**
- Try reducing `--top-k` value
- Use a smaller embedding model

**If you run out of memory:**
- Use `--embedding-model sentence-transformers/all-MiniLM-L6-v2` (smaller model)
- Build indexes in smaller batches

---

## Quick Reference

| Task | Command |
|------|---------|
| List indexes | `leann list` |
| Search | `leann search INDEX_NAME "query"` |
| Ask question | `leann ask INDEX_NAME "question" --model gemma3:4b` |
| Interactive chat | `leann ask INDEX_NAME --interactive --model gemma3:4b` |
| Build index | `leann build INDEX_NAME --docs ./path/to/documents` |
| Remove index | `leann remove INDEX_NAME` |

---

## Next Steps

1. Try searching your existing `my-docs` index
2. Build an index from a folder of your documents
3. Use interactive chat to explore your data
4. Check out `demo.py` and `COMPREHENSIVE_GUIDE.py` for more examples
