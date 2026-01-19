# LEANN Setup Fixes Applied

## Issue 1: Segmentation Fault in COMPREHENSIVE_GUIDE.py ✅ FIXED
**Problem:** DiskANN backend was causing segmentation faults on your system.

**Solution:** Changed DiskANN backend to HNSW in COMPREHENSIVE_GUIDE.py (line 125).
- HNSW is more stable and still provides excellent performance
- Maximum storage savings (97%+)
- Full recomputation support

## Issue 2: Missing Ollama Model ✅ SOLUTION PROVIDED
**Problem:** Default model `qwen3:8b` is not installed, only `gemma3:4b` is available.

**Solution:** Use `--model` flag to specify the available model:

```bash
# Instead of:
leann ask my-docs --interactive

# Use:
leann ask my-docs --interactive --model gemma3:4b
```

Or install the requested model:
```bash
ollama pull qwen3:8b
```

## Quick Commands Reference

### Search indexes:
```bash
leann search my-docs "your query"
```

### Ask questions (with correct model):
```bash
leann ask my-docs "your question" --model gemma3:4b
```

### Interactive chat:
```bash
leann ask my-docs --interactive --model gemma3:4b
```

### Build new index:
```bash
leann build my-index --docs ./path/to/documents
```

## Available Ollama Models
- `gemma3:4b` ✅ (installed)
- `qwen3:8b` (needs: `ollama pull qwen3:8b`)
- `llama3:latest` (needs: `ollama pull llama3:latest`)

