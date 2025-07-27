# FAQ

## 1. My building time seems long

You can speed up the process by using a lightweight embedding model. Add this to your arguments:

```bash
--embedding-model sentence-transformers/all-MiniLM-L6-v2
```
**Model sizes:** `all-MiniLM-L6-v2` (30M parameters), `facebook/contriever` (~100M parameters), `Qwen3-0.6B` (600M parameters)
