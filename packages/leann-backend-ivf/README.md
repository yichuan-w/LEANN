# LEANN IVF Backend

FAISS **IndexIVFFlat** backend for LEANN with **add** and **delete** APIs for incremental updates (#231, #89, #141).

## Install

```bash
uv sync --package leann-backend-ivf
# or
pip install leann-backend-ivf
```

Requires `faiss-cpu`. For query embedding during search, the IVF searcher uses the same embedding server as HNSW; install `leann-backend-hnsw` if you use recompute/query embedding.

## Build

```bash
leann build my-index --docs ./src --backend-name ivf
```

Optional backend kwargs (e.g. via API): `nlist` (default 100), `distance_metric` ("l2" or "cosine"/"mips").

## Add / Delete API

Follows the same idea as HNSW’s update path; use these for incremental workflows (e.g. after Merkle tree / file-change API):

- **`add_vectors(index_path, embeddings, passage_ids)`**
  Appends vectors to an existing IVF index. `embeddings`: `(N, D)` float32; `passage_ids`: list of N passage id strings (must not already exist).

- **`remove_ids(index_path, passage_ids)`**
  Removes vectors by passage id. Returns the number of vectors removed. Use after detecting changed chunk ids: delete old, then re-insert new.

Example:

```python
from leann_backend_ivf import add_vectors, remove_ids

# After file-change API says chunk id "abc123" changed:
remove_ids("/path/to/index.leann", ["abc123"])
# Re-embed and add new chunk with same or new id
add_vectors("/path/to/index.leann", new_embeddings, ["abc123"])
```

## Core integration (#89, #141)

`LeannBuilder.update_index(index_path, remove_passage_ids=None)` in leann-core supports IVF: it calls `add_vectors` for appends and, when `remove_passage_ids` is set, calls `remove_ids` first (e.g. from a file-change / Merkle API). Use for incremental build or reindex: detect changed chunk ids → `update_index(path, remove_passage_ids=changed_ids)` then add chunks for the new content.

## Notes

- IVF stores vectors in flat lists per cluster; no compact/recompute mode like HNSW. Good for frequent add/delete.
- ID mapping is stored in `ivf_id_map.json` next to the index; passage ids are stable across add/delete.
- Search uses `nprobe` (default from `complexity`, capped by `nlist`). Tune `nlist` at build time for speed/recall.
