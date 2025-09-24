from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_repo_paths_importable(current_file: str) -> None:
    _repo_root = Path(current_file).resolve().parents[3]
    _leann_core_src = _repo_root / "packages" / "leann-core" / "src"
    _leann_hnsw_pkg = _repo_root / "packages" / "leann-backend-hnsw"
    if str(_leann_core_src) not in sys.path:
        sys.path.append(str(_leann_core_src))
    if str(_leann_hnsw_pkg) not in sys.path:
        sys.path.append(str(_leann_hnsw_pkg))


_ensure_repo_paths_importable(__file__)

from leann_backend_hnsw.hnsw_backend import HNSWBuilder, HNSWSearcher  # noqa: E402


class LeannMultiVector:
    def __init__(
        self,
        index_path: str,
        dim: int = 128,
        distance_metric: str = "mips",
        m: int = 16,
        ef_construction: int = 500,
        is_compact: bool = False,
        is_recompute: bool = False,
        embedding_model_name: str = "colvision",
    ) -> None:
        self.index_path = index_path
        self.dim = dim
        self.embedding_model_name = embedding_model_name
        self._pending_items: list[dict] = []
        self._backend_kwargs = {
            "distance_metric": distance_metric,
            "M": m,
            "efConstruction": ef_construction,
            "is_compact": is_compact,
            "is_recompute": is_recompute,
        }
        self._labels_meta: list[dict] = []

    def _meta_dict(self) -> dict:
        return {
            "version": "1.0",
            "backend_name": "hnsw",
            "embedding_model": self.embedding_model_name,
            "embedding_mode": "custom",
            "dimensions": self.dim,
            "backend_kwargs": self._backend_kwargs,
            "is_compact": self._backend_kwargs.get("is_compact", True),
            "is_pruned": self._backend_kwargs.get("is_compact", True)
            and self._backend_kwargs.get("is_recompute", True),
        }

    def create_collection(self) -> None:
        path = Path(self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def insert(self, data: dict) -> None:
        self._pending_items.append(
            {
                "doc_id": int(data["doc_id"]),
                "filepath": data.get("filepath", ""),
                "colbert_vecs": [np.asarray(v, dtype=np.float32) for v in data["colbert_vecs"]],
            }
        )

    def _labels_path(self) -> Path:
        index_path_obj = Path(self.index_path)
        return index_path_obj.parent / f"{index_path_obj.name}.labels.json"

    def _meta_path(self) -> Path:
        index_path_obj = Path(self.index_path)
        return index_path_obj.parent / f"{index_path_obj.name}.meta.json"

    def create_index(self) -> None:
        if not self._pending_items:
            return

        embeddings: list[np.ndarray] = []
        labels_meta: list[dict] = []

        for item in self._pending_items:
            doc_id = int(item["doc_id"])
            filepath = item.get("filepath", "")
            colbert_vecs = item["colbert_vecs"]
            for seq_id, vec in enumerate(colbert_vecs):
                vec_np = np.asarray(vec, dtype=np.float32)
                embeddings.append(vec_np)
                labels_meta.append(
                    {
                        "id": f"{doc_id}:{seq_id}",
                        "doc_id": doc_id,
                        "seq_id": int(seq_id),
                        "filepath": filepath,
                    }
                )

        if not embeddings:
            return

        embeddings_np = np.vstack(embeddings).astype(np.float32)
        # print shape of embeddings_np
        print(embeddings_np.shape)

        builder = HNSWBuilder(**{**self._backend_kwargs, "dimensions": self.dim})
        ids = [str(i) for i in range(embeddings_np.shape[0])]
        builder.build(embeddings_np, ids, self.index_path)

        import json as _json

        with open(self._meta_path(), "w", encoding="utf-8") as f:
            _json.dump(self._meta_dict(), f, indent=2)
        with open(self._labels_path(), "w", encoding="utf-8") as f:
            _json.dump(labels_meta, f)

        self._labels_meta = labels_meta

    def _load_labels_meta_if_needed(self) -> None:
        if self._labels_meta:
            return
        labels_path = self._labels_path()
        if labels_path.exists():
            import json as _json

            with open(labels_path, encoding="utf-8") as f:
                self._labels_meta = _json.load(f)

    def search(
        self, data: np.ndarray, topk: int, first_stage_k: int = 50
    ) -> list[tuple[float, int]]:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        self._load_labels_meta_if_needed()

        searcher = HNSWSearcher(self.index_path, meta=self._meta_dict())
        raw = searcher.search(
            data,
            first_stage_k,
            recompute_embeddings=False,
            complexity=128,
            beam_width=1,
            prune_ratio=0.0,
            batch_size=0,
        )

        labels = raw.get("labels")
        distances = raw.get("distances")
        if labels is None or distances is None:
            return []

        doc_scores: dict[int, float] = {}
        B = len(labels)
        for b in range(B):
            per_doc_best: dict[int, float] = {}
            for k, sid in enumerate(labels[b]):
                try:
                    idx = int(sid)
                except Exception:
                    continue
                if 0 <= idx < len(self._labels_meta):
                    doc_id = int(self._labels_meta[idx]["doc_id"])  # type: ignore[index]
                else:
                    continue
                score = float(distances[b][k])
                if (doc_id not in per_doc_best) or (score > per_doc_best[doc_id]):
                    per_doc_best[doc_id] = score
            for doc_id, best_score in per_doc_best.items():
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + best_score

        scores = sorted(((v, k) for k, v in doc_scores.items()), key=lambda x: x[0], reverse=True)
        return scores[:topk] if len(scores) >= topk else scores
