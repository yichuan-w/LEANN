"""Score fusion utilities for hybrid (dense + BM25) search."""


def _min_max_normalize(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Normalize scores to [0, 1] via min-max scaling."""
    if not items:
        return []
    scores = [s for _, s in items]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span == 0:
        return [(doc_id, 1.0) for doc_id, _ in items]
    return [(doc_id, (s - lo) / span) for doc_id, s in items]


def weighted_score_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Fuse dense and sparse results using weighted score combination.

    Both result lists are min-max normalized to [0, 1] before combining so
    that scores from different systems (e.g. FAISS distances vs BM25) are
    comparable.

    Parameters
    ----------
    dense_results:
        ``(id, score)`` pairs from the dense (vector) search, higher is better.
    sparse_results:
        ``(id, score)`` pairs from BM25/FTS5 search, higher is better.
    dense_weight:
        Weight for dense scores.
    sparse_weight:
        Weight for sparse/BM25 scores.
    top_k:
        Maximum number of fused results to return.

    Returns
    -------
    list[tuple[str, float]]
        ``(id, fused_score)`` pairs sorted by combined score, descending.
    """
    dense_norm = dict(_min_max_normalize(dense_results))
    sparse_norm = dict(_min_max_normalize(sparse_results))

    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    fused: dict[str, float] = {}
    for doc_id in all_ids:
        d = dense_norm.get(doc_id, 0.0)
        s = sparse_norm.get(doc_id, 0.0)
        fused[doc_id] = dense_weight * d + sparse_weight * s

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
