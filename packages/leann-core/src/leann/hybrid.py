"""Score fusion utilities for hybrid (dense + BM25) search."""


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 10,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each input list contains ``(document_id, score)`` tuples sorted
    best-first.  The original scores are ignored — only *rank position*
    matters.  This makes RRF robust to differing score distributions
    (e.g. FAISS distances vs. BM25 scores).

    Parameters
    ----------
    ranked_lists:
        One or more result lists, each a sequence of ``(id, score)`` pairs
        in descending relevance order.
    k:
        RRF smoothing constant.  Standard value for web-scale fusion is 60,
        but for small result sets (5–30 items) a lower value like 10 gives
        better rank discrimination.
    top_k:
        Maximum number of fused results to return.

    Returns
    -------
    list[tuple[str, float]]
        ``(id, rrf_score)`` pairs sorted by fused score, descending.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _original_score) in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]
