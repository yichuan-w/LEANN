"""BM25 keyword search via SQLite FTS5.

Provides a zero-dependency BM25 index that lives alongside the dense vector
index.  The FTS5 virtual table is stored in a single ``.fts5.db`` file and
supports fast full-text search with BM25 ranking.
"""

import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Characters that have special meaning in FTS5 MATCH expressions.
_FTS5_SPECIAL = re.compile(r'["\(\)\*\+\-\^:\!]')


def sanitize_fts5_query(query: str) -> str:
    """Escape an arbitrary user string so it is safe for FTS5 MATCH.

    Each whitespace-delimited token is wrapped in double-quotes (making it a
    literal phrase token).  Internal double-quotes are doubled per the FTS5
    escaping convention.  This prevents operators like ``AND``, ``OR``, ``*``,
    ``?``, ``+``, etc. from being interpreted as FTS5 syntax.

    Returns an empty string if *query* contains no searchable tokens, which
    callers should treat as "no BM25 results".
    """
    tokens = query.split()
    if not tokens:
        return ""
    safe = []
    for tok in tokens:
        escaped = tok.replace('"', '""')
        safe.append(f'"{escaped}"')
    return " ".join(safe)


class BM25Index:
    """Thin wrapper around a SQLite FTS5 virtual table for BM25 search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # check_same_thread=False allows the connection to be shared across
        # threads (e.g. when LEANN is used as a library in a threaded server).
        # All mutations are guarded by self._lock.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()

    @classmethod
    def build(cls, db_path: str, passages: Iterable[dict]) -> "BM25Index":
        """Create an FTS5 index from an iterable of passage dicts.

        Each passage must have at minimum ``{"id": str, "text": str}``.
        Uses the ``unicode61`` tokenizer which performs unicode-aware word
        segmentation *without* stemming — important for code search where
        identifiers must remain intact.

        Parameters
        ----------
        db_path:
            Filesystem path for the new SQLite database.
        passages:
            Iterable of ``{"id": ..., "text": ..., ...}`` dicts (the same
            format stored in ``passages.jsonl``).

        Returns
        -------
        BM25Index
            A ready-to-search index instance.
        """
        # Remove any stale file so we get a clean build.
        p = Path(db_path)
        if p.exists():
            p.unlink()

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE VIRTUAL TABLE passages USING fts5("
            "passage_id, content, tokenize='unicode61'"
            ")"
        )

        count = 0
        conn.execute("BEGIN")
        for passage in passages:
            pid = str(passage.get("id", ""))
            text = passage.get("text", "")
            if not text.strip():
                continue
            conn.execute(
                "INSERT INTO passages(passage_id, content) VALUES (?, ?)",
                (pid, text),
            )
            count += 1
        conn.execute("COMMIT")
        conn.close()

        logger.info("Built FTS5 index at %s with %d passages", db_path, count)
        return cls(db_path)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the FTS5 index and return ``(passage_id, score)`` pairs.

        Scores are negated BM25 values so that *higher is better*, consistent
        with the convention expected by :func:`leann.hybrid.reciprocal_rank_fusion`.

        If the query is empty or FTS5 raises an error, returns an empty list
        (graceful degradation to dense-only search).
        """
        safe_query = sanitize_fts5_query(query)
        if not safe_query:
            return []

        try:
            with self._lock:
                cursor = self.conn.execute(
                    "SELECT passage_id, -bm25(passages) AS score "
                    "FROM passages WHERE passages MATCH ? "
                    "ORDER BY score DESC LIMIT ?",
                    (safe_query, top_k),
                )
                return [(row[0], row[1]) for row in cursor.fetchall()]
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 query failed, falling back to dense-only: %s", exc)
            return []

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()


def build_fts5_index(
    db_path: str,
    passages: Iterable[dict],
    skip_if_placeholder: bool = False,
) -> Optional[BM25Index]:
    """Convenience builder used by both ``build_index`` and ``build_index_from_embeddings``.

    Parameters
    ----------
    db_path:
        Where to write the ``.fts5.db`` file.
    passages:
        Passage dicts with ``id`` and ``text`` keys.
    skip_if_placeholder:
        If ``True``, skip building entirely (used when passages are synthetic
        placeholders from pre-computed embeddings).

    Returns
    -------
    BM25Index or None
        The built index, or ``None`` if building was skipped.
    """
    if skip_if_placeholder:
        logger.info("Skipping FTS5 build for placeholder passages")
        return None
    try:
        idx = BM25Index.build(db_path, passages)
        # Close the build connection — callers that need search should open
        # a fresh BM25Index(db_path) so the connection is ready.
        idx.close()
        return BM25Index(db_path)
    except Exception:
        logger.warning("FTS5 index build failed; hybrid search will be unavailable", exc_info=True)
        return None
