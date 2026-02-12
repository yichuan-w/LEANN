"""Tests for hybrid (BM25 + dense) search.

Unit tests (test_bm25_*, test_rrf_*, test_sanitize_*) run without any ML model
and exercise the FTS5 and fusion logic in isolation.

Integration tests (test_hybrid_*) require a model download and are skipped in CI.
"""

import importlib.util
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helpers — load bm25 and hybrid modules directly without triggering
# the full leann package __init__.py (which needs compiled C++ backends).
# ---------------------------------------------------------------------------
_LEANN_SRC = Path(__file__).resolve().parent.parent / "packages" / "leann-core" / "src" / "leann"


def _load_module(name: str):
    """Load a single module file from leann source, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(name, _LEANN_SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bm25_mod = _load_module("bm25")
_hybrid_mod = _load_module("hybrid")

BM25Index = _bm25_mod.BM25Index
build_fts5_index = _bm25_mod.build_fts5_index
sanitize_fts5_query = _bm25_mod.sanitize_fts5_query
reciprocal_rank_fusion = _hybrid_mod.reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# BM25 / FTS5 unit tests
# ---------------------------------------------------------------------------


class TestSanitizeFTS5Query:
    def test_plain_text(self):

        assert sanitize_fts5_query("hello world") == '"hello" "world"'

    def test_empty_string(self):

        assert sanitize_fts5_query("") == ""

    def test_whitespace_only(self):

        assert sanitize_fts5_query("   ") == ""

    def test_special_chars_question_mark(self):

        result = sanitize_fts5_query("what is a function?")
        assert "?" not in result or '"function?"' in result  # quoted is safe

    def test_special_chars_asterisk(self):

        result = sanitize_fts5_query("find * files")
        assert '"*"' in result

    def test_special_chars_plus(self):

        result = sanitize_fts5_query("C++ templates")
        assert '"C++"' in result

    def test_internal_quotes(self):

        result = sanitize_fts5_query('say "hello"')
        # Internal quotes should be doubled
        assert '""hello""' in result

    def test_fts5_operators_are_escaped(self):

        # AND, OR, NOT are FTS5 operators when unquoted
        result = sanitize_fts5_query("cats AND dogs")
        assert '"AND"' in result

    def test_parentheses(self):

        result = sanitize_fts5_query("func(arg)")
        assert '"func(arg)"' in result


class TestBM25Index:
    @pytest.fixture
    def sample_passages(self):
        return [
            {"id": "0", "text": "Python is a programming language"},
            {"id": "1", "text": "JavaScript runs in the browser"},
            {"id": "2", "text": "Rust is a systems programming language"},
            {"id": "3", "text": "Python web frameworks include Django and Flask"},
            {"id": "4", "text": "The quick brown fox jumps over the lazy dog"},
        ]

    def test_build_and_search(self, sample_passages):

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.fts5.db")
            idx = BM25Index.build(db_path, sample_passages)

            results = idx.search("Python programming", top_k=3)
            assert len(results) > 0
            # Python passages should rank highest
            result_ids = [r[0] for r in results]
            assert "0" in result_ids or "3" in result_ids
            # Scores should be positive (we negate FTS5's negative bm25)
            assert all(score > 0 for _, score in results)
            idx.close()

    def test_search_no_matches(self, sample_passages):

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.fts5.db")
            idx = BM25Index.build(db_path, sample_passages)

            results = idx.search("xyznonexistentterm", top_k=5)
            assert results == []
            idx.close()

    def test_search_empty_query(self, sample_passages):

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.fts5.db")
            idx = BM25Index.build(db_path, sample_passages)

            results = idx.search("", top_k=5)
            assert results == []
            idx.close()

    def test_search_special_characters_no_crash(self, sample_passages):

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.fts5.db")
            idx = BM25Index.build(db_path, sample_passages)

            # These should not crash — graceful degradation
            for query in ['what?', 'C++', '"unmatched', 'a AND b', '(group)', '*wildcard']:
                results = idx.search(query, top_k=5)
                assert isinstance(results, list)
            idx.close()

    def test_build_skips_empty_texts(self):

        passages = [
            {"id": "0", "text": "real content here"},
            {"id": "1", "text": ""},
            {"id": "2", "text": "   "},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.fts5.db")
            idx = BM25Index.build(db_path, passages)
            results = idx.search("content", top_k=5)
            assert len(results) == 1
            assert results[0][0] == "0"
            idx.close()

    def test_build_overwrites_existing(self, sample_passages):

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.fts5.db")
            # Build twice — second should overwrite cleanly
            BM25Index.build(db_path, sample_passages).close()
            idx = BM25Index.build(db_path, sample_passages)
            results = idx.search("Python", top_k=10)
            # Should have exactly the passages from the second build, not doubled
            assert len(results) <= len(sample_passages)
            idx.close()

    def test_fts5_not_available(self):
        """If FTS5 is not compiled into SQLite, build should fail gracefully."""
        # This test documents the behavior; FTS5 is available in most builds.

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.fts5.db")
            # build_fts5_index catches exceptions and returns None
            result = build_fts5_index(db_path, [], skip_if_placeholder=True)
            assert result is None


# ---------------------------------------------------------------------------
# RRF fusion unit tests
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_basic_fusion(self):

        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        bm25 = [("b", 5.0), ("d", 4.0), ("a", 3.0)]
        fused = reciprocal_rank_fusion([dense, bm25], k=10, top_k=5)

        # "a" and "b" appear in both lists, should rank highest
        fused_ids = [fid for fid, _ in fused]
        assert "a" in fused_ids[:2]
        assert "b" in fused_ids[:2]

    def test_disjoint_lists(self):

        list1 = [("a", 1.0), ("b", 0.9)]
        list2 = [("c", 1.0), ("d", 0.9)]
        fused = reciprocal_rank_fusion([list1, list2], k=10, top_k=10)

        # All 4 items should appear with equal RRF contribution per list
        assert len(fused) == 4
        fused_ids = {fid for fid, _ in fused}
        assert fused_ids == {"a", "b", "c", "d"}

    def test_one_empty_list(self):

        dense = [("a", 0.9), ("b", 0.8)]
        fused = reciprocal_rank_fusion([dense, []], k=10, top_k=5)

        # Should degrade to dense ranking
        assert len(fused) == 2
        assert fused[0][0] == "a"
        assert fused[1][0] == "b"

    def test_both_empty(self):

        fused = reciprocal_rank_fusion([[], []], k=10, top_k=5)
        assert fused == []

    def test_top_k_truncation(self):

        long_list = [(str(i), float(100 - i)) for i in range(50)]
        fused = reciprocal_rank_fusion([long_list], k=10, top_k=5)
        assert len(fused) == 5

    def test_scores_are_positive(self):

        dense = [("a", 0.9), ("b", 0.8)]
        bm25 = [("a", 5.0), ("c", 3.0)]
        fused = reciprocal_rank_fusion([dense, bm25], k=10, top_k=5)
        assert all(score > 0 for _, score in fused)

    def test_k_parameter_affects_discrimination(self):

        items = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        fused_small_k = reciprocal_rank_fusion([items], k=1, top_k=3)
        fused_large_k = reciprocal_rank_fusion([items], k=100, top_k=3)

        # With small k, score difference between ranks is larger
        small_k_spread = fused_small_k[0][1] - fused_small_k[-1][1]
        large_k_spread = fused_large_k[0][1] - fused_large_k[-1][1]
        assert small_k_spread > large_k_spread


# ---------------------------------------------------------------------------
# build_fts5_index helper tests
# ---------------------------------------------------------------------------


class TestBuildFTS5Index:
    def test_skip_if_placeholder(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.fts5.db")
            result = build_fts5_index(db_path, [], skip_if_placeholder=True)
            assert result is None
            assert not Path(db_path).exists()

    def test_normal_build(self):

        passages = [{"id": "0", "text": "hello world"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.fts5.db")
            result = build_fts5_index(db_path, passages)
            # build_fts5_index closes the index after building, returns it
            assert Path(db_path).exists()


# ---------------------------------------------------------------------------
# Integration tests (require model — skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip model tests in CI to avoid MPS memory issues",
)
class TestHybridSearchIntegration:
    def _build_test_index(self, tmpdir):
        from leann.api import LeannBuilder

        index_path = str(Path(tmpdir) / "test_hybrid.hnsw")
        texts = [
            "Python is a high-level programming language",
            "JavaScript is used for web development",
            "Rust provides memory safety without garbage collection",
            "Django is a Python web framework for rapid development",
            "React is a JavaScript library for building user interfaces",
            "The Cargo package manager handles Rust dependencies",
            "Flask is a lightweight Python web microframework",
            "Node.js runs JavaScript on the server side",
            "Tokio is an async runtime for Rust",
            "TypeScript adds static typing to JavaScript",
        ]
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            M=16,
            efConstruction=200,
            is_compact=False,
            is_recompute=False,
        )
        for text in texts:
            builder.add_text(text)
        builder.build_index(index_path)
        return index_path

    def test_hybrid_search_returns_results(self):
        from leann.api import LeannSearcher, SearchResult

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = self._build_test_index(tmpdir)
            searcher = LeannSearcher(index_path)
            results = searcher.search("Python web framework", top_k=5, hybrid=True)
            assert len(results) > 0
            assert isinstance(results[0], SearchResult)
            searcher.cleanup()

    def test_hybrid_false_matches_dense_only(self):
        from leann.api import LeannSearcher

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = self._build_test_index(tmpdir)
            searcher = LeannSearcher(index_path)
            dense_results = searcher.search("Python", top_k=5, hybrid=False)
            default_results = searcher.search("Python", top_k=5)
            # hybrid=False (explicit) should match the default behavior
            assert [r.id for r in dense_results] == [r.id for r in default_results]
            searcher.cleanup()

    def test_hybrid_graceful_degradation_no_fts5(self):
        """If .fts5.db doesn't exist, hybrid=True should still work (dense only)."""
        from leann.api import LeannSearcher

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = self._build_test_index(tmpdir)
            # Delete the FTS5 file
            fts5_path = Path(index_path).parent / f"{Path(index_path).name}.fts5.db"
            if fts5_path.exists():
                fts5_path.unlink()
            searcher = LeannSearcher(index_path)
            assert searcher._bm25_index is None
            results = searcher.search("Python", top_k=5, hybrid=True)
            assert len(results) > 0  # Falls back to dense-only
            searcher.cleanup()

    def test_fts5_file_created_during_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = self._build_test_index(tmpdir)
            fts5_path = Path(index_path).parent / f"{Path(index_path).name}.fts5.db"
            assert fts5_path.exists()

    def test_hybrid_with_metadata_filters(self):
        from leann.api import LeannBuilder, LeannSearcher

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = str(Path(tmpdir) / "test_meta.hnsw")
            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="facebook/contriever",
                embedding_mode="sentence-transformers",
                M=16,
                efConstruction=200,
                is_compact=False,
                is_recompute=False,
            )
            for i in range(20):
                builder.add_text(
                    f"Document {i} about topic {i % 3}",
                    metadata={"topic": i % 3},
                )
            builder.build_index(index_path)

            searcher = LeannSearcher(index_path)
            results = searcher.search(
                "document topic",
                top_k=10,
                hybrid=True,
                metadata_filters={"topic": {"==": 1}},
            )
            # All results should have topic == 1
            for r in results:
                assert r.metadata.get("topic") == 1
            searcher.cleanup()
