"""
Comprehensive tests for metadata filtering functionality.

This module tests the MetadataFilterEngine class and its integration
with the LEANN search system.
"""

import os

# Import the modules we're testing
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../packages/leann-core/src"))

from leann.api import PassageManager, SearchResult
from leann.metadata_filter import MetadataFilterEngine


class TestMetadataFilterEngine:
    """Test suite for the MetadataFilterEngine class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.engine = MetadataFilterEngine()

        # Sample search results for testing
        self.sample_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "text": "This is chapter 1 content",
                "metadata": {
                    "chapter": 1,
                    "character": "Alice",
                    "tags": ["adventure", "fantasy"],
                    "word_count": 150,
                    "is_published": True,
                    "genre": "fiction",
                },
            },
            {
                "id": "doc2",
                "score": 0.87,
                "text": "This is chapter 3 content",
                "metadata": {
                    "chapter": 3,
                    "character": "Bob",
                    "tags": ["mystery", "thriller"],
                    "word_count": 250,
                    "is_published": True,
                    "genre": "fiction",
                },
            },
            {
                "id": "doc3",
                "score": 0.82,
                "text": "This is chapter 5 content",
                "metadata": {
                    "chapter": 5,
                    "character": "Alice",
                    "tags": ["romance", "drama"],
                    "word_count": 300,
                    "is_published": False,
                    "genre": "non-fiction",
                },
            },
            {
                "id": "doc4",
                "score": 0.78,
                "text": "This is chapter 10 content",
                "metadata": {
                    "chapter": 10,
                    "character": "Charlie",
                    "tags": ["action", "adventure"],
                    "word_count": 400,
                    "is_published": True,
                    "genre": "fiction",
                },
            },
        ]

    def test_engine_initialization(self):
        """Test that the filter engine initializes correctly."""
        assert self.engine is not None
        assert len(self.engine.operators) > 0
        assert "==" in self.engine.operators
        assert "contains" in self.engine.operators
        assert "in" in self.engine.operators

    def test_direct_instantiation(self):
        """Test direct instantiation of the engine."""
        engine = MetadataFilterEngine()
        assert isinstance(engine, MetadataFilterEngine)

    def test_no_filters_returns_all_results(self):
        """Test that passing None or empty filters returns all results."""
        # Test with None
        result = self.engine.apply_filters(self.sample_results, None)
        assert len(result) == len(self.sample_results)

        # Test with empty dict
        result = self.engine.apply_filters(self.sample_results, {})
        assert len(result) == len(self.sample_results)

    # Test comparison operators
    def test_equals_filter(self):
        """Test equals (==) filter."""
        filters = {"chapter": {"==": 1}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 1
        assert result[0]["id"] == "doc1"

    def test_not_equals_filter(self):
        """Test not equals (!=) filter."""
        filters = {"genre": {"!=": "fiction"}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 1
        assert result[0]["metadata"]["genre"] == "non-fiction"

    def test_less_than_filter(self):
        """Test less than (<) filter."""
        filters = {"chapter": {"<": 5}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 2
        chapters = [r["metadata"]["chapter"] for r in result]
        assert all(ch < 5 for ch in chapters)

    def test_less_than_or_equal_filter(self):
        """Test less than or equal (<=) filter."""
        filters = {"chapter": {"<=": 5}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 3
        chapters = [r["metadata"]["chapter"] for r in result]
        assert all(ch <= 5 for ch in chapters)

    def test_greater_than_filter(self):
        """Test greater than (>) filter."""
        filters = {"word_count": {">": 200}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 3  # Documents with word_count 250, 300, 400
        word_counts = [r["metadata"]["word_count"] for r in result]
        assert all(wc > 200 for wc in word_counts)

    def test_greater_than_or_equal_filter(self):
        """Test greater than or equal (>=) filter."""
        filters = {"word_count": {">=": 250}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 3
        word_counts = [r["metadata"]["word_count"] for r in result]
        assert all(wc >= 250 for wc in word_counts)

    # Test membership operators
    def test_in_filter(self):
        """Test in filter."""
        filters = {"character": {"in": ["Alice", "Bob"]}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 3
        characters = [r["metadata"]["character"] for r in result]
        assert all(ch in ["Alice", "Bob"] for ch in characters)

    def test_not_in_filter(self):
        """Test not_in filter."""
        filters = {"character": {"not_in": ["Alice", "Bob"]}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 1
        assert result[0]["metadata"]["character"] == "Charlie"

    # Test string operators
    def test_contains_filter(self):
        """Test contains filter."""
        filters = {"genre": {"contains": "fiction"}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 4  # Both "fiction" and "non-fiction"

    def test_starts_with_filter(self):
        """Test starts_with filter."""
        filters = {"genre": {"starts_with": "non"}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 1
        assert result[0]["metadata"]["genre"] == "non-fiction"

    def test_ends_with_filter(self):
        """Test ends_with filter."""
        filters = {"text": {"ends_with": "content"}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 4  # All sample texts end with "content"

    # Test boolean operators
    def test_is_true_filter(self):
        """Test is_true filter."""
        filters = {"is_published": {"is_true": True}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 3
        assert all(r["metadata"]["is_published"] for r in result)

    def test_is_false_filter(self):
        """Test is_false filter."""
        filters = {"is_published": {"is_false": False}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 1
        assert not result[0]["metadata"]["is_published"]

    # Test compound filters (AND logic)
    def test_compound_filters(self):
        """Test multiple filters applied together (AND logic)."""
        filters = {"genre": {"==": "fiction"}, "chapter": {"<=": 5}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 2
        for r in result:
            assert r["metadata"]["genre"] == "fiction"
            assert r["metadata"]["chapter"] <= 5

    def test_multiple_operators_same_field(self):
        """Test multiple operators on the same field."""
        filters = {"word_count": {">=": 200, "<=": 350}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 2
        for r in result:
            wc = r["metadata"]["word_count"]
            assert 200 <= wc <= 350

    # Test edge cases
    def test_missing_field_fails_filter(self):
        """Test that missing metadata fields fail filters."""
        filters = {"nonexistent_field": {"==": "value"}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 0

    def test_invalid_operator(self):
        """Test that invalid operators are handled gracefully."""
        filters = {"chapter": {"invalid_op": 1}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 0  # Should filter out all results

    def test_type_coercion_numeric(self):
        """Test numeric type coercion in comparisons."""
        # Add a result with string chapter number
        test_results = [
            *self.sample_results,
            {
                "id": "doc5",
                "score": 0.75,
                "text": "String chapter test",
                "metadata": {"chapter": "2", "genre": "test"},
            },
        ]

        filters = {"chapter": {"<": 3}}
        result = self.engine.apply_filters(test_results, filters)
        # Should include doc1 (chapter=1) and doc5 (chapter="2")
        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert "doc1" in ids
        assert "doc5" in ids

    def test_list_membership_with_nested_tags(self):
        """Test membership operations with list metadata."""
        # Note: This tests the metadata structure, not list field filtering
        # For list field filtering, we'd need to modify the test data
        filters = {"character": {"in": ["Alice"]}}
        result = self.engine.apply_filters(self.sample_results, filters)
        assert len(result) == 2
        assert all(r["metadata"]["character"] == "Alice" for r in result)

        result = self.engine.apply_filters([], filters)
        assert len(result) == 0

    def test_equality_fast_path_optimization(self):
        """Test the fast-path optimization for equality checks."""
        # This test ensures the optimized code path works correctly
        # Note: self.engine is initialized in setup_method, but we can make a new one or use self.engine

        result = {"category": "A", "val": 10}

        # 1. Basic equality check (Fast Path)
        filters = {"category": {"==": "A"}}
        # Access protected method for direct verification if needed,
        # or just use public apply_filters
        assert self.engine._evaluate_filters(result, filters) is True

        filters_fail = {"category": {"==": "B"}}
        assert self.engine._evaluate_filters(result, filters_fail) is False


class TestPassageManagerFiltering:
    """Test suite for PassageManager filtering integration."""

    def setup_method(self):
        """Setup test fixtures."""
        # Mock the passage manager without actual file I/O
        self.passage_manager = Mock(spec=PassageManager)
        self.passage_manager.filter_engine = MetadataFilterEngine()

        # Sample SearchResult objects
        self.search_results = [
            SearchResult(
                id="doc1",
                score=0.95,
                text="Chapter 1 content",
                metadata={"chapter": 1, "character": "Alice"},
            ),
            SearchResult(
                id="doc2",
                score=0.87,
                text="Chapter 5 content",
                metadata={"chapter": 5, "character": "Bob"},
            ),
            SearchResult(
                id="doc3",
                score=0.82,
                text="Chapter 10 content",
                metadata={"chapter": 10, "character": "Alice"},
            ),
        ]

    def test_search_result_filtering(self):
        """Test filtering SearchResult objects."""
        # Create a real PassageManager instance just for the filtering method
        # We'll mock the file operations
        with patch("builtins.open"), patch("json.loads"), patch("pickle.load"):
            pm = PassageManager([{"type": "jsonl", "path": "test.jsonl"}])

            filters = {"chapter": {"<=": 5}}
            result = pm.filter_search_results(self.search_results, filters)

            assert len(result) == 2
            chapters = [r.metadata["chapter"] for r in result]
            assert all(ch <= 5 for ch in chapters)

    def test_filter_search_results_no_filters(self):
        """Test that None filters return all results."""
        with patch("builtins.open"), patch("json.loads"), patch("pickle.load"):
            pm = PassageManager([{"type": "jsonl", "path": "test.jsonl"}])

            result = pm.filter_search_results(self.search_results, None)
            assert len(result) == len(self.search_results)

    def test_filter_maintains_search_result_type(self):
        """Test that filtering returns SearchResult objects."""
        with patch("builtins.open"), patch("json.loads"), patch("pickle.load"):
            pm = PassageManager([{"type": "jsonl", "path": "test.jsonl"}])

            filters = {"character": {"==": "Alice"}}
            result = pm.filter_search_results(self.search_results, filters)

            assert len(result) == 2
            for r in result:
                assert isinstance(r, SearchResult)
                assert r.metadata["character"] == "Alice"


# Integration tests would go here, but they require actual LEANN backend setup
# These would test the full pipeline from LeannSearcher.search() with metadata_filters

if __name__ == "__main__":
    # Run basic smoke tests
    engine = MetadataFilterEngine()

    sample_data = [
        {
            "id": "test1",
            "score": 0.9,
            "text": "Test content",
            "metadata": {"chapter": 1, "published": True},
        }
    ]

    # Test basic filtering
    result = engine.apply_filters(sample_data, {"chapter": {"==": 1}})
    assert len(result) == 1
    print("✅ Basic filtering test passed")

    result = engine.apply_filters(sample_data, {"chapter": {"==": 2}})
    assert len(result) == 0
    print("✅ No match filtering test passed")

    print("🎉 All smoke tests passed!")
