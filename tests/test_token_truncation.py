"""Unit tests for token-aware truncation functionality.

This test suite defines the contract for token truncation functions that prevent
500 errors from Ollama when text exceeds model token limits. These tests verify:

1. Model token limit retrieval (known and unknown models)
2. Text truncation behavior for single and multiple texts
3. Token counting and truncation accuracy using tiktoken

All tests are written in Red Phase - they should FAIL initially because the
implementation does not exist yet.
"""

import pytest
import tiktoken
from leann.embedding_compute import (
    EMBEDDING_MODEL_LIMITS,
    get_model_token_limit,
    truncate_to_token_limit,
)


class TestModelTokenLimits:
    """Tests for retrieving model-specific token limits."""

    def test_get_model_token_limit_known_model(self):
        """Verify correct token limit is returned for known models.

        Known models should return their specific token limits from
        EMBEDDING_MODEL_LIMITS dictionary.
        """
        # Test nomic-embed-text (2048 tokens)
        limit = get_model_token_limit("nomic-embed-text")
        assert limit == 2048, "nomic-embed-text should have 2048 token limit"

        # Test nomic-embed-text-v1.5 (2048 tokens)
        limit = get_model_token_limit("nomic-embed-text-v1.5")
        assert limit == 2048, "nomic-embed-text-v1.5 should have 2048 token limit"

        # Test nomic-embed-text-v2 (512 tokens)
        limit = get_model_token_limit("nomic-embed-text-v2")
        assert limit == 512, "nomic-embed-text-v2 should have 512 token limit"

        # Test OpenAI models (8192 tokens)
        limit = get_model_token_limit("text-embedding-3-small")
        assert limit == 8192, "text-embedding-3-small should have 8192 token limit"

    def test_get_model_token_limit_unknown_model(self):
        """Verify default token limit is returned for unknown models.

        Unknown models should return the default limit (2048) to allow
        operation with reasonable safety margin.
        """
        # Test with completely unknown model
        limit = get_model_token_limit("unknown-model-xyz")
        assert limit == 2048, "Unknown models should return default 2048"

        # Test with empty string
        limit = get_model_token_limit("")
        assert limit == 2048, "Empty model name should return default 2048"

    def test_get_model_token_limit_custom_default(self):
        """Verify custom default can be specified for unknown models.

        Allow callers to specify their own default token limit when
        model is not in the known models dictionary.
        """
        limit = get_model_token_limit("unknown-model", default=4096)
        assert limit == 4096, "Should return custom default for unknown models"

        # Known model should ignore custom default
        limit = get_model_token_limit("nomic-embed-text", default=4096)
        assert limit == 2048, "Known model should ignore custom default"

    def test_embedding_model_limits_dictionary_exists(self):
        """Verify EMBEDDING_MODEL_LIMITS dictionary contains expected models.

        The dictionary should be importable and contain at least the
        known nomic models with correct token limits.
        """
        assert isinstance(EMBEDDING_MODEL_LIMITS, dict), "Should be a dictionary"
        assert "nomic-embed-text" in EMBEDDING_MODEL_LIMITS, "Should contain nomic-embed-text"
        assert "nomic-embed-text-v1.5" in EMBEDDING_MODEL_LIMITS, (
            "Should contain nomic-embed-text-v1.5"
        )
        assert EMBEDDING_MODEL_LIMITS["nomic-embed-text"] == 2048
        assert EMBEDDING_MODEL_LIMITS["nomic-embed-text-v1.5"] == 2048
        assert EMBEDDING_MODEL_LIMITS["nomic-embed-text-v2"] == 512
        # OpenAI models
        assert EMBEDDING_MODEL_LIMITS["text-embedding-3-small"] == 8192


class TestTokenTruncation:
    """Tests for truncating texts to token limits."""

    @pytest.fixture
    def tokenizer(self):
        """Provide tiktoken tokenizer for token counting verification."""
        return tiktoken.get_encoding("cl100k_base")

    def test_truncate_single_text_under_limit(self, tokenizer):
        """Verify text under token limit remains unchanged.

        When text is already within the token limit, it should be
        returned unchanged with no truncation.
        """
        text = "This is a short text that is well under the token limit."
        token_count = len(tokenizer.encode(text))
        assert token_count < 100, f"Test setup: text should be short (has {token_count} tokens)"

        # Truncate with generous limit
        result = truncate_to_token_limit([text], token_limit=512)

        assert len(result) == 1, "Should return same number of texts"
        assert result[0] == text, "Text under limit should be unchanged"

    def test_truncate_single_text_over_limit(self, tokenizer):
        """Verify text over token limit is truncated correctly.

        When text exceeds the token limit, it should be truncated to
        fit within the limit while maintaining valid token boundaries.
        """
        # Create a text that definitely exceeds limit
        text = "word " * 200  # ~200 tokens (each "word " is typically 1-2 tokens)
        original_token_count = len(tokenizer.encode(text))
        assert original_token_count > 50, (
            f"Test setup: text should be long (has {original_token_count} tokens)"
        )

        # Truncate to 50 tokens
        result = truncate_to_token_limit([text], token_limit=50)

        assert len(result) == 1, "Should return same number of texts"
        assert result[0] != text, "Text over limit should be truncated"
        assert len(result[0]) < len(text), "Truncated text should be shorter"

        # Verify truncated text is within token limit
        truncated_token_count = len(tokenizer.encode(result[0]))
        assert truncated_token_count <= 50, (
            f"Truncated text should be ≤50 tokens, got {truncated_token_count}"
        )

    def test_truncate_multiple_texts_mixed_lengths(self, tokenizer):
        """Verify multiple texts with mixed lengths are handled correctly.

        When processing multiple texts:
        - Texts under limit should remain unchanged
        - Texts over limit should be truncated independently
        - Output list should maintain same order and length
        """
        texts = [
            "Short text.",  # Under limit
            "word " * 200,  # Over limit
            "Another short one.",  # Under limit
            "token " * 150,  # Over limit
        ]

        # Verify test setup
        for i, text in enumerate(texts):
            token_count = len(tokenizer.encode(text))
            if i in [1, 3]:
                assert token_count > 50, f"Text {i} should be over limit (has {token_count} tokens)"
            else:
                assert token_count < 50, (
                    f"Text {i} should be under limit (has {token_count} tokens)"
                )

        # Truncate with 50 token limit
        result = truncate_to_token_limit(texts, token_limit=50)

        assert len(result) == len(texts), "Should return same number of texts"

        # Verify each text individually
        for i, (original, truncated) in enumerate(zip(texts, result)):
            token_count = len(tokenizer.encode(truncated))
            assert token_count <= 50, f"Text {i} should be ≤50 tokens, got {token_count}"

            # Short texts should be unchanged
            if i in [0, 2]:
                assert truncated == original, f"Short text {i} should be unchanged"
            # Long texts should be truncated
            else:
                assert len(truncated) < len(original), f"Long text {i} should be truncated"

    def test_truncate_empty_list(self):
        """Verify empty input list returns empty output list.

        Edge case: empty list should return empty list without errors.
        """
        result = truncate_to_token_limit([], token_limit=512)
        assert result == [], "Empty input should return empty output"

    def test_truncate_preserves_order(self, tokenizer):
        """Verify truncation preserves original text order.

        Output list should maintain the same order as input list,
        regardless of which texts were truncated.
        """
        texts = [
            "First text " * 50,  # Will be truncated
            "Second text.",  # Won't be truncated
            "Third text " * 50,  # Will be truncated
        ]

        result = truncate_to_token_limit(texts, token_limit=20)

        assert len(result) == 3, "Should preserve list length"
        # Check that order is maintained by looking for distinctive words
        assert "First" in result[0], "First text should remain in first position"
        assert "Second" in result[1], "Second text should remain in second position"
        assert "Third" in result[2], "Third text should remain in third position"

    def test_truncate_extremely_long_text(self, tokenizer):
        """Verify extremely long texts are truncated efficiently.

        Test with text that far exceeds token limit to ensure
        truncation handles extreme cases without performance issues.
        """
        # Create very long text (simulate real-world scenario)
        text = "token " * 5000  # ~5000+ tokens
        original_token_count = len(tokenizer.encode(text))
        assert original_token_count > 1000, "Test setup: text should be very long"

        # Truncate to small limit
        result = truncate_to_token_limit([text], token_limit=100)

        assert len(result) == 1
        truncated_token_count = len(tokenizer.encode(result[0]))
        assert truncated_token_count <= 100, (
            f"Should truncate to ≤100 tokens, got {truncated_token_count}"
        )
        assert len(result[0]) < len(text) // 10, "Should significantly reduce text length"

    def test_truncate_exact_token_limit(self, tokenizer):
        """Verify text at exactly token limit is handled correctly.

        Edge case: text with exactly the token limit should either
        remain unchanged or be safely truncated by 1 token.
        """
        # Create text with approximately 50 tokens
        # We'll adjust to get exactly 50
        target_tokens = 50
        text = "word " * 50
        tokens = tokenizer.encode(text)

        # Adjust to get exactly target_tokens
        if len(tokens) > target_tokens:
            tokens = tokens[:target_tokens]
            text = tokenizer.decode(tokens)
        elif len(tokens) < target_tokens:
            # Add more words
            while len(tokenizer.encode(text)) < target_tokens:
                text += "word "
            tokens = tokenizer.encode(text)[:target_tokens]
            text = tokenizer.decode(tokens)

        # Verify we have exactly target_tokens
        assert len(tokenizer.encode(text)) == target_tokens, (
            "Test setup: should have exactly 50 tokens"
        )

        result = truncate_to_token_limit([text], token_limit=target_tokens)

        assert len(result) == 1
        result_tokens = len(tokenizer.encode(result[0]))
        assert result_tokens <= target_tokens, (
            f"Should be ≤{target_tokens} tokens, got {result_tokens}"
        )
