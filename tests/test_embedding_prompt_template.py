"""Unit tests for prompt template prepending in OpenAI embeddings.

This test suite defines the contract for prompt template functionality that allows
users to prepend a consistent prompt to all embedding inputs. These tests verify:

1. Template prepending to all input texts before embedding computation
2. Graceful handling of None/missing provider_options
3. Empty string template behavior (no-op)
4. Logging of template application for observability
5. Template application before token truncation

All tests are written in Red Phase - they should FAIL initially because the
implementation does not exist yet.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from leann.embedding_compute import compute_embeddings_openai


class TestPromptTemplatePrepending:
    """Tests for prompt template prepending in compute_embeddings_openai."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client that captures input texts."""
        mock_client = MagicMock()

        # Mock the embeddings.create response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        return mock_client

    @pytest.fixture
    def mock_openai_module(self, mock_openai_client, monkeypatch):
        """Mock the openai module to return our mock client."""
        # Mock the API key environment variable
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key-for-mocking")

        # openai is imported inside the function, so we need to patch it there
        with patch("openai.OpenAI", return_value=mock_openai_client) as mock_openai:
            yield mock_openai

    def test_prompt_template_prepended_to_all_texts(self, mock_openai_module, mock_openai_client):
        """Verify template is prepended to all input texts.

        When provider_options contains "prompt_template", that template should
        be prepended to every text in the input list before sending to OpenAI API.

        This is the core functionality: the template acts as a consistent prefix
        that provides context or instruction for the embedding model.
        """
        texts = ["First document", "Second document"]
        template = "search_document: "
        provider_options = {"prompt_template": template}

        # Call compute_embeddings_openai with provider_options
        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify embeddings.create was called with templated texts
        mock_openai_client.embeddings.create.assert_called_once()
        call_args = mock_openai_client.embeddings.create.call_args

        # Extract the input texts sent to API
        sent_texts = call_args.kwargs["input"]

        # Verify template was prepended to all texts
        assert len(sent_texts) == 2, "Should send same number of texts"
        assert sent_texts[0] == "search_document: First document", (
            "Template should be prepended to first text"
        )
        assert sent_texts[1] == "search_document: Second document", (
            "Template should be prepended to second text"
        )

        # Verify result is valid embeddings array
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3), "Should return correct shape"

    def test_template_not_applied_when_missing_or_empty(
        self, mock_openai_module, mock_openai_client
    ):
        """Verify template not applied when provider_options is None, missing key, or empty string.

        This consolidated test covers three scenarios where templates should NOT be applied:
        1. provider_options is None (default behavior)
        2. provider_options exists but missing 'prompt_template' key
        3. prompt_template is explicitly set to empty string ""

        In all cases, texts should be sent to the API unchanged.
        """
        # Scenario 1: None provider_options
        texts = ["Original text one", "Original text two"]
        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=None,
        )
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]
        assert sent_texts[0] == "Original text one", (
            "Text should be unchanged with None provider_options"
        )
        assert sent_texts[1] == "Original text two"
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

        # Reset mock for next scenario
        mock_openai_client.reset_mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_openai_client.embeddings.create.return_value = mock_response

        # Scenario 2: Missing 'prompt_template' key
        texts = ["Text without template", "Another text"]
        provider_options = {"base_url": "https://api.openai.com/v1"}
        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]
        assert sent_texts[0] == "Text without template", "Text should be unchanged with missing key"
        assert sent_texts[1] == "Another text"
        assert isinstance(result, np.ndarray)

        # Reset mock for next scenario
        mock_openai_client.reset_mock()
        mock_openai_client.embeddings.create.return_value = mock_response

        # Scenario 3: Empty string template
        texts = ["Text one", "Text two"]
        provider_options = {"prompt_template": ""}
        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]
        assert sent_texts[0] == "Text one", "Empty template should not modify text"
        assert sent_texts[1] == "Text two"
        assert isinstance(result, np.ndarray)

    def test_prompt_template_with_multiple_batches(self, mock_openai_module, mock_openai_client):
        """Verify template is prepended in all batches when texts exceed batch size.

        OpenAI API has batch size limits. When input texts are split into
        multiple batches, the template should be prepended to texts in every batch.

        This ensures consistency across all API calls.
        """
        # Create many texts that will be split into multiple batches
        texts = [f"Document {i}" for i in range(1000)]
        template = "passage: "
        provider_options = {"prompt_template": template}

        # Mock multiple batch responses
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in range(1000)]
        mock_openai_client.embeddings.create.return_value = mock_response

        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify embeddings.create was called multiple times (batching)
        assert mock_openai_client.embeddings.create.call_count >= 2, (
            "Should make multiple API calls for large text list"
        )

        # Verify template was prepended in ALL batches
        for call in mock_openai_client.embeddings.create.call_args_list:
            sent_texts = call.kwargs["input"]
            for text in sent_texts:
                assert text.startswith(template), (
                    f"All texts in all batches should start with template. Got: {text}"
                )

        # Verify result shape
        assert result.shape[0] == 1000, "Should return embeddings for all texts"

    def test_gemini_openai_compat_caps_batch_size_to_100(
        self, mock_openai_module, mock_openai_client
    ):
        texts = [f"Doc {i}" for i in range(250)]
        provider_options = {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai"}

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in range(250)]
        mock_openai_client.embeddings.create.return_value = mock_response

        result = compute_embeddings_openai(
            texts=texts,
            model_name="gemini-embedding-001",
            provider_options=provider_options,
        )

        # Should chunk into <=100 inputs per request.
        assert mock_openai_client.embeddings.create.call_count == 3
        batch_sizes = [
            len(call.kwargs["input"])
            for call in mock_openai_client.embeddings.create.call_args_list
        ]
        assert batch_sizes == [100, 100, 50]
        assert result.shape[0] == 250

    def test_prompt_template_with_special_characters(self, mock_openai_module, mock_openai_client):
        """Verify template with special characters is handled correctly.

        Templates may contain special characters, Unicode, newlines, etc.
        These should all be prepended correctly without encoding issues.
        """
        texts = ["Document content"]
        # Template with various special characters
        template = "🔍 Search query [EN]: "
        provider_options = {"prompt_template": template}

        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify special characters in template were preserved
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]

        assert sent_texts[0] == "🔍 Search query [EN]: Document content", (
            "Special characters in template should be preserved"
        )

        assert isinstance(result, np.ndarray)

    def test_prompt_template_integration_with_existing_validation(
        self, mock_openai_module, mock_openai_client
    ):
        """Verify template works with existing input validation.

        compute_embeddings_openai has validation for empty texts and whitespace.
        Template prepending should happen AFTER validation, so validation errors
        are thrown based on original texts, not templated texts.

        This ensures users get clear error messages about their input.
        """
        # Empty text should still raise ValueError even with template
        texts = [""]
        provider_options = {"prompt_template": "prefix: "}

        with pytest.raises(ValueError, match="empty/invalid"):
            compute_embeddings_openai(
                texts=texts,
                model_name="text-embedding-3-small",
                provider_options=provider_options,
            )

    def test_prompt_template_with_api_key_and_base_url(
        self, mock_openai_module, mock_openai_client
    ):
        """Verify template works alongside other provider_options.

        provider_options may contain multiple settings: prompt_template,
        base_url, api_key. All should work together correctly.
        """
        texts = ["Test document"]
        provider_options = {
            "prompt_template": "embed: ",
            "base_url": "https://custom.api.com/v1",
            "api_key": "test-key-123",
        }

        result = compute_embeddings_openai(
            texts=texts,
            model_name="text-embedding-3-small",
            provider_options=provider_options,
        )

        # Verify template was applied
        call_args = mock_openai_client.embeddings.create.call_args
        sent_texts = call_args.kwargs["input"]
        assert sent_texts[0] == "embed: Test document"

        # Verify OpenAI client was created with correct base_url
        mock_openai_module.assert_called()
        client_init_kwargs = mock_openai_module.call_args.kwargs
        assert client_init_kwargs["base_url"] == "https://custom.api.com/v1"
        assert client_init_kwargs["api_key"] == "test-key-123"

        assert isinstance(result, np.ndarray)
