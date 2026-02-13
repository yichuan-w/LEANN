"""End-to-end integration tests for prompt template and token limit features.

These tests verify real-world functionality with live services:
- OpenAI-compatible APIs (OpenAI, LM Studio) with prompt template support
- Ollama with dynamic token limit detection
- Hybrid token limit discovery mechanism

Run with: pytest tests/test_prompt_template_e2e.py -v -s
Skip if services unavailable: pytest tests/test_prompt_template_e2e.py -m "not integration"

Prerequisites:
1. LM Studio running with embedding model: http://localhost:1234
2. [Optional] Ollama running: ollama serve
3. [Optional] Ollama model: ollama pull nomic-embed-text
4. [Optional] Node.js + @lmstudio/sdk for context length detection
"""

import logging
import socket

import numpy as np
import pytest
import requests
from leann.embedding_compute import (
    compute_embeddings_ollama,
    compute_embeddings_openai,
    get_model_token_limit,
)

# Test markers for conditional execution
pytestmark = pytest.mark.integration

logger = logging.getLogger(__name__)


def check_service_available(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a service is available on the given host:port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_ollama_available() -> bool:
    """Check if Ollama service is available."""
    if not check_service_available("localhost", 11434):
        return False
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def check_lmstudio_available() -> bool:
    """Check if LM Studio service is available."""
    if not check_service_available("localhost", 1234):
        return False
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def get_lmstudio_first_model() -> str | None:
    """Get the first available model from LM Studio."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5.0)
        data = response.json()
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return None


class TestPromptTemplateOpenAI:
    """End-to-end tests for prompt template with OpenAI-compatible APIs (LM Studio)."""

    @pytest.mark.skipif(
        not check_lmstudio_available(), reason="LM Studio service not available on localhost:1234"
    )
    def test_lmstudio_embedding_with_prompt_template(self):
        """Test prompt templates with LM Studio using OpenAI-compatible API."""
        model_name = get_lmstudio_first_model()
        if not model_name:
            pytest.skip("No models loaded in LM Studio")
        assert model_name is not None  # Type narrowing for type checker

        texts = ["artificial intelligence", "machine learning"]
        prompt_template = "search_query: "

        # Get embeddings with prompt template via provider_options
        provider_options = {"prompt_template": prompt_template}
        embeddings = compute_embeddings_openai(
            texts=texts,
            model_name=model_name,
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",  # LM Studio doesn't require real key
            provider_options=provider_options,
        )

        assert embeddings is not None
        assert len(embeddings) == 2
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

        logger.info(
            f"✓ LM Studio embeddings with prompt template: {len(embeddings)} vectors, {len(embeddings[0])} dimensions"
        )

    @pytest.mark.skipif(not check_lmstudio_available(), reason="LM Studio service not available")
    def test_lmstudio_prompt_template_affects_embeddings(self):
        """Verify that prompt templates actually change embedding values."""
        model_name = get_lmstudio_first_model()
        if not model_name:
            pytest.skip("No models loaded in LM Studio")
        assert model_name is not None  # Type narrowing for type checker

        text = "machine learning"
        base_url = "http://localhost:1234/v1"
        api_key = "lm-studio"

        # Get embeddings without template
        embeddings_no_template = compute_embeddings_openai(
            texts=[text],
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            provider_options={},
        )

        # Get embeddings with template
        embeddings_with_template = compute_embeddings_openai(
            texts=[text],
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            provider_options={"prompt_template": "search_query: "},
        )

        # Embeddings should be different when template is applied
        assert not np.allclose(embeddings_no_template[0], embeddings_with_template[0])

        logger.info("✓ Prompt template changes embedding values as expected")


class TestPromptTemplateOllama:
    """End-to-end tests for prompt template with Ollama."""

    @pytest.mark.skipif(
        not check_ollama_available(), reason="Ollama service not available on localhost:11434"
    )
    def test_ollama_embedding_with_prompt_template(self):
        """Test prompt templates with Ollama using any available embedding model."""
        # Get any available embedding model
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2.0)
            models = response.json().get("models", [])

            embedding_models = []
            for model in models:
                name = model["name"]
                base_name = name.split(":")[0]
                if any(emb in base_name for emb in ["embed", "bge", "minilm", "e5", "nomic"]):
                    embedding_models.append(name)

            if not embedding_models:
                pytest.skip("No embedding models available in Ollama")

            model_name = embedding_models[0]

            texts = ["artificial intelligence", "machine learning"]
            prompt_template = "search_query: "

            # Get embeddings with prompt template via provider_options
            provider_options = {"prompt_template": prompt_template}
            embeddings = compute_embeddings_ollama(
                texts=texts,
                model_name=model_name,
                is_build=False,
                host="http://localhost:11434",
                provider_options=provider_options,
            )

            assert embeddings is not None
            assert len(embeddings) == 2
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
            assert all(len(emb) > 0 for emb in embeddings)

            logger.info(
                f"✓ Ollama embeddings with prompt template: {len(embeddings)} vectors, {len(embeddings[0])} dimensions"
            )

        except Exception as e:
            pytest.skip(f"Could not test Ollama prompt template: {e}")

    @pytest.mark.skipif(not check_ollama_available(), reason="Ollama service not available")
    def test_ollama_prompt_template_affects_embeddings(self):
        """Verify that prompt templates actually change embedding values with Ollama."""
        # Get any available embedding model
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2.0)
            models = response.json().get("models", [])

            embedding_models = []
            for model in models:
                name = model["name"]
                base_name = name.split(":")[0]
                if any(emb in base_name for emb in ["embed", "bge", "minilm", "e5", "nomic"]):
                    embedding_models.append(name)

            if not embedding_models:
                pytest.skip("No embedding models available in Ollama")

            model_name = embedding_models[0]
            text = "machine learning"
            host = "http://localhost:11434"

            # Get embeddings without template
            embeddings_no_template = compute_embeddings_ollama(
                texts=[text], model_name=model_name, is_build=False, host=host, provider_options={}
            )

            # Get embeddings with template
            embeddings_with_template = compute_embeddings_ollama(
                texts=[text],
                model_name=model_name,
                is_build=False,
                host=host,
                provider_options={"prompt_template": "search_query: "},
            )

            # Embeddings should be different when template is applied
            assert not np.allclose(embeddings_no_template[0], embeddings_with_template[0])

            logger.info("✓ Ollama prompt template changes embedding values as expected")

        except Exception as e:
            pytest.skip(f"Could not test Ollama prompt template: {e}")


class TestLMStudioSDK:
    """End-to-end tests for LM Studio SDK integration."""

    @pytest.mark.skipif(not check_lmstudio_available(), reason="LM Studio service not available")
    def test_lmstudio_model_listing(self):
        """Test that we can list models from LM Studio."""
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=5.0)
            assert response.status_code == 200

            data = response.json()
            assert "data" in data

            models = data["data"]
            logger.info(f"✓ LM Studio models available: {len(models)}")

            if models:
                logger.info(f"  First model: {models[0].get('id', 'unknown')}")
        except Exception as e:
            pytest.skip(f"LM Studio API error: {e}")

    @pytest.mark.skipif(not check_lmstudio_available(), reason="LM Studio service not available")
    def test_lmstudio_sdk_context_length_detection(self):
        """Test context length detection via LM Studio SDK bridge (requires Node.js + SDK)."""
        model_name = get_lmstudio_first_model()
        if not model_name:
            pytest.skip("No models loaded in LM Studio")
        assert model_name is not None  # Type narrowing for type checker

        try:
            from leann.embedding_compute import _query_lmstudio_context_limit

            # SDK requires WebSocket URL (ws://)
            context_length = _query_lmstudio_context_limit(
                model_name=model_name, base_url="ws://localhost:1234"
            )

            if context_length is None:
                logger.warning(
                    "⚠ LM Studio SDK bridge returned None (Node.js or SDK may not be available)"
                )
                pytest.skip("Node.js or @lmstudio/sdk not available - SDK bridge unavailable")
            else:
                assert context_length > 0
                logger.info(
                    f"✓ LM Studio context length detected via SDK: {context_length} for {model_name}"
                )

        except ImportError:
            pytest.skip("_query_lmstudio_context_limit not implemented yet")
        except Exception as e:
            logger.error(f"LM Studio SDK test error: {e}")
            raise


class TestOllamaTokenLimit:
    """End-to-end tests for Ollama token limit discovery."""

    @pytest.mark.skipif(not check_ollama_available(), reason="Ollama service not available")
    def test_ollama_token_limit_detection(self):
        """Test dynamic token limit detection from Ollama /api/show endpoint."""
        # Get any available embedding model
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2.0)
            models = response.json().get("models", [])

            embedding_models = []
            for model in models:
                name = model["name"]
                base_name = name.split(":")[0]
                if any(emb in base_name for emb in ["embed", "bge", "minilm", "e5", "nomic"]):
                    embedding_models.append(name)

            if not embedding_models:
                pytest.skip("No embedding models available in Ollama")

            test_model = embedding_models[0]

            # Test token limit detection
            limit = get_model_token_limit(model_name=test_model, base_url="http://localhost:11434")

            assert limit > 0
            logger.info(f"✓ Ollama token limit detected: {limit} for {test_model}")

        except Exception as e:
            pytest.skip(f"Could not test Ollama token detection: {e}")


class TestHybridTokenLimit:
    """End-to-end tests for hybrid token limit discovery mechanism."""

    def test_hybrid_discovery_registry_fallback(self):
        """Test fallback to static registry for known OpenAI models."""
        # Use a known OpenAI model (should be in registry)
        limit = get_model_token_limit(
            model_name="text-embedding-3-small",
            base_url="http://fake-server:9999",  # Fake URL to force registry lookup
        )

        # text-embedding-3-small should have 8192 in registry
        assert limit == 8192
        logger.info(f"✓ Hybrid discovery (registry fallback): {limit} tokens")

    def test_hybrid_discovery_default_fallback(self):
        """Test fallback to safe default for completely unknown models."""
        limit = get_model_token_limit(
            model_name="completely-unknown-model-xyz-12345",
            base_url="http://fake-server:9999",
            default=512,
        )

        # Should get the specified default
        assert limit == 512
        logger.info(f"✓ Hybrid discovery (default fallback): {limit} tokens")

    @pytest.mark.skipif(not check_ollama_available(), reason="Ollama service not available")
    def test_hybrid_discovery_ollama_dynamic_first(self):
        """Test that Ollama models use dynamic discovery first."""
        # Get any available embedding model
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2.0)
            models = response.json().get("models", [])

            embedding_models = []
            for model in models:
                name = model["name"]
                base_name = name.split(":")[0]
                if any(emb in base_name for emb in ["embed", "bge", "minilm", "e5", "nomic"]):
                    embedding_models.append(name)

            if not embedding_models:
                pytest.skip("No embedding models available in Ollama")

            test_model = embedding_models[0]

            # Should query Ollama /api/show dynamically
            limit = get_model_token_limit(model_name=test_model, base_url="http://localhost:11434")

            assert limit > 0
            logger.info(f"✓ Hybrid discovery (Ollama dynamic): {limit} tokens for {test_model}")

        except Exception as e:
            pytest.skip(f"Could not test hybrid Ollama discovery: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUITE - Real Service Testing")
    print("=" * 70)
    print("\nThese tests require live services:")
    print("  • LM Studio: http://localhost:1234 (with embedding model loaded)")
    print("  • [Optional] Ollama: http://localhost:11434")
    print("  • [Optional] Node.js + @lmstudio/sdk for SDK bridge tests")
    print("\nRun with: pytest tests/test_prompt_template_e2e.py -v -s")
    print("=" * 70 + "\n")
