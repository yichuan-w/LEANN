"""Ollama embedding provider."""

import logging
from typing import Any, Optional

import numpy as np

from ..embedding_compute import get_model_token_limit, truncate_to_token_limit
from ..settings import resolve_ollama_host

logger = logging.getLogger(__name__)


def compute_embeddings_ollama(
    texts: list[str],
    model_name: str,
    is_build: bool = False,
    host: Optional[str] = None,
    provider_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """
    Compute embeddings using Ollama API with true batch processing.

    Uses the /api/embed endpoint which supports batch inputs.
    Batch size: 32 for MPS/CPU, 128 for CUDA to optimize performance.

    Args:
        texts: List of texts to compute embeddings for
        model_name: Ollama model name (e.g., "nomic-embed-text", "mxbai-embed-large")
        is_build: Whether this is a build operation (shows progress bar)
        host: Ollama host URL (defaults to environment or http://localhost:11434)
        provider_options: Optional provider-specific options (e.g., prompt_template)

    Returns:
        Normalized embeddings array, shape: (len(texts), embedding_dim)
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The 'requests' library is required for Ollama embeddings. Install with: uv pip install requests"
        )

    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")

    resolved_host = resolve_ollama_host(host)

    logger.info(
        f"Computing embeddings for {len(texts)} texts using Ollama API, model: '{model_name}', host: '{resolved_host}'"
    )

    # Check if Ollama is running
    try:
        response = requests.get(f"{resolved_host}/api/version", timeout=5)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        error_msg = (
            f"❌ Could not connect to Ollama at {resolved_host}.\n\n"
            "Please ensure Ollama is running:\n"
            "  • macOS/Linux: ollama serve\n"
            "  • Windows: Make sure Ollama is running in the system tray\n\n"
            "Installation: https://ollama.com/download"
        )
        raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"Unexpected error connecting to Ollama: {e}")

    # Check if model exists and provide helpful suggestions
    try:
        response = requests.get(f"{resolved_host}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json()
        model_names = [model["name"] for model in models.get("models", [])]

        # Filter for embedding models (models that support embeddings)
        embedding_models = []
        suggested_embedding_models = [
            "nomic-embed-text",
            "mxbai-embed-large",
            "bge-m3",
            "all-minilm",
            "snowflake-arctic-embed",
        ]

        for model in model_names:
            # Check if it's an embedding model (by name patterns or known models)
            base_name = model.split(":")[0]
            if any(emb in base_name for emb in ["embed", "bge", "minilm", "e5"]):
                embedding_models.append(model)

        # Check if model exists (handle versioned names) and resolve to full name
        resolved_model_name = None
        for name in model_names:
            # Exact match
            if model_name == name:
                resolved_model_name = name
                break
            # Match without version tag (use the versioned name)
            elif model_name == name.split(":")[0]:
                resolved_model_name = name
                break

        if not resolved_model_name:
            error_msg = f"❌ Model '{model_name}' not found in local Ollama.\n\n"

            # Suggest pulling the model
            error_msg += "📦 To install this embedding model:\n"
            error_msg += f"   ollama pull {model_name}\n\n"

            # Show available embedding models
            if embedding_models:
                error_msg += "✅ Available embedding models:\n"
                for model in embedding_models[:5]:
                    error_msg += f"   • {model}\n"
                if len(embedding_models) > 5:
                    error_msg += f"   ... and {len(embedding_models) - 5} more\n"
            else:
                error_msg += "💡 Popular embedding models to install:\n"
                for model in suggested_embedding_models[:3]:
                    error_msg += f"   • ollama pull {model}\n"

            error_msg += "\n📚 Browse more: https://ollama.com/library"
            raise ValueError(error_msg)

        # Use the resolved model name for all subsequent operations
        if resolved_model_name != model_name:
            logger.info(f"Resolved model name '{model_name}' to '{resolved_model_name}'")
        model_name = resolved_model_name

        # Verify the model supports embeddings by testing it with /api/embed
        try:
            test_response = requests.post(
                f"{resolved_host}/api/embed",
                json={"model": model_name, "input": "test"},
                timeout=10,
            )
            if test_response.status_code != 200:
                error_msg = (
                    f"⚠️ Model '{model_name}' exists but may not support embeddings.\n\n"
                    f"Please use an embedding model like:\n"
                )
                for model in suggested_embedding_models[:3]:
                    error_msg += f"   • {model}\n"
                raise ValueError(error_msg)
        except requests.exceptions.RequestException:
            # If test fails, continue anyway - model might still work
            pass

    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not verify model existence: {e}")

    # Determine batch size based on device availability
    # Check for CUDA/MPS availability using torch if available
    batch_size = 32  # Default for MPS/CPU
    try:
        import torch

        if torch.cuda.is_available():
            batch_size = 128  # CUDA gets larger batch size
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            batch_size = 32  # MPS gets smaller batch size
    except ImportError:
        # If torch is not available, use conservative batch size
        batch_size = 32

    logger.info(f"Using batch size: {batch_size} for true batch processing")

    # Apply prompt template if provided
    provider_options = provider_options or {}
    # Priority: build_prompt_template (new format) > prompt_template (old format)
    prompt_template = provider_options.get("build_prompt_template") or provider_options.get(
        "prompt_template"
    )

    if prompt_template:
        logger.warning(f"Applying prompt template: '{prompt_template}'")
        texts = [f"{prompt_template}{text}" for text in texts]

    # Get model token limit and apply truncation before batching
    token_limit = get_model_token_limit(model_name, base_url=resolved_host)
    logger.info(f"Model '{model_name}' token limit: {token_limit}")

    # Apply truncation to all texts before batch processing
    # Function logs truncation details internally
    texts = truncate_to_token_limit(texts, token_limit)

    def get_batch_embeddings(batch_texts):
        """Get embeddings for a batch of texts using /api/embed endpoint."""
        max_retries = 3
        retry_count = 0

        # Texts are already truncated to token limit by the outer function
        while retry_count < max_retries:
            try:
                # Use /api/embed endpoint with "input" parameter for batch processing
                response = requests.post(
                    f"{resolved_host}/api/embed",
                    json={"model": model_name, "input": batch_texts},
                    timeout=60,  # Increased timeout for batch processing
                )
                response.raise_for_status()

                result = response.json()
                batch_embeddings = result.get("embeddings")

                if batch_embeddings is None:
                    raise ValueError("No embeddings returned from API")

                if not isinstance(batch_embeddings, list):
                    raise ValueError(f"Invalid embeddings format: {type(batch_embeddings)}")

                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError(
                        f"Mismatch: requested {len(batch_texts)} embeddings, got {len(batch_embeddings)}"
                    )

                return batch_embeddings, []

            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.warning(f"Timeout for batch after {max_retries} retries")
                    return None, list(range(len(batch_texts)))

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    # Enhanced error detection for token limit violations
                    error_msg = str(e).lower()
                    if "token" in error_msg and (
                        "limit" in error_msg or "exceed" in error_msg or "length" in error_msg
                    ):
                        logger.error(
                            f"Token limit exceeded for batch. Error: {e}. "
                            f"Consider reducing chunk sizes or check token truncation."
                        )
                    else:
                        logger.error(f"Failed to get embeddings for batch: {e}")
                    return None, list(range(len(batch_texts)))

        return None, list(range(len(batch_texts)))

    # Process texts in batches
    all_embeddings = []
    all_failed_indices = []

    # Setup progress bar if needed
    show_progress = is_build or len(texts) > 10
    try:
        if show_progress:
            from tqdm import tqdm
    except ImportError:
        show_progress = False

    # Process batches
    num_batches = (len(texts) + batch_size - 1) // batch_size

    if show_progress:
        batch_iterator = tqdm(range(num_batches), desc="Computing Ollama embeddings (batched)")
    else:
        batch_iterator = range(num_batches)

    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        batch_embeddings, batch_failed = get_batch_embeddings(batch_texts)

        if batch_embeddings is not None:
            all_embeddings.extend(batch_embeddings)
        else:
            # Entire batch failed, add None placeholders
            all_embeddings.extend([None] * len(batch_texts))
            # Adjust failed indices to global indices
            global_failed = [start_idx + idx for idx in batch_failed]
            all_failed_indices.extend(global_failed)

    # Handle failed embeddings
    if all_failed_indices:
        if len(all_failed_indices) == len(texts):
            raise RuntimeError("Failed to compute any embeddings")

        logger.warning(
            f"Failed to compute embeddings for {len(all_failed_indices)}/{len(texts)} texts"
        )

        # Use zero embeddings as fallback for failed ones
        valid_embedding = next((e for e in all_embeddings if e is not None), None)
        if valid_embedding:
            embedding_dim = len(valid_embedding)
            for i, embedding in enumerate(all_embeddings):
                if embedding is None:
                    all_embeddings[i] = [0.0] * embedding_dim

    # Remove None values
    all_embeddings = [e for e in all_embeddings if e is not None]

    if not all_embeddings:
        raise RuntimeError("No valid embeddings were computed")

    # Validate embedding dimensions
    expected_dim = len(all_embeddings[0])
    inconsistent_dims = []
    for i, embedding in enumerate(all_embeddings):
        if len(embedding) != expected_dim:
            inconsistent_dims.append((i, len(embedding)))

    if inconsistent_dims:
        error_msg = f"Ollama returned inconsistent embedding dimensions. Expected {expected_dim}, but got:\n"
        for idx, dim in inconsistent_dims[:10]:  # Show first 10 inconsistent ones
            error_msg += f"  - Text {idx}: {dim} dimensions\n"
        if len(inconsistent_dims) > 10:
            error_msg += f"  ... and {len(inconsistent_dims) - 10} more\n"
        error_msg += f"\nThis is likely an Ollama API bug with model '{model_name}'. Please try:\n"
        error_msg += "1. Restart Ollama service: 'ollama serve'\n"
        error_msg += f"2. Re-pull the model: 'ollama pull {model_name}'\n"
        error_msg += (
            "3. Use sentence-transformers instead: --embedding-mode sentence-transformers\n"
        )
        error_msg += "4. Report this issue to Ollama: https://github.com/ollama/ollama/issues"
        raise ValueError(error_msg)

    # Convert to numpy array and normalize
    embeddings = np.array(all_embeddings, dtype=np.float32)

    # Normalize embeddings (L2 normalization)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero

    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    return embeddings
