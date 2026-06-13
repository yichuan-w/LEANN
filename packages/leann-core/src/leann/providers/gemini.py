"""Google Gemini embedding provider."""

import logging

import numpy as np

from ..embedding_compute import _model_cache

logger = logging.getLogger(__name__)


def compute_embeddings_gemini(
    texts: list[str], model_name: str = "text-embedding-004", is_build: bool = False
) -> np.ndarray:
    """
    Compute embeddings using Google Gemini API.

    Args:
        texts: List of texts to compute embeddings for
        model_name: Gemini model name (default: "text-embedding-004")
        is_build: Whether this is a build operation (shows progress bar)

    Returns:
        Embeddings array, shape: (len(texts), embedding_dim)
    """
    try:
        import os

        import google.genai as genai
    except ImportError as e:
        raise ImportError(f"Google GenAI package not installed: {e}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    # Cache Gemini client
    cache_key = "gemini_client"
    if cache_key in _model_cache:
        client = _model_cache[cache_key]
    else:
        client = genai.Client(api_key=api_key)
        _model_cache[cache_key] = client
        logger.info("Gemini client cached")

    logger.info(
        f"Computing embeddings for {len(texts)} texts using Gemini API, model: '{model_name}'"
    )

    # Gemini supports batch embedding
    max_batch_size = 100  # Conservative batch size for Gemini
    all_embeddings = []

    try:
        from tqdm import tqdm

        total_batches = (len(texts) + max_batch_size - 1) // max_batch_size
        batch_range = range(0, len(texts), max_batch_size)
        batch_iterator = tqdm(
            batch_range, desc="Computing embeddings", unit="batch", total=total_batches
        )
    except ImportError:
        # Fallback when tqdm is not available
        batch_iterator = range(0, len(texts), max_batch_size)

    for i in batch_iterator:
        batch_texts = texts[i : i + max_batch_size]

        try:
            # Use the embed_content method from the new Google GenAI SDK
            response = client.models.embed_content(
                model=model_name,
                contents=batch_texts,
                config=genai.types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"  # For document embedding
                ),
            )

            # Extract embeddings from response
            for embedding_data in response.embeddings:
                all_embeddings.append(embedding_data.values)
        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    return embeddings
