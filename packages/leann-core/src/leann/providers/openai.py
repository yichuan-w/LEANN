"""OpenAI embedding provider."""
import logging
import time
from typing import Any, Optional

import numpy as np

from ..embedding_compute import get_model_token_limit, truncate_to_token_limit
from ..settings import resolve_openai_api_key, resolve_openai_base_url

logger = logging.getLogger(__name__)

def compute_embeddings_openai(
    texts: list[str],
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    # TODO: @yichuan-w add progress bar only in build mode
    """Compute embeddings using OpenAI API"""
    try:
        import openai
    except ImportError as e:
        raise ImportError(f"OpenAI package not installed: {e}")

    # Validate input list
    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")
    # Extra validation: abort early if any item is empty/whitespace
    invalid_count = sum(1 for t in texts if not isinstance(t, str) or not t.strip())
    if invalid_count > 0:
        raise ValueError(
            f"Found {invalid_count} empty/invalid text(s) in input. Upstream should filter before calling OpenAI."
        )

    # Extract base_url and api_key from provider_options if not provided directly
    provider_options = provider_options or {}
    effective_base_url = base_url or provider_options.get("base_url")
    effective_api_key = api_key or provider_options.get("api_key")

    resolved_base_url = resolve_openai_base_url(effective_base_url)
    resolved_api_key = resolve_openai_api_key(effective_api_key)

    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    # Create OpenAI client
    client = openai.OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

    logger.info(
        f"Computing embeddings for {len(texts)} texts using OpenAI API, model: '{model_name}'"
    )

    # Apply prompt template if provided
    # Priority: build_prompt_template (new format) > prompt_template (old format)
    prompt_template = provider_options.get("build_prompt_template") or provider_options.get(
        "prompt_template"
    )

    if prompt_template:
        logger.warning(f"Applying prompt template: '{prompt_template}'")
        texts = [f"{prompt_template}{text}" for text in texts]

    # Query token limit and apply truncation
    token_limit = get_model_token_limit(model_name, base_url=effective_base_url)
    logger.info(f"Using token limit: {token_limit} for model '{model_name}'")
    texts = truncate_to_token_limit(texts, token_limit)

    # OpenAI has limits on batch size and input length
    max_batch_size = 800  # Conservative batch size because the token limit is 300K
    all_embeddings = []
    # get the avg len of texts
    avg_len = sum(len(text) for text in texts) / len(texts)
    # if avg len is less than 1000, use the max batch size
    if avg_len > 300:
        max_batch_size = 500

    # Gemini's OpenAI-compatible endpoint hard-limits embedding batches to 100 inputs per request.
    # If we exceed this, the API returns:
    #   "BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch"
    if "generativelanguage.googleapis.com" in (resolved_base_url or ""):
        max_batch_size = min(max_batch_size, 100)
        logger.info(
            "Detected Gemini OpenAI-compatible base_url; capping embedding batch_size to %d.",
            max_batch_size,
        )

    # Alibaba Cloud DashScope's OpenAI-compatible endpoint hard-limits embedding batches
    # to 10 inputs per request (e.g. text-embedding-v4). Exceeding this returns HTTP 400:
    #   "InternalError.Algo.InvalidParameter: Value error, batch size is invalid,
    #    it should not be larger than 10.: input.contents"
    if "dashscope" in (resolved_base_url or ""):
        max_batch_size = min(max_batch_size, 10)
        logger.info(
            "Detected DashScope OpenAI-compatible base_url; capping embedding batch_size to %d.",
            max_batch_size,
        )

    # if avg len is less than 1000, use the max batch size

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
            response = client.embeddings.create(model=model_name, input=batch_texts)
            batch_embeddings = [embedding.embedding for embedding in response.data]

            # Verify we got the expected number of embeddings
            if len(batch_embeddings) != len(batch_texts):
                logger.warning(
                    f"Expected {len(batch_texts)} embeddings but got {len(batch_embeddings)}"
                )

            # Only take the number of embeddings that match the batch size
            all_embeddings.extend(batch_embeddings[: len(batch_texts)])
        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")
    return embeddings

