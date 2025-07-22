"""
Unified embedding computation module
Consolidates all embedding computation logic using SentenceTransformer
Preserves all optimization parameters to ensure performance
"""

import numpy as np
import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Global model cache to avoid repeated loading
_model_cache: Dict[str, Any] = {}


def compute_embeddings(
    texts: List[str],
    model_name: str,
    mode: str = "sentence-transformers",
    is_build: bool = False,
) -> np.ndarray:
    """
    Unified embedding computation entry point

    Args:
        texts: List of texts to compute embeddings for
        model_name: Model name
        mode: Computation mode ('sentence-transformers', 'openai', 'mlx')

    Returns:
        Normalized embeddings array, shape: (len(texts), embedding_dim)
    """
    if mode == "sentence-transformers":
        return compute_embeddings_sentence_transformers(
            texts, model_name, is_build=is_build
        )
    elif mode == "openai":
        return compute_embeddings_openai(texts, model_name)
    elif mode == "mlx":
        return compute_embeddings_mlx(texts, model_name)
    else:
        raise ValueError(f"Unsupported embedding mode: {mode}")


def compute_embeddings_sentence_transformers(
    texts: List[str],
    model_name: str,
    use_fp16: bool = True,
    device: str = "auto",
    batch_size: int = 32,
    is_build: bool = False,
) -> np.ndarray:
    """
    Compute embeddings using SentenceTransformer with model caching
    """
    logger.info(
        f"Computing embeddings for {len(texts)} texts using SentenceTransformer, model: '{model_name}'"
    )

    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Create cache key
    cache_key = f"sentence_transformers_{model_name}_{device}_{use_fp16}"

    # Check if model is already cached
    if cache_key in _model_cache:
        logger.info(f"Using cached model: {model_name}")
        model = _model_cache[cache_key]
    else:
        logger.info(f"Loading and caching SentenceTransformer model: {model_name}")
        from sentence_transformers import SentenceTransformer

        logger.info(f"Using device: {device}")

        # Prepare model and tokenizer optimization parameters
        model_kwargs = {
            "torch_dtype": torch.float16 if use_fp16 else torch.float32,
            "low_cpu_mem_usage": True,
            "_fast_init": True,
        }

        tokenizer_kwargs = {
            "use_fast": True,
        }

        try:
            # Try local loading first
            model_kwargs["local_files_only"] = True
            tokenizer_kwargs["local_files_only"] = True

            model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                local_files_only=True,
            )
            logger.info("Model loaded successfully! (local + optimized)")
        except Exception as e:
            logger.warning(f"Local loading failed ({e}), trying network download...")
            # Fallback to network loading
            model_kwargs["local_files_only"] = False
            tokenizer_kwargs["local_files_only"] = False

            model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                local_files_only=False,
            )
            logger.info("Model loaded successfully! (network + optimized)")

        # Apply additional optimizations (if supported)
        if use_fp16 and device in ["cuda", "mps"]:
            try:
                model = model.half()
                model = torch.compile(model)
                logger.info(f"Using FP16 precision and compile optimization: {model_name}")
            except Exception as e:
                logger.warning(f"FP16 or compile optimization failed: {e}")

        # Cache the model
        _model_cache[cache_key] = model
        logger.info(f"Model cached: {cache_key}")

    # Compute embeddings
    logger.info("Starting embedding computation...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=is_build,  # Don't show progress bar in server environment
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    )

    print(
        f"INFO: Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}"
    )

    # Validate results
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise RuntimeError(
            f"Detected NaN or Inf values in embeddings, model: {model_name}"
        )

    return embeddings


def compute_embeddings_openai(texts: List[str], model_name: str) -> np.ndarray:
    # TODO: @yichuan-w add progress bar only in build mode
    """Compute embeddings using OpenAI API"""
    try:
        import openai
        import os
    except ImportError as e:
        raise ImportError(f"OpenAI package not installed: {e}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    # Cache OpenAI client
    cache_key = "openai_client"
    if cache_key in _model_cache:
        client = _model_cache[cache_key]
    else:
        client = openai.OpenAI(api_key=api_key)
        _model_cache[cache_key] = client
        logger.info("OpenAI client cached")

    logger.info(
        f"Computing embeddings for {len(texts)} texts using OpenAI API, model: '{model_name}'"
    )

    # OpenAI has limits on batch size and input length
    max_batch_size = 100  # Conservative batch size
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
            response = client.embeddings.create(model=model_name, input=batch_texts)
            batch_embeddings = [embedding.embedding for embedding in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(
        f"INFO: Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}"
    )
    return embeddings


def compute_embeddings_mlx(
    chunks: List[str], model_name: str, batch_size: int = 16
) -> np.ndarray:
    # TODO: @yichuan-w add progress bar only in build mode
    """Computes embeddings using an MLX model."""
    try:
        import mlx.core as mx
        from mlx_lm.utils import load
    except ImportError as e:
        raise RuntimeError(
            "MLX or related libraries not available. Install with: uv pip install mlx mlx-lm"
        ) from e

    logger.info(
        f"Computing embeddings for {len(chunks)} chunks using MLX model '{model_name}' with batch_size={batch_size}..."
    )

    # Cache MLX model and tokenizer
    cache_key = f"mlx_{model_name}"
    if cache_key in _model_cache:
        logger.info(f"Using cached MLX model: {model_name}")
        model, tokenizer = _model_cache[cache_key]
    else:
        logger.info(f"Loading and caching MLX model: {model_name}")
        model, tokenizer = load(model_name)
        _model_cache[cache_key] = (model, tokenizer)
        logger.info(f"MLX model cached: {cache_key}")

    # Process chunks in batches with progress bar
    all_embeddings = []

    try:
        from tqdm import tqdm

        batch_iterator = tqdm(
            range(0, len(chunks), batch_size), desc="Computing embeddings", unit="batch"
        )
    except ImportError:
        batch_iterator = range(0, len(chunks), batch_size)

    for i in batch_iterator:
        batch_chunks = chunks[i : i + batch_size]

        # Tokenize all chunks in the batch
        batch_token_ids = []
        for chunk in batch_chunks:
            token_ids = tokenizer.encode(chunk)  # type: ignore
            batch_token_ids.append(token_ids)

        # Pad sequences to the same length for batch processing
        max_length = max(len(ids) for ids in batch_token_ids)
        padded_token_ids = []
        for token_ids in batch_token_ids:
            # Pad with tokenizer.pad_token_id or 0
            padded = token_ids + [0] * (max_length - len(token_ids))
            padded_token_ids.append(padded)

        # Convert to MLX array with batch dimension
        input_ids = mx.array(padded_token_ids)

        # Get embeddings for the batch
        embeddings = model(input_ids)

        # Mean pooling for each sequence in the batch
        pooled = embeddings.mean(axis=1)  # Shape: (batch_size, hidden_size)

        # Convert batch embeddings to numpy
        for j in range(len(batch_chunks)):
            pooled_list = pooled[j].tolist()  # Convert to list
            pooled_numpy = np.array(pooled_list, dtype=np.float32)
            all_embeddings.append(pooled_numpy)

    # Stack numpy arrays
    return np.stack(all_embeddings)
