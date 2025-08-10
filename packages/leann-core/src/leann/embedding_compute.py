"""
Unified embedding computation module
Consolidates all embedding computation logic using SentenceTransformer
Preserves all optimization parameters to ensure performance
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
import torch

# Set up logger with proper level
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logger.setLevel(log_level)

# Global model cache to avoid repeated loading
_model_cache: Dict[str, Any] = {}


def compute_embeddings(
    texts: List[str],
    model_name: str,
    mode: str = "sentence-transformers",
    is_build: bool = False,
    batch_size: int = 32,
    adaptive_optimization: bool = True,
) -> np.ndarray:
    """
    Unified embedding computation entry point

    Args:
        texts: List of texts to compute embeddings for
        model_name: Model name
        mode: Computation mode ('sentence-transformers', 'openai', 'mlx', 'ollama')
        is_build: Whether this is a build operation (shows progress bar)
        batch_size: Batch size for processing
        adaptive_optimization: Whether to use adaptive optimization based on batch size

    Returns:
        Normalized embeddings array, shape: (len(texts), embedding_dim)
    """
    if mode == "sentence-transformers":
        return compute_embeddings_sentence_transformers(
            texts,
            model_name,
            is_build=is_build,
            batch_size=batch_size,
            adaptive_optimization=adaptive_optimization,
        )
    elif mode == "openai":
        return compute_embeddings_openai(texts, model_name)
    elif mode == "mlx":
        return compute_embeddings_mlx(texts, model_name)
    elif mode == "ollama":
        return compute_embeddings_ollama(texts, model_name, is_build=is_build)
    else:
        raise ValueError(f"Unsupported embedding mode: {mode}")


def compute_embeddings_sentence_transformers(
    texts: List[str],
    model_name: str,
    use_fp16: bool = True,
    device: str = "auto",
    batch_size: int = 32,
    is_build: bool = False,
    adaptive_optimization: bool = True,
) -> np.ndarray:
    """
    Compute embeddings using SentenceTransformer with model caching and adaptive optimization

    Args:
        texts: List of texts to compute embeddings for
        model_name: Model name
        use_fp16: Whether to use FP16 precision
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        batch_size: Batch size for processing
        is_build: Whether this is a build operation (shows progress bar)
        adaptive_optimization: Whether to use adaptive optimization based on batch size
    """
    # Handle empty input
    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")
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

    # Apply optimizations based on benchmark results
    if adaptive_optimization:
        # Use optimal batch_size constants for different devices based on benchmark results
        if device == "mps":
            batch_size = 128  # MPS optimal batch size from benchmark
            if model_name == "Qwen/Qwen3-Embedding-0.6B":
                batch_size = 32
        elif device == "cuda":
            batch_size = 256  # CUDA optimal batch size
        # Keep original batch_size for CPU

    # Create cache key
    cache_key = f"sentence_transformers_{model_name}_{device}_{use_fp16}_optimized"

    # Check if model is already cached
    if cache_key in _model_cache:
        logger.info(f"Using cached optimized model: {model_name}")
        model = _model_cache[cache_key]
    else:
        logger.info(f"Loading and caching optimized SentenceTransformer model: {model_name}")
        from sentence_transformers import SentenceTransformer

        logger.info(f"Using device: {device}")

        # Apply hardware optimizations
        if device == "cuda":
            # TODO: Haven't tested this yet
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.set_per_process_memory_fraction(0.9)
        elif device == "mps":
            try:
                if hasattr(torch.mps, "set_per_process_memory_fraction"):
                    torch.mps.set_per_process_memory_fraction(0.9)
            except AttributeError:
                logger.warning("Some MPS optimizations not available in this PyTorch version")
        elif device == "cpu":
            # TODO: Haven't tested this yet
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            try:
                torch.backends.mkldnn.enabled = True
            except AttributeError:
                pass

        # Prepare optimized model and tokenizer parameters
        model_kwargs = {
            "torch_dtype": torch.float16 if use_fp16 else torch.float32,
            "low_cpu_mem_usage": True,
            "_fast_init": True,
            "attn_implementation": "eager",  # Use eager attention for speed
        }

        tokenizer_kwargs = {
            "use_fast": True,
            "padding": True,
            "truncation": True,
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

        # Apply additional optimizations based on mode
        if use_fp16 and device in ["cuda", "mps"]:
            try:
                model = model.half()
                logger.info(f"Applied FP16 precision: {model_name}")
            except Exception as e:
                logger.warning(f"FP16 optimization failed: {e}")

        # Apply torch.compile optimization
        if device in ["cuda", "mps"]:
            try:
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
                logger.info(f"Applied torch.compile optimization: {model_name}")
            except Exception as e:
                logger.warning(f"torch.compile optimization failed: {e}")

        # Set model to eval mode and disable gradients for inference
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        # Cache the model
        _model_cache[cache_key] = model
        logger.info(f"Model cached: {cache_key}")

    # Compute embeddings with optimized inference mode
    logger.info(f"Starting embedding computation... (batch_size: {batch_size})")

    # Use torch.inference_mode for optimal performance
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=is_build,  # Don't show progress bar in server environment
            convert_to_numpy=True,
            normalize_embeddings=False,
            device=device,
        )

    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    # Validate results
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise RuntimeError(f"Detected NaN or Inf values in embeddings, model: {model_name}")

    return embeddings


def compute_embeddings_openai(texts: List[str], model_name: str) -> np.ndarray:
    # TODO: @yichuan-w add progress bar only in build mode
    """Compute embeddings using OpenAI API"""
    try:
        import os

        import openai
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
    print(f"len of texts: {len(texts)}")

    # OpenAI has limits on batch size and input length
    max_batch_size = 1000  # Conservative batch size
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
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")
    print(f"len of embeddings: {len(embeddings)}")
    return embeddings


def compute_embeddings_mlx(chunks: List[str], model_name: str, batch_size: int = 16) -> np.ndarray:
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


def compute_embeddings_ollama(
    texts: List[str], model_name: str, is_build: bool = False, host: str = "http://localhost:11434"
) -> np.ndarray:
    """
    Compute embeddings using Ollama API.

    Args:
        texts: List of texts to compute embeddings for
        model_name: Ollama model name (e.g., "nomic-embed-text", "mxbai-embed-large")
        is_build: Whether this is a build operation (shows progress bar)
        host: Ollama host URL (default: http://localhost:11434)

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

    logger.info(
        f"Computing embeddings for {len(texts)} texts using Ollama API, model: '{model_name}'"
    )

    # Check if Ollama is running
    try:
        response = requests.get(f"{host}/api/version", timeout=5)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        error_msg = (
            f"❌ Could not connect to Ollama at {host}.\n\n"
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
        response = requests.get(f"{host}/api/tags", timeout=5)
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

        # Check if model exists (handle versioned names)
        model_found = any(
            model_name == name.split(":")[0] or model_name == name for name in model_names
        )

        if not model_found:
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

        # Verify the model supports embeddings by testing it
        try:
            test_response = requests.post(
                f"{host}/api/embeddings", json={"model": model_name, "prompt": "test"}, timeout=10
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

    # Process embeddings with optimized concurrent processing
    import requests

    def get_single_embedding(text_idx_tuple):
        """Helper function to get embedding for a single text."""
        text, idx = text_idx_tuple
        max_retries = 3
        retry_count = 0

        # Truncate very long texts to avoid API issues
        truncated_text = text[:8000] if len(text) > 8000 else text

        while retry_count < max_retries:
            try:
                response = requests.post(
                    f"{host}/api/embeddings",
                    json={"model": model_name, "prompt": truncated_text},
                    timeout=30,
                )
                response.raise_for_status()

                result = response.json()
                embedding = result.get("embedding")

                if embedding is None:
                    raise ValueError(f"No embedding returned for text {idx}")

                return idx, embedding

            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.warning(f"Timeout for text {idx} after {max_retries} retries")
                    return idx, None

            except Exception as e:
                if retry_count >= max_retries - 1:
                    logger.error(f"Failed to get embedding for text {idx}: {e}")
                    return idx, None
                retry_count += 1

        return idx, None

    # Determine if we should use concurrent processing
    use_concurrent = (
        len(texts) > 5 and not is_build
    )  # Don't use concurrent in build mode to avoid overwhelming
    max_workers = min(4, len(texts))  # Limit concurrent requests to avoid overwhelming Ollama

    all_embeddings = [None] * len(texts)  # Pre-allocate list to maintain order
    failed_indices = []

    if use_concurrent:
        logger.info(
            f"Using concurrent processing with {max_workers} workers for {len(texts)} texts"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(get_single_embedding, (text, idx)): idx
                for idx, text in enumerate(texts)
            }

            # Add progress bar for concurrent processing
            try:
                if is_build or len(texts) > 10:
                    from tqdm import tqdm

                    futures_iterator = tqdm(
                        as_completed(future_to_idx),
                        total=len(texts),
                        desc="Computing Ollama embeddings",
                    )
                else:
                    futures_iterator = as_completed(future_to_idx)
            except ImportError:
                futures_iterator = as_completed(future_to_idx)

            # Collect results as they complete
            for future in futures_iterator:
                try:
                    idx, embedding = future.result()
                    if embedding is not None:
                        all_embeddings[idx] = embedding
                    else:
                        failed_indices.append(idx)
                except Exception as e:
                    idx = future_to_idx[future]
                    logger.error(f"Exception for text {idx}: {e}")
                    failed_indices.append(idx)

    else:
        # Sequential processing with progress bar
        show_progress = is_build or len(texts) > 10

        try:
            if show_progress:
                from tqdm import tqdm

                iterator = tqdm(
                    enumerate(texts), total=len(texts), desc="Computing Ollama embeddings"
                )
            else:
                iterator = enumerate(texts)
        except ImportError:
            iterator = enumerate(texts)

        for idx, text in iterator:
            result_idx, embedding = get_single_embedding((text, idx))
            if embedding is not None:
                all_embeddings[idx] = embedding
            else:
                failed_indices.append(idx)

    # Handle failed embeddings
    if failed_indices:
        if len(failed_indices) == len(texts):
            raise RuntimeError("Failed to compute any embeddings")

        logger.warning(f"Failed to compute embeddings for {len(failed_indices)}/{len(texts)} texts")

        # Use zero embeddings as fallback for failed ones
        valid_embedding = next((e for e in all_embeddings if e is not None), None)
        if valid_embedding:
            embedding_dim = len(valid_embedding)
            for idx in failed_indices:
                all_embeddings[idx] = [0.0] * embedding_dim

    # Remove None values and convert to numpy array
    all_embeddings = [e for e in all_embeddings if e is not None]

    # Convert to numpy array and normalize
    embeddings = np.array(all_embeddings, dtype=np.float32)

    # Normalize embeddings (L2 normalization)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero

    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    return embeddings
