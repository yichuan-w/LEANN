"""
Unified embedding computation module
Consolidates all embedding computation logic using SentenceTransformer
Preserves all optimization parameters to ensure performance
"""

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch

from .settings import resolve_ollama_host, resolve_openai_api_key, resolve_openai_base_url

# Set up logger with proper level
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logger.setLevel(log_level)

# Global model cache to avoid repeated loading
_model_cache: dict[str, Any] = {}


def compute_embeddings(
    texts: list[str],
    model_name: str,
    mode: str = "sentence-transformers",
    is_build: bool = False,
    batch_size: int = 32,
    adaptive_optimization: bool = True,
    manual_tokenize: bool = False,
    max_length: int = 512,
    provider_options: Optional[dict[str, Any]] = None,
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
    provider_options = provider_options or {}

    if mode == "sentence-transformers":
        return compute_embeddings_sentence_transformers(
            texts,
            model_name,
            is_build=is_build,
            batch_size=batch_size,
            adaptive_optimization=adaptive_optimization,
            manual_tokenize=manual_tokenize,
            max_length=max_length,
        )
    elif mode == "openai":
        return compute_embeddings_openai(
            texts,
            model_name,
            base_url=provider_options.get("base_url"),
            api_key=provider_options.get("api_key"),
        )
    elif mode == "mlx":
        return compute_embeddings_mlx(texts, model_name)
    elif mode == "ollama":
        return compute_embeddings_ollama(
            texts,
            model_name,
            is_build=is_build,
            host=provider_options.get("host"),
        )
    elif mode == "gemini":
        return compute_embeddings_gemini(texts, model_name, is_build=is_build)
    else:
        raise ValueError(f"Unsupported embedding mode: {mode}")


def compute_embeddings_sentence_transformers(
    texts: list[str],
    model_name: str,
    use_fp16: bool = True,
    device: str = "auto",
    batch_size: int = 32,
    is_build: bool = False,
    adaptive_optimization: bool = True,
    manual_tokenize: bool = False,
    max_length: int = 512,
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
    logger.info(
        f"Starting embedding computation... (batch_size: {batch_size}, manual_tokenize={manual_tokenize})"
    )

    start_time = time.time()
    if not manual_tokenize:
        # Use SentenceTransformer's optimized encode path (default)
        with torch.inference_mode():
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=is_build,  # Don't show progress bar in server environment
                convert_to_numpy=True,
                normalize_embeddings=False,
                device=device,
            )
        # Synchronize if CUDA to measure accurate wall time
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    else:
        # Manual tokenization + forward pass using HF AutoTokenizer/AutoModel
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as e:
            raise ImportError(f"transformers is required for manual_tokenize=True: {e}")

        # Cache tokenizer and model
        tok_cache_key = f"hf_tokenizer_{model_name}"
        mdl_cache_key = f"hf_model_{model_name}_{device}_{use_fp16}"
        if tok_cache_key in _model_cache and mdl_cache_key in _model_cache:
            hf_tokenizer = _model_cache[tok_cache_key]
            hf_model = _model_cache[mdl_cache_key]
            logger.info("Using cached HF tokenizer/model for manual path")
        else:
            logger.info("Loading HF tokenizer/model for manual tokenization path")
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            torch_dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
            hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
            hf_model.to(device)
            hf_model.eval()
            # Optional compile on supported devices
            if device in ["cuda", "mps"]:
                try:
                    hf_model = torch.compile(hf_model, mode="reduce-overhead", dynamic=True)  # type: ignore
                except Exception:
                    pass
            _model_cache[tok_cache_key] = hf_tokenizer
            _model_cache[mdl_cache_key] = hf_model

        all_embeddings: list[np.ndarray] = []
        # Progress bar when building or for large inputs
        show_progress = is_build or len(texts) > 32
        try:
            if show_progress:
                from tqdm import tqdm  # type: ignore

                batch_iter = tqdm(
                    range(0, len(texts), batch_size),
                    desc="Embedding (manual)",
                    unit="batch",
                )
            else:
                batch_iter = range(0, len(texts), batch_size)
        except Exception:
            batch_iter = range(0, len(texts), batch_size)

        start_time_manual = time.time()
        with torch.inference_mode():
            for start_index in batch_iter:
                end_index = min(start_index + batch_size, len(texts))
                batch_texts = texts[start_index:end_index]
                tokenize_start_time = time.time()
                inputs = hf_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokenize_end_time = time.time()
                logger.info(
                    f"Tokenize time taken: {tokenize_end_time - tokenize_start_time} seconds"
                )
                # Print shapes of all input tensors for debugging
                for k, v in inputs.items():
                    print(f"inputs[{k!r}] shape: {getattr(v, 'shape', type(v))}")
                to_device_start_time = time.time()
                inputs = {k: v.to(device) for k, v in inputs.items()}
                to_device_end_time = time.time()
                logger.info(
                    f"To device time taken: {to_device_end_time - to_device_start_time} seconds"
                )
                forward_start_time = time.time()
                outputs = hf_model(**inputs)
                forward_end_time = time.time()
                logger.info(f"Forward time taken: {forward_end_time - forward_start_time} seconds")
                last_hidden_state = outputs.last_hidden_state  # (B, L, H)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    # Fallback: assume all tokens are valid
                    pooled = last_hidden_state.mean(dim=1)
                else:
                    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
                    masked = last_hidden_state * mask
                    lengths = mask.sum(dim=1).clamp(min=1)
                    pooled = masked.sum(dim=1) / lengths
                # Move to CPU float32
                batch_embeddings = pooled.detach().to("cpu").float().numpy()
                all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings).astype(np.float32, copy=False)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        end_time = time.time()
        logger.info(f"Manual tokenize time taken: {end_time - start_time_manual} seconds")
    end_time = time.time()
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")
    logger.info(f"Time taken: {end_time - start_time} seconds")

    # Validate results
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise RuntimeError(f"Detected NaN or Inf values in embeddings, model: {model_name}")

    return embeddings


def compute_embeddings_openai(
    texts: list[str],
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
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

    resolved_base_url = resolve_openai_base_url(base_url)
    resolved_api_key = resolve_openai_api_key(api_key)

    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    # Cache OpenAI client
    cache_key = f"openai_client::{resolved_base_url}"
    if cache_key in _model_cache:
        client = _model_cache[cache_key]
    else:
        client = openai.OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        _model_cache[cache_key] = client
        logger.info("OpenAI client cached")

    logger.info(
        f"Computing embeddings for {len(texts)} texts using OpenAI API, model: '{model_name}'"
    )
    print(f"len of texts: {len(texts)}")

    # OpenAI has limits on batch size and input length
    max_batch_size = 800  # Conservative batch size because the token limit is 300K
    all_embeddings = []
    # get the avg len of texts
    avg_len = sum(len(text) for text in texts) / len(texts)
    print(f"avg len of texts: {avg_len}")
    # if avg len is less than 1000, use the max batch size
    if avg_len > 300:
        max_batch_size = 500

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
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")
    print(f"len of embeddings: {len(embeddings)}")
    return embeddings


def compute_embeddings_mlx(chunks: list[str], model_name: str, batch_size: int = 16) -> np.ndarray:
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
    texts: list[str],
    model_name: str,
    is_build: bool = False,
    host: Optional[str] = None,
) -> np.ndarray:
    """
    Compute embeddings using Ollama API with simplified batch processing.

    Uses batch size of 32 for MPS/CPU and 128 for CUDA to optimize performance.

    Args:
        texts: List of texts to compute embeddings for
        model_name: Ollama model name (e.g., "nomic-embed-text", "mxbai-embed-large")
        is_build: Whether this is a build operation (shows progress bar)
        host: Ollama host URL (defaults to environment or http://localhost:11434)

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

        # Verify the model supports embeddings by testing it
        try:
            test_response = requests.post(
                f"{resolved_host}/api/embeddings",
                json={"model": model_name, "prompt": "test"},
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

    logger.info(f"Using batch size: {batch_size}")

    def get_batch_embeddings(batch_texts):
        """Get embeddings for a batch of texts."""
        all_embeddings = []
        failed_indices = []

        for i, text in enumerate(batch_texts):
            max_retries = 3
            retry_count = 0

            # Truncate very long texts to avoid API issues
            truncated_text = text[:8000] if len(text) > 8000 else text
            while retry_count < max_retries:
                try:
                    response = requests.post(
                        f"{resolved_host}/api/embeddings",
                        json={"model": model_name, "prompt": truncated_text},
                        timeout=30,
                    )
                    response.raise_for_status()

                    result = response.json()
                    embedding = result.get("embedding")

                    if embedding is None:
                        raise ValueError(f"No embedding returned for text {i}")

                    if not isinstance(embedding, list) or len(embedding) == 0:
                        raise ValueError(f"Invalid embedding format for text {i}")

                    all_embeddings.append(embedding)
                    break

                except requests.exceptions.Timeout:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.warning(f"Timeout for text {i} after {max_retries} retries")
                        failed_indices.append(i)
                        all_embeddings.append(None)
                        break

                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to get embedding for text {i}: {e}")
                        failed_indices.append(i)
                        all_embeddings.append(None)
                        break
        return all_embeddings, failed_indices

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
        batch_iterator = tqdm(range(num_batches), desc="Computing Ollama embeddings")
    else:
        batch_iterator = range(num_batches)

    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        batch_embeddings, batch_failed = get_batch_embeddings(batch_texts)

        # Adjust failed indices to global indices
        global_failed = [start_idx + idx for idx in batch_failed]
        all_failed_indices.extend(global_failed)
        all_embeddings.extend(batch_embeddings)

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
