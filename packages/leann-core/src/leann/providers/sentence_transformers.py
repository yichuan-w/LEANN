"""Sentence-transformers embedding provider."""

import logging
import os
import time
from typing import Any, Protocol, cast

import numpy as np

from ..embedding_compute import _model_cache

logger = logging.getLogger(__name__)

_DEFAULT_CUDA_BATCH_SIZE = 256
_DEFAULT_MPS_BATCH_SIZE = 128


class _SentenceTransformerLike(Protocol):
    def eval(self) -> Any: ...
    def parameters(self) -> Any: ...
    def encode(self, *args: Any, **kwargs: Any) -> Any: ...
    def half(self) -> Any: ...


# Token limit registry for embedding models
# Used as fallback when dynamic discovery fails (e.g., LM Studio, OpenAI)
# Ollama models use dynamic discovery via /api/show
EMBEDDING_MODEL_LIMITS = {
    # Nomic models (common across servers)
    "nomic-embed-text": 2048,  # Corrected from 512 - verified via /api/show
    "nomic-embed-text-v1.5": 2048,
    "nomic-embed-text-v2": 512,
    # Other embedding models
    "mxbai-embed-large": 512,
    "all-minilm": 512,
    "bge-m3": 8192,
    "snowflake-arctic-embed": 512,
    # OpenAI models
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-ada-002": 8192,
}

# Runtime cache for dynamically discovered token limits
# Key: (model_name, base_url), Value: token_limit
# Prevents repeated SDK/API calls for the same model
_token_limit_cache: dict[tuple[str, str], int] = {}


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s must be >= 1; using default %d", name, default)
        return default
    return value


def _resolve_cpu_thread_count() -> int:
    return _parse_positive_int_env("LEANN_CPU_THREADS", min(8, os.cpu_count() or 4))


def _resolve_adaptive_batch_size(device: str, model_name: str) -> int:
    if device == "cuda":
        return _parse_positive_int_env("LEANN_CUDA_BATCH_SIZE", _DEFAULT_CUDA_BATCH_SIZE)
    if device == "mps":
        default = 32 if model_name == "Qwen/Qwen3-Embedding-0.6B" else _DEFAULT_MPS_BATCH_SIZE
        return _parse_positive_int_env("LEANN_MPS_BATCH_SIZE", default)
    return 32


def _cap_cuda_batch_by_vram(requested: int, max_length: int = 512) -> int:
    auto = os.getenv("LEANN_CUDA_AUTO_BATCH", "1").lower()
    if auto in ("0", "false", "no"):
        return requested
    import torch

    if not torch.cuda.is_available():
        return requested
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:
        return requested

    # Eager-attention peak memory scales ~O(seq^2) per sequence in the batch.
    bytes_per_seq = max(8_000_000, max_length * max_length * 32)
    budget = int(free_bytes * 0.2)
    max_by_vram = max(1, budget // bytes_per_seq)
    capped = min(requested, max_by_vram)
    if capped < requested:
        logger.info(
            "Capping CUDA embedding batch size %d -> %d (%.2f GiB free VRAM)",
            requested,
            capped,
            free_bytes / (1024**3),
        )
    return capped


def _encode_with_oom_retry(
    model: _SentenceTransformerLike,
    texts: list[str],
    batch_size: int,
    *,
    is_build: bool,
    device: str,
) -> Any:
    import torch

    current = batch_size
    while True:
        try:
            with torch.inference_mode():
                return model.encode(
                    texts,
                    batch_size=current,
                    show_progress_bar=is_build,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    device=device,
                )
        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            if torch.cuda.is_available():
                oom = oom or isinstance(exc, torch.cuda.OutOfMemoryError)
            if not oom or current <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(1, current // 2)
            logger.warning(
                "CUDA OOM at embedding batch_size=%d; retrying with batch_size=%d",
                current,
                next_batch,
            )
            current = next_batch


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
    import torch

    outer_start_time = time.time()
    # Handle empty input
    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")
    logger.info(
        f"Computing embeddings for {len(texts)} texts using SentenceTransformer, model: '{model_name}'"
    )

    # Auto-detect device
    if device == "auto":
        # Check environment variable first
        env_device = os.getenv("LEANN_EMBEDDING_DEVICE")
        if env_device:
            device = env_device
            logger.info(f"Using device from LEANN_EMBEDDING_DEVICE: {device}")
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Apply optimizations based on benchmark results
    if adaptive_optimization:
        batch_size = _resolve_adaptive_batch_size(device, model_name)

    # Create cache key
    cache_key = f"sentence_transformers_{model_name}_{device}_{use_fp16}_optimized"

    pre_model_init_end_time = time.time()
    logger.debug(
        "compute_embeddings_sentence_transformers pre-model-init time "
        f"(device/batch selection etc.): {pre_model_init_end_time - outer_start_time:.6f}s"
    )

    # Check if model is already cached
    start_time = time.time()
    if cache_key in _model_cache:
        logger.info(f"Using cached optimized model: {model_name}")
        model = cast(_SentenceTransformerLike, _model_cache[cache_key])
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
            # No device-level init for MPS. set_per_process_memory_fraction causes
            # greedy allocation; torch.compile causes graph buffer bloat. Cache
            # clearing is handled per-batch in the compute loop below.
            pass
        elif device == "cpu":
            # TODO: Haven't tested this yet
            torch.set_num_threads(_resolve_cpu_thread_count())
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
            # Try loading with advanced parameters first (newer versions)
            local_model_kwargs = model_kwargs.copy()
            local_tokenizer_kwargs = tokenizer_kwargs.copy()
            local_model_kwargs["local_files_only"] = True
            local_tokenizer_kwargs["local_files_only"] = True

            model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=local_model_kwargs,
                tokenizer_kwargs=local_tokenizer_kwargs,
                local_files_only=True,
            )
            logger.info("Model loaded successfully! (local + optimized)")
        except TypeError as e:
            if "model_kwargs" in str(e) or "tokenizer_kwargs" in str(e):
                logger.warning(
                    f"Advanced parameters not supported ({e}), using basic initialization..."
                )
                # Fallback to basic initialization for older versions
                try:
                    model = SentenceTransformer(
                        model_name,
                        device=device,
                        local_files_only=True,
                    )
                    logger.info("Model loaded successfully! (local + basic)")
                except Exception as e2:
                    logger.warning(f"Local loading failed ({e2}), trying network download...")
                    model = SentenceTransformer(
                        model_name,
                        device=device,
                        local_files_only=False,
                    )
                    logger.info("Model loaded successfully! (network + basic)")
            else:
                raise
        except Exception as e:
            logger.warning(f"Local loading failed ({e}), trying network download...")
            # Fallback to network loading with advanced parameters
            try:
                network_model_kwargs = model_kwargs.copy()
                network_tokenizer_kwargs = tokenizer_kwargs.copy()
                network_model_kwargs["local_files_only"] = False
                network_tokenizer_kwargs["local_files_only"] = False

                model = SentenceTransformer(
                    model_name,
                    device=device,
                    model_kwargs=network_model_kwargs,
                    tokenizer_kwargs=network_tokenizer_kwargs,
                    local_files_only=False,
                )
                logger.info("Model loaded successfully! (network + optimized)")
            except TypeError as e2:
                if "model_kwargs" in str(e2) or "tokenizer_kwargs" in str(e2):
                    logger.warning(
                        f"Advanced parameters not supported ({e2}), using basic network loading..."
                    )
                    model = SentenceTransformer(
                        model_name,
                        device=device,
                        local_files_only=False,
                    )
                    logger.info("Model loaded successfully! (network + basic)")
                else:
                    raise

        # Apply additional optimizations based on mode
        if use_fp16 and device in ["cuda", "mps"]:
            try:
                model = model.half()
                logger.info(f"Applied FP16 precision: {model_name}")
            except Exception as e:
                logger.warning(f"FP16 optimization failed: {e}")

        # Apply torch.compile optimization
        if device == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
                logger.info(f"Applied torch.compile optimization: {model_name}")
            except Exception as e:
                logger.warning(f"torch.compile optimization failed: {e}")

        model = cast(_SentenceTransformerLike, model)

        # Set model to eval mode and disable gradients for inference
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        # Cache the model
        _model_cache[cache_key] = model
        logger.info(f"Model cached: {cache_key}")

        end_time = time.time()

        # Compute embeddings with optimized inference mode
        logger.info(
            f"Starting embedding computation... (batch_size: {batch_size}, manual_tokenize={manual_tokenize})"
        )
        logger.info(f"start sentence transformers {model} takes {end_time - start_time}")

    if device == "cuda" and adaptive_optimization:
        batch_size = _cap_cuda_batch_by_vram(batch_size, max_length=max_length)

    start_time = time.time()
    if not manual_tokenize:
        # Use SentenceTransformer's optimized encode path (default)
        embeddings = _encode_with_oom_retry(
            model,
            texts,
            batch_size,
            is_build=is_build,
            device=device,
        )
        # Synchronize if CUDA to measure accurate wall time
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    else:
        # Manual tokenization + forward pass using HF AutoTokenizer/AutoModel.
        # This path is reserved for an aggressively optimized FP pipeline
        # (no quantization), mainly for experimentation.
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise ImportError(f"transformers is required for manual_tokenize=True: {e}")

        tok_cache_key = f"hf_tokenizer_{model_name}"
        mdl_cache_key = f"hf_model_{model_name}_{device}_{use_fp16}_fp"

        if tok_cache_key in _model_cache and mdl_cache_key in _model_cache:
            hf_tokenizer = _model_cache[tok_cache_key]
            hf_model = _model_cache[mdl_cache_key]
            logger.info("Using cached HF tokenizer/model for manual FP path")
        else:
            logger.info("Loading HF tokenizer/model for manual FP path")
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            torch_dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
            hf_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )
            hf_model.to(device)

            hf_model.eval()
            # Optional compile on supported devices
            if device == "cuda":
                try:
                    hf_model = torch.compile(hf_model, mode="reduce-overhead", dynamic=True)
                    logger.info(
                        f"Applied torch.compile to HF model for {model_name} "
                        f"(device={device}, dtype={torch_dtype})"
                    )
                except Exception as exc:
                    logger.warning(f"torch.compile optimization failed: {exc}")

            _model_cache[tok_cache_key] = hf_tokenizer
            _model_cache[mdl_cache_key] = hf_model

        all_embeddings: list[np.ndarray] = []
        # Progress bar when building or for large inputs
        show_progress = is_build or len(texts) > 32
        try:
            if show_progress:
                from tqdm import tqdm

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
                inputs = hf_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = hf_model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # (B, L, H)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    pooled = last_hidden_state.mean(dim=1)
                else:
                    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
                    masked = last_hidden_state * mask
                    lengths = mask.sum(dim=1).clamp(min=1)
                    pooled = masked.sum(dim=1) / lengths
                batch_embeddings = pooled.detach().to("cpu").float().numpy()
                all_embeddings.append(batch_embeddings)
                if device == "mps":
                    try:
                        torch.mps.empty_cache()
                    except (RuntimeError, AttributeError):
                        pass

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

    outer_end_time = time.time()
    logger.debug(
        "compute_embeddings_sentence_transformers total time "
        f"(function entry -> return): {outer_end_time - outer_start_time:.6f}s"
    )

    return embeddings
