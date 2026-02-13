import concurrent.futures
import glob
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _ensure_repo_paths_importable(current_file: str) -> None:
    """Make local leann packages importable without installing (mirrors multi-vector-leann.py)."""
    _repo_root = Path(current_file).resolve().parents[3]
    _leann_core_src = _repo_root / "packages" / "leann-core" / "src"
    _leann_hnsw_pkg = _repo_root / "packages" / "leann-backend-hnsw"
    if str(_leann_core_src) not in sys.path:
        sys.path.append(str(_leann_core_src))
    if str(_leann_hnsw_pkg) not in sys.path:
        sys.path.append(str(_leann_hnsw_pkg))


def _find_backend_module_file() -> Optional[Path]:
    """Best-effort locate the backend leann_multi_vector.py file, avoiding this file."""
    this_file = Path(__file__).resolve()
    candidates: list[Path] = []

    # Common in-repo location
    repo_root = this_file.parents[3]
    candidates.append(repo_root / "packages" / "leann-backend-hnsw" / "leann_multi_vector.py")
    candidates.append(
        repo_root / "packages" / "leann-backend-hnsw" / "src" / "leann_multi_vector.py"
    )

    for cand in candidates:
        try:
            if cand.exists() and cand.resolve() != this_file:
                return cand.resolve()
        except Exception:
            pass

    # Fallback: scan sys.path for another leann_multi_vector.py different from this file
    for p in list(sys.path):
        try:
            cand = Path(p) / "leann_multi_vector.py"
            if cand.exists() and cand.resolve() != this_file:
                return cand.resolve()
        except Exception:
            continue
    return None


_BACKEND_LEANN_CLASS: Optional[type] = None


def _get_backend_leann_multi_vector() -> type:
    """Load backend LeannMultiVector class even if this file shadows its module name."""
    global _BACKEND_LEANN_CLASS
    if _BACKEND_LEANN_CLASS is not None:
        return _BACKEND_LEANN_CLASS

    backend_path = _find_backend_module_file()
    if backend_path is None:
        # Fallback to local implementation in this module
        try:
            cls = LeannMultiVector
            _BACKEND_LEANN_CLASS = cls
            return cls
        except Exception as e:
            raise ImportError(
                "Could not locate backend 'leann_multi_vector.py' and no local implementation found. "
                "Ensure the leann backend is available under packages/leann-backend-hnsw or installed."
            ) from e

    import importlib.util

    module_name = "leann_hnsw_backend_module"
    spec = importlib.util.spec_from_file_location(module_name, str(backend_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for backend module at {backend_path}")
    backend_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = backend_module
    spec.loader.exec_module(backend_module)

    if not hasattr(backend_module, "LeannMultiVector"):
        raise ImportError(f"'LeannMultiVector' not found in backend module at {backend_path}")
    _BACKEND_LEANN_CLASS = backend_module.LeannMultiVector
    return _BACKEND_LEANN_CLASS


def _natural_sort_key(name: str) -> int:
    m = re.search(r"\d+", name)
    return int(m.group()) if m else 0


def _load_images_from_dir(
    pages_dir: str, recursive: bool = False
) -> tuple[list[str], list[Image.Image]]:
    """
    Load images from a directory.

    Args:
        pages_dir: Directory path containing images
        recursive: If True, recursively search subdirectories (default: False)

    Returns:
        Tuple of (filepaths, images)
    """

    # Supported image extensions
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.webp", "*.WEBP")

    if recursive:
        # Recursive search
        filepaths = []
        for ext in extensions:
            pattern = os.path.join(pages_dir, "**", ext)
            filepaths.extend(glob.glob(pattern, recursive=True))
    else:
        # Non-recursive search (only top-level directory)
        filepaths = []
        for ext in extensions:
            pattern = os.path.join(pages_dir, ext)
            filepaths.extend(glob.glob(pattern))

    # Sort files naturally
    filepaths = sorted(filepaths, key=lambda x: _natural_sort_key(os.path.basename(x)))

    # Load images with error handling
    images = []
    valid_filepaths = []
    failed_count = 0

    for filepath in filepaths:
        try:
            img = Image.open(filepath)
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
            valid_filepaths.append(filepath)
        except Exception as e:
            failed_count += 1
            print(f"Warning: Failed to load image {filepath}: {e}")
            continue

    if failed_count > 0:
        print(
            f"Warning: Failed to load {failed_count} image(s) out of {len(filepaths)} total files"
        )

    return valid_filepaths, images


def _maybe_convert_pdf_to_images(pdf_path: Optional[str], pages_dir: str, dpi: int = 200) -> None:
    if not pdf_path:
        return
    os.makedirs(pages_dir, exist_ok=True)
    try:
        from pdf2image import convert_from_path
    except Exception as e:
        raise RuntimeError(
            "pdf2image is required to convert PDF to images. Install via pip install pdf2image"
        ) from e
    images = convert_from_path(pdf_path, dpi=dpi)
    for i, image in enumerate(images):
        image.save(os.path.join(pages_dir, f"page_{i + 1}.png"), "PNG")


def _select_device_and_dtype():
    import torch
    from colpali_engine.utils.torch_utils import get_torch_device

    device_str = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    device = get_torch_device(device_str)
    # Stable dtype selection to avoid NaNs:
    # - CUDA: prefer bfloat16 if supported, else float16
    # - MPS: use float32 (fp16 on MPS can produce NaNs in some ops)
    # - CPU: float32
    if device_str == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # Better stability/perf on Ampere+
        except Exception:
            pass
    elif device_str == "mps":
        dtype = torch.float32
    else:
        dtype = torch.float32
    return device_str, device, dtype


def _load_colvision(model_choice: str):
    import os

    import torch
    from colpali_engine.models import (
        ColPali,
        ColQwen2,
        ColQwen2_5,
        ColQwen2_5_Processor,
        ColQwen2Processor,
    )
    from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
    from transformers.utils.import_utils import is_flash_attn_2_available

    # Force HuggingFace Hub to use HF endpoint, avoid Google Drive
    # Set environment variables to ensure models are downloaded from HuggingFace
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Log model loading info
    logger.info(f"Loading ColVision model: {model_choice}")
    logger.info(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'not set')}")
    logger.info("Models will be downloaded from HuggingFace Hub, not Google Drive")

    device_str, device, dtype = _select_device_and_dtype()

    # Determine model name and type
    # IMPORTANT: Check colqwen2.5 BEFORE colqwen2 to avoid false matches
    model_choice_lower = model_choice.lower()
    if model_choice == "colqwen2":
        model_name = "vidore/colqwen2-v1.0"
        model_type = "colqwen2"
    elif model_choice == "colqwen2.5" or model_choice == "colqwen25":
        model_name = "vidore/colqwen2.5-v0.2"
        model_type = "colqwen2.5"
    elif model_choice == "colpali":
        model_name = "vidore/colpali-v1.2"
        model_type = "colpali"
    elif (
        "colqwen2.5" in model_choice_lower
        or "colqwen25" in model_choice_lower
        or "colqwen2_5" in model_choice_lower
    ):
        # Handle HuggingFace model names like "vidore/colqwen2.5-v0.2"
        model_name = model_choice
        model_type = "colqwen2.5"
    elif "colqwen2" in model_choice_lower and "colqwen2-v1.0" in model_choice_lower:
        # Handle HuggingFace model names like "vidore/colqwen2-v1.0" (but not colqwen2.5)
        model_name = model_choice
        model_type = "colqwen2"
    elif "colpali" in model_choice_lower:
        # Handle HuggingFace model names like "vidore/colpali-v1.2"
        model_name = model_choice
        model_type = "colpali"
    else:
        # Default to colpali for backward compatibility
        model_name = "vidore/colpali-v1.2"
        model_type = "colpali"

    # Load model based on type
    attn_implementation = (
        "flash_attention_2" if (device_str == "cuda" and is_flash_attn_2_available()) else "eager"
    )

    # Load model from HuggingFace Hub (not Google Drive)
    # Use local_files_only=False to ensure download from HF if not cached
    if model_type == "colqwen2.5":
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
            local_files_only=False,  # Ensure download from HuggingFace Hub
        ).eval()
        processor = ColQwen2_5_Processor.from_pretrained(model_name, local_files_only=False)
    elif model_type == "colqwen2":
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
            local_files_only=False,  # Ensure download from HuggingFace Hub
        ).eval()
        processor = ColQwen2Processor.from_pretrained(model_name, local_files_only=False)
    else:  # colpali
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=False,  # Ensure download from HuggingFace Hub
        ).eval()
        processor = cast(
            ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name, local_files_only=False)
        )

    return model_name, model, processor, device_str, device, dtype


def _embed_images(model, processor, images: list[Image.Image]) -> list[Any]:
    import torch
    from colpali_engine.utils.torch_utils import ListDataset
    from torch.utils.data import DataLoader

    # Ensure deterministic eval and autocast for stability
    model.eval()

    dataloader = DataLoader(
        dataset=ListDataset[Image.Image](images),
        batch_size=32,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )

    doc_vecs: list[Any] = []
    for batch_doc in tqdm(dataloader, desc="Embedding images"):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            # autocast on CUDA for bf16/fp16; on CPU/MPS stay in fp32
            if model.device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=model.dtype if model.dtype.is_floating_point else torch.bfloat16,
                ):
                    embeddings_doc = model(**batch_doc)
            else:
                embeddings_doc = model(**batch_doc)
        doc_vecs.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return doc_vecs


def _embed_queries(model, processor, queries: list[str]) -> list[Any]:
    import torch

    model.eval()

    # Match MTEB's exact query processing from ColPaliEngineWrapper.get_text_embeddings:
    # 1. MTEB receives batch["text"] which already includes instruction/prompt (from _combine_queries_with_instruction_text)
    # 2. Manually adds: query_prefix + text + query_augmentation_token * 10
    # 3. Calls processor.process_queries(batch) where batch is now a list of strings
    # 4. process_queries adds: query_prefix + text + suffix (suffix = query_augmentation_token * 10)
    #
    # This results in duplicate addition: query_prefix is added twice, query_augmentation_token * 20 total
    # We need to match this exactly to reproduce MTEB results

    all_embeds = []
    batch_size = 32  # Match MTEB's default batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size), desc="Embedding queries"):
            batch_queries = queries[i : i + batch_size]

            # Match MTEB: manually add query_prefix + text + query_augmentation_token * 10
            # Then process_queries will add them again (resulting in 20 augmentation tokens total)
            batch = [
                processor.query_prefix + t + processor.query_augmentation_token * 10
                for t in batch_queries
            ]
            inputs = processor.process_queries(batch)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            if model.device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=model.dtype if model.dtype.is_floating_point else torch.bfloat16,
                ):
                    outs = model(**inputs)
            else:
                outs = model(**inputs)

            # Match MTEB: convert to float32 on CPU
            all_embeds.extend(list(torch.unbind(outs.cpu().to(torch.float32))))

    return all_embeds


def _build_index(
    index_path: str, doc_vecs: list[Any], filepaths: list[str], images: list[Image.Image]
) -> Any:
    LeannMultiVector = _get_backend_leann_multi_vector()
    dim = int(doc_vecs[0].shape[-1])
    retriever = LeannMultiVector(index_path=index_path, dim=dim)
    retriever.create_collection()
    for i, vec in enumerate(doc_vecs):
        data = {
            "colbert_vecs": vec.float().numpy(),
            "doc_id": i,
            "filepath": filepaths[i],
            "image": images[i],  # Include the original image
        }
        retriever.insert(data)
    retriever.create_index()
    return retriever


def _load_retriever_if_index_exists(index_path: str) -> Optional[Any]:
    LeannMultiVector = _get_backend_leann_multi_vector()
    index_base = Path(index_path)
    # Check for the actual HNSW index file written by the backend + our sidecar files
    index_file = index_base.parent / f"{index_base.stem}.index"
    meta = index_base.parent / f"{index_base.name}.meta.json"
    labels = index_base.parent / f"{index_base.name}.labels.json"
    if index_file.exists() and meta.exists() and labels.exists():
        try:
            with open(meta, encoding="utf-8") as f:
                meta_json = json.load(f)
            dim = int(meta_json.get("dimensions", 128))
        except Exception:
            dim = 128
        return LeannMultiVector(index_path=index_path, dim=dim)
    return None


def _build_fast_plaid_index(
    index_path: str,
    doc_vecs: list[Any],
    filepaths: list[str],
    images: list[Image.Image],
) -> tuple[Any, float]:
    """
    Build a Fast-Plaid index from document embeddings.

    Args:
        index_path: Path to save the Fast-Plaid index
        doc_vecs: List of document embeddings (each is a tensor with shape [num_tokens, embedding_dim])
        filepaths: List of filepath identifiers for each document
        images: List of PIL Images corresponding to each document

    Returns:
        Tuple of (FastPlaid index object, build_time_in_seconds)
    """
    import torch
    from fast_plaid import search as fast_plaid_search

    print(f"    Preparing {len(doc_vecs)} document embeddings for Fast-Plaid...")
    _t0 = time.perf_counter()

    # Convert doc_vecs to list of tensors
    documents_embeddings = []
    for i, vec in enumerate(doc_vecs):
        if i % 1000 == 0:
            print(f"      Converting embedding {i}/{len(doc_vecs)}...")
        if not isinstance(vec, torch.Tensor):
            vec = (
                torch.tensor(vec)
                if isinstance(vec, np.ndarray)
                else torch.from_numpy(np.array(vec))
            )
        # Ensure float32 for Fast-Plaid
        if vec.dtype != torch.float32:
            vec = vec.float()
        documents_embeddings.append(vec)

    print(f"    Converted {len(documents_embeddings)} embeddings")
    if len(documents_embeddings) > 0:
        print(f"    First embedding shape: {documents_embeddings[0].shape}")
        print(f"    First embedding dtype: {documents_embeddings[0].dtype}")

    # Prepare metadata for Fast-Plaid
    print(f"    Preparing metadata for {len(filepaths)} documents...")
    metadata_list = []
    for i, filepath in enumerate(filepaths):
        metadata_list.append(
            {
                "filepath": filepath,
                "index": i,
            }
        )

    # Create Fast-Plaid index
    print(f"    Creating FastPlaid object with index path: {index_path}")
    try:
        fast_plaid_index = fast_plaid_search.FastPlaid(index=index_path)
        print("    FastPlaid object created successfully")
    except Exception as e:
        print(f"    Error creating FastPlaid object: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise

    print(f"    Calling fast_plaid_index.create() with {len(documents_embeddings)} documents...")
    try:
        fast_plaid_index.create(
            documents_embeddings=documents_embeddings,
            metadata=metadata_list,
        )
        print("    Fast-Plaid index created successfully")
    except Exception as e:
        print(f"    Error creating Fast-Plaid index: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise

    build_secs = time.perf_counter() - _t0

    # Save images separately (Fast-Plaid doesn't store images)
    print(f"    Saving {len(images)} images...")
    images_dir = Path(index_path) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(tqdm(images, desc="Saving images")):
        img_path = images_dir / f"doc_{i}.png"
        img.save(str(img_path))

    return fast_plaid_index, build_secs


def _fast_plaid_index_exists(index_path: str) -> bool:
    """
    Check if Fast-Plaid index exists by checking for key files.
    This avoids creating the FastPlaid object which may trigger memory allocation.

    Args:
        index_path: Path to the Fast-Plaid index

    Returns:
        True if index appears to exist, False otherwise
    """
    index_path_obj = Path(index_path)
    if not index_path_obj.exists() or not index_path_obj.is_dir():
        return False

    # Fast-Plaid creates a SQLite database file for metadata
    # Check for metadata.db as the most reliable indicator
    metadata_db = index_path_obj / "metadata.db"
    if metadata_db.exists() and metadata_db.stat().st_size > 0:
        return True

    # Also check if directory has any files (might be incomplete index)
    try:
        if any(index_path_obj.iterdir()):
            return True
    except Exception:
        pass

    return False


def _load_fast_plaid_index_if_exists(index_path: str) -> Optional[Any]:
    """
    Load Fast-Plaid index if it exists.
    First checks if index files exist, then creates the FastPlaid object.
    The actual index data loading happens lazily when search is called.

    Args:
        index_path: Path to the Fast-Plaid index

    Returns:
        FastPlaid index object if exists, None otherwise
    """
    try:
        from fast_plaid import search as fast_plaid_search

        # First check if index files exist without creating the object
        if not _fast_plaid_index_exists(index_path):
            return None

        # Now try to create FastPlaid object
        # This may trigger some memory allocation, but the full index loading is deferred
        fast_plaid_index = fast_plaid_search.FastPlaid(index=index_path)
        return fast_plaid_index
    except ImportError:
        # fast-plaid not installed
        return None
    except Exception as e:
        # Any error (including memory errors from Rust backend) - return None
        # The error will be caught and index will be rebuilt
        print(f"Warning: Could not load Fast-Plaid index: {type(e).__name__}: {e}")
        return None


def _search_fast_plaid(
    fast_plaid_index: Any,
    query_vec: Any,
    top_k: int,
) -> tuple[list[tuple[float, int]], float]:
    """
    Search Fast-Plaid index with a query embedding.

    Args:
        fast_plaid_index: FastPlaid index object
        query_vec: Query embedding tensor with shape [num_tokens, embedding_dim]
        top_k: Number of top results to return

    Returns:
        Tuple of (results_list, search_time_in_seconds)
        results_list: List of (score, doc_id) tuples
    """
    import torch

    _t0 = time.perf_counter()

    # Ensure query is a torch tensor
    if not isinstance(query_vec, torch.Tensor):
        q_vec_tensor = (
            torch.tensor(query_vec)
            if isinstance(query_vec, np.ndarray)
            else torch.from_numpy(np.array(query_vec))
        )
    else:
        q_vec_tensor = query_vec

    # Fast-Plaid expects shape [num_queries, num_tokens, embedding_dim]
    if q_vec_tensor.dim() == 2:
        q_vec_tensor = q_vec_tensor.unsqueeze(0)  # [1, num_tokens, embedding_dim]

    # Perform search
    scores = fast_plaid_index.search(
        queries_embeddings=q_vec_tensor,
        top_k=top_k,
        show_progress=True,
    )

    search_secs = time.perf_counter() - _t0

    # Convert Fast-Plaid results to same format as LEANN: list of (score, doc_id) tuples
    results = []
    if scores and len(scores) > 0:
        query_results = scores[0]
        # Fast-Plaid returns (doc_id, score), convert to (score, doc_id) to match LEANN format
        results = [(float(score), int(doc_id)) for doc_id, score in query_results]

    return results, search_secs


def _get_fast_plaid_image(index_path: str, doc_id: int) -> Optional[Image.Image]:
    """
    Retrieve image for a document from Fast-Plaid index.

    Args:
        index_path: Path to the Fast-Plaid index
        doc_id: Document ID returned by Fast-Plaid search

    Returns:
        PIL Image if found, None otherwise

    Note: Uses metadata['index'] to get the actual file index, as Fast-Plaid
    doc_id may differ from the file naming index.
    """
    # First get metadata to find the actual index used for file naming
    metadata = _get_fast_plaid_metadata(index_path, doc_id)
    if metadata is None:
        # Fallback: try using doc_id directly
        file_index = doc_id
    else:
        # Use the 'index' field from metadata, which matches the file naming
        file_index = metadata.get("index", doc_id)

    images_dir = Path(index_path) / "images"
    image_path = images_dir / f"doc_{file_index}.png"

    if image_path.exists():
        return Image.open(image_path)

    # If not found with index, try doc_id as fallback
    if file_index != doc_id:
        fallback_path = images_dir / f"doc_{doc_id}.png"
        if fallback_path.exists():
            return Image.open(fallback_path)

    return None


def _get_fast_plaid_metadata(index_path: str, doc_id: int) -> Optional[dict]:
    """
    Retrieve metadata for a document from Fast-Plaid index.

    Args:
        index_path: Path to the Fast-Plaid index
        doc_id: Document ID

    Returns:
        Dictionary with metadata if found, None otherwise
    """
    try:
        from fast_plaid import filtering

        metadata_list = filtering.get(index=index_path, subset=[doc_id])
        if metadata_list and len(metadata_list) > 0:
            return metadata_list[0]
    except Exception:
        pass
    return None


def _generate_similarity_map(
    model,
    processor,
    image: Image.Image,
    query: str,
    token_idx: Optional[int] = None,
    output_path: Optional[str] = None,
) -> tuple[int, float]:
    import torch
    from colpali_engine.interpretability import (
        get_similarity_maps_from_embeddings,
        plot_similarity_map,
    )

    batch_images = processor.process_images([image]).to(model.device)
    batch_queries = processor.process_queries([query]).to(model.device)

    with torch.no_grad():
        image_embeddings = model.forward(**batch_images)
        query_embeddings = model.forward(**batch_queries)

    n_patches = processor.get_n_patches(
        image_size=image.size,
        spatial_merge_size=getattr(model, "spatial_merge_size", None),
    )
    image_mask = processor.get_image_mask(batch_images)

    batched_similarity_maps = get_similarity_maps_from_embeddings(
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask,
    )

    similarity_maps = batched_similarity_maps[0]

    # Determine token index if not provided: choose the token with highest max score
    if token_idx is None:
        per_token_max = similarity_maps.view(similarity_maps.shape[0], -1).max(dim=1).values
        token_idx = int(per_token_max.argmax().item())

    max_sim_score = similarity_maps[token_idx, :, :].max().item()

    if output_path:
        import matplotlib.pyplot as plt

        fig, ax = plot_similarity_map(
            image=image,
            similarity_map=similarity_maps[token_idx],
            figsize=(14, 14),
            show_colorbar=False,
        )
        ax.set_title(f"Token #{token_idx}. MaxSim score: {max_sim_score:.2f}", fontsize=12)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    return token_idx, float(max_sim_score)


class QwenVL:
    def __init__(self, device: str):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers.utils.import_utils import is_flash_attn_2_available

        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "eager"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype="auto",
            device_map=device,
            attn_implementation=attn_implementation,
        )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
        )

    def answer(self, query: str, images: list[Image.Image], max_new_tokens: int = 128) -> str:
        import base64
        from io import BytesIO

        from qwen_vl_utils import process_vision_info

        content = []
        for img in images:
            buffer = BytesIO()
            img.save(buffer, format="jpeg")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            content.append({"type": "image", "image": f"data:image;base64,{img_base64}"})
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# Ensure repo paths are importable for dynamic backend loading
_ensure_repo_paths_importable(__file__)

from leann_backend_hnsw.hnsw_backend import HNSWBuilder, HNSWSearcher  # noqa: E402


class LeannMultiVector:
    def __init__(
        self,
        index_path: str,
        dim: int = 128,
        distance_metric: str = "mips",
        m: int = 16,
        ef_construction: int = 500,
        is_compact: bool = False,
        is_recompute: bool = False,
        embedding_model_name: str = "colvision",
    ) -> None:
        self.index_path = index_path
        self.dim = dim
        self.embedding_model_name = embedding_model_name
        self._pending_items: list[dict] = []
        self._backend_kwargs = {
            "distance_metric": distance_metric,
            "M": m,
            "efConstruction": ef_construction,
            "is_compact": is_compact,
            "is_recompute": is_recompute,
        }
        self._labels_meta: list[dict] = []
        self._docid_to_indices: dict[int, list[int]] | None = None

    def _meta_dict(self) -> dict:
        return {
            "version": "1.0",
            "backend_name": "hnsw",
            "embedding_model": self.embedding_model_name,
            "embedding_mode": "custom",
            "dimensions": self.dim,
            "backend_kwargs": self._backend_kwargs,
            "is_compact": self._backend_kwargs.get("is_compact", True),
            "is_pruned": self._backend_kwargs.get("is_compact", True)
            and self._backend_kwargs.get("is_recompute", True),
        }

    def create_collection(self) -> None:
        path = Path(self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def insert(self, data: dict) -> None:
        self._pending_items.append(
            {
                "doc_id": int(data["doc_id"]),
                "filepath": data.get("filepath", ""),
                "colbert_vecs": [np.asarray(v, dtype=np.float32) for v in data["colbert_vecs"]],
                "image": data.get("image"),  # PIL Image object (optional)
            }
        )

    def _labels_path(self) -> Path:
        index_path_obj = Path(self.index_path)
        return index_path_obj.parent / f"{index_path_obj.name}.labels.json"

    def _meta_path(self) -> Path:
        index_path_obj = Path(self.index_path)
        return index_path_obj.parent / f"{index_path_obj.name}.meta.json"

    def _embeddings_path(self) -> Path:
        index_path_obj = Path(self.index_path)
        return index_path_obj.parent / f"{index_path_obj.name}.emb.npy"

    def _images_dir_path(self) -> Path:
        """Directory where original images are stored."""
        index_path_obj = Path(self.index_path)
        return index_path_obj.parent / f"{index_path_obj.name}.images"

    def create_index(self) -> None:
        if not self._pending_items:
            return

        embeddings: list[np.ndarray] = []
        labels_meta: list[dict] = []

        # Create images directory if needed
        images_dir = self._images_dir_path()
        images_dir.mkdir(parents=True, exist_ok=True)

        for item in self._pending_items:
            doc_id = int(item["doc_id"])
            filepath = item.get("filepath", "")
            colbert_vecs = item["colbert_vecs"]
            image = item.get("image")

            # Save image if provided
            image_path = ""
            if image is not None and isinstance(image, Image.Image):
                image_filename = f"doc_{doc_id}.png"
                image_path = str(images_dir / image_filename)
                image.save(image_path, "PNG")

            for seq_id, vec in enumerate(colbert_vecs):
                vec_np = np.asarray(vec, dtype=np.float32)
                embeddings.append(vec_np)
                labels_meta.append(
                    {
                        "id": f"{doc_id}:{seq_id}",
                        "doc_id": doc_id,
                        "seq_id": int(seq_id),
                        "filepath": filepath,
                        "image_path": image_path,  # Store the path to the saved image
                    }
                )

        if not embeddings:
            return

        embeddings_np = np.vstack(embeddings).astype(np.float32)
        print(embeddings_np.shape)

        builder = HNSWBuilder(**{**self._backend_kwargs, "dimensions": self.dim})
        ids = [str(i) for i in range(embeddings_np.shape[0])]
        builder.build(embeddings_np, ids, self.index_path)

        import json as _json

        with open(self._meta_path(), "w", encoding="utf-8") as f:
            _json.dump(self._meta_dict(), f, indent=2)
        with open(self._labels_path(), "w", encoding="utf-8") as f:
            _json.dump(labels_meta, f)

        # Persist embeddings for exact reranking
        np.save(self._embeddings_path(), embeddings_np)

        self._labels_meta = labels_meta

    def _load_labels_meta_if_needed(self) -> None:
        if self._labels_meta:
            return
        labels_path = self._labels_path()
        if labels_path.exists():
            import json as _json

            with open(labels_path, encoding="utf-8") as f:
                self._labels_meta = _json.load(f)

    def _build_docid_to_indices_if_needed(self) -> None:
        if self._docid_to_indices is not None:
            return
        self._load_labels_meta_if_needed()
        mapping: dict[int, list[int]] = {}
        for idx, meta in enumerate(self._labels_meta):
            try:
                doc_id = int(meta["doc_id"])
            except Exception:
                continue
            mapping.setdefault(doc_id, []).append(idx)
        self._docid_to_indices = mapping

    def search(
        self, data: np.ndarray, topk: int, first_stage_k: int = 50
    ) -> list[tuple[float, int]]:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        self._load_labels_meta_if_needed()

        searcher = HNSWSearcher(self.index_path, meta=self._meta_dict())
        raw = searcher.search(
            data,
            first_stage_k,
            recompute_embeddings=False,
            complexity=128,
            beam_width=1,
            prune_ratio=0.0,
            batch_size=0,
        )

        labels = raw.get("labels")
        distances = raw.get("distances")
        if labels is None or distances is None:
            return []

        doc_scores: dict[int, float] = {}
        B = len(labels)
        for b in range(B):
            per_doc_best: dict[int, float] = {}
            for k, sid in enumerate(labels[b]):
                try:
                    idx = int(sid)
                except Exception:
                    continue
                if 0 <= idx < len(self._labels_meta):
                    doc_id = int(self._labels_meta[idx]["doc_id"])
                else:
                    continue
                score = float(distances[b][k])
                if (doc_id not in per_doc_best) or (score > per_doc_best[doc_id]):
                    per_doc_best[doc_id] = score
            for doc_id, best_score in per_doc_best.items():
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + best_score

        scores = sorted(((v, k) for k, v in doc_scores.items()), key=lambda x: x[0], reverse=True)
        return scores[:topk] if len(scores) >= topk else scores

    def search_exact(
        self,
        data: np.ndarray,
        topk: int,
        *,
        first_stage_k: int = 200,
        max_workers: int = 32,
    ) -> list[tuple[float, int]]:
        """
        High-precision MaxSim reranking over candidate documents.

        Steps:
        1) Run a first-stage ANN to collect candidate doc_ids (using seq-level neighbors).
        2) For each candidate doc, load all its token embeddings and compute
           MaxSim(query_tokens, doc_tokens) exactly: sum(max(dot(q_i, d_j))).

        Returns top-k list of (score, doc_id).
        """
        # Normalize inputs
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        self._load_labels_meta_if_needed()
        self._build_docid_to_indices_if_needed()

        emb_path = self._embeddings_path()
        if not emb_path.exists():
            # Fallback to approximate if we don't have persisted embeddings
            return self.search(data, topk, first_stage_k=first_stage_k)

        # Memory-map embeddings to avoid loading all into RAM
        all_embeddings = np.load(emb_path, mmap_mode="r")
        if all_embeddings.dtype != np.float32:
            all_embeddings = all_embeddings.astype(np.float32)

        # First-stage ANN to collect candidate doc_ids
        searcher = HNSWSearcher(self.index_path, meta=self._meta_dict())
        raw = searcher.search(
            data,
            first_stage_k,
            recompute_embeddings=False,
            complexity=128,
            beam_width=1,
            prune_ratio=0.0,
            batch_size=0,
        )
        labels = raw.get("labels")
        if labels is None:
            return []
        candidate_doc_ids: set[int] = set()
        for batch in labels:
            for sid in batch:
                try:
                    idx = int(sid)
                except Exception:
                    continue
                if 0 <= idx < len(self._labels_meta):
                    candidate_doc_ids.add(int(self._labels_meta[idx]["doc_id"]))

        # Exact scoring per doc (parallelized)
        assert self._docid_to_indices is not None

        def _score_one(doc_id: int) -> tuple[float, int]:
            token_indices = self._docid_to_indices.get(doc_id, [])
            if not token_indices:
                return (0.0, doc_id)
            doc_vecs = np.asarray(all_embeddings[token_indices], dtype=np.float32)
            # (Q, D) x (P, D)^T -> (Q, P) then MaxSim over P, sum over Q
            sim = np.dot(data, doc_vecs.T)
            # nan-safe
            sim = np.nan_to_num(sim, nan=-1e30, posinf=1e30, neginf=-1e30)
            score = sim.max(axis=2).sum(axis=1) if sim.ndim == 3 else sim.max(axis=1).sum()
            return (float(score), doc_id)

        scores: list[tuple[float, int]] = []
        # load and core time
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_score_one, doc_id) for doc_id in candidate_doc_ids]
            for fut in concurrent.futures.as_completed(futures):
                scores.append(fut.result())
        end_time = time.time()
        print(f"Number of candidate doc ids: {len(candidate_doc_ids)}")
        print(f"Time taken in load and core time: {end_time - start_time} seconds")
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:topk] if len(scores) >= topk else scores

    def search_exact_all(
        self,
        data: np.ndarray,
        topk: int,
        *,
        max_workers: int = 32,
    ) -> list[tuple[float, int]]:
        """
        Exact MaxSim over ALL documents (no ANN pre-filtering).

        This computes, for each document, sum_i max_j dot(q_i, d_j).
        It memory-maps the persisted token-embedding matrix for scalability.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        self._load_labels_meta_if_needed()
        self._build_docid_to_indices_if_needed()

        emb_path = self._embeddings_path()
        if not emb_path.exists():
            return self.search(data, topk)
        all_embeddings = np.load(emb_path, mmap_mode="r")
        if all_embeddings.dtype != np.float32:
            all_embeddings = all_embeddings.astype(np.float32)

        assert self._docid_to_indices is not None
        candidate_doc_ids = list(self._docid_to_indices.keys())

        def _score_one(doc_id: int, _all_embeddings=all_embeddings) -> tuple[float, int]:
            token_indices = self._docid_to_indices.get(doc_id, [])
            if not token_indices:
                return (0.0, doc_id)
            doc_vecs = np.asarray(_all_embeddings[token_indices], dtype=np.float32)
            sim = np.dot(data, doc_vecs.T)
            sim = np.nan_to_num(sim, nan=-1e30, posinf=1e30, neginf=-1e30)
            score = sim.max(axis=2).sum(axis=1) if sim.ndim == 3 else sim.max(axis=1).sum()
            return (float(score), doc_id)

        scores: list[tuple[float, int]] = []
        # load and core time
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_score_one, d) for d in candidate_doc_ids]
            for fut in concurrent.futures.as_completed(futures):
                scores.append(fut.result())
        end_time = time.time()
        # print number of candidate doc ids
        print(f"Number of candidate doc ids: {len(candidate_doc_ids)}")
        print(f"Time taken in load and core time: {end_time - start_time} seconds")
        scores.sort(key=lambda x: x[0], reverse=True)
        del all_embeddings
        return scores[:topk] if len(scores) >= topk else scores

    def get_image(self, doc_id: int) -> Optional[Image.Image]:
        """
        Retrieve the original image for a given doc_id from the index.

        Args:
            doc_id: The document ID

        Returns:
            PIL Image object if found, None otherwise
        """
        self._load_labels_meta_if_needed()

        # Find the image_path for this doc_id (all seq_ids for same doc share the same image_path)
        for meta in self._labels_meta:
            if meta.get("doc_id") == doc_id:
                image_path = meta.get("image_path", "")
                if image_path and Path(image_path).exists():
                    return Image.open(image_path)
                break
        return None

    def get_metadata(self, doc_id: int) -> Optional[dict]:
        """
        Retrieve metadata for a given doc_id.

        Args:
            doc_id: The document ID

        Returns:
            Dictionary with metadata (filepath, image_path, etc.) if found, None otherwise
        """
        self._load_labels_meta_if_needed()

        for meta in self._labels_meta:
            if meta.get("doc_id") == doc_id:
                return {
                    "doc_id": doc_id,
                    "filepath": meta.get("filepath", ""),
                    "image_path": meta.get("image_path", ""),
                }
        return None


class ViDoReBenchmarkEvaluator:
    """
    A reusable class for evaluating ViDoRe benchmarks (v1 and v2).
    This class encapsulates common functionality for building indexes, searching, and evaluating.
    """

    def __init__(
        self,
        model_name: str,
        use_fast_plaid: bool = False,
        top_k: int = 100,
        first_stage_k: int = 500,
        k_values: Optional[list[int]] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: Model name ("colqwen2" or "colpali")
            use_fast_plaid: Whether to use Fast-Plaid instead of LEANN
            top_k: Top-k results to retrieve
            first_stage_k: First stage k for LEANN search
            k_values: List of k values for evaluation metrics
        """
        self.model_name = model_name
        self.use_fast_plaid = use_fast_plaid
        self.top_k = top_k
        self.first_stage_k = first_stage_k
        self.k_values = k_values if k_values is not None else [1, 3, 5, 10, 100]

        # Load model once (can be reused across tasks)
        self._model = None
        self._processor = None
        self._model_name_actual = None

    def _load_model_if_needed(self):
        """Lazy load the model."""
        if self._model is None:
            print(f"\nLoading model: {self.model_name}")
            self._model_name_actual, self._model, self._processor, _, _, _ = _load_colvision(
                self.model_name
            )
            print(f"Model loaded: {self._model_name_actual}")

    def build_index_from_corpus(
        self,
        corpus: dict[str, Image.Image],
        index_path: str,
        rebuild: bool = False,
    ) -> tuple[Any, list[str]]:
        """
        Build index from corpus images.

        Args:
            corpus: dict mapping corpus_id to PIL Image
            index_path: Path to save/load the index
            rebuild: Whether to rebuild even if index exists

        Returns:
            tuple: (retriever or fast_plaid_index object, list of corpus_ids in order)
        """
        self._load_model_if_needed()

        # Ensure consistent ordering
        corpus_ids = sorted(corpus.keys())
        images = [corpus[cid] for cid in corpus_ids]

        if self.use_fast_plaid:
            # Check if Fast-Plaid index exists
            if not rebuild and _load_fast_plaid_index_if_exists(index_path) is not None:
                print(f"Fast-Plaid index already exists at {index_path}")
                return _load_fast_plaid_index_if_exists(index_path), corpus_ids

            print(f"Building Fast-Plaid index at {index_path}...")
            print("Embedding images...")
            doc_vecs = _embed_images(self._model, self._processor, images)

            fast_plaid_index, build_time = _build_fast_plaid_index(
                index_path, doc_vecs, corpus_ids, images
            )
            print(f"Fast-Plaid index built in {build_time:.2f}s")
            return fast_plaid_index, corpus_ids
        else:
            # Check if LEANN index exists
            if not rebuild:
                retriever = _load_retriever_if_index_exists(index_path)
                if retriever is not None:
                    print(f"LEANN index already exists at {index_path}")
                    return retriever, corpus_ids

            print(f"Building LEANN index at {index_path}...")
            print("Embedding images...")
            doc_vecs = _embed_images(self._model, self._processor, images)

            retriever = _build_index(index_path, doc_vecs, corpus_ids, images)
            print("LEANN index built")
            return retriever, corpus_ids

    def search_queries(
        self,
        queries: dict[str, str],
        corpus_ids: list[str],
        index_or_retriever: Any,
        fast_plaid_index_path: Optional[str] = None,
        task_prompt: Optional[dict[str, str]] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Search queries against the index.

        Args:
            queries: dict mapping query_id to query text
            corpus_ids: list of corpus_ids in the same order as the index
            index_or_retriever: index or retriever object
            fast_plaid_index_path: path to Fast-Plaid index (for metadata)
            task_prompt: Optional dict with prompt for query (e.g., {"query": "..."})

        Returns:
            results: dict mapping query_id to dict of {corpus_id: score}
        """
        self._load_model_if_needed()

        print(f"Searching {len(queries)} queries (top_k={self.top_k})...")

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        # Note: ColPaliEngineWrapper does NOT use task prompt from metadata
        # It uses query_prefix + text + query_augmentation_token (handled in _embed_queries)
        # So we don't append task_prompt here to match MTEB behavior

        # Embed queries
        print("Embedding queries...")
        query_vecs = _embed_queries(self._model, self._processor, query_texts)

        results = {}

        for query_id, query_vec in zip(tqdm(query_ids, desc="Searching"), query_vecs):
            if self.use_fast_plaid:
                # Fast-Plaid search
                search_results, _ = _search_fast_plaid(index_or_retriever, query_vec, self.top_k)
                query_results = {}
                for score, doc_id in search_results:
                    if doc_id < len(corpus_ids):
                        corpus_id = corpus_ids[doc_id]
                        query_results[corpus_id] = float(score)
            else:
                # LEANN search
                import torch

                query_np = (
                    query_vec.float().numpy() if isinstance(query_vec, torch.Tensor) else query_vec
                )
                search_results = index_or_retriever.search_exact(query_np, topk=self.top_k)
                query_results = {}
                for score, doc_id in search_results:
                    if doc_id < len(corpus_ids):
                        corpus_id = corpus_ids[doc_id]
                        query_results[corpus_id] = float(score)

            results[query_id] = query_results

        return results

    @staticmethod
    def evaluate_results(
        results: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        k_values: Optional[list[int]] = None,
    ) -> dict[str, float]:
        """
        Evaluate retrieval results using NDCG and other metrics.

        Args:
            results: dict mapping query_id to dict of {corpus_id: score}
            qrels: dict mapping query_id to dict of {corpus_id: relevance_score}
            k_values: List of k values for evaluation metrics

        Returns:
            Dictionary of metric scores
        """
        try:
            from mteb._evaluators.retrieval_metrics import (
                calculate_retrieval_scores,
                make_score_dict,
            )
        except ImportError:
            raise ImportError(
                "pytrec_eval is required for evaluation. Install with: pip install pytrec-eval"
            )

        if k_values is None:
            k_values = [1, 3, 5, 10, 100]

        # Check if we have any queries to evaluate
        if len(results) == 0:
            print("Warning: No queries to evaluate. Returning zero scores.")
            scores = {}
            for k in k_values:
                scores[f"ndcg_at_{k}"] = 0.0
                scores[f"map_at_{k}"] = 0.0
                scores[f"recall_at_{k}"] = 0.0
                scores[f"precision_at_{k}"] = 0.0
                scores[f"mrr_at_{k}"] = 0.0
            return scores

        print(f"Evaluating results with k_values={k_values}...")
        print(f"Before filtering: {len(results)} results, {len(qrels)} qrels")

        # Filter to ensure qrels and results have the same query set
        # This matches MTEB behavior: only evaluate queries that exist in both
        # pytrec_eval only evaluates queries in qrels, so we need to ensure
        # results contains all queries in qrels, and filter out queries not in qrels
        results_filtered = {qid: res for qid, res in results.items() if qid in qrels}
        qrels_filtered = {
            qid: rel_docs for qid, rel_docs in qrels.items() if qid in results_filtered
        }

        print(f"After filtering: {len(results_filtered)} results, {len(qrels_filtered)} qrels")

        if len(results_filtered) != len(qrels_filtered):
            print(
                f"Warning: Mismatch between results ({len(results_filtered)}) and qrels ({len(qrels_filtered)}) queries"
            )
            missing_in_results = set(qrels.keys()) - set(results.keys())
            if missing_in_results:
                print(f"Queries in qrels but not in results: {len(missing_in_results)} queries")
                print(f"First 5 missing queries: {list(missing_in_results)[:5]}")

        # Convert qrels to pytrec_eval format
        qrels_pytrec = {}
        for qid, rel_docs in qrels_filtered.items():
            qrels_pytrec[qid] = dict(rel_docs.items())

        # Evaluate
        eval_result = calculate_retrieval_scores(
            results=results_filtered,
            qrels=qrels_pytrec,
            k_values=k_values,
        )

        # Format scores
        scores = make_score_dict(
            ndcg=eval_result.ndcg,
            _map=eval_result.map,
            recall=eval_result.recall,
            precision=eval_result.precision,
            mrr=eval_result.mrr,
            naucs=eval_result.naucs,
            naucs_mrr=eval_result.naucs_mrr,
            cv_recall=eval_result.cv_recall,
            task_scores={},
        )

        return scores
