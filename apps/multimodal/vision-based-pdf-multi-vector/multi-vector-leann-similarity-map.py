## Jupyter-style notebook script
# %%
# uv pip install matplotlib qwen_vl_utils
import argparse
import faulthandler
import os
import time
from typing import Any, Optional, cast

import numpy as np
from PIL import Image
from tqdm import tqdm

# Enable faulthandler to get stack trace on segfault
faulthandler.enable()


from leann_multi_vector import (  # utility functions/classes
    _ensure_repo_paths_importable,
    _load_images_from_dir,
    _maybe_convert_pdf_to_images,
    _load_colvision,
    _embed_images,
    _embed_queries,
    _build_index,
    _load_retriever_if_index_exists,
    _generate_similarity_map,
    _build_fast_plaid_index,
    _load_fast_plaid_index_if_exists,
    _search_fast_plaid,
    _get_fast_plaid_image,
    _get_fast_plaid_metadata,
    QwenVL,
)

_ensure_repo_paths_importable(__file__)

# %%
# Config
os.environ["TOKENIZERS_PARALLELISM"] = "false"
QUERY = "The paper talk about the latent video generative model and data curation in the related work part?"
MODEL: str = "colqwen2"  # "colpali" or "colqwen2"

# Data source: set to True to use the Hugging Face dataset example (recommended)
USE_HF_DATASET: bool = True
# Single dataset name (used when DATASET_NAMES is None)
DATASET_NAME: str = "weaviate/arXiv-AI-papers-multi-vector"
# Multiple datasets to combine (if provided, DATASET_NAME is ignored)
# Can be:
# - List of strings: ["dataset1", "dataset2"]
# - List of tuples: [("dataset1", "config1"), ("dataset2", None)]  # None = no config needed
# - Mixed: ["dataset1", ("dataset2", "config2")]
#
# Some potential datasets with images (may need IMAGE_FIELD_NAME adjustment):
# - "weaviate/arXiv-AI-papers-multi-vector" (current, has "page_image" field)
# - ("lmms-lab/DocVQA", "DocVQA") (has "image" field, document images, needs config)
# - ("lmms-lab/DocVQA", "InfographicVQA") (has "image" field, infographic images)
# - "pixparse/arxiv-papers" (if available, arXiv papers)
# - "allenai/ai2d" (AI2D diagram dataset, has "image" field)
# - "huggingface/document-images" (if available)
# Note: Check dataset structure first - some may need IMAGE_FIELD_NAME specified
# DATASET_NAMES: Optional[list[str | tuple[str, Optional[str]]]] = None
DATASET_NAMES = [
    "weaviate/arXiv-AI-papers-multi-vector",
    # ("lmms-lab/DocVQA", "DocVQA"),  # Specify config name for datasets with multiple configs
]
# Load multiple splits to get more data (e.g., ["train", "test", "validation"])
# Set to None to try loading all available splits automatically
DATASET_SPLITS: Optional[list[str]] = ["train", "test"]  # None = auto-detect all splits
# Image field name in the dataset (auto-detect if None)
# Common names: "page_image", "image", "images", "img"
IMAGE_FIELD_NAME: Optional[str] = None  # None = auto-detect
MAX_DOCS: Optional[int] = None  # limit number of pages to index; None = all

# Local pages (used when USE_HF_DATASET == False)
PDF: Optional[str] = None  # e.g., "./pdfs/2004.12832v2.pdf"
PAGES_DIR: str = "./pages"
# Custom folder path (takes precedence over USE_HF_DATASET and PAGES_DIR)
# If set, images will be loaded directly from this folder
CUSTOM_FOLDER_PATH: Optional[str] = None  # e.g., "/home/ubuntu/dr-tulu/agent/screenshots"
# Whether to recursively search subdirectories when loading from custom folder
CUSTOM_FOLDER_RECURSIVE: bool = False  # Set to True to search subdirectories

# Index + retrieval settings
# Use a different index path for larger dataset to avoid overwriting existing index
INDEX_PATH: str = "./indexes/colvision_large.leann"
# Fast-Plaid index settings (alternative to LEANN index)
# These are now command-line arguments (see CLI overrides section)
TOPK: int = 3
FIRST_STAGE_K: int = 500
REBUILD_INDEX: bool = False  # Set to True to force rebuild even if index exists

# Artifacts
SAVE_TOP_IMAGE: Optional[str] = "./figures/retrieved_page.png"
SIMILARITY_MAP: bool = True
SIM_TOKEN_IDX: int = 13  # -1 means auto-select the most salient token
SIM_OUTPUT: str = "./figures/similarity_map.png"
ANSWER: bool = True
MAX_NEW_TOKENS: int = 1024


# %%
# CLI overrides
parser = argparse.ArgumentParser(description="Multi-vector LEANN similarity map demo")
parser.add_argument(
    "--search-method",
    type=str,
    choices=["ann", "exact", "exact-all"],
    default="ann",
    help="Which search method to use: 'ann' (fast ANN), 'exact' (ANN + exact rerank), or 'exact-all' (exact over all docs).",
)
parser.add_argument(
    "--query",
    type=str,
    default=QUERY,
    help=f"Query string to search for. Default: '{QUERY}'",
)
parser.add_argument(
    "--use-fast-plaid",
    action="store_true",
    default=False,
    help="Set to True to use fast-plaid instead of LEANN. Default: False",
)
parser.add_argument(
    "--fast-plaid-index-path",
    type=str,
    default="./indexes/colvision_fastplaid",
    help="Path to the Fast-Plaid index. Default: './indexes/colvision_fastplaid'",
)
parser.add_argument(
    "--topk",
    type=int,
    default=TOPK,
    help=f"Number of top results to retrieve. Default: {TOPK}",
)
parser.add_argument(
    "--custom-folder",
    type=str,
    default=None,
    help="Path to a custom folder containing images to search. Takes precedence over dataset loading. Default: None",
)
parser.add_argument(
    "--recursive",
    action="store_true",
    default=False,
    help="Recursively search subdirectories when loading images from custom folder. Default: False",
)
parser.add_argument(
    "--rebuild-index",
    action="store_true",
    default=False,
    help="Force rebuild the index even if it already exists. Default: False (reuse existing index if available)",
)
cli_args, _unknown = parser.parse_known_args()
SEARCH_METHOD: str = cli_args.search_method
QUERY = cli_args.query  # Override QUERY with CLI argument if provided
USE_FAST_PLAID: bool = cli_args.use_fast_plaid
FAST_PLAID_INDEX_PATH: str = cli_args.fast_plaid_index_path
TOPK: int = cli_args.topk  # Override TOPK with CLI argument if provided
CUSTOM_FOLDER_PATH = cli_args.custom_folder if cli_args.custom_folder else CUSTOM_FOLDER_PATH  # Override with CLI argument if provided
CUSTOM_FOLDER_RECURSIVE = cli_args.recursive if cli_args.recursive else CUSTOM_FOLDER_RECURSIVE  # Override with CLI argument if provided
REBUILD_INDEX = cli_args.rebuild_index  # Override REBUILD_INDEX with CLI argument

# %%

# Step 1: Check if we can skip data loading (index already exists)
retriever: Optional[Any] = None
fast_plaid_index: Optional[Any] = None
need_to_build_index = REBUILD_INDEX

if USE_FAST_PLAID:
    # Fast-Plaid index handling
    if not REBUILD_INDEX:
        try:
            fast_plaid_index = _load_fast_plaid_index_if_exists(FAST_PLAID_INDEX_PATH)
            if fast_plaid_index is not None:
                print(f"✓ Fast-Plaid index found at {FAST_PLAID_INDEX_PATH}")
                need_to_build_index = False
            else:
                print(f"Fast-Plaid index not found, will build new index")
                need_to_build_index = True
        except Exception as e:
            # If loading fails (e.g., memory error, corrupted index), rebuild
            print(f"Warning: Failed to load Fast-Plaid index: {e}")
            print("Will rebuild the index...")
            need_to_build_index = True
            fast_plaid_index = None
    else:
        print(f"REBUILD_INDEX=True, will rebuild Fast-Plaid index")
        need_to_build_index = True
else:
    # Original LEANN index handling
    if not REBUILD_INDEX:
        retriever = _load_retriever_if_index_exists(INDEX_PATH)
        if retriever is not None:
            print(f"✓ Index loaded from {INDEX_PATH}")
            print(f"✓ Images available at: {retriever._images_dir_path()}")
            need_to_build_index = False
        else:
            print(f"Index not found, will build new index")
            need_to_build_index = True
    else:
        print(f"REBUILD_INDEX=True, will rebuild index")
        need_to_build_index = True

# Step 2: Load data only if we need to build the index
if need_to_build_index:
    print("Loading dataset...")
    # Check for custom folder path first (takes precedence)
    if CUSTOM_FOLDER_PATH:
        if not os.path.isdir(CUSTOM_FOLDER_PATH):
            raise RuntimeError(f"Custom folder path does not exist: {CUSTOM_FOLDER_PATH}")
        print(f"Loading images from custom folder: {CUSTOM_FOLDER_PATH}")
        if CUSTOM_FOLDER_RECURSIVE:
            print("  (recursive mode: searching subdirectories)")
        filepaths, images = _load_images_from_dir(CUSTOM_FOLDER_PATH, recursive=CUSTOM_FOLDER_RECURSIVE)
        print(f"  Found {len(filepaths)} image files")
        if not images:
            raise RuntimeError(
                f"No images found in {CUSTOM_FOLDER_PATH}. Ensure the folder contains image files (.png, .jpg, .jpeg, .webp)."
            )
        print(f"  Successfully loaded {len(images)} images")
        # Use filenames as identifiers instead of full paths for cleaner metadata
        filepaths = [os.path.basename(fp) for fp in filepaths]
    elif USE_HF_DATASET:
        from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

        # Determine which datasets to load
        if DATASET_NAMES is not None:
            dataset_names_to_load = DATASET_NAMES
            print(f"Loading {len(dataset_names_to_load)} datasets: {dataset_names_to_load}")
        else:
            dataset_names_to_load = [DATASET_NAME]
            print(f"Loading single dataset: {DATASET_NAME}")

        # Load and combine datasets
        all_datasets_to_concat = []

        for dataset_entry in dataset_names_to_load:
            # Handle both string and tuple formats
            if isinstance(dataset_entry, tuple):
                dataset_name, config_name = dataset_entry
            else:
                dataset_name = dataset_entry
                config_name = None

            print(f"\nProcessing dataset: {dataset_name}" + (f" (config: {config_name})" if config_name else ""))

            # Load dataset to check available splits
            # If config_name is provided, use it; otherwise try without config
            try:
                if config_name:
                    dataset_dict = load_dataset(dataset_name, config_name)
                else:
                    dataset_dict = load_dataset(dataset_name)
            except ValueError as e:
                if "Config name is missing" in str(e):
                    # Try to get available configs and suggest
                    from datasets import get_dataset_config_names
                    try:
                        available_configs = get_dataset_config_names(dataset_name)
                        raise ValueError(
                            f"Dataset '{dataset_name}' requires a config name. "
                            f"Available configs: {available_configs}. "
                            f"Please specify as: ('{dataset_name}', 'config_name')"
                        ) from e
                    except Exception:
                        raise ValueError(
                            f"Dataset '{dataset_name}' requires a config name. "
                            f"Please specify as: ('{dataset_name}', 'config_name')"
                        ) from e
                raise

            # Determine which splits to load
            if DATASET_SPLITS is None:
                # Auto-detect: try to load all available splits
                available_splits = list(dataset_dict.keys())
                print(f"  Auto-detected splits: {available_splits}")
                splits_to_load = available_splits
            else:
                splits_to_load = DATASET_SPLITS

            # Load and concatenate multiple splits for this dataset
            datasets_to_concat: list[Dataset] = []
            for split in splits_to_load:
                if split not in dataset_dict:
                    print(f"  Warning: Split '{split}' not found in dataset. Available splits: {list(dataset_dict.keys())}")
                    continue
                split_dataset = cast(Dataset, dataset_dict[split])
                print(f"  Loaded split '{split}': {len(split_dataset)} pages")
                datasets_to_concat.append(split_dataset)

            if not datasets_to_concat:
                print(f"  Warning: No valid splits found for {dataset_name}. Skipping.")
                continue

            # Concatenate splits for this dataset
            if len(datasets_to_concat) > 1:
                combined_dataset = concatenate_datasets(datasets_to_concat)
                print(f"  Concatenated {len(datasets_to_concat)} splits into {len(combined_dataset)} pages")
            else:
                combined_dataset = datasets_to_concat[0]

            all_datasets_to_concat.append(combined_dataset)

        if not all_datasets_to_concat:
            raise RuntimeError("No valid datasets or splits found.")

        # Concatenate all datasets
        if len(all_datasets_to_concat) > 1:
            dataset = concatenate_datasets(all_datasets_to_concat)
            print(f"\nConcatenated {len(all_datasets_to_concat)} datasets into {len(dataset)} total pages")
        else:
            dataset = all_datasets_to_concat[0]

        # Apply MAX_DOCS limit if specified
        N = len(dataset) if MAX_DOCS is None else min(MAX_DOCS, len(dataset))
        if N < len(dataset):
            print(f"Limiting to {N} pages (from {len(dataset)} total)")
            dataset = dataset.select(range(N))

        # Auto-detect image field name if not specified
        if IMAGE_FIELD_NAME is None:
            # Check multiple samples to find the most common image field
            # (useful when datasets are merged and may have different field names)
            possible_image_fields = ["page_image", "image", "images", "img", "page", "document_image"]
            field_counts = {}

            # Check first few samples to find image fields
            num_samples_to_check = min(10, len(dataset))
            for sample_idx in range(num_samples_to_check):
                sample = dataset[sample_idx]
                for field in possible_image_fields:
                    if field in sample and sample[field] is not None:
                        value = sample[field]
                        if isinstance(value, Image.Image) or (hasattr(value, 'size') and hasattr(value, 'mode')):
                            field_counts[field] = field_counts.get(field, 0) + 1

            # Choose the most common field, or first found if tied
            if field_counts:
                image_field = max(field_counts.items(), key=lambda x: x[1])[0]
                print(f"Auto-detected image field: '{image_field}' (found in {field_counts[image_field]}/{num_samples_to_check} samples)")
            else:
                # Fallback: check first sample only
                sample = dataset[0]
                image_field = None
                for field in possible_image_fields:
                    if field in sample:
                        value = sample[field]
                        if isinstance(value, Image.Image) or (hasattr(value, 'size') and hasattr(value, 'mode')):
                            image_field = field
                            break
                if image_field is None:
                    raise RuntimeError(
                        f"Could not auto-detect image field. Available fields: {list(sample.keys())}. "
                        f"Please specify IMAGE_FIELD_NAME manually."
                    )
                print(f"Auto-detected image field: '{image_field}'")
        else:
            image_field = IMAGE_FIELD_NAME
            if image_field not in dataset[0]:
                raise RuntimeError(
                    f"Image field '{image_field}' not found. Available fields: {list(dataset[0].keys())}"
                )

        filepaths: list[str] = []
        images: list[Image.Image] = []
        for i in tqdm(range(len(dataset)), desc="Loading dataset", total=len(dataset)):
            p = dataset[i]
            # Try to compose a descriptive identifier
            # Handle different dataset structures
            identifier_parts = []

            # Helper function to safely get field value
            def safe_get(field_name, default=None):
                if field_name in p and p[field_name] is not None:
                    return p[field_name]
                return default

            # Try to get various identifier fields
            if safe_get("paper_arxiv_id"):
                identifier_parts.append(f"arXiv:{p['paper_arxiv_id']}")
            if safe_get("paper_title"):
                identifier_parts.append(f"title:{p['paper_title']}")
            if safe_get("page_number") is not None:
                try:
                    identifier_parts.append(f"page:{int(p['page_number'])}")
                except (ValueError, TypeError):
                    # If conversion fails, use the raw value or skip
                    if p['page_number']:
                        identifier_parts.append(f"page:{p['page_number']}")
            if safe_get("page_id"):
                identifier_parts.append(f"id:{p['page_id']}")
            elif safe_get("questionId"):
                identifier_parts.append(f"qid:{p['questionId']}")
            elif safe_get("docId"):
                identifier_parts.append(f"docId:{p['docId']}")
            elif safe_get("id"):
                identifier_parts.append(f"id:{p['id']}")

            # If no identifier parts found, create one from index
            if identifier_parts:
                identifier = "|".join(identifier_parts)
            else:
                # Create identifier from available fields or index
                fallback_parts = []
                # Try common fields that might exist
                for field in ["ucsf_document_id", "docId", "questionId", "id"]:
                    if safe_get(field):
                        fallback_parts.append(f"{field}:{p[field]}")
                        break
                if fallback_parts:
                    identifier = "|".join(fallback_parts) + f"|idx:{i}"
                else:
                    identifier = f"doc_{i}"

            filepaths.append(identifier)

            # Get image - try detected field first, then fallback to other common fields
            img = None
            if image_field in p and p[image_field] is not None:
                img = p[image_field]
            else:
                # Fallback: try other common image field names
                for fallback_field in ["image", "page_image", "images", "img"]:
                    if fallback_field in p and p[fallback_field] is not None:
                        img = p[fallback_field]
                        break

            if img is None:
                raise RuntimeError(
                    f"No image found for sample {i}. Available fields: {list(p.keys())}. "
                    f"Expected field: {image_field}"
                )

            # Ensure it's a PIL Image
            if not isinstance(img, Image.Image):
                if hasattr(img, 'convert'):
                    img = img.convert('RGB')
                else:
                    img = Image.fromarray(img) if hasattr(img, '__array__') else Image.open(img)
            images.append(img)
    else:
        _maybe_convert_pdf_to_images(PDF, PAGES_DIR)
        filepaths, images = _load_images_from_dir(PAGES_DIR)
        if not images:
            raise RuntimeError(
                f"No images found in {PAGES_DIR}. Provide PDF path in PDF variable or ensure images exist."
            )
    print(f"Loaded {len(images)} images")

    # Memory check before loading model
    try:
        import psutil
        import torch
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"Memory usage after loading images: {mem_info.rss / 1024 / 1024 / 1024:.2f} GB")
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    except ImportError:
        pass
else:
    print("Skipping dataset loading (using existing index)")
    filepaths = []  # Not needed when using existing index
    images = []  # Not needed when using existing index


# %%
# Step 3: Load model and processor (only if we need to build index or perform search)
print("Step 3: Loading model and processor...")
print(f"  Model: {MODEL}")
try:
    import sys
    print(f"  Python version: {sys.version}")
    print(f"  Python executable: {sys.executable}")

    model_name, model, processor, device_str, device, dtype = _load_colvision(MODEL)
    print(f"✓ Using model={model_name}, device={device_str}, dtype={dtype}")

    # Memory check after loading model
    try:
        import psutil
        import torch
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"  Memory usage after loading model: {mem_info.rss / 1024 / 1024 / 1024:.2f} GB")
        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    except ImportError:
        pass
except Exception as e:
    print(f"✗ Error loading model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise


# %%

# %%
# Step 4: Build index if needed
if need_to_build_index:
    print("Step 4: Building index...")
    print(f"  Number of images: {len(images)}")
    print(f"  Number of filepaths: {len(filepaths)}")

    try:
        print("  Embedding images...")
        doc_vecs = _embed_images(model, processor, images)
        print(f"  Embedded {len(doc_vecs)} documents")
        print(f"  First doc vec shape: {doc_vecs[0].shape if len(doc_vecs) > 0 else 'N/A'}")
    except Exception as e:
        print(f"Error embedding images: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    if USE_FAST_PLAID:
        # Build Fast-Plaid index
        print("  Building Fast-Plaid index...")
        try:
            fast_plaid_index, build_secs = _build_fast_plaid_index(
                FAST_PLAID_INDEX_PATH, doc_vecs, filepaths, images
            )
            from pathlib import Path
            print(f"✓ Fast-Plaid index built in {build_secs:.3f}s")
            print(f"✓ Index saved to: {FAST_PLAID_INDEX_PATH}")
            print(f"✓ Images saved to: {Path(FAST_PLAID_INDEX_PATH) / 'images'}")
        except Exception as e:
            print(f"Error building Fast-Plaid index: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Clear memory
            print("  Clearing memory...")
            del images, filepaths, doc_vecs
    else:
        # Build original LEANN index
        try:
            retriever = _build_index(INDEX_PATH, doc_vecs, filepaths, images)
            print(f"✓ Index built and images saved to: {retriever._images_dir_path()}")
        except Exception as e:
            print(f"Error building LEANN index: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Clear memory
            print("  Clearing memory...")
            del images, filepaths, doc_vecs

# Note: Images are now stored separately, retriever/fast_plaid_index will reference them


# %%
# Step 5: Embed query and search
_t0 = time.perf_counter()
q_vec = _embed_queries(model, processor, [QUERY])[0]
query_embed_secs = time.perf_counter() - _t0

print(f"[Search] Method: {SEARCH_METHOD}")
print(f"[Timing] Query embedding: {query_embed_secs:.3f}s")

# Run the selected search method and time it
if USE_FAST_PLAID:
    # Fast-Plaid search
    if fast_plaid_index is None:
        fast_plaid_index = _load_fast_plaid_index_if_exists(FAST_PLAID_INDEX_PATH)
        if fast_plaid_index is None:
            raise RuntimeError(f"Fast-Plaid index not found at {FAST_PLAID_INDEX_PATH}")

    results, search_secs = _search_fast_plaid(fast_plaid_index, q_vec, TOPK)
    print(f"[Timing] Fast-Plaid Search: {search_secs:.3f}s")
else:
    # Original LEANN search
    query_np = q_vec.float().numpy()

    if SEARCH_METHOD == "ann":
        results = retriever.search(query_np, topk=TOPK, first_stage_k=FIRST_STAGE_K)
        search_secs = time.perf_counter() - _t0
        print(f"[Timing] Search (ANN): {search_secs:.3f}s (first_stage_k={FIRST_STAGE_K})")
    elif SEARCH_METHOD == "exact":
        results = retriever.search_exact(query_np, topk=TOPK, first_stage_k=FIRST_STAGE_K)
        search_secs = time.perf_counter() - _t0
        print(f"[Timing] Search (Exact rerank): {search_secs:.3f}s (first_stage_k={FIRST_STAGE_K})")
    elif SEARCH_METHOD == "exact-all":
        results = retriever.search_exact_all(query_np, topk=TOPK)
        search_secs = time.perf_counter() - _t0
        print(f"[Timing] Search (Exact all): {search_secs:.3f}s")
    else:
        results = []
if not results:
    print("No results found.")
else:
    print(f'Top {len(results)} results for query: "{QUERY}"')
    print("\n[DEBUG] Retrieval details:")
    top_images: list[Image.Image] = []
    image_hashes = {}  # Track image hashes to detect duplicates

    for rank, (score, doc_id) in enumerate(results, start=1):
        # Retrieve image and metadata based on index type
        if USE_FAST_PLAID:
            # Fast-Plaid: load image and get metadata
            image = _get_fast_plaid_image(FAST_PLAID_INDEX_PATH, doc_id)
            if image is None:
                print(f"Warning: Could not find image for doc_id {doc_id}")
                continue

            metadata = _get_fast_plaid_metadata(FAST_PLAID_INDEX_PATH, doc_id)
            path = metadata.get("filepath", f"doc_{doc_id}") if metadata else f"doc_{doc_id}"
            top_images.append(image)
        else:
            # Original LEANN: retrieve from retriever
            image = retriever.get_image(doc_id)
            if image is None:
                print(f"Warning: Could not retrieve image for doc_id {doc_id}")
                continue

            metadata = retriever.get_metadata(doc_id)
            path = metadata.get("filepath", "unknown") if metadata else "unknown"
            top_images.append(image)

        # Calculate image hash to detect duplicates
        import hashlib
        import io
        # Convert image to bytes for hashing
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        image_bytes = img_bytes.getvalue()
        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]

        # Check if this image was already seen
        duplicate_info = ""
        if image_hash in image_hashes:
            duplicate_info = f" [DUPLICATE of rank {image_hashes[image_hash]}]"
        else:
            image_hashes[image_hash] = rank

        # Print detailed information
        print(f"{rank}) doc_id={doc_id}, MaxSim={score:.4f}, Page={path}, ImageHash={image_hash}{duplicate_info}")
        if metadata:
            print(f"   Metadata: {metadata}")

    if SAVE_TOP_IMAGE:
        from pathlib import Path as _Path

        base = _Path(SAVE_TOP_IMAGE)
        base.parent.mkdir(parents=True, exist_ok=True)
        for rank, img in enumerate(top_images[:TOPK], start=1):
            if base.suffix:
                out_path = base.parent / f"{base.stem}_rank{rank}{base.suffix}"
            else:
                out_path = base / f"retrieved_page_rank{rank}.png"
            img.save(str(out_path))
            # Print the retrieval score (document-level MaxSim) alongside the saved path
            try:
                score, _doc_id = results[rank - 1]
                print(f"Saved retrieved page (rank {rank}) [MaxSim={score:.4f}] to: {out_path}")
            except Exception:
                print(f"Saved retrieved page (rank {rank}) to: {out_path}")


# %%
# Step 6: Similarity maps for top-K results
if results and SIMILARITY_MAP:
    token_idx = None if SIM_TOKEN_IDX < 0 else int(SIM_TOKEN_IDX)
    from pathlib import Path as _Path

    output_base = _Path(SIM_OUTPUT) if SIM_OUTPUT else None
    for rank, img in enumerate(top_images[:TOPK], start=1):
        if output_base:
            if output_base.suffix:
                out_dir = output_base.parent
                out_name = f"{output_base.stem}_rank{rank}{output_base.suffix}"
                out_path = str(out_dir / out_name)
            else:
                out_dir = output_base
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = str(out_dir / f"similarity_map_rank{rank}.png")
        else:
            out_path = None
        chosen_idx, max_sim = _generate_similarity_map(
            model=model,
            processor=processor,
            image=img,
            query=QUERY,
            token_idx=token_idx,
            output_path=out_path,
        )
        if out_path:
            print(
                f"Saved similarity map for rank {rank}, token #{chosen_idx} (max={max_sim:.2f}) to: {out_path}"
            )
        else:
            print(
                f"Computed similarity map for rank {rank}, token #{chosen_idx} (max={max_sim:.2f})"
            )


# %%
# Step 7: Optional answer generation
if results and ANSWER:
    qwen = QwenVL(device=device_str)
    _t0 = time.perf_counter()
    response = qwen.answer(QUERY, top_images[:TOPK], max_new_tokens=MAX_NEW_TOKENS)
    gen_secs = time.perf_counter() - _t0
    print(f"[Timing] Generation: {gen_secs:.3f}s")
    print("\nAnswer:")
    print(response)
