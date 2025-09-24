## Jupyter-style notebook script
# %%
# uv pip install matplotlib qwen_vl_utils
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, cast

from PIL import Image
from tqdm import tqdm


def _ensure_repo_paths_importable(current_file: str) -> None:
    """Make local leann packages importable without installing (mirrors multi-vector-leann.py)."""
    _repo_root = Path(current_file).resolve().parents[3]
    _leann_core_src = _repo_root / "packages" / "leann-core" / "src"
    _leann_hnsw_pkg = _repo_root / "packages" / "leann-backend-hnsw"
    if str(_leann_core_src) not in sys.path:
        sys.path.append(str(_leann_core_src))
    if str(_leann_hnsw_pkg) not in sys.path:
        sys.path.append(str(_leann_hnsw_pkg))


_ensure_repo_paths_importable(__file__)

from leann_multi_vector import LeannMultiVector  # noqa: E402

# %%
# Config
os.environ["TOKENIZERS_PARALLELISM"] = "false"
QUERY = "How does DeepSeek-V2 compare against the LLaMA family of LLMs?"
MODEL: str = "colqwen2"  # "colpali" or "colqwen2"

# Data source: set to True to use the Hugging Face dataset example (recommended)
USE_HF_DATASET: bool = True
DATASET_NAME: str = "weaviate/arXiv-AI-papers-multi-vector"
DATASET_SPLIT: str = "train"
MAX_DOCS: Optional[int] = None  # limit number of pages to index; None = all

# Local pages (used when USE_HF_DATASET == False)
PDF: Optional[str] = None  # e.g., "./pdfs/2004.12832v2.pdf"
PAGES_DIR: str = "./pages"

# Index + retrieval settings
INDEX_PATH: str = "./indexes/colvision.leann"
TOPK: int = 1
FIRST_STAGE_K: int = 500
REBUILD_INDEX: bool = False

# Artifacts
SAVE_TOP_IMAGE: Optional[str] = "./figures/retrieved_page.png"
SIMILARITY_MAP: bool = True
SIM_TOKEN_IDX: int = 13  # -1 means auto-select the most salient token
SIM_OUTPUT: str = "./figures/similarity_map.png"
ANSWER: bool = True
MAX_NEW_TOKENS: int = 128


# %%
# Helpers
def _natural_sort_key(name: str) -> int:
    m = re.search(r"\d+", name)
    return int(m.group()) if m else 0


def _load_images_from_dir(pages_dir: str) -> tuple[list[str], list[Image.Image]]:
    filenames = [n for n in os.listdir(pages_dir) if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    filenames = sorted(filenames, key=_natural_sort_key)
    filepaths = [os.path.join(pages_dir, n) for n in filenames]
    images = [Image.open(p) for p in filepaths]
    return filepaths, images


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
    import torch
    from colpali_engine.models import ColPali, ColQwen2, ColQwen2Processor
    from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
    from transformers.utils.import_utils import is_flash_attn_2_available

    device_str, device, dtype = _select_device_and_dtype()

    if model_choice == "colqwen2":
        model_name = "vidore/colqwen2-v1.0"
        # On CPU/MPS we must avoid flash-attn and stay eager; on CUDA prefer flash-attn if available
        attn_implementation = (
            "flash_attention_2"
            if (device_str == "cuda" and is_flash_attn_2_available())
            else "eager"
        )
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
        ).eval()
        processor = ColQwen2Processor.from_pretrained(model_name)
    else:
        model_name = "vidore/colpali-v1.2"
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

    return model_name, model, processor, device_str, device, dtype


def _embed_images(model, processor, images: list[Image.Image]) -> list[Any]:
    import torch
    from colpali_engine.utils.torch_utils import ListDataset
    from torch.utils.data import DataLoader

    # Ensure deterministic eval and autocast for stability
    model.eval()

    dataloader = DataLoader(
        dataset=ListDataset[Image.Image](images),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )

    doc_vecs: list[Any] = []
    for batch_doc in dataloader:
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
    from colpali_engine.utils.torch_utils import ListDataset
    from torch.utils.data import DataLoader

    model.eval()

    dataloader = DataLoader(
        dataset=ListDataset[str](queries),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    q_vecs: list[Any] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            if model.device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=model.dtype if model.dtype.is_floating_point else torch.bfloat16,
                ):
                    embeddings_query = model(**batch_query)
            else:
                embeddings_query = model(**batch_query)
        q_vecs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
    return q_vecs


def _build_index(index_path: str, doc_vecs: list[Any], filepaths: list[str]) -> LeannMultiVector:
    dim = int(doc_vecs[0].shape[-1])
    retriever = LeannMultiVector(index_path=index_path, dim=dim)
    retriever.create_collection()
    for i, vec in enumerate(doc_vecs):
        data = {
            "colbert_vecs": vec.float().numpy(),
            "doc_id": i,
            "filepath": filepaths[i],
        }
        retriever.insert(data)
    retriever.create_index()
    return retriever


def _load_retriever_if_index_exists(index_path: str, dim: int) -> Optional[LeannMultiVector]:
    index_base = Path(index_path)
    # Rough heuristic: index dir exists AND meta+labels files exist
    meta = index_base.parent / f"{index_base.name}.meta.json"
    labels = index_base.parent / f"{index_base.name}.labels.json"
    if index_base.exists() and meta.exists() and labels.exists():
        return LeannMultiVector(index_path=index_path, dim=dim)
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


# %%

# Step 1: Prepare data
if USE_HF_DATASET:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    N = len(dataset) if MAX_DOCS is None else min(MAX_DOCS, len(dataset))
    filepaths: list[str] = []
    images: list[Image.Image] = []
    for i in tqdm(range(N), desc="Loading dataset"):
        p = dataset[i]
        # Compose a descriptive identifier for printing later
        identifier = f"arXiv:{p['paper_arxiv_id']}|title:{p['paper_title']}|page:{int(p['page_number'])}|id:{p['page_id']}"
        print(identifier)
        filepaths.append(identifier)
        images.append(p["page_image"])  # PIL Image
else:
    _maybe_convert_pdf_to_images(PDF, PAGES_DIR)
    filepaths, images = _load_images_from_dir(PAGES_DIR)
    if not images:
        raise RuntimeError(
            f"No images found in {PAGES_DIR}. Provide PDF path in PDF variable or ensure images exist."
        )


# %%
# Step 2: Load model and processor
model_name, model, processor, device_str, device, dtype = _load_colvision(MODEL)
print(f"Using model={model_name}, device={device_str}, dtype={dtype}")


# %%

# %%
# Step 3: Build or load index
retriever: Optional[LeannMultiVector] = None
if not REBUILD_INDEX:
    try:
        one_vec = _embed_images(model, processor, [images[0]])[0]
        retriever = _load_retriever_if_index_exists(INDEX_PATH, dim=int(one_vec.shape[-1]))
    except Exception:
        retriever = None

if retriever is None:
    doc_vecs = _embed_images(model, processor, images)
    retriever = _build_index(INDEX_PATH, doc_vecs, filepaths)


# %%
# Step 4: Embed query and search
q_vec = _embed_queries(model, processor, [QUERY])[0]
results = retriever.search(q_vec.float().numpy(), topk=TOPK, first_stage_k=FIRST_STAGE_K)
if not results:
    print("No results found.")
else:
    print(f'Top {len(results)} results for query: "{QUERY}"')
    top_images: list[Image.Image] = []
    for rank, (score, doc_id) in enumerate(results, start=1):
        path = filepaths[doc_id]
        # For HF dataset, path is a descriptive identifier, not a real file path
        print(f"{rank}) MaxSim: {score:.4f}, Page: {path}")
        top_images.append(images[doc_id])

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
            print(f"Saved retrieved page (rank {rank}) to: {out_path}")

## TODO stange results of second page of DeepSeek-V2 rather than the first page

# %%
# Step 5: Similarity maps for top-K results
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
# Step 6: Optional answer generation
if results and ANSWER:
    qwen = QwenVL(device=device_str)
    response = qwen.answer(QUERY, top_images[:TOPK], max_new_tokens=MAX_NEW_TOKENS)
    print("\nAnswer:")
    print(response)
