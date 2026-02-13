# pip install pdf2image
# pip install pymilvus
# pip install colpali_engine
# pip install tqdm
# pip install pillow

import os
import re
import sys
from pathlib import Path
from typing import cast

from PIL import Image
from tqdm import tqdm

# Ensure local leann packages are importable before importing them
_repo_root = Path(__file__).resolve().parents[3]
_leann_core_src = _repo_root / "packages" / "leann-core" / "src"
_leann_hnsw_pkg = _repo_root / "packages" / "leann-backend-hnsw"
if str(_leann_core_src) not in sys.path:
    sys.path.insert(0, str(_leann_core_src))
if str(_leann_hnsw_pkg) not in sys.path:
    sys.path.insert(0, str(_leann_hnsw_pkg))

from leann_multi_vector import LeannMultiVector

import torch
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader

# Auto-select device: CUDA > MPS (mac) > CPU
_device_str = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
)
device = get_torch_device(_device_str)
# Prefer fp16 on GPU/MPS, bfloat16 on CPU
_dtype = torch.float16 if _device_str in ("cuda", "mps") else torch.bfloat16
model_name = "vidore/colpali-v1.2"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=_dtype,
    device_map=device,
).eval()
print(f"Using device={_device_str}, dtype={_dtype}")

queries = [
    "How to end-to-end retrieval with ColBert",
    "Where is ColBERT performance Table, including text representation results?",
]

processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

dataloader = DataLoader(
    dataset=ListDataset[str](queries),
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: processor.process_queries(x),
)

qs: list[torch.Tensor] = []
for batch_query in dataloader:
    with torch.no_grad():
        batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
    qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
print(qs[0].shape)
# %%
page_filenames = sorted(os.listdir("./pages"), key=lambda n: int(re.search(r"\d+", n).group()))
images = [Image.open(os.path.join("./pages", name)) for name in page_filenames]

dataloader = DataLoader(
    dataset=ListDataset[str](images),
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: processor.process_images(x),
)

ds: list[torch.Tensor] = []
for batch_doc in tqdm(dataloader):
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
        embeddings_doc = model(**batch_doc)
    ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

print(ds[0].shape)

# %%
# Build HNSW index via LeannMultiVector primitives and run search
index_path = "./indexes/colpali.leann"
retriever = LeannMultiVector(index_path=index_path, dim=int(ds[0].shape[-1]))
retriever.create_collection()
filepaths = [os.path.join("./pages", name) for name in page_filenames]
for i in range(len(filepaths)):
    data = {
        "colbert_vecs": ds[i].float().numpy(),
        "doc_id": i,
        "filepath": filepaths[i],
    }
    retriever.insert(data)
retriever.create_index()
for query in qs:
    query_np = query.float().numpy()
    result = retriever.search(query_np, topk=1)
    print(filepaths[result[0][1]])
