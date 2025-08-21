---
license: mit
---

# LEANN-RAG Evaluation Data

This repository contains the necessary data to run the recall evaluation scripts for the [LEANN-RAG](https://huggingface.co/LEANN-RAG) project.

## Dataset Components

This dataset is structured into three main parts:

1.  **Pre-built LEANN Indices**:
    *   `dpr/`: A pre-built index for the DPR dataset.
    *   `rpj_wiki/`: A pre-built index for the RPJ-Wiki dataset.
    These indices were created using the `leann-core` library and are required by the `LeannSearcher`.

2.  **Ground Truth Data**:
    *   `ground_truth/`: Contains the ground truth files (`flat_results_nq_k3.json`) for both the DPR and RPJ-Wiki datasets. These files map queries to the original passage IDs from the Natural Questions benchmark, evaluated using the Contriever model.

3.  **Queries**:
    *   `queries/`: Contains the `nq_open.jsonl` file with the Natural Questions queries used for the evaluation.

## Usage

To use this data, you can download it locally using the `huggingface-hub` library. First, install the library:

```bash
pip install huggingface-hub
```

Then, you can download the entire dataset to a local directory (e.g., `data/`) with the following Python script:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LEANN-RAG/leann-rag-evaluation-data",
    repo_type="dataset",
    local_dir="data"
)
```

This will download all the necessary files into a local `data` folder, preserving the repository structure. The evaluation scripts in the main [LEANN-RAG Space](https://huggingface.co/LEANN-RAG) are configured to work with this data structure.
