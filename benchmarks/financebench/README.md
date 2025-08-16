# FinanceBench Benchmark for LEANN-RAG

FinanceBench is a benchmark for evaluating retrieval-augmented generation (RAG) systems on financial document question-answering tasks.

## Dataset

- **Source**: [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench)
- **Questions**: 150 financial Q&A examples
- **Documents**: 368 PDF files (10-K, 10-Q, 8-K, earnings reports)
- **Companies**: Major public companies (3M, Apple, Microsoft, Amazon, etc.)
- **Paper**: [FinanceBench: A New Benchmark for Financial Question Answering](https://arxiv.org/abs/2311.11944)

## Structure

```
benchmarks/financebench/
├── setup_financebench.py        # Downloads PDFs and builds index
├── evaluate_financebench.py     # Intelligent evaluation script
├── data/
│   ├── financebench_merged.jsonl     # Q&A dataset
│   ├── pdfs/                         # Downloaded financial documents
│   └── index/                        # LEANN indexes
│       └── financebench_full_hnsw.leann
└── README.md
```

## Usage

### 1. Setup (Download & Build Index)

```bash
cd benchmarks/financebench
python setup_financebench.py
```

This will:
- Download the 150 Q&A examples
- Download all 368 PDF documents (parallel processing)
- Build a LEANN index from 53K+ text chunks
- Verify setup with test query

### 2. Evaluation

```bash
# Basic retrieval evaluation
python evaluate_financebench.py --index data/index/financebench_full_hnsw.leann

# Include QA evaluation with OpenAI
export OPENAI_API_KEY="your-key"
python evaluate_financebench.py --index data/index/financebench_full_hnsw.leann --qa-samples 20
```

## Evaluation Methods

### Retrieval Evaluation
Uses intelligent matching with three strategies:
1. **Exact text overlap** - Direct substring matches
2. **Number matching** - Key financial figures ($1,577, 1.2B, etc.)
3. **Semantic similarity** - Word overlap with 20% threshold

### QA Evaluation
LLM-based answer evaluation using GPT-4o:
- Handles numerical rounding and equivalent representations
- Considers fractions, percentages, and decimal equivalents
- Evaluates semantic meaning rather than exact text match

## Expected Results

Previous runs show:
- **Question Coverage**: ~65-75% (questions with relevant docs retrieved)
- **Index Size**: 53,985 chunks from 368 PDFs
- **Search Time**: ~0.1-0.2s per query
- **Build Time**: ~5-10 minutes with sentence-transformers/all-mpnet-base-v2

## Options

```bash
# Use different backends
python setup_financebench.py --backend diskann
python evaluate_financebench.py --index data/index/financebench_full_diskann.leann

# Use different embedding models
python setup_financebench.py --embedding-model facebook/contriever
```
