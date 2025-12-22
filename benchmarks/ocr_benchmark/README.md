# OCR Benchmark Evaluation with olmOCR-Bench

This benchmark evaluates OCR accuracy using the [olmOCR-Bench dataset](https://huggingface.co/datasets/allenai/olmOCR-bench) from AllenAI.

## Dataset Information

- **Dataset**: [allenai/olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
- **Size**: 1,403 PDF files with 7,010 test cases
- **Splits**: arxiv_math, headers_footers, long_tiny_text, multi_column, old_scans, old_scans_math, table_tests
- **Purpose**: Evaluates OCR systems' ability to accurately convert PDFs to markdown while preserving textual and structural information

## Setup

1. Install dependencies:
```bash
pip install datasets huggingface_hub
```

2. Download the dataset (automatically done by setup script):
```bash
python benchmarks/ocr_benchmark/setup_ocr_bench.py
```

## Evaluation

Run the evaluation:
```bash
# Evaluate on all splits
python benchmarks/ocr_benchmark/evaluate_ocr_bench.py

# Evaluate on specific split
python benchmarks/ocr_benchmark/evaluate_ocr_bench.py --split arxiv_math

# Limit number of samples
python benchmarks/ocr_benchmark/evaluate_ocr_bench.py --max-samples 50
```

## Metrics

- **Character Error Rate (CER)**: Percentage of character-level errors
- **Word Error Rate (WER)**: Percentage of word-level errors
- **Extraction Success Rate**: Percentage of PDFs successfully processed
- **Processing Time**: Time taken for standard vs OCR extraction
- **Test Case Pass Rate**: Percentage of test cases passed (if ground truth available)

## Reference

Based on the olmOCR-Bench paper and dataset from AllenAI.

