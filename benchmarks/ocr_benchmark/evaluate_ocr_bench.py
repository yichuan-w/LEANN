#!/usr/bin/env python3
"""
Evaluate OCR accuracy using olmOCR-Bench dataset.
Compares standard PDF extraction vs OCR extraction (MinerU).
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps"))

from document_rag import extract_pdf_with_ocr_fallback, OCR_AVAILABLE


def calculate_cer(predicted: str, reference: str) -> float:
    """Calculate Character Error Rate (CER)."""
    if not reference:
        return 1.0 if predicted else 0.0
    
    # Simple Levenshtein distance for CER
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    distance = levenshtein(predicted, reference)
    return distance / len(reference) if reference else 1.0


def calculate_wer(predicted: str, reference: str) -> float:
    """Calculate Word Error Rate (WER)."""
    if not reference:
        return 1.0 if predicted else 0.0
    
    pred_words = predicted.split()
    ref_words = reference.split()
    
    if not ref_words:
        return 1.0 if pred_words else 0.0
    
    # Simple word-level Levenshtein
    def word_levenshtein(words1: list, words2: list) -> int:
        if len(words1) < len(words2):
            return word_levenshtein(words2, words1)
        if len(words2) == 0:
            return len(words1)
        
        previous_row = range(len(words2) + 1)
        for i, w1 in enumerate(words1):
            current_row = [i + 1]
            for j, w2 in enumerate(words2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (w1 != w2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    distance = word_levenshtein(pred_words, ref_words)
    return distance / len(ref_words)


def evaluate_pdf(pdf_path: str, ground_truth: Optional[str] = None) -> dict:
    """Evaluate a single PDF with both standard and OCR extraction."""
    results = {
        "pdf_path": pdf_path,
        "standard_extraction": {"text": "", "time": 0, "success": False},
        "ocr_extraction": {"text": "", "time": 0, "success": False},
    }
    
    # Test standard extraction
    try:
        start = time.time()
        text_standard = extract_pdf_with_ocr_fallback(pdf_path, use_ocr=False)
        results["standard_extraction"]["time"] = time.time() - start
        results["standard_extraction"]["text"] = text_standard
        results["standard_extraction"]["success"] = len(text_standard.strip()) > 0
    except Exception as e:
        results["standard_extraction"]["error"] = str(e)
    
    # Test OCR extraction
    if OCR_AVAILABLE:
        try:
            start = time.time()
            text_ocr = extract_pdf_with_ocr_fallback(pdf_path, use_ocr=True)
            results["ocr_extraction"]["time"] = time.time() - start
            results["ocr_extraction"]["text"] = text_ocr
            results["ocr_extraction"]["success"] = len(text_ocr.strip()) > 0
        except Exception as e:
            results["ocr_extraction"]["error"] = str(e)
    else:
        results["ocr_extraction"]["error"] = "MinerU not installed"
    
    # Calculate accuracy if ground truth provided
    if ground_truth:
        if results["standard_extraction"]["success"]:
            results["standard_extraction"]["cer"] = calculate_cer(
                results["standard_extraction"]["text"], ground_truth
            )
            results["standard_extraction"]["wer"] = calculate_wer(
                results["standard_extraction"]["text"], ground_truth
            )
        
        if results["ocr_extraction"]["success"]:
            results["ocr_extraction"]["cer"] = calculate_cer(
                results["ocr_extraction"]["text"], ground_truth
            )
            results["ocr_extraction"]["wer"] = calculate_wer(
                results["ocr_extraction"]["text"], ground_truth
            )
    
    return results


def load_olmocr_bench(data_dir: Path, split: Optional[str] = None, max_samples: int = 0):
    """Load olmOCR-Bench dataset using HuggingFace Datasets Server API."""
    import requests
    import json
    from urllib.parse import quote
    
    print(f"Loading olmOCR-Bench dataset from HuggingFace Datasets Server API...")
    
    pdf_files = []
    ground_truths = {}
    
    # API configuration
    base_url = "https://datasets-server.huggingface.co"
    dataset_name = "allenai/olmOCR-bench"
    config = "olmocr-bench"
    
    # Define available splits
    splits_to_load = [split] if split else [
        "arxiv_math",
        "headers_footers",
        "long_tiny_text", 
        "multi_column",
        "old_scans",
        "old_scans_math",
        "table_tests"
    ]
    
    for split_name in splits_to_load:
        print(f"  Loading split: {split_name}...")
        count = 0
        
        try:
            # Fetch data from API (paginated)
            offset = 0
            length = 100  # Fetch 100 rows at a time
            
            while True:
                # Get rows from API
                url = f"{base_url}/rows?dataset={quote(dataset_name)}&config={quote(config)}&split={quote(split_name)}&offset={offset}&length={length}"
                
                try:
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                except requests.exceptions.RequestException as e:
                    print(f"    ⚠ API request failed: {e}")
                    break
                
                if "rows" not in data:
                    break
                
                rows = data["rows"]
                if not rows:
                    break
                
                # Process each row
                for row in rows:
                    if "row" not in row:
                        continue
                    
                    item = row["row"]
                    
                    # Extract PDF path - try different field names
                    pdf_path = None
                    for field in ["pdf", "pdf_path", "image_path"]:
                        if field in item:
                            pdf_path = item[field]
                            break
                    
                    if not pdf_path:
                        continue
                    
                    # Extract ground truth - try different field names
                    gt = None
                    for field in ["text", "ground_truth", "math"]:
                        if field in item:
                            gt = item[field]
                            break
                    
                    # For now, we'll store the PDF path and ground truth
                    # Note: The actual PDF files may need to be downloaded separately
                    # or accessed via URL if available
                    pdf_info = {
                        "path": pdf_path,
                        "split": split_name,
                        "url": item.get("url", ""),
                    }
                    
                    # Try to construct a local path (PDFs might be in HuggingFace cache)
                    # Check if it's a URL we can download from
                    if pdf_path.startswith("http"):
                        # It's a URL - we'd need to download it
                        pdf_info["is_url"] = True
                    else:
                        # Try to find in local cache
                        cache_dir = data_dir / "allenai___olmOCR-bench"
                        possible_paths = [
                            cache_dir / pdf_path,
                            cache_dir / "bench_data" / pdf_path,
                            cache_dir / "pdfs" / pdf_path,
                        ]
                        
                        pdf_file = None
                        for pp in possible_paths:
                            if pp.exists() and pp.suffix == '.pdf':
                                pdf_file = pp
                                break
                        
                        if pdf_file:
                            pdf_info["local_path"] = str(pdf_file)
                            pdf_files.append(pdf_file)
                            if gt:
                                ground_truths[str(pdf_file)] = gt
                            count += 1
                    
                    # Limit per split if max_samples specified
                    if max_samples > 0 and len(pdf_files) >= max_samples:
                        break
                
                # Check if we should continue fetching
                if max_samples > 0 and len(pdf_files) >= max_samples:
                    break
                
                # Check if there are more rows
                if len(rows) < length:
                    break
                
                offset += length
                
            print(f"    ✓ Processed {count} items from {split_name}")
            
            if max_samples > 0 and len(pdf_files) >= max_samples:
                break
                
        except Exception as e:
            print(f"    ⚠ Error loading {split_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Limit total samples if specified
    if max_samples > 0 and len(pdf_files) > max_samples:
        pdf_files = pdf_files[:max_samples]
    
    print(f"\n  Total PDFs found: {len(pdf_files)}")
    print(f"  Ground truth available for: {len(ground_truths)} PDFs")
    
    if len(pdf_files) == 0:
        print("\n  ⚠ Note: PDF files not found locally.")
        print("  The dataset contains references to PDFs that may need to be downloaded separately.")
        print("  You can test with your own PDFs using --pdf-dir option.")
    
    return pdf_files, ground_truths


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy with olmOCR-Bench")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="benchmarks/ocr_benchmark/data/olmocr_bench",
        help="Directory for dataset cache",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["arxiv_math", "headers_footers", "long_tiny_text", "multi_column", 
                 "old_scans", "old_scans_math", "table_tests"],
        help="Specific split to evaluate (default: all splits)",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Directory with PDFs to test (alternative to dataset, for quick testing)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to evaluate (0 = all)",
    )
    
    args = parser.parse_args()
    
    print("OCR Benchmark Evaluation with olmOCR-Bench")
    print("=" * 60)
    print(f"MinerU available: {OCR_AVAILABLE}")
    if not OCR_AVAILABLE:
        print("⚠ Install MinerU for OCR evaluation: pip install mineru")
    print()
    
    # Load dataset or use provided PDFs
    pdf_files = []
    ground_truths = {}
    
    if args.pdf_dir:
        # Use provided PDF directory (for quick testing)
        pdf_dir = Path(args.pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"Using PDF directory: {pdf_dir}")
        print(f"Found {len(pdf_files)} PDF files")
    else:
        # Load from olmOCR-Bench dataset
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files, ground_truths = load_olmocr_bench(
            data_dir, 
            split=args.split,
            max_samples=args.max_samples if args.max_samples > 0 else 0
        )
        
        if not pdf_files:
            # Fallback to default data directory
            print("⚠ No PDFs found in dataset, trying default data directory...")
            pdf_files = list(Path("data").glob("*.pdf"))
            if pdf_files:
                print(f"  Found {len(pdf_files)} PDFs in data/ directory")
    
    if not pdf_files:
        print("❌ No PDF files found to evaluate")
        print("\nOptions:")
        print("  1. Run setup first: python benchmarks/ocr_benchmark/setup_ocr_bench.py")
        print("  2. Use --pdf-dir to specify a directory with PDFs")
        print("  3. Place PDFs in the 'data/' directory")
        return
    
    # Limit samples
    if args.max_samples > 0:
        pdf_files = pdf_files[: args.max_samples]
    
    print(f"\nEvaluating {len(pdf_files)} PDFs...")
    print(f"Ground truth available for {len(ground_truths)} PDFs")
    print("-" * 60)
    
    # Evaluate each PDF
    all_results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_str = str(pdf_path)
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        gt = ground_truths.get(pdf_str)
        result = evaluate_pdf(pdf_str, gt)
        all_results.append(result)
        
        # Print quick summary
        std = result["standard_extraction"]
        ocr = result["ocr_extraction"]
        
        print(f"  Standard: {len(std['text'])} chars, {std['time']:.2f}s, "
              f"{'✓' if std['success'] else '✗'}")
        if OCR_AVAILABLE:
            print(f"  OCR:      {len(ocr['text'])} chars, {ocr['time']:.2f}s, "
                  f"{'✓' if ocr['success'] else '✗'}")
            if gt and ocr['success']:
                print(f"  OCR CER:  {ocr.get('cer', 0):.4f}")
                print(f"  OCR WER:  {ocr.get('wer', 0):.4f}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    std_success = sum(1 for r in all_results if r["standard_extraction"]["success"])
    ocr_success = sum(1 for r in all_results if OCR_AVAILABLE and r["ocr_extraction"]["success"])
    
    std_avg_time = sum(r["standard_extraction"]["time"] for r in all_results) / len(all_results)
    ocr_avg_time = (
        sum(r["ocr_extraction"]["time"] for r in all_results) / len(all_results)
        if OCR_AVAILABLE
        else 0
    )
    
    print(f"Standard Extraction:")
    print(f"  Success rate: {std_success}/{len(all_results)} ({100*std_success/len(all_results):.1f}%)")
    print(f"  Avg time: {std_avg_time:.2f}s per PDF")
    
    if OCR_AVAILABLE:
        print(f"\nOCR Extraction:")
        print(f"  Success rate: {ocr_success}/{len(all_results)} ({100*ocr_success/len(all_results):.1f}%)")
        print(f"  Avg time: {ocr_avg_time:.2f}s per PDF")
        
        # Calculate average CER/WER if ground truth available
        ocr_cers = [r["ocr_extraction"].get("cer") for r in all_results 
                    if r["ocr_extraction"].get("cer") is not None]
        ocr_wers = [r["ocr_extraction"].get("wer") for r in all_results 
                    if r["ocr_extraction"].get("wer") is not None]
        
        if ocr_cers:
            print(f"  Avg CER: {sum(ocr_cers)/len(ocr_cers):.4f}")
        if ocr_wers:
            print(f"  Avg WER: {sum(ocr_wers)/len(ocr_wers):.4f}")
    
    print("\n✓ Evaluation complete!")
    print(f"\nReference: https://huggingface.co/datasets/allenai/olmOCR-bench")


if __name__ == "__main__":
    main()

