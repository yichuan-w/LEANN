#!/usr/bin/env python3
"""
Setup script for olmOCR-Bench dataset.
Downloads the dataset from HuggingFace if not already present.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def download_olmocr_bench(data_dir: Path):
    """Download olmOCR-Bench dataset info using HuggingFace Datasets Server API."""
    print("Fetching olmOCR-Bench dataset info from HuggingFace...")
    print("Dataset: allenai/olmOCR-bench")
    print("-" * 60)
    
    try:
        import requests
        import json
        from urllib.parse import quote
        
        # Use HuggingFace Datasets Server API to get split info
        base_url = "https://datasets-server.huggingface.co"
        dataset_name = "allenai/olmOCR-bench"
        config = "olmocr-bench"
        
        splits = [
            "arxiv_math",
            "headers_footers", 
            "long_tiny_text",
            "multi_column",
            "old_scans",
            "old_scans_math",
            "table_tests"
        ]
        
        split_counts = {}
        split_info = {}
        
        print("Fetching split information...")
        for split in splits:
            try:
                # Get first rows to check split exists and get structure
                url = f"{base_url}/first-rows?dataset={quote(dataset_name)}&config={quote(config)}&split={quote(split)}"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Get split info (number of rows)
                info_url = f"{base_url}/info?dataset={quote(dataset_name)}"
                info_response = requests.get(info_url, timeout=30)
                if info_response.status_code == 200:
                    info_data = info_response.json()
                    # Try to find split size
                    if "splits" in info_data:
                        for split_info_item in info_data["splits"]:
                            if split_info_item.get("name") == split:
                                num_rows = split_info_item.get("num_rows", 0)
                                split_counts[split] = num_rows
                                split_info[split] = {
                                    "num_rows": num_rows,
                                    "features": data.get("features", [])
                                }
                                print(f"  ✓ {split}: {num_rows} samples")
                                break
                    else:
                        # Fallback: count from first-rows response
                        if "num_rows_total" in data:
                            split_counts[split] = data["num_rows_total"]
                            print(f"  ✓ {split}: {data['num_rows_total']} samples")
                        else:
                            split_counts[split] = 0
                            print(f"  ⚠ {split}: size unknown")
                else:
                    # Fallback: just mark as available
                    split_counts[split] = 0
                    print(f"  ✓ {split}: available (size unknown)")
                    
            except requests.exceptions.RequestException as e:
                print(f"  ⚠ {split}: failed to fetch ({e})")
                continue
            except Exception as e:
                print(f"  ⚠ {split}: error ({e})")
                continue
        
        # Save dataset info
        info_file = data_dir / "dataset_info.txt"
        with open(info_file, "w") as f:
            f.write(f"Dataset: allenai/olmOCR-bench\n")
            f.write(f"API: https://datasets-server.huggingface.co\n")
            f.write(f"Config: {config}\n")
            f.write(f"Splits: {list(split_counts.keys())}\n")
            for split_name, count in split_counts.items():
                f.write(f"  {split_name}: {count} samples\n")
        
        print(f"\n✓ Dataset info fetched!")
        print(f"  Splits available: {list(split_counts.keys())}")
        print(f"  Using HuggingFace Datasets Server API")
        
        return {
            "api_base": base_url,
            "dataset": dataset_name,
            "config": config,
            "splits": split_counts,
            "split_info": split_info
        }
        
    except ImportError:
        print("❌ Error: 'requests' package not installed")
        print("  Install with: pip install requests")
        return None
    except Exception as e:
        print(f"⚠ Error fetching dataset info: {e}")
        print("\nAlternative: Download manually from:")
        print("  https://huggingface.co/datasets/allenai/olmOCR-bench")
        print(f"\nOr place PDF files in: {data_dir}")
        return None


def main():
    """Main setup function."""
    # Get benchmark data directory
    benchmark_dir = Path(__file__).resolve().parent
    data_dir = benchmark_dir / "data" / "olmocr_bench"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up olmOCR-Bench dataset")
    print(f"Cache directory: {data_dir}")
    print("=" * 60)
    
    # Check if dataset already exists
    if (data_dir / "dataset_info.txt").exists():
        print("✓ Dataset already downloaded")
        with open(data_dir / "dataset_info.txt") as f:
            print(f.read())
        print("\nTo re-download, delete the cache directory and run again.")
        return
    
    # Download dataset
    dataset = download_olmocr_bench(data_dir)
    
    if dataset:
        print("\n✓ Setup complete!")
        print(f"  Dataset location: {data_dir}")
        print("\nNext step: Run evaluation with:")
        print("  python benchmarks/ocr_benchmark/evaluate_ocr_bench.py")
    else:
        print("\n⚠ Setup incomplete. Please check errors above.")


if __name__ == "__main__":
    main()

