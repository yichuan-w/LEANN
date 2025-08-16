#!/usr/bin/env python3
"""
FinanceBench Complete Setup Script
Downloads all PDFs and builds full LEANN datastore
"""

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pymupdf
import requests
from leann import LeannBuilder, LeannSearcher
from tqdm import tqdm


class FinanceBenchSetup:
    def __init__(self, data_dir: str = "data"):
        self.base_dir = Path(__file__).parent  # benchmarks/financebench/
        self.data_dir = self.base_dir / data_dir
        self.pdf_dir = self.data_dir / "pdfs"
        self.dataset_file = self.data_dir / "financebench_merged.jsonl"
        self.index_dir = self.data_dir / "index"
        self.download_lock = Lock()

    def download_dataset(self):
        """Download the main FinanceBench dataset"""
        print("📊 Downloading FinanceBench dataset...")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset_file.exists():
            print(f"✅ Dataset already exists: {self.dataset_file}")
            return

        url = "https://huggingface.co/datasets/PatronusAI/financebench/raw/main/financebench_merged.jsonl"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(self.dataset_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ Dataset downloaded: {self.dataset_file}")

    def get_pdf_list(self):
        """Get list of all PDF files from GitHub"""
        print("📋 Fetching PDF list from GitHub...")

        response = requests.get(
            "https://api.github.com/repos/patronus-ai/financebench/contents/pdfs"
        )
        response.raise_for_status()
        pdf_files = response.json()

        print(f"Found {len(pdf_files)} PDF files")
        return pdf_files

    def download_single_pdf(self, pdf_info, position):
        """Download a single PDF file"""
        pdf_name = pdf_info["name"]
        pdf_path = self.pdf_dir / pdf_name

        # Skip if already downloaded
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return f"✅ {pdf_name} (cached)"

        try:
            # Download PDF
            response = requests.get(pdf_info["download_url"], timeout=60)
            response.raise_for_status()

            # Write to file
            with self.download_lock:
                with open(pdf_path, "wb") as f:
                    f.write(response.content)

            return f"✅ {pdf_name} ({len(response.content) // 1024}KB)"

        except Exception as e:
            return f"❌ {pdf_name}: {e!s}"

    def download_all_pdfs(self, max_workers: int = 5):
        """Download all PDF files with parallel processing"""
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = self.get_pdf_list()

        print(f"📥 Downloading {len(pdf_files)} PDFs with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_pdf = {
                executor.submit(self.download_single_pdf, pdf_info, i): pdf_info["name"]
                for i, pdf_info in enumerate(pdf_files)
            }

            # Process completed downloads with progress bar
            with tqdm(total=len(pdf_files), desc="Downloading PDFs") as pbar:
                for future in as_completed(future_to_pdf):
                    result = future.result()
                    pbar.set_postfix_str(result.split()[-1] if "✅" in result else "Error")
                    pbar.update(1)

        # Verify downloads
        downloaded_pdfs = list(self.pdf_dir.glob("*.pdf"))
        print(f"✅ Successfully downloaded {len(downloaded_pdfs)}/{len(pdf_files)} PDFs")

        # Show any failures
        missing_pdfs = []
        for pdf_info in pdf_files:
            pdf_path = self.pdf_dir / pdf_info["name"]
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                missing_pdfs.append(pdf_info["name"])

        if missing_pdfs:
            print(f"⚠️  Failed to download {len(missing_pdfs)} PDFs:")
            for pdf in missing_pdfs[:5]:  # Show first 5
                print(f"   - {pdf}")
            if len(missing_pdfs) > 5:
                print(f"   ... and {len(missing_pdfs) - 5} more")

    def build_leann_index(
        self,
        backend: str = "hnsw",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    ):
        """Build LEANN index from all PDFs"""
        print(f"🏗️  Building LEANN index with {backend} backend...")

        # Check if we have PDFs
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError("No PDF files found! Run download first.")

        print(f"Found {len(pdf_files)} PDF files to process")

        start_time = time.time()

        # Initialize builder
        builder = LeannBuilder(
            backend_name=backend,
            embedding_model=embedding_model,
            embedding_mode="sentence-transformers",
            graph_degree=32,
            complexity=64,
            is_recompute=False,  # Store embeddings for speed
            num_threads=4,
        )

        # Process PDFs and extract text
        total_chunks = 0
        failed_pdfs = []

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                chunks = self.extract_pdf_text(pdf_path)
                for chunk in chunks:
                    builder.add_text(chunk["text"], metadata=chunk["metadata"])
                    total_chunks += 1

            except Exception as e:
                print(f"❌ Failed to process {pdf_path.name}: {e}")
                failed_pdfs.append(pdf_path.name)
                continue

        # Build index in index directory
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / f"financebench_full_{backend}.leann"
        print(f"🔨 Building index: {index_path}")
        builder.build_index(str(index_path))

        build_time = time.time() - start_time

        print("✅ Index built successfully!")
        print(f"   📁 Index path: {index_path}")
        print(f"   📊 Total chunks: {total_chunks:,}")
        print(f"   📄 Processed PDFs: {len(pdf_files) - len(failed_pdfs)}/{len(pdf_files)}")
        print(f"   ⏱️  Build time: {build_time:.1f}s")

        if failed_pdfs:
            print(f"   ⚠️  Failed PDFs: {failed_pdfs}")

        return str(index_path)

    def extract_pdf_text(self, pdf_path: Path) -> list[dict]:
        """Extract and chunk text from a PDF file"""
        chunks = []
        doc = pymupdf.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()  # type: ignore

            if not text.strip():
                continue

            # Create metadata
            metadata = {
                "source_file": pdf_path.name,
                "page_number": page_num + 1,
                "document_type": "10K" if "10K" in pdf_path.name else "10Q",
                "company": pdf_path.name.split("_")[0],
                "doc_period": self.extract_year_from_filename(pdf_path.name),
            }

            # Use recursive character splitting like LangChain
            if len(text.split()) > 500:
                # Split by double newlines (paragraphs)
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

                current_chunk = ""
                for para in paragraphs:
                    # If adding this paragraph would make chunk too long, save current chunk
                    if current_chunk and len((current_chunk + " " + para).split()) > 300:
                        if current_chunk.strip():
                            chunks.append(
                                {
                                    "text": current_chunk.strip(),
                                    "metadata": {
                                        **metadata,
                                        "chunk_id": f"page_{page_num + 1}_chunk_{len(chunks)}",
                                    },
                                }
                            )
                        current_chunk = para
                    else:
                        current_chunk = (current_chunk + " " + para).strip()

                # Add the last chunk
                if current_chunk.strip():
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "metadata": {
                                **metadata,
                                "chunk_id": f"page_{page_num + 1}_chunk_{len(chunks)}",
                            },
                        }
                    )
            else:
                # Page is short enough, use as single chunk
                chunks.append(
                    {
                        "text": text.strip(),
                        "metadata": {**metadata, "chunk_id": f"page_{page_num + 1}"},
                    }
                )

        doc.close()
        return chunks

    def extract_year_from_filename(self, filename: str) -> str:
        """Extract year from PDF filename"""
        # Try to find 4-digit year in filename

        match = re.search(r"(\d{4})", filename)
        return match.group(1) if match else "unknown"

    def verify_setup(self, index_path: str):
        """Verify the setup by testing a simple query"""
        print("🧪 Verifying setup with test query...")

        try:
            searcher = LeannSearcher(index_path)

            # Test query
            test_query = "What is the capital expenditure for 3M in 2018?"
            results = searcher.search(test_query, top_k=3)

            print(f"✅ Test query successful! Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                company = result.metadata.get("company", "Unknown")
                year = result.metadata.get("doc_period", "Unknown")
                page = result.metadata.get("page_number", "Unknown")
                print(f"   {i}. {company} {year} (page {page}) - Score: {result.score:.3f}")
                print(f"      {result.text[:100]}...")

            searcher.cleanup()
            print("✅ Setup verification completed successfully!")

        except Exception as e:
            print(f"❌ Setup verification failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Setup FinanceBench with full PDF datastore")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--backend", choices=["hnsw", "diskann"], default="hnsw", help="LEANN backend"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model",
    )
    parser.add_argument("--max-workers", type=int, default=5, help="Parallel download workers")
    parser.add_argument("--skip-download", action="store_true", help="Skip PDF download")
    parser.add_argument("--skip-build", action="store_true", help="Skip index building")

    args = parser.parse_args()

    print("🏦 FinanceBench Complete Setup")
    print("=" * 50)

    setup = FinanceBenchSetup(args.data_dir)

    try:
        # Step 1: Download dataset
        setup.download_dataset()

        # Step 2: Download PDFs
        if not args.skip_download:
            setup.download_all_pdfs(max_workers=args.max_workers)
        else:
            print("⏭️  Skipping PDF download")

        # Step 3: Build LEANN index
        if not args.skip_build:
            index_path = setup.build_leann_index(
                backend=args.backend, embedding_model=args.embedding_model
            )

            # Step 4: Verify setup
            setup.verify_setup(index_path)
        else:
            print("⏭️  Skipping index building")

        print("\n🎉 FinanceBench setup completed!")
        print(f"📁 Data directory: {setup.data_dir.absolute()}")
        print("\nNext steps:")
        print(
            "1. Run evaluation: python evaluate_financebench.py --index data/index/financebench_full_hnsw.leann"
        )
        print(
            "2. Or test manually: python -c \"from leann import LeannSearcher; s = LeannSearcher('data/index/financebench_full_hnsw.leann'); print(s.search('3M capital expenditure 2018'))\""
        )

    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
