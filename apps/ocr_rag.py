"""
OCR RAG example for scanned and image-heavy PDFs.

This app extracts embedded PDF text first, then OCRs only pages without text.
Install the optional OCR dependencies with `pip install "leann[ocr]"` and make
sure the Tesseract binary is available on PATH.
"""

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks
from leann.cli import extract_pdf_text_with_pymupdf
from llama_index.core import Document


def load_ocr_pdf_documents(data_dir: str | Path) -> list[Document]:
    """Load PDFs from a directory using opt-in OCR for image-only pages."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_path}")

    pdf_paths = sorted(data_path.rglob("*.pdf"))
    documents = []
    for pdf_path in pdf_paths:
        text = extract_pdf_text_with_pymupdf(str(pdf_path), use_ocr=True)
        if not text or not text.strip():
            continue
        documents.append(
            Document(
                text=text,
                metadata={
                    "source": str(pdf_path),
                    "file_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "ocr_enabled": True,
                },
            )
        )
    return documents


class OcrRAG(BaseRAGExample):
    """RAG example for scanned and image-heavy PDFs."""

    def __init__(self):
        super().__init__(
            name="OCR Document",
            description="Process scanned and image-heavy PDFs with LEANN",
            default_index_name="ocr_docs",
        )

    def _add_specific_arguments(self, parser):
        """Add OCR-specific arguments."""
        ocr_group = parser.add_argument_group("OCR Parameters")
        ocr_group.add_argument(
            "--data-dir",
            type=str,
            default="data",
            help="Directory containing PDF files to index (default: data)",
        )
        ocr_group.add_argument(
            "--chunk-size", type=int, default=256, help="Text chunk size (default: 256)"
        )
        ocr_group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load OCR documents and convert them to text chunks."""
        print(f"Loading OCR PDFs from: {args.data_dir}")
        documents = load_ocr_pdf_documents(args.data_dir)
        if not documents:
            print(f"No OCR text extracted from PDFs in {args.data_dir}")
            return []

        all_texts = create_text_chunks(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if args.max_items > 0 and len(all_texts) > args.max_items:
            print(f"Limiting to {args.max_items} chunks (from {len(all_texts)})")
            all_texts = all_texts[: args.max_items]
        return all_texts


def main():
    """Run the OCR RAG example."""
    import asyncio

    rag = OcrRAG()
    asyncio.run(rag.run())


if __name__ == "__main__":
    main()
