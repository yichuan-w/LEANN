"""
OCR RAG pipeline — index scanned PDFs with MinerU OCR and query them.

Processes PDFs through MinerU (magic-pdf) to extract text from scanned
or image-heavy documents, chunks the results, and builds a LEANN index
for retrieval-augmented generation.

Requires: pip install leann-core[ocr]
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks


def _extract_pdf_with_ocr(pdf_path: str) -> str:
    """Extract text from a PDF, using MinerU OCR when needed."""
    try:
        from leann.cli import extract_pdf_text_with_pymupdf
        text = extract_pdf_text_with_pymupdf(pdf_path, use_ocr=True)
        if text:
            return text
    except Exception:
        pass

    # bare minimum fallback via pymupdf without OCR
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception:
        return ""


class OcrRAG(BaseRAGExample):
    """RAG example for scanned / image-heavy PDFs."""

    def __init__(self):
        super().__init__(
            name="OCR Document",
            description="Process scanned PDFs with OCR and query them with LEANN",
            default_index_name="ocr_docs",
        )

    def _add_specific_arguments(self, parser):
        ocr_group = parser.add_argument_group("OCR Parameters")
        ocr_group.add_argument(
            "--data-dir",
            type=str,
            default="data",
            help="Directory containing PDF files (default: data)",
        )
        ocr_group.add_argument(
            "--chunk-size",
            type=int,
            default=256,
            help="Text chunk size (default: 256)",
        )
        ocr_group.add_argument(
            "--chunk-overlap",
            type=int,
            default=128,
            help="Text chunk overlap (default: 128)",
        )

    async def load_data(self, args) -> list[str]:
        """Load PDFs with OCR and return text chunks."""
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {args.data_dir}")

        pdfs = sorted(data_path.rglob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {args.data_dir}")
            return []

        print(f"Found {len(pdfs)} PDF(s), extracting text with OCR...")

        from llama_index.core import Document

        documents = []
        for pdf in pdfs:
            print(f"  Processing: {pdf.name}")
            text = _extract_pdf_with_ocr(str(pdf))
            if text.strip():
                documents.append(
                    Document(text=text, metadata={"source": str(pdf)})
                )
                print(f"    extracted {len(text)} chars")
            else:
                print(f"    (no text extracted)")

        if not documents:
            return []

        print(f"Loaded {len(documents)} documents, chunking...")

        all_texts = create_text_chunks(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        if args.max_items > 0 and len(all_texts) > args.max_items:
            print(f"Limiting to {args.max_items} chunks (from {len(all_texts)})")
            all_texts = all_texts[: args.max_items]

        return all_texts


if __name__ == "__main__":
    import asyncio

    print("\nOCR RAG Pipeline")
    print("=" * 50)
    print("\nProcesses scanned PDFs with MinerU OCR, indexes them,")
    print("and lets you query the extracted text.")
    print("\nUsage:")
    print("  python ocr_rag.py --data-dir ./pdfs --query 'what does the paper say about X?'")
    print("  python ocr_rag.py --data-dir ./pdfs  # interactive mode")
    print()

    rag = OcrRAG()
    asyncio.run(rag.run())
