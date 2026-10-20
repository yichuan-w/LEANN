"""
Document RAG example using the unified interface.
Supports PDF, TXT, MD, and other document formats.
"""

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks
from llama_index.core import SimpleDirectoryReader

# MinerU powers OCR for scanned PDFs. Probe for it rather than importing it:
# the real import happens lazily in extract_pdf_with_ocr_fallback, so a run
# that never touches a scanned PDF does not pay for it.
OCR_AVAILABLE = importlib.util.find_spec("mineru") is not None


def extract_pdf_with_ocr_fallback(pdf_path: str, use_ocr: bool = False) -> str:
    """
    Extract text from PDF with OCR fallback.
    Used as a custom file extractor for SimpleDirectoryReader.

    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to try OCR if standard extraction fails

    Returns:
        Extracted text string
    """
    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if text and len(text.strip()) > 100:
            return text
    except Exception:
        pass

    # Try pdfplumber
    try:
        import pdfplumber

        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        if text and len(text.strip()) > 100:
            return text
    except Exception:
        pass

    # Try OCR if enabled
    if use_ocr and OCR_AVAILABLE:
        try:
            from mineru import parse

            # ocr_mode="ocr" forces the OCR pipeline instead of letting MinerU
            # pick: we only get here once the text layer has already proven
            # unusable, so its "auto" heuristic has nothing left to decide.
            return parse(pdf_path, ocr_mode="ocr").markdown()
        except Exception as e:
            print(f"  OCR failed for {pdf_path}: {e}")

    return ""  # Return empty if all fail


class DocumentRAG(BaseRAGExample):
    """RAG example for document processing (PDF, TXT, MD, etc.)."""

    def __init__(self):
        super().__init__(
            name="Document",
            description="Process and query documents (PDF, TXT, MD, etc.) with LEANN",
            default_index_name="test_doc_files",
        )

    def _add_specific_arguments(self, parser):
        """Add document-specific arguments."""
        doc_group = parser.add_argument_group("Document Parameters")
        doc_group.add_argument(
            "--data-dir",
            type=str,
            default="data",
            help="Directory containing documents to index (default: data)",
        )
        doc_group.add_argument(
            "--file-types",
            nargs="+",
            default=None,
            help="Filter by file types (e.g., .pdf .txt .md). If not specified, all supported types are processed",
        )
        doc_group.add_argument(
            "--chunk-size", type=int, default=256, help="Text chunk size (default: 256)"
        )
        doc_group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )
        doc_group.add_argument(
            "--enable-code-chunking",
            action="store_true",
            help="Enable AST-aware chunking for code files in the data directory",
        )

        # OCR parameters
        ocr_group = parser.add_argument_group("OCR Parameters (for scanned PDFs)")
        ocr_group.add_argument(
            "--use-ocr",
            action="store_true",
            help="Force OCR processing for all PDFs (even if they contain text)",
        )
        ocr_group.add_argument(
            "--auto-detect-scanned",
            action="store_true",
            default=True,
            help="Automatically detect and OCR scanned PDFs (default: True)",
        )
        ocr_group.add_argument(
            "--no-auto-detect-scanned",
            dest="auto_detect_scanned",
            action="store_false",
            help="Disable automatic detection of scanned PDFs",
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load documents and convert to text chunks."""
        print(f"Loading documents from: {args.data_dir}")
        if args.file_types:
            print(f"Filtering by file types: {args.file_types}")
        else:
            print("Processing all supported file types")

        # Check if data directory exists
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {args.data_dir}")

        use_ocr_for_all = args.use_ocr
        auto_detect_scanned = args.auto_detect_scanned and OCR_AVAILABLE

        # Create custom PDF extractor with OCR fallback
        def pdf_extractor(file_path: str) -> str:
            """Custom extractor for PDFs with OCR support."""
            # Check if we should try OCR
            try_ocr = use_ocr_for_all

            if not try_ocr and auto_detect_scanned:
                # Quick check: try standard extraction first
                text = extract_pdf_with_ocr_fallback(file_path, use_ocr=False)
                # If we got very little text, it's likely scanned
                if len(text.strip()) < 100:
                    try_ocr = True
                    print(f"Detected scanned PDF: {Path(file_path).name}")

            # Extract with OCR if needed
            text = extract_pdf_with_ocr_fallback(file_path, use_ocr=try_ocr)
            if try_ocr and text:
                print(f"✓ OCR: {Path(file_path).name}")
            return text

        # Load documents with custom PDF extractor
        reader_kwargs = {
            "recursive": True,
            "encoding": "utf-8",
        }
        if args.file_types:
            reader_kwargs["required_exts"] = args.file_types

        # Add custom PDF extractor if we need OCR
        if use_ocr_for_all or auto_detect_scanned:
            reader_kwargs["file_extractor"] = {".pdf": pdf_extractor}

        documents = SimpleDirectoryReader(args.data_dir, **reader_kwargs).load_data(
            show_progress=True
        )

        if not documents:
            print(f"No documents found in {args.data_dir} with extensions {args.file_types}")
            return []

        print(f"Loaded {len(documents)} documents")

        # Determine chunking strategy
        use_ast = args.enable_code_chunking or getattr(args, "use_ast_chunking", False)

        if use_ast:
            print("Using AST-aware chunking for code files")

        # Convert to text chunks with optional AST support
        all_texts = create_text_chunks(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_ast_chunking=use_ast,
            ast_chunk_size=getattr(args, "ast_chunk_size", 512),
            ast_chunk_overlap=getattr(args, "ast_chunk_overlap", 64),
            code_file_extensions=getattr(args, "code_file_extensions", None),
            ast_fallback_traditional=getattr(args, "ast_fallback_traditional", True),
        )

        # Apply max_items limit if specified
        if args.max_items > 0 and len(all_texts) > args.max_items:
            print(f"Limiting to {args.max_items} chunks (from {len(all_texts)})")
            all_texts = all_texts[: args.max_items]

        return all_texts


if __name__ == "__main__":
    import asyncio

    # Example queries for document RAG
    print("\nDocument RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    print("- 'What are the main techniques LEANN uses?'")
    print("- 'What is the technique DLPM?'")
    print("- 'Who does Elizabeth Bennet marry?'")
    print("- 'What challenges did Huawei face while developing the Pangu model?'")
    print("\nNEW: Code-aware chunking available!")
    print("- Use --enable-code-chunking to enable AST-aware chunking for code files")
    print("- Supports Python, Java, C#, TypeScript files")
    print("- Better semantic understanding of code structure")
    if OCR_AVAILABLE:
        print("\n📄 OCR Support: Scanned PDF processing available!")
        print("- Use --use-ocr to force OCR for all PDFs")
        print("- Use --auto-detect-scanned (default) to automatically detect scanned PDFs")
    else:
        print("\n📄 OCR Support: Install mineru for scanned PDF processing:")
        print('  pip install "mineru>=4.0,<5"')
    print("\nOr run without --query for interactive mode\n")

    rag = DocumentRAG()
    asyncio.run(rag.run())
