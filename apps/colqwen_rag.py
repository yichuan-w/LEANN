#!/usr/bin/env python3
"""
ColQwen RAG - Easy-to-use multimodal PDF retrieval with ColQwen2/ColPali

Usage:
    python -m apps.colqwen_rag build --pdfs ./my_pdfs/ --index my_index
    python -m apps.colqwen_rag search my_index "How does attention work?"
    python -m apps.colqwen_rag ask my_index --interactive
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, cast

# Add LEANN packages to path
_repo_root = Path(__file__).resolve().parents[1]
_leann_core_src = _repo_root / "packages" / "leann-core" / "src"
_leann_hnsw_pkg = _repo_root / "packages" / "leann-backend-hnsw"
if str(_leann_core_src) not in sys.path:
    sys.path.append(str(_leann_core_src))
if str(_leann_hnsw_pkg) not in sys.path:
    sys.path.append(str(_leann_hnsw_pkg))

import torch  # noqa: E402
from colpali_engine import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor  # noqa: E402
from colpali_engine.utils.torch_utils import ListDataset  # noqa: E402
from pdf2image import convert_from_path  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Import the existing multi-vector implementation
sys.path.append(str(_repo_root / "apps" / "multimodal" / "vision-based-pdf-multi-vector"))
from leann_multi_vector import LeannMultiVector  # noqa: E402


class ColQwenRAG:
    """Easy-to-use ColQwen RAG system for multimodal PDF retrieval."""

    def __init__(self, model_type: str = "colpali"):
        """
        Initialize ColQwen RAG system.

        Args:
            model_type: "colqwen2" or "colpali"
        """
        self.model_type = model_type
        self.device = self._get_device()
        # Use float32 on MPS to avoid memory issues, float16 on CUDA, bfloat16 on CPU
        if self.device.type == "mps":
            self.dtype = torch.float32
        elif self.device.type == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.bfloat16

        print(f"🚀 Initializing {model_type.upper()} on {self.device} with {self.dtype}")

        # Load model and processor with MPS-optimized settings
        try:
            if model_type == "colqwen2":
                self.model_name = "vidore/colqwen2-v1.0"
                if self.device.type == "mps":
                    # For MPS, load on CPU first then move to avoid memory allocation issues
                    self.model = ColQwen2.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                    ).eval()
                    self.model = self.model.to(self.device)
                else:
                    self.model = ColQwen2.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        device_map=self.device,
                        low_cpu_mem_usage=True,
                    ).eval()
                self.processor = ColQwen2Processor.from_pretrained(self.model_name)
            else:  # colpali
                self.model_name = "vidore/colpali-v1.2"
                if self.device.type == "mps":
                    # For MPS, load on CPU first then move to avoid memory allocation issues
                    self.model = ColPali.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                    ).eval()
                    self.model = self.model.to(self.device)
                else:
                    self.model = ColPali.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        device_map=self.device,
                        low_cpu_mem_usage=True,
                    ).eval()
                self.processor = ColPaliProcessor.from_pretrained(self.model_name)
        except Exception as e:
            if "memory" in str(e).lower() or "offload" in str(e).lower():
                print(f"⚠️  Memory constraint on {self.device}, using CPU with optimizations...")
                self.device = torch.device("cpu")
                self.dtype = torch.float32

                if model_type == "colqwen2":
                    self.model = ColQwen2.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                    ).eval()
                else:
                    self.model = ColPali.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                    ).eval()
            else:
                raise

    def _get_device(self):
        """Auto-select best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def build_index(self, pdf_paths: list[str], index_name: str, pages_dir: Optional[str] = None):
        """
        Build multimodal index from PDF files.

        Args:
            pdf_paths: List of PDF file paths
            index_name: Name for the index
            pages_dir: Directory to save page images (optional)
        """
        print(f"Building index '{index_name}' from {len(pdf_paths)} PDFs...")

        # Convert PDFs to images
        all_images = []
        all_metadata = []

        if pages_dir:
            os.makedirs(pages_dir, exist_ok=True)

        for pdf_path in tqdm(pdf_paths, desc="Converting PDFs"):
            try:
                images = convert_from_path(pdf_path, dpi=150)
                pdf_name = Path(pdf_path).stem

                for i, image in enumerate(images):
                    # Save image if pages_dir specified
                    if pages_dir:
                        image_path = Path(pages_dir) / f"{pdf_name}_page_{i + 1}.png"
                        image.save(image_path)

                    all_images.append(image)
                    all_metadata.append(
                        {
                            "pdf_path": pdf_path,
                            "pdf_name": pdf_name,
                            "page_number": i + 1,
                            "image_path": str(image_path) if pages_dir else None,
                        }
                    )

            except Exception as e:
                print(f"❌ Error processing {pdf_path}: {e}")
                continue

        print(f"📄 Converted {len(all_images)} pages from {len(pdf_paths)} PDFs")
        print(f"All metadata: {all_metadata}")

        # Generate embeddings
        print("🧠 Generating embeddings...")
        embeddings = self._embed_images(all_images)

        # Build LEANN index
        print("🔍 Building LEANN index...")
        leann_mv = LeannMultiVector(
            index_path=index_name,
            dim=embeddings.shape[-1],
            embedding_model_name=self.model_type,
        )

        # Create collection and insert data
        leann_mv.create_collection()
        for i, (embedding, metadata) in enumerate(zip(embeddings, all_metadata)):
            data = {
                "doc_id": i,
                "filepath": metadata.get("image_path", ""),
                "colbert_vecs": embedding.numpy(),  # Convert tensor to numpy
            }
            leann_mv.insert(data)

        # Build the index
        leann_mv.create_index()
        print(f"✅ Index '{index_name}' built successfully!")

        return leann_mv

    def search(self, index_name: str, query: str, top_k: int = 5):
        """
        Search the index with a text query.

        Args:
            index_name: Name of the index to search
            query: Text query
            top_k: Number of results to return
        """
        print(f"🔍 Searching '{index_name}' for: '{query}'")

        # Load index
        leann_mv = LeannMultiVector(
            index_path=index_name,
            dim=128,  # Will be updated when loading
            embedding_model_name=self.model_type,
        )

        # Generate query embedding
        query_embedding = self._embed_query(query)

        # Search (returns list of (score, doc_id) tuples)
        search_results = leann_mv.search(query_embedding.numpy(), topk=top_k)

        # Display results
        print(f"\n📋 Top {len(search_results)} results:")
        for i, (score, doc_id) in enumerate(search_results, 1):
            # Get metadata for this doc_id (we need to load the metadata)
            print(f"{i}. Score: {score:.3f} | Doc ID: {doc_id}")

        return search_results

    def _validate_query_input(self, query: str) -> bool:
        """
        Validate user input to prevent injection attacks (command, SQL, NoSQL, etc).
        
        Args:
            query: The user input query to validate
            
        Returns:
            True if query is safe, False otherwise
        """
        if not query:
            return False
        
        # Enforce maximum length to prevent buffer overflow and DoS
        if len(query) > 10000:
            print("⚠️  Input blocked: query exceeds maximum length")
            return False
        
        # Block command injection patterns
        command_injection_patterns = [
            ";",  # Command separator
            "|",  # Pipe operator
            "&",  # Background/AND operator
            "$(",  # Command substitution
            "`",  # Backticks (command substitution)
            "eval",  # eval function
            "__import__",  # Import manipulation
            "exec",  # exec function
            "os.system",  # System calls
            "subprocess",  # Subprocess calls
            "open(",  # File access
        ]
        
        # Block SQL/NoSQL injection patterns
        sql_injection_patterns = [
            " or ",  # SQL OR condition
            " and ",  # SQL AND condition
            " union ",  # SQL UNION
            " select ",  # SQL SELECT
            " drop ",  # SQL DROP
            " insert ",  # SQL INSERT
            " update ",  # SQL UPDATE
            " delete ",  # SQL DELETE
            "--",  # SQL comment
            "/*",  # SQL comment
            "*/",  # SQL comment
            "$where",  # MongoDB $where
            "$ne",  # MongoDB $ne operator
            "$gt",  # MongoDB $gt operator
            "$lt",  # MongoDB $lt operator
            "$regex",  # MongoDB regex
            "$or",  # MongoDB $or
            "$and",  # MongoDB $and
        ]
        
        query_lower = query.lower()
        
        # Check command injection patterns
        for pattern in command_injection_patterns:
            if pattern in query_lower:
                print(f"⚠️  Input blocked: contains potentially dangerous pattern '{pattern}'")
                return False
        
        # Check SQL/NoSQL injection patterns
        for pattern in sql_injection_patterns:
            if pattern in query_lower:
                print(f"⚠️  Input blocked: contains potentially dangerous pattern '{pattern}'")
                return False
        
        return True

    def ask(self, index_name: str, interactive: bool = False):
        """
        Interactive Q&A with the indexed documents.
        
        SECURITY: All user input from input() is validated via _validate_query_input()
        before being passed to search(). This prevents command injection, SQL injection,
        NoSQL injection, and other injection attacks.

        Args:
            index_name: Name of the index to query
            interactive: Whether to run in interactive mode
        """
        print(f"💬 ColQwen Chat with '{index_name}'")

        if interactive:
            print("Type 'quit' to exit, 'help' for commands")
            while True:
                try:
                    # Collect user input
                    user_input = input("\n🤔 Your question: ").strip()
                    
                    # Check for exit commands first (these bypass validation as they're control flow)
                    if user_input.lower() in ["quit", "exit", "q"]:
                        break
                    elif user_input.lower() == "help":
                        print("Commands: quit/exit/q (exit), help (this message)")
                        continue
                    elif not user_input:
                        continue
                    
                    # SECURITY: Validate user input before processing to prevent injection attacks
                    if not self._validate_query_input(user_input):
                        print("❌ Invalid input. Please ask a natural language question.")
                        continue
                    
                    # Pass validated query to search - input is now safe
                    self.search(index_name, user_input, top_k=3)

                    # TODO: Add answer generation with Qwen-VL
                    print("\n💡 For detailed answers, we can integrate Qwen-VL here!")

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
        else:
            # Collect user input
            user_input = input("🤔 Your question: ").strip()
            
            # SECURITY: Validate user input before processing to prevent injection attacks
            if user_input and self._validate_query_input(user_input):
                # Pass validated query to search - input is now safe
                self.search(index_name, user_input)

    def _embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Generate embeddings for a list of images."""
        dataset = ListDataset(images)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Embedding images"):
                batch_images = cast(list, batch)
                batch_inputs = self.processor.process_images(batch_images).to(self.device)
                batch_embeddings = self.model(**batch_inputs)
                embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    def _embed_query(self, query: str) -> torch.Tensor:
        """Generate embedding for a text query."""
        with torch.no_grad():
            query_inputs = self.processor.process_queries([query]).to(self.device)
            query_embedding = self.model(**query_inputs)
            return query_embedding.cpu()


def main():
    parser = argparse.ArgumentParser(description="ColQwen RAG - Easy multimodal PDF retrieval")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build index from PDFs")
    build_parser.add_argument("--pdfs", required=True, help="Directory containing PDF files")
    build_parser.add_argument("--index", required=True, help="Index name")
    build_parser.add_argument(
        "--model", choices=["colqwen2", "colpali"], default="colqwen2", help="Model to use"
    )
    build_parser.add_argument("--pages-dir", help="Directory to save page images")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("index", help="Index name")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument(
        "--model", choices=["colqwen2", "colpali"], default="colqwen2", help="Model to use"
    )

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Interactive Q&A")
    ask_parser.add_argument("index", help="Index name")
    ask_parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    ask_parser.add_argument(
        "--model", choices=["colqwen2", "colpali"], default="colqwen2", help="Model to use"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize ColQwen RAG
    if args.command == "build":
        colqwen = ColQwenRAG(args.model)

        # Get PDF files
        pdf_dir = Path(args.pdfs)
        if pdf_dir.is_file() and pdf_dir.suffix.lower() == ".pdf":
            pdf_paths = [str(pdf_dir)]
        elif pdf_dir.is_dir():
            pdf_paths = [str(p) for p in pdf_dir.glob("*.pdf")]
        else:
            print(f"❌ Invalid PDF path: {args.pdfs}")
            return

        if not pdf_paths:
            print(f"❌ No PDF files found in {args.pdfs}")
            return

        colqwen.build_index(pdf_paths, args.index, args.pages_dir)

    elif args.command == "search":
        colqwen = ColQwenRAG(args.model)
        colqwen.search(args.index, args.query, args.top_k)

    elif args.command == "ask":
        colqwen = ColQwenRAG(args.model)
        colqwen.ask(args.index, args.interactive)


if __name__ == "__main__":
    main()
