"""
Enhanced chunking utilities with AST-aware code chunking support.
Packaged within leann-core so installed wheels can import it reliably.
"""

import logging
import os
import concurrent.futures
from multiprocessing import get_context, cpu_count
from pathlib import Path
from typing import Any, Optional

from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)

# Flag to ensure AST token warning only shown once per session
_ast_token_warning_shown = False


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for a text string.
    Uses conservative estimation: ~4 characters per token for natural text,
    ~1.2 tokens per character for code (worse tokenization).

    Args:
        text: Input text to estimate tokens for

    Returns:
        Estimated token count
    """
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except ImportError:
        # Fallback: Conservative character-based estimation
        # Assume worst case for code: 1.2 tokens per character
        return int(len(text) * 1.2)


def calculate_safe_chunk_size(
    model_token_limit: int,
    overlap_tokens: int,
    chunking_mode: str = "traditional",
    safety_factor: float = 0.9,
) -> int:
    """
    Calculate safe chunk size accounting for overlap and safety margin.

    Args:
        model_token_limit: Maximum tokens supported by embedding model
        overlap_tokens: Overlap size (tokens for traditional, chars for AST)
        chunking_mode: "traditional" (tokens) or "ast" (characters)
        safety_factor: Safety margin (0.9 = 10% safety margin)

    Returns:
        Safe chunk size: tokens for traditional, characters for AST
    """
    safe_limit = int(model_token_limit * safety_factor)

    if chunking_mode == "traditional":
        # Traditional chunking uses tokens
        # Max chunk = chunk_size + overlap, so chunk_size = limit - overlap
        return max(1, safe_limit - overlap_tokens)
    else:  # AST chunking
        # AST uses characters, need to convert
        # Conservative estimate: 1.2 tokens per char for code
        overlap_chars = int(overlap_tokens * 3)  # ~3 chars per token for code
        safe_chars = int(safe_limit / 1.2)
        return max(1, safe_chars - overlap_chars)


def validate_chunk_token_limits(chunks: list[str], max_tokens: int = 512) -> tuple[list[str], int]:
    """
    Validate that chunks don't exceed token limits and truncate if necessary.

    Args:
        chunks: List of text chunks to validate
        max_tokens: Maximum tokens allowed per chunk

    Returns:
        Tuple of (validated_chunks, num_truncated)
    """
    validated_chunks = []
    num_truncated = 0

    for i, chunk in enumerate(chunks):
        estimated_tokens = estimate_token_count(chunk)

        if estimated_tokens > max_tokens:
            # Truncate chunk to fit token limit
            try:
                import tiktoken

                encoder = tiktoken.get_encoding("cl100k_base")
                tokens = encoder.encode(chunk)
                if len(tokens) > max_tokens:
                    truncated_tokens = tokens[:max_tokens]
                    truncated_chunk = encoder.decode(truncated_tokens)
                    validated_chunks.append(truncated_chunk)
                    num_truncated += 1
                    logger.warning(
                        f"Truncated chunk {i} from {len(tokens)} to {max_tokens} tokens "
                        f"(from {len(chunk)} to {len(truncated_chunk)} characters)"
                    )
                else:
                    validated_chunks.append(chunk)
            except ImportError:
                # Fallback: Conservative character truncation
                char_limit = int(max_tokens / 1.2)  # Conservative for code
                if len(chunk) > char_limit:
                    truncated_chunk = chunk[:char_limit]
                    validated_chunks.append(truncated_chunk)
                    num_truncated += 1
                    logger.warning(
                        f"Truncated chunk {i} from {len(chunk)} to {char_limit} characters "
                        f"(conservative estimate for {max_tokens} tokens)"
                    )
                else:
                    validated_chunks.append(chunk)
        else:
            validated_chunks.append(chunk)

    if num_truncated > 0:
        logger.warning(f"Truncated {num_truncated}/{len(chunks)} chunks to fit token limits")

    return validated_chunks, num_truncated


# Code file extensions supported by astchunk
CODE_EXTENSIONS = {
    ".py": "python",
    ".java": "java",
    ".cs": "csharp",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
}


def detect_code_files(documents, code_extensions=None) -> tuple[list, list]:
    """Separate documents into code files and regular text files."""
    if code_extensions is None:
        code_extensions = CODE_EXTENSIONS

    code_docs = []
    text_docs = []

    for doc in documents:
        file_path = doc.metadata.get("file_path", "") or doc.metadata.get("file_name", "")
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in code_extensions:
                doc.metadata["language"] = code_extensions[file_ext]
                doc.metadata["is_code"] = True
                code_docs.append(doc)
            else:
                doc.metadata["is_code"] = False
                text_docs.append(doc)
        else:
            doc.metadata["is_code"] = False
            text_docs.append(doc)

    logger.info(f"Detected {len(code_docs)} code files and {len(text_docs)} text files")
    return code_docs, text_docs


def get_language_from_extension(file_path: str) -> Optional[str]:
    """Return language string from a filename/extension using CODE_EXTENSIONS."""
    ext = Path(file_path).suffix.lower()
    return CODE_EXTENSIONS.get(ext)


def create_ast_chunks(
    documents,
    max_chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata_template: str = "default",
) -> list[dict[str, Any]]:
    """Create AST-aware chunks from code documents using CodeAnalyzer.

    Delegates to leann.analysis.CodeAnalyzer which uses astchunk under the hood.
    Falls back to traditional chunking if AST analysis fails or is unavailable.

    Returns:
        List of dicts with {"text": str, "metadata": dict}
    """
    try:
        from leann.analysis import ASTCHUNK_AVAILABLE, CodeAnalyzer

        if not ASTCHUNK_AVAILABLE:
            raise ImportError("astchunk not available via CodeAnalyzer")
    except ImportError as e:
        logger.error(f"AST chunking unavailable: {e}")
        logger.info("Falling back to traditional chunking for code files")
        return _traditional_chunks_as_dicts(documents, max_chunk_size, chunk_overlap)

    all_chunks = []

    # Cache analyzers by language to avoid repeated re-initialization overhead
    analyzers = {}

    for doc in documents:
        language = doc.metadata.get("language")
        if not language:
            logger.warning("No language detected; falling back to traditional chunking")
            all_chunks.extend(_traditional_chunks_as_dicts([doc], max_chunk_size, chunk_overlap))
            continue

        try:
            # 1. Get or create analyzer for this language
            if language not in analyzers:
                analyzers[language] = CodeAnalyzer(language)

            analyzer = analyzers[language]

            # 2. Get content and basic metadata
            code_content = doc.get_content()
            if not code_content or not code_content.strip():
                continue

            file_path = doc.metadata.get("file_path", "") or doc.metadata.get("file_name", "")

            # 3. Base metadata from document
            doc_metadata = {
                "file_path": file_path,
                "file_name": doc.metadata.get("file_name", ""),
                "language": language,
            }
            if "creation_date" in doc.metadata:
                doc_metadata["creation_date"] = doc.metadata["creation_date"]
            if "last_modified_date" in doc.metadata:
                doc_metadata["last_modified_date"] = doc.metadata["last_modified_date"]

            # 4. Generate Semantic Chunks
            # CodeAnalyzer handles the astchunk call + rich context injection (global imports)
            chunks = analyzer.get_semantic_chunks(
                code=code_content,
                file_path=file_path,
                metadata=doc_metadata,  # Passed as repo-level metadata
            )

            if chunks:
                all_chunks.extend(chunks)
                logger.debug(f"Created {len(chunks)} AST chunks for {file_path}")
            else:
                # Fallback if analyzer returns empty (e.g. parse error) but content exists
                logger.warning(f"AST analysis yielded no chunks for {file_path}, falling back.")
                all_chunks.extend(
                    _traditional_chunks_as_dicts([doc], max_chunk_size, chunk_overlap)
                )

        except Exception as e:
            logger.warning(
                f"AST chunking failed for {language} file {doc.metadata.get('file_path')}: {e}"
            )
            logger.info("Falling back to traditional chunking")
            all_chunks.extend(_traditional_chunks_as_dicts([doc], max_chunk_size, chunk_overlap))

    return all_chunks


def create_traditional_chunks(
    documents, chunk_size: int = 256, chunk_overlap: int = 128
) -> list[dict[str, Any]]:
    """Create traditional text chunks using LlamaIndex SentenceSplitter.

    Returns:
        List of dicts with {"text": str, "metadata": dict}
    """
    if chunk_size <= 0:
        logger.warning(f"Invalid chunk_size={chunk_size}, using default value of 256")
        chunk_size = 256
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2

    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
        paragraph_separator="\n\n",
    )

    result = []
    for doc in documents:
        # Extract document-level metadata
        doc_metadata = {
            "file_path": doc.metadata.get("file_path", ""),
            "file_name": doc.metadata.get("file_name", ""),
        }
        if "creation_date" in doc.metadata:
            doc_metadata["creation_date"] = doc.metadata["creation_date"]
        if "last_modified_date" in doc.metadata:
            doc_metadata["last_modified_date"] = doc.metadata["last_modified_date"]

        try:
            nodes = node_parser.get_nodes_from_documents([doc])
            if nodes:
                for node in nodes:
                    result.append({"text": node.get_content(), "metadata": doc_metadata})
        except Exception as e:
            logger.error(f"Traditional chunking failed for document: {e}")
            content = doc.get_content()
            if content and content.strip():
                result.append({"text": content.strip(), "metadata": doc_metadata})
    return result


def _traditional_chunks_as_dicts(
    documents, chunk_size: int = 256, chunk_overlap: int = 128
) -> list[dict[str, Any]]:
    """Helper: Traditional chunking that returns dict format for consistency.

    This is now just an alias for create_traditional_chunks for backwards compatibility.
    """
    return create_traditional_chunks(documents, chunk_size, chunk_overlap)


def create_text_chunks(
    documents,
    chunk_size: int = 256,
    chunk_overlap: int = 128,
    use_ast_chunking: bool = False,
    ast_chunk_size: int = 512,
    ast_chunk_overlap: int = 64,
    code_file_extensions: Optional[list[str]] = None,
    ast_fallback_traditional: bool = True,
) -> list[dict[str, Any]]:
    """Create text chunks from documents with optional AST support for code files.

    Returns:
        List of dicts with {"text": str, "metadata": dict}
    """
    if not documents:
        logger.warning("No documents provided for chunking")
        return []

    local_code_extensions = CODE_EXTENSIONS.copy()
    if code_file_extensions:
        ext_mapping = {
            ".py": "python",
            ".java": "java",
            ".cs": "c_sharp",
            ".ts": "typescript",
            ".tsx": "typescript",
        }
        for ext in code_file_extensions:
            if ext.lower() not in local_code_extensions:
                if ext.lower() in ext_mapping:
                    local_code_extensions[ext.lower()] = ext_mapping[ext.lower()]
                else:
                    logger.warning(f"Unsupported extension {ext}, will use traditional chunking")

    all_chunks = []

    # helper for parallel processing
    def process_docs_parallel(docs, chunk_func, **kwargs):
        """Internal helper to process documents in parallel batches."""
        if len(docs) <= 5: # Small sets are faster serial
            return chunk_func(docs, **kwargs)

        # 1. Determine worker count
        cpu_total = cpu_count() or 4
        num_workers = int(os.getenv("LEANN_INDEXING_WORKERS", min(cpu_total, 8)))
        
        # 2. Calculate batch size (target ~4 batches per worker for load balancing)
        target_batches = num_workers * 4
        batch_size = max(5, len(docs) // target_batches)
        batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]
        
        logger.info(f"Parallelizing {len(docs)} docs across {num_workers} workers (batch_size={batch_size})")

        # 3. Use 'spawn' for safety with C-extensions (tree-sitter/faiss)
        ctx = get_context("spawn")
        all_chunks = []
        
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(batches), desc="Processing AST chunks (parallel)", unit="batch", leave=False)
        except ImportError:
            pbar = None

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            # Note: chunk_func must be top-level and picklable
            future_to_batch = {executor.submit(chunk_func, batch, **kwargs): batch for batch in batches}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                if pbar:
                    pbar.update(1)
                try:
                    results = future.result()
                    if results:
                        all_chunks.extend(results)
                except Exception as e:
                    batch_sample = future_to_batch[future][0].metadata.get("file_path", "unknown")
                    logger.error(f"Parallel worker failed on batch starting with {batch_sample}: {e}")
        
        if pbar:
            pbar.close()
        
        return all_chunks

    if use_ast_chunking:
        code_docs, text_docs = detect_code_files(documents, local_code_extensions)
        if code_docs:
            try:
                # AST chunking is CPU heavy, but running serial to be safe
                all_chunks.extend(
                    process_docs_parallel(
                        code_docs,
                        create_ast_chunks,
                        max_chunk_size=ast_chunk_size,
                        chunk_overlap=ast_chunk_overlap,
                    )
                )
            except Exception as e:
                logger.error(f"AST chunking failed: {e}")
                if ast_fallback_traditional:
                    all_chunks.extend(
                        process_docs_parallel(
                            code_docs,
                            _traditional_chunks_as_dicts,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                    )
                else:
                    raise
        if text_docs:
            all_chunks.extend(
                process_docs_parallel(
                    text_docs,
                    _traditional_chunks_as_dicts,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
    else:
        all_chunks.extend(
            process_docs_parallel(
                documents,
                _traditional_chunks_as_dicts,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
