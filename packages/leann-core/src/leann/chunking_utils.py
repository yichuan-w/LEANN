"""
Enhanced chunking utilities with AST-aware code chunking support.
Packaged within leann-core so installed wheels can import it reliably.
"""

import logging
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
    """Create AST-aware chunks from code documents using astchunk.

    Falls back to traditional chunking if astchunk is unavailable.

    Returns:
        List of dicts with {"text": str, "metadata": dict}
    """
    try:
        from astchunk import ASTChunkBuilder  # optional dependency
    except ImportError as e:
        logger.error(f"astchunk not available: {e}")
        logger.info("Falling back to traditional chunking for code files")
        return _traditional_chunks_as_dicts(documents, max_chunk_size, chunk_overlap)

    all_chunks = []
    for doc in documents:
        language = doc.metadata.get("language")
        if not language:
            logger.warning("No language detected; falling back to traditional chunking")
            all_chunks.extend(_traditional_chunks_as_dicts([doc], max_chunk_size, chunk_overlap))
            continue

        try:
            # Warn once if AST chunk size + overlap might exceed common token limits
            # Note: Actual truncation happens at embedding time with dynamic model limits
            global _ast_token_warning_shown
            estimated_max_tokens = int(
                (max_chunk_size + chunk_overlap) * 1.2
            )  # Conservative estimate
            if estimated_max_tokens > 512 and not _ast_token_warning_shown:
                logger.warning(
                    f"AST chunk size ({max_chunk_size}) + overlap ({chunk_overlap}) = {max_chunk_size + chunk_overlap} chars "
                    f"may exceed 512 token limit (~{estimated_max_tokens} tokens estimated). "
                    f"Consider reducing --ast-chunk-size to {int(400 / 1.2)} or --ast-chunk-overlap to {int(50 / 1.2)}. "
                    f"Note: Chunks will be auto-truncated at embedding time based on your model's actual token limit."
                )
                _ast_token_warning_shown = True

            configs = {
                "max_chunk_size": max_chunk_size,
                "language": language,
                "metadata_template": metadata_template,
                "chunk_overlap": chunk_overlap if chunk_overlap > 0 else 0,
            }

            repo_metadata = {
                "file_path": doc.metadata.get("file_path", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "creation_date": doc.metadata.get("creation_date", ""),
                "last_modified_date": doc.metadata.get("last_modified_date", ""),
            }
            configs["repo_level_metadata"] = repo_metadata

            chunk_builder = ASTChunkBuilder(**configs)
            code_content = doc.get_content()
            if not code_content or not code_content.strip():
                logger.warning("Empty code content, skipping")
                continue

            chunks = chunk_builder.chunkify(code_content)
            for chunk in chunks:
                chunk_text = None
                astchunk_metadata = {}

                if hasattr(chunk, "text"):
                    chunk_text = chunk.text
                elif isinstance(chunk, str):
                    chunk_text = chunk
                elif isinstance(chunk, dict):
                    # Handle astchunk format: {"content": "...", "metadata": {...}}
                    if "content" in chunk:
                        chunk_text = chunk["content"]
                        astchunk_metadata = chunk.get("metadata", {})
                    elif "text" in chunk:
                        chunk_text = chunk["text"]
                    else:
                        chunk_text = str(chunk)  # Last resort
                else:
                    chunk_text = str(chunk)

                if chunk_text and chunk_text.strip():
                    # Extract document-level metadata
                    doc_metadata = {
                        "file_path": doc.metadata.get("file_path", ""),
                        "file_name": doc.metadata.get("file_name", ""),
                    }
                    if "creation_date" in doc.metadata:
                        doc_metadata["creation_date"] = doc.metadata["creation_date"]
                    if "last_modified_date" in doc.metadata:
                        doc_metadata["last_modified_date"] = doc.metadata["last_modified_date"]

                    # Merge document metadata + astchunk metadata
                    combined_metadata = {**doc_metadata, **astchunk_metadata}

                    all_chunks.append({"text": chunk_text.strip(), "metadata": combined_metadata})

            logger.info(
                f"Created {len(chunks)} AST chunks from {language} file: {doc.metadata.get('file_name', 'unknown')}"
            )
        except Exception as e:
            logger.warning(f"AST chunking failed for {language} file: {e}")
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
    if use_ast_chunking:
        code_docs, text_docs = detect_code_files(documents, local_code_extensions)
        if code_docs:
            try:
                all_chunks.extend(
                    create_ast_chunks(
                        code_docs, max_chunk_size=ast_chunk_size, chunk_overlap=ast_chunk_overlap
                    )
                )
            except Exception as e:
                logger.error(f"AST chunking failed: {e}")
                if ast_fallback_traditional:
                    all_chunks.extend(
                        _traditional_chunks_as_dicts(code_docs, chunk_size, chunk_overlap)
                    )
                else:
                    raise
        if text_docs:
            all_chunks.extend(_traditional_chunks_as_dicts(text_docs, chunk_size, chunk_overlap))
    else:
        all_chunks = _traditional_chunks_as_dicts(documents, chunk_size, chunk_overlap)

    logger.info(f"Total chunks created: {len(all_chunks)}")

    # Note: Token truncation is now handled at embedding time with dynamic model limits
    # See get_model_token_limit() and truncate_to_token_limit() in embedding_compute.py
    return all_chunks
