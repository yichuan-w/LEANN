"""
Enhanced chunking utilities with AST-aware code chunking support.
Provides unified interface for both traditional and AST-based text chunking.
"""

import logging
from pathlib import Path
from typing import Optional

from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)

# Code file extensions supported by astchunk
CODE_EXTENSIONS = {
    ".py": "python",
    ".java": "java",
    ".cs": "csharp",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".go": "go",
}

# Default chunk parameters for different content types
DEFAULT_CHUNK_PARAMS = {
    "code": {
        "max_chunk_size": 512,
        "chunk_overlap": 64,
    },
    "text": {
        "chunk_size": 256,
        "chunk_overlap": 128,
    },
}


def detect_code_files(documents, code_extensions=None) -> tuple[list, list]:
    """
    Separate documents into code files and regular text files.

    Args:
        documents: List of LlamaIndex Document objects
        code_extensions: Dict mapping file extensions to languages (defaults to CODE_EXTENSIONS)

    Returns:
        Tuple of (code_documents, text_documents)
    """
    if code_extensions is None:
        code_extensions = CODE_EXTENSIONS

    code_docs = []
    text_docs = []

    for doc in documents:
        # Get file path from metadata
        file_path = doc.metadata.get("file_path", "")
        if not file_path:
            # Fallback to file_name
            file_path = doc.metadata.get("file_name", "")

        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in code_extensions:
                # Add language info to metadata
                doc.metadata["language"] = code_extensions[file_ext]
                doc.metadata["is_code"] = True
                code_docs.append(doc)
            else:
                doc.metadata["is_code"] = False
                text_docs.append(doc)
        else:
            # If no file path, treat as text
            doc.metadata["is_code"] = False
            text_docs.append(doc)

    logger.info(f"Detected {len(code_docs)} code files and {len(text_docs)} text files")
    return code_docs, text_docs


def get_language_from_extension(file_path: str) -> Optional[str]:
    """Get the programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    return CODE_EXTENSIONS.get(ext)


def create_ast_chunks(
    documents,
    max_chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata_template: str = "default",
) -> list[str]:
    """
    Create AST-aware chunks from code documents using astchunk.

    Args:
        documents: List of code documents
        max_chunk_size: Maximum characters per chunk
        chunk_overlap: Number of AST nodes to overlap between chunks
        metadata_template: Template for chunk metadata

    Returns:
        List of text chunks with preserved code structure
    """
    try:
        from astchunk import ASTChunkBuilder
    except ImportError as e:
        logger.error(f"astchunk not available: {e}")
        logger.info("Trying local AST chunkers before falling back to traditional chunking")
        return create_ast_chunks_with_local_chunkers(documents, max_chunk_size, chunk_overlap)

    all_chunks = []

    for doc in documents:
        # Get language from metadata (set by detect_code_files)
        language = doc.metadata.get("language")
        if not language:
            logger.warning(
                "No language detected for document, falling back to traditional chunking"
            )
            traditional_chunks = create_traditional_chunks([doc], max_chunk_size, chunk_overlap)
            all_chunks.extend(traditional_chunks)
            continue

        # Special handling for Go - use local chunker since external astchunk doesn't support it
        if language == "go":
            logger.debug("Using local Go AST chunker for Go file")
            try:
                local_chunks = create_ast_chunks_with_local_chunkers(
                    [doc], max_chunk_size, chunk_overlap
                )
                all_chunks.extend(local_chunks)
                continue
            except Exception as e:
                logger.warning(f"Local Go AST chunking failed: {e}")
                logger.info("Falling back to traditional chunking for Go file")
                traditional_chunks = create_traditional_chunks([doc], max_chunk_size, chunk_overlap)
                all_chunks.extend(traditional_chunks)
                continue

        try:
            # Configure astchunk
            configs = {
                "max_chunk_size": max_chunk_size,
                "language": language,
                "metadata_template": metadata_template,
                "chunk_overlap": chunk_overlap if chunk_overlap > 0 else 0,
            }

            # Add repository-level metadata if available
            repo_metadata = {
                "file_path": doc.metadata.get("file_path", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "creation_date": doc.metadata.get("creation_date", ""),
                "last_modified_date": doc.metadata.get("last_modified_date", ""),
            }
            configs["repo_level_metadata"] = repo_metadata

            # Create chunk builder and process
            chunk_builder = ASTChunkBuilder(**configs)
            code_content = doc.get_content()

            if not code_content or not code_content.strip():
                logger.warning("Empty code content, skipping")
                continue

            chunks = chunk_builder.chunkify(code_content)

            # Extract text content from chunks
            for chunk in chunks:
                if hasattr(chunk, "text"):
                    chunk_text = chunk.text
                elif isinstance(chunk, dict) and "text" in chunk:
                    chunk_text = chunk["text"]
                elif isinstance(chunk, str):
                    chunk_text = chunk
                else:
                    # Try to convert to string
                    chunk_text = str(chunk)

                if chunk_text and chunk_text.strip():
                    all_chunks.append(chunk_text.strip())

            logger.info(
                f"Created {len(chunks)} AST chunks from {language} file: {doc.metadata.get('file_name', 'unknown')}"
            )

        except Exception as e:
            logger.warning(f"AST chunking failed for {language} file: {e}")

            # For Go files, try local chunker before falling back to traditional chunking
            if language == "go":
                logger.info("Trying local Go AST chunker as fallback")
                try:
                    local_chunks = create_ast_chunks_with_local_chunkers(
                        [doc], max_chunk_size, chunk_overlap
                    )
                    all_chunks.extend(local_chunks)
                    continue
                except Exception as local_e:
                    logger.warning(f"Local Go AST chunker also failed: {local_e}")

            logger.info("Falling back to traditional chunking")
            traditional_chunks = create_traditional_chunks([doc], max_chunk_size, chunk_overlap)
            all_chunks.extend(traditional_chunks)

    return all_chunks


def create_ast_chunks_with_local_chunkers(
    documents,
    max_chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """
    Create AST-aware chunks using local chunker implementations.

    Args:
        documents: List of code documents
        max_chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks with preserved code structure
    """
    all_chunks = []

    for doc in documents:
        language = doc.metadata.get("language")
        if not language:
            logger.warning(
                "No language detected for document, falling back to traditional chunking"
            )
            traditional_chunks = create_traditional_chunks([doc], max_chunk_size, chunk_overlap)
            all_chunks.extend(traditional_chunks)
            continue

        try:
            code_content = doc.get_content()
            if not code_content or not code_content.strip():
                logger.warning("Empty code content, skipping")
                continue

            chunks = []

            # Use appropriate local chunker based on language
            if language == "go":
                logger.debug("Using local Go AST chunker")
                try:
                    from .ast_chunkers.go import chunk_go_code

                    chunk_dicts = chunk_go_code(code_content, max_chunk_size, chunk_overlap)
                    chunks = [
                        chunk_dict["text"] for chunk_dict in chunk_dicts if chunk_dict.get("text")
                    ]
                except ImportError as e:
                    logger.warning(f"Local Go chunker not available: {e}")
                    raise
            else:
                # No local chunker available for this language
                logger.info(
                    f"No local AST chunker available for {language}, using traditional chunking"
                )
                raise ValueError(f"No local chunker for {language}")

            if chunks:
                all_chunks.extend(chunks)
                logger.info(
                    f"Created {len(chunks)} local AST chunks from {language} file: {doc.metadata.get('file_name', 'unknown')}"
                )
            else:
                logger.warning(
                    f"No chunks created from {language} file, falling back to traditional chunking"
                )
                traditional_chunks = create_traditional_chunks([doc], max_chunk_size, chunk_overlap)
                all_chunks.extend(traditional_chunks)

        except Exception as e:
            logger.warning(f"Local AST chunking failed for {language} file: {e}")
            logger.info("Falling back to traditional chunking")
            traditional_chunks = create_traditional_chunks([doc], max_chunk_size, chunk_overlap)
            all_chunks.extend(traditional_chunks)

    return all_chunks


def create_traditional_chunks(
    documents, chunk_size: int = 256, chunk_overlap: int = 128
) -> list[str]:
    """
    Create traditional text chunks using LlamaIndex SentenceSplitter.

    Args:
        documents: List of documents to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    # Handle invalid chunk_size values
    if chunk_size <= 0:
        logger.warning(f"Invalid chunk_size={chunk_size}, using default value of 256")
        chunk_size = 256

    # Ensure chunk_overlap is not negative and not larger than chunk_size
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

    all_texts = []
    for doc in documents:
        try:
            nodes = node_parser.get_nodes_from_documents([doc])
            if nodes:
                chunk_texts = [node.get_content() for node in nodes]
                all_texts.extend(chunk_texts)
                logger.debug(f"Created {len(chunk_texts)} traditional chunks from document")
        except Exception as e:
            logger.error(f"Traditional chunking failed for document: {e}")
            # As last resort, add the raw content
            content = doc.get_content()
            if content and content.strip():
                all_texts.append(content.strip())

    return all_texts


def create_text_chunks(
    documents,
    chunk_size: int = 256,
    chunk_overlap: int = 128,
    use_ast_chunking: bool = False,
    ast_chunk_size: int = 512,
    ast_chunk_overlap: int = 64,
    code_file_extensions: Optional[list[str]] = None,
    ast_fallback_traditional: bool = True,
) -> list[str]:
    """
    Create text chunks from documents with optional AST support for code files.

    Args:
        documents: List of LlamaIndex Document objects
        chunk_size: Size for traditional text chunks
        chunk_overlap: Overlap for traditional text chunks
        use_ast_chunking: Whether to use AST chunking for code files
        ast_chunk_size: Size for AST chunks
        ast_chunk_overlap: Overlap for AST chunks
        code_file_extensions: Custom list of code file extensions
        ast_fallback_traditional: Fall back to traditional chunking on AST errors

    Returns:
        List of text chunks
    """
    if not documents:
        logger.warning("No documents provided for chunking")
        return []

    # Create a local copy of supported extensions for this function call
    local_code_extensions = CODE_EXTENSIONS.copy()

    # Update supported extensions if provided
    if code_file_extensions:
        # Map extensions to languages (simplified mapping)
        ext_mapping = {
            ".py": "python",
            ".java": "java",
            ".cs": "c_sharp",
            ".ts": "typescript",
            ".tsx": "typescript",
        }
        for ext in code_file_extensions:
            if ext.lower() not in local_code_extensions:
                # Try to guess language from extension
                if ext.lower() in ext_mapping:
                    local_code_extensions[ext.lower()] = ext_mapping[ext.lower()]
                else:
                    logger.warning(f"Unsupported extension {ext}, will use traditional chunking")

    all_chunks = []

    if use_ast_chunking:
        # Separate code and text documents using local extensions
        code_docs, text_docs = detect_code_files(documents, local_code_extensions)

        # Process code files with AST chunking
        if code_docs:
            logger.info(f"Processing {len(code_docs)} code files with AST chunking")
            try:
                ast_chunks = create_ast_chunks(
                    code_docs, max_chunk_size=ast_chunk_size, chunk_overlap=ast_chunk_overlap
                )
                all_chunks.extend(ast_chunks)
                logger.info(f"Created {len(ast_chunks)} AST chunks from code files")
            except Exception as e:
                logger.error(f"AST chunking failed: {e}")
                if ast_fallback_traditional:
                    logger.info("Falling back to traditional chunking for code files")
                    traditional_code_chunks = create_traditional_chunks(
                        code_docs, chunk_size, chunk_overlap
                    )
                    all_chunks.extend(traditional_code_chunks)
                else:
                    raise

        # Process text files with traditional chunking
        if text_docs:
            logger.info(f"Processing {len(text_docs)} text files with traditional chunking")
            text_chunks = create_traditional_chunks(text_docs, chunk_size, chunk_overlap)
            all_chunks.extend(text_chunks)
            logger.info(f"Created {len(text_chunks)} traditional chunks from text files")
    else:
        # Use traditional chunking for all files
        logger.info(f"Processing {len(documents)} documents with traditional chunking")
        all_chunks = create_traditional_chunks(documents, chunk_size, chunk_overlap)

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
