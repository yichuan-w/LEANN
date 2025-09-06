"""
Bridge module to expose AST chunking functionality to LEANN CLI.

This module imports and re-exports the AST chunking functionality from apps.chunking
so that the CLI can find it with a simple 'import chunking'.
"""

# Import all chunking utilities
try:
    # Specifically import the AST chunkers
    from apps.chunking.ast_chunkers.go import GoASTChunker
    from apps.chunking.utils import (
        create_ast_chunks,
        create_ast_chunks_with_local_chunkers,
        create_text_chunks,
        create_traditional_chunks,
        detect_code_files,
        get_language_from_extension,
    )

    # Make them available at module level
    __all__ = [
        "GoASTChunker",
        "create_ast_chunks",
        "create_ast_chunks_with_local_chunkers",
        "create_text_chunks",
        "create_traditional_chunks",
        "detect_code_files",
        "get_language_from_extension",
    ]

    # Flag to indicate AST chunking is available
    AST_CHUNKING_AVAILABLE = True

except ImportError as e:
    # Fallback if chunking modules aren't available
    AST_CHUNKING_AVAILABLE = False
    print(f"Warning: AST chunking not available: {e}")
