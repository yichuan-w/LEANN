"""
Bridge module to expose AST chunking functionality to LEANN CLI.

This module imports and re-exports the AST chunking functionality from apps.chunking
so that the CLI can find it with a simple 'import chunking'.
"""

# Import all chunking utilities
try:
    from apps.chunking.utils import *
    from apps.chunking.ast_chunkers import *
    
    # Specifically import the AST chunkers
    from apps.chunking.ast_chunkers.go import GoASTChunker
    
    # Make them available at module level
    __all__ = ['GoASTChunker']
    
    # Flag to indicate AST chunking is available
    AST_CHUNKING_AVAILABLE = True
    
except ImportError as e:
    # Fallback if chunking modules aren't available
    AST_CHUNKING_AVAILABLE = False
    print(f"Warning: AST chunking not available: {e}")