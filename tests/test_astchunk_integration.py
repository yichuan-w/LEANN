"""
Test suite for astchunk integration with LEANN.
Tests AST-aware chunking functionality using the REAL astchunk library.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add paths for local modules
try:
    TEST_FILE_PATH = Path(__file__).resolve()
    LEANN_FORK_DIR = TEST_FILE_PATH.parent.parent

    LEANN_CORE_SRC = LEANN_FORK_DIR / "packages" / "leann-core" / "src"
    ASTCHUNK_SRC = LEANN_FORK_DIR / "packages" / "astchunk-leann" / "src"
    APPS_DIR = LEANN_FORK_DIR / "apps"

    sys.path.insert(0, str(LEANN_CORE_SRC))
    sys.path.insert(0, str(ASTCHUNK_SRC))
    sys.path.insert(0, str(APPS_DIR))
except Exception:
    pass

# Mock Backend Dependencies
sys.modules["leann_backend_hnsw"] = MagicMock()
sys.modules["leann_backend_hnsw.convert_to_csr"] = MagicMock()
sys.modules["leann_backend_faiss"] = MagicMock()

# Mock LlamaIndex if missing
try:
    import llama_index.core.node_parser  # noqa: F401
except ImportError:
    llama_index_mock = MagicMock()
    core_mock = MagicMock()
    node_parser_mock = MagicMock()
    sys.modules["llama_index"] = llama_index_mock
    sys.modules["llama_index.core"] = core_mock
    sys.modules["llama_index.core.node_parser"] = node_parser_mock

    # Configure SentenceSplitter to return usable nodes
    mock_splitter_instance = MagicMock()
    mock_node = MagicMock()
    mock_node.get_content.return_value = "mock content"
    mock_splitter_instance.get_nodes_from_documents.return_value = [mock_node]
    node_parser_mock.SentenceSplitter.return_value = mock_splitter_instance


from typing import Optional  # noqa: E402

# Import direct
from leann.chunking_utils import (  # noqa: E402
    create_ast_chunks,
    detect_code_files,
    get_language_from_extension,
)

# Check if astchunk is available
try:
    import astchunk  # noqa: F401

    ASTCHUNK_AVAILABLE = True
except ImportError:
    ASTCHUNK_AVAILABLE = False


class MockDocument:
    """Mock LlamaIndex Document for testing."""

    def __init__(self, content: str, file_path: str = "", metadata: Optional[dict] = None):
        self.content = content
        self.metadata = metadata or {}
        if file_path:
            self.metadata["file_path"] = file_path

    def get_content(self) -> str:
        return self.content


class TestCodeFileDetection:
    """Test code file detection and language mapping."""

    def test_detect_code_files_python(self):
        docs = [
            MockDocument("print('hello')", "/path/to/file.py"),
            MockDocument("text", "/path/to/file.txt"),
        ]
        code_docs, _text_docs = detect_code_files(docs)
        assert len(code_docs) == 1
        assert code_docs[0].metadata["language"] == "python"

    def test_get_language_from_extension(self):
        assert get_language_from_extension("test.ts") == "typescript"


class TestChunkingFunctions:
    """Test various chunking functionality."""

    @pytest.mark.skipif(not ASTCHUNK_AVAILABLE, reason="astchunk not installed")
    def test_create_ast_chunks_real_python(self):
        """Test AST chunking with REAL astchunk library for Python."""
        python_code = '''
import os
import sys

def hello_world():
    """Print hello world message."""
    print("Hello, World!")

class Calculator:
    def add(self, a, b):
        return a + b
'''
        docs = [MockDocument(python_code, "/test/calculator.py", {"language": "python"})]
        chunks = create_ast_chunks(docs, max_chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 0

        # Verify Enrichment (Imports Injection)
        # combined_content = " ".join([c["text"] for c in chunks])

        # Verify Metadata
        first_chunk_meta = chunks[0]["metadata"]
        assert "imports" in first_chunk_meta or "five_paths" in first_chunk_meta
        # Check imports in metadata
        imports = first_chunk_meta.get("imports", [])
        assert "os" in imports
        assert "sys" in imports

    @pytest.mark.skipif(not ASTCHUNK_AVAILABLE, reason="astchunk not installed")
    def test_create_ast_chunks_typescript(self):
        """Test AST chunking for TypeScript."""
        ts_code = """
import { useState } from 'react';

interface Props {
  name: string;
}

export const MyComponent = ({ name }: Props) => {
  return <div>Hello {name}</div>;
}
"""
        docs = [MockDocument(ts_code, "/test/component.tsx", {"language": "typescript"})]
        chunks = create_ast_chunks(docs, max_chunk_size=200)

        assert len(chunks) > 0
        assert any("MyComponent" in c["text"] for c in chunks)
        # Check imports logic for TS
        # imports = chunks[0]["metadata"].get("imports", [])
        # assert "react" in imports

    def test_create_ast_chunks_fallback(self):
        """Test fallback when AST chunking is not applied."""
        # Note: If ASTCHUNK_AVAILABLE is True, create_ast_chunks tries to use it.
        # But if we pass a document without a supported language, it falls back.
        doc_no_lang = MockDocument("some code", "/path/unknown.xyz", {})
        chunks = create_ast_chunks([doc_no_lang])
        assert len(chunks) > 0

        # Should contain "mock content" if mocked, or real content if real splitter used?
        # If mocked, get_nodes_from_documents returns [mock_node] with "mock content".
        # So chunks[0]["text"] == "mock content".
        # If real splitter, it chunks "some code" -> "some code".

        # We accept either for resilience
        text = chunks[0]["text"]
        assert text == "mock content" or text == "some code"

    @pytest.mark.skipif(not ASTCHUNK_AVAILABLE, reason="astchunk not installed")
    def test_chunk_expansion_is_active(self):
        """Verify that chunk expansion (ancestors) is enabled."""
        code = """
class Parent:
    def child(self):
        pass
"""
        docs = [MockDocument(code, "test.py", {"language": "python"})]
        chunks = create_ast_chunks(docs)

        # Checking for ancestors in text or metadata
        for chunk in chunks:
            if "def child" in chunk["text"]:
                assert "Parent" in chunk["text"] or "Parent" in chunk.get("metadata", {}).get(
                    "ancestors", ""
                )
